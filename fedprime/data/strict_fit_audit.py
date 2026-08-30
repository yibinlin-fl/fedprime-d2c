from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from fedprime.data.loaders import (
    CorruptionSkewClientDataset,
    TwoViewTransform,
    _private_test_transform,
    _prepared_private_dataset_name,
    _rahfl_augmix_view_transforms,
)
from fedprime.utils.env import add_vendor_paths


@dataclass(frozen=True)
class StrictClientSplit:
    """Fixed client-local fit/audit split and its deterministic probe dataset."""

    client_id: int
    fit_indices: np.ndarray
    audit_indices: np.ndarray
    labels: np.ndarray
    fit_loader: data.DataLoader
    probe_dataset: data.Dataset

    def class_indices(self, class_id: int, *, split: str) -> np.ndarray:
        indices = self.fit_indices if split == "fit" else self.audit_indices
        return indices[self.labels[indices] == int(class_id)]


def stratified_fit_audit_indices(
    labels: np.ndarray,
    *,
    audit_ratio: float,
    min_audit_per_class: int,
    min_fit_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a client by class while preserving extremely rare classes for fit."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0.0 < float(audit_ratio) < 1.0:
        raise ValueError("audit_ratio must be between zero and one")
    if int(min_audit_per_class) < 1:
        raise ValueError("min_audit_per_class must be positive")
    if int(min_fit_per_class) < 1:
        raise ValueError("min_fit_per_class must be positive")

    rng = np.random.default_rng(int(seed))
    fit_parts: list[np.ndarray] = []
    audit_parts: list[np.ndarray] = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == int(class_id))
        class_indices = rng.permutation(class_indices)
        count = int(class_indices.size)
        if count < int(min_audit_per_class) + int(min_fit_per_class):
            fit_parts.append(class_indices)
            continue

        proposed = max(
            int(min_audit_per_class),
            int(round(float(audit_ratio) * count)),
        )
        audit_count = min(proposed, count - int(min_fit_per_class))
        audit_parts.append(class_indices[:audit_count])
        fit_parts.append(class_indices[audit_count:])

    fit = np.sort(np.concatenate(fit_parts)).astype(np.int64, copy=False)
    if audit_parts:
        audit = np.sort(np.concatenate(audit_parts)).astype(np.int64, copy=False)
    else:
        audit = np.empty(0, dtype=np.int64)
    return fit, audit


def _save_split_file(
    path: Path,
    splits: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    audit_ratio: float,
    min_audit_per_class: int,
    min_fit_per_class: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "audit_ratio": np.asarray([float(audit_ratio)], dtype=np.float64),
        "min_audit_per_class": np.asarray([int(min_audit_per_class)], dtype=np.int64),
        "min_fit_per_class": np.asarray([int(min_fit_per_class)], dtype=np.int64),
        "seed": np.asarray([int(seed)], dtype=np.int64),
    }
    for client_id, (fit, audit) in splits.items():
        payload[f"client_{client_id}_fit"] = np.asarray(fit, dtype=np.int64)
        payload[f"client_{client_id}_audit"] = np.asarray(audit, dtype=np.int64)
    np.savez_compressed(path, **payload)


def _load_split_file(
    path: Path,
    *,
    num_clients: int,
    audit_ratio: float,
    min_audit_per_class: int,
    min_fit_per_class: int,
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "audit_ratio": float(audit_ratio),
            "min_audit_per_class": int(min_audit_per_class),
            "min_fit_per_class": int(min_fit_per_class),
            "seed": int(seed),
        }
        for key, value in expected.items():
            if key not in archive:
                raise ValueError(f"Split metadata '{key}' is missing in {path}")
            stored = archive[key][0].item()
            if stored != value:
                raise ValueError(
                    f"Split metadata mismatch for {key}: stored={stored}, expected={value}"
                )
        splits = {}
        for client_id in range(int(num_clients)):
            fit_key = f"client_{client_id}_fit"
            audit_key = f"client_{client_id}_audit"
            if fit_key not in archive or audit_key not in archive:
                raise KeyError(f"Missing {fit_key}/{audit_key} in {path}")
            splits[client_id] = (
                np.asarray(archive[fit_key], dtype=np.int64),
                np.asarray(archive[audit_key], dtype=np.int64),
            )
    return splits


def _validate_split(
    *,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    audit_indices: np.ndarray,
    client_id: int,
) -> None:
    size = int(len(labels))
    fit = np.asarray(fit_indices, dtype=np.int64)
    audit = np.asarray(audit_indices, dtype=np.int64)
    if np.any(fit < 0) or np.any(fit >= size) or np.any(audit < 0) or np.any(audit >= size):
        raise ValueError(f"Client {client_id} split contains an out-of-range index")
    if np.intersect1d(fit, audit).size:
        raise ValueError(f"Client {client_id} fit/audit indices overlap")
    combined = np.sort(np.concatenate([fit, audit]))
    if not np.array_equal(combined, np.arange(size, dtype=np.int64)):
        raise ValueError(f"Client {client_id} fit/audit split does not cover the training set")


def build_strict_fit_audit_loaders(
    *,
    root: str | Path,
    num_clients: int,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    split_path: str | Path,
    audit_ratio: float,
    min_audit_per_class: int,
    min_fit_per_class: int,
    seed: int,
    num_classes: int = 10,
    augmix_module: str = "jsd",
    loader_seed: int | None = None,
) -> tuple[
    list[data.DataLoader],
    data.DataLoader,
    dict[int, StrictClientSplit],
    dict[int, torch.Tensor],
]:
    """Build disjoint fit/audit data without touching the final-test labels."""

    add_vendor_paths()
    from Dataset.dataaug import AugMixDataset

    root = Path(root)
    split_path = Path(split_path)
    labels_by_client = {
        client_id: np.load(root / f"client_{client_id}" / "train_labels.npy").astype(
            np.int64,
            copy=False,
        )
        for client_id in range(int(num_clients))
    }
    if split_path.is_file():
        raw_splits = _load_split_file(
            split_path,
            num_clients=num_clients,
            audit_ratio=audit_ratio,
            min_audit_per_class=min_audit_per_class,
            min_fit_per_class=min_fit_per_class,
            seed=seed,
        )
        print(f"[setup] loaded fixed strict fit/audit split: {split_path}", flush=True)
    else:
        raw_splits = {
            client_id: stratified_fit_audit_indices(
                labels,
                audit_ratio=audit_ratio,
                min_audit_per_class=min_audit_per_class,
                min_fit_per_class=min_fit_per_class,
                seed=int(seed) * 1009 + client_id,
            )
            for client_id, labels in labels_by_client.items()
        }
        _save_split_file(
            split_path,
            raw_splits,
            audit_ratio=audit_ratio,
            min_audit_per_class=min_audit_per_class,
            min_fit_per_class=min_fit_per_class,
            seed=seed,
        )
        print(f"[setup] wrote fixed strict fit/audit split: {split_path}", flush=True)

    dataset_name = _prepared_private_dataset_name(root)
    base, weak, preprocess = _rahfl_augmix_view_transforms(dataset_name)
    fit_loaders: list[data.DataLoader] = []
    client_splits: dict[int, StrictClientSplit] = {}
    class_counts: dict[int, torch.Tensor] = {}
    for client_id in range(int(num_clients)):
        labels = labels_by_client[client_id]
        fit_indices, audit_indices = raw_splits[client_id]
        _validate_split(
            labels=labels,
            fit_indices=fit_indices,
            audit_indices=audit_indices,
            client_id=client_id,
        )

        train_base = CorruptionSkewClientDataset(
            root=root,
            client_id=client_id,
            train=True,
            transform=TwoViewTransform(base, weak),
            return_corruption=False,
        )
        fit_augmix = AugMixDataset(
            data.Subset(train_base, fit_indices.tolist()),
            preprocess,
            jsd_or_nojsd=augmix_module,
        )
        loader_generator = None
        if loader_seed is not None:
            loader_generator = torch.Generator()
            loader_generator.manual_seed(int(loader_seed) * 1009 + client_id)
        fit_loader = data.DataLoader(
            fit_augmix,
            batch_size=int(train_batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
            generator=loader_generator,
        )
        probe_dataset = CorruptionSkewClientDataset(
            root=root,
            client_id=client_id,
            train=True,
            transform=_private_test_transform(dataset_name),
            return_corruption=False,
        )
        client_splits[client_id] = StrictClientSplit(
            client_id=client_id,
            fit_indices=fit_indices,
            audit_indices=audit_indices,
            labels=labels,
            fit_loader=fit_loader,
            probe_dataset=probe_dataset,
        )
        fit_loaders.append(fit_loader)
        class_counts[client_id] = torch.bincount(
            torch.as_tensor(labels[fit_indices], dtype=torch.long),
            minlength=int(num_classes),
        ).float()

        fit_counts = np.bincount(labels[fit_indices], minlength=int(num_classes))
        audit_counts = np.bincount(labels[audit_indices], minlength=int(num_classes))
        print(
            f"[setup] strict split client={client_id} fit={len(fit_indices)} "
            f"audit={len(audit_indices)} fit_per_class={fit_counts.tolist()} "
            f"audit_per_class={audit_counts.tolist()}",
            flush=True,
        )

    test_dataset = CorruptionSkewClientDataset(
        root=root,
        train=False,
        transform=_private_test_transform(dataset_name),
        return_corruption=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=int(test_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return fit_loaders, test_loader, client_splits, class_counts


def build_client_audit_loaders(
    client_splits: dict[int, StrictClientSplit],
    *,
    batch_size: int,
    num_workers: int,
) -> dict[int, data.DataLoader]:
    """Build deterministic client-private audit loaders for routing only."""

    loaders: dict[int, data.DataLoader] = {}
    for client_id, split in sorted(client_splits.items()):
        if split.audit_indices.size == 0:
            raise ValueError(f"Client {client_id} has no audit samples for strict routing")
        dataset = data.Subset(split.probe_dataset, split.audit_indices.tolist())
        loaders[int(client_id)] = data.DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
