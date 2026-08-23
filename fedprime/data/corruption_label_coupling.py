from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from fedprime.data.loaders import (
    TwoViewTransform,
    _private_test_transform,
    _private_train_transform,
    _rahfl_augmix_view_transforms,
)
from fedprime.utils.env import add_vendor_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _largest_remainder_quotas(sizes: np.ndarray, total: int) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.ndim != 1 or np.any(sizes < 0):
        raise ValueError("sizes must be a one-dimensional non-negative array")
    if not 0 <= int(total) <= int(sizes.sum()):
        raise ValueError("total quota is outside the available capacity")
    if sizes.sum() == 0:
        return np.zeros_like(sizes)
    raw = sizes.astype(np.float64) * (float(total) / float(sizes.sum()))
    quotas = np.floor(raw).astype(np.int64)
    remaining = int(total) - int(quotas.sum())
    order = np.lexsort((np.arange(sizes.size), -(raw - quotas)))
    for idx in order:
        if remaining == 0:
            break
        if quotas[idx] < sizes[idx]:
            quotas[idx] += 1
            remaining -= 1
    if remaining != 0 or int(quotas.sum()) != int(total):
        raise RuntimeError("largest-remainder allocation failed")
    return quotas


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_metadata(metadata_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_root = metadata_root / "train"
    types = np.load(train_root / "corruption_type.npy").astype(np.int64, copy=False)
    severities = np.load(train_root / "corruption_severity.npy").astype(np.int64, copy=False)
    manifest = json.loads((train_root / "corruption_manifest.json").read_text(encoding="utf-8"))
    names = [str(item) for item in manifest["corruption_names"]]
    if types.shape != severities.shape:
        raise ValueError("corruption type/severity metadata shapes differ")
    if np.any(types < 0) or np.any(severities < 1) or np.any(severities > 4):
        raise ValueError("Phase-A requires corruption_rate=1 and legacy severities 1..4")
    return types, severities, names


def _make_disjoint_partition(
    *,
    num_samples: int,
    num_clients: int,
    samples_per_client: int,
    seed: int,
) -> dict[int, np.ndarray]:
    required = int(num_clients) * int(samples_per_client)
    if required > int(num_samples):
        raise ValueError("disjoint client partition requests more samples than available")
    shuffled = np.random.default_rng(int(seed)).permutation(int(num_samples))[:required]
    return {
        client_id: np.sort(
            shuffled[client_id * samples_per_client : (client_id + 1) * samples_per_client]
        ).astype(np.int64, copy=False)
        for client_id in range(int(num_clients))
    }


def _stratified_fit_audit(
    *,
    labels: np.ndarray,
    audit_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    audit_parts: list[np.ndarray] = []
    fit_parts: list[np.ndarray] = []
    class_ids = np.unique(labels)
    class_sizes = np.asarray(
        [np.count_nonzero(labels == int(class_id)) for class_id in class_ids],
        dtype=np.int64,
    )
    audit_total = int(round(float(audit_ratio) * int(labels.size)))
    audit_quotas = _largest_remainder_quotas(class_sizes, audit_total)
    if np.any(audit_quotas < 1) or np.any(audit_quotas >= class_sizes):
        raise ValueError("stratified fit/audit requires at least one fit and audit item per class")
    for class_id, audit_count in zip(class_ids, audit_quotas):
        indices = rng.permutation(np.flatnonzero(labels == int(class_id)))
        audit_count = int(audit_count)
        audit_parts.append(indices[:audit_count])
        fit_parts.append(indices[audit_count:])
    fit = np.sort(np.concatenate(fit_parts)).astype(np.int64, copy=False)
    audit = np.sort(np.concatenate(audit_parts)).astype(np.int64, copy=False)
    return fit, audit


def prepare_coupling_artifacts(
    *,
    data_root: str | Path,
    metadata_root: str | Path | None = None,
    artifact_root: str | Path,
    num_clients: int = 4,
    samples_per_client: int = 10000,
    audit_ratio: float = 0.10,
    noise_rate: float = 0.20,
    betas: tuple[float, ...] = (0.0, 4.0),
    partition_seed: int = 0,
    split_seed: int = 0,
    noise_seed: int = 0,
) -> dict[str, object]:
    """Create paired Independent/Coupled label manifests without copying images."""

    data_root = Path(data_root)
    metadata_root = data_root if metadata_root is None else Path(metadata_root)
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    train_root = data_root / "train"
    image_path = train_root / "random_corrupt_1.npy"
    clean_labels = np.load(train_root / "labels.npy").astype(np.int64, copy=False)
    corruption_types, severities, corruption_names = _load_metadata(metadata_root)
    if clean_labels.size != corruption_types.size:
        raise ValueError("training labels and corruption metadata have different lengths")

    partition = _make_disjoint_partition(
        num_samples=clean_labels.size,
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        seed=partition_seed,
    )
    partition_payload: dict[str, np.ndarray] = {}
    split_payload: dict[str, np.ndarray] = {}
    splits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for client_id, global_indices in partition.items():
        local_clean = clean_labels[global_indices]
        fit, audit = _stratified_fit_audit(
            labels=local_clean,
            audit_ratio=audit_ratio,
            seed=int(split_seed) * 1009 + client_id,
        )
        partition_payload[f"client_{client_id}_global"] = global_indices
        split_payload[f"client_{client_id}_fit"] = fit
        split_payload[f"client_{client_id}_audit"] = audit
        splits[client_id] = (fit, audit)
    np.savez_compressed(artifact_root / "partition_disjoint_iid.npz", **partition_payload)
    np.savez_compressed(artifact_root / "fit_audit_split.npz", **split_payload)

    shared_gumbels: dict[int, np.ndarray] = {}
    shared_destinations: dict[tuple[int, int, int], np.ndarray] = {}
    rng_score = np.random.default_rng(int(noise_seed))
    rng_destination = np.random.default_rng(int(noise_seed) + 7919)
    for client_id, global_indices in partition.items():
        shared_gumbels[client_id] = rng_score.gumbel(size=global_indices.size)

    summaries: dict[str, dict[str, object]] = {}
    transition_references: dict[int, np.ndarray] = {}
    for beta in betas:
        beta_token = f"beta{int(beta) if float(beta).is_integer() else beta}"
        regime_root = artifact_root / beta_token
        regime_root.mkdir(parents=True, exist_ok=True)
        stratum_rows: list[dict[str, object]] = []
        client_rows: list[dict[str, object]] = []
        transition_total = np.zeros((10, 10), dtype=np.int64)
        selected_severities: list[np.ndarray] = []
        clean_severities: list[np.ndarray] = []

        for client_id, global_indices in partition.items():
            local_clean = clean_labels[global_indices]
            local_types = corruption_types[global_indices]
            local_severity = severities[global_indices]
            fit, audit = splits[client_id]
            requested = int(round(float(noise_rate) * int(fit.size)))

            keys = sorted({(int(local_types[pos]), int(local_clean[pos])) for pos in fit})
            cells = [
                fit[(local_types[fit] == corruption_id) & (local_clean[fit] == class_id)]
                for corruption_id, class_id in keys
            ]
            quotas = _largest_remainder_quotas(
                np.asarray([cell.size for cell in cells], dtype=np.int64), requested
            )
            noisy_mask = np.zeros(global_indices.size, dtype=np.bool_)
            noisy_labels = local_clean.copy()

            for (corruption_id, class_id), cell, quota in zip(keys, cells, quotas):
                quota = int(quota)
                destination_key = (client_id, corruption_id, class_id)
                if destination_key not in shared_destinations:
                    draws = rng_destination.integers(0, 9, size=quota, dtype=np.int64)
                    shared_destinations[destination_key] = draws + (draws >= class_id)
                destinations = shared_destinations[destination_key]
                if destinations.size != quota:
                    raise RuntimeError("destination quota changed across beta regimes")

                normalized = (local_severity[cell].astype(np.float64) - 1.0) / 3.0
                scores = shared_gumbels[client_id][cell] + float(beta) * normalized
                chosen_order = np.argsort(-scores, kind="stable")[:quota]
                chosen = cell[chosen_order]
                noisy_mask[chosen] = True
                noisy_labels[chosen] = destinations

                counts = np.bincount(local_severity[cell], minlength=5)
                chosen_counts = np.bincount(local_severity[chosen], minlength=5)
                stratum_rows.append({
                    "client_id": client_id,
                    "corruption_id": corruption_id,
                    "corruption_name": corruption_names[corruption_id],
                    "true_class": class_id,
                    "n": int(cell.size),
                    "count_s1": int(counts[1]),
                    "count_s2": int(counts[2]),
                    "count_s3": int(counts[3]),
                    "count_s4": int(counts[4]),
                    "noise_quota": quota,
                    "noisy_s1": int(chosen_counts[1]),
                    "noisy_s2": int(chosen_counts[2]),
                    "noisy_s3": int(chosen_counts[3]),
                    "noisy_s4": int(chosen_counts[4]),
                })

            if int(noisy_mask[fit].sum()) != requested or bool(noisy_mask[audit].any()):
                raise RuntimeError("fit noise count or clean-audit invariant failed")
            if np.any(noisy_labels[noisy_mask] == local_clean[noisy_mask]):
                raise RuntimeError("a selected noisy label did not change")
            transition = np.zeros((10, 10), dtype=np.int64)
            np.add.at(transition, (local_clean[noisy_mask], noisy_labels[noisy_mask]), 1)
            transition_total += transition
            if client_id not in transition_references:
                transition_references[client_id] = transition.copy()
            elif not np.array_equal(transition_references[client_id], transition):
                raise RuntimeError("flip transition matrix changed across beta regimes")

            np.save(regime_root / f"client_{client_id}_labels.npy", noisy_labels)
            np.save(regime_root / f"client_{client_id}_noisy_mask.npy", noisy_mask)
            np.save(regime_root / f"client_{client_id}_transition.npy", transition)
            selected_severities.append(local_severity[fit][noisy_mask[fit]])
            clean_severities.append(local_severity[fit][~noisy_mask[fit]])
            client_rows.append({
                "client_id": client_id,
                "fit_samples": int(fit.size),
                "audit_samples": int(audit.size),
                "num_noisy": int(noisy_mask.sum()),
                "noise_rate": float(noisy_mask.sum() / fit.size),
                "mean_severity_noisy": float(local_severity[noisy_mask].mean()),
                "mean_severity_non_noisy_fit": float(local_severity[fit][~noisy_mask[fit]].mean()),
            })

        _write_csv(regime_root / "stratum_statistics.csv", stratum_rows)
        _write_csv(regime_root / "client_statistics.csv", client_rows)
        noisy_severity = np.concatenate(selected_severities)
        non_noisy_severity = np.concatenate(clean_severities)
        summary = {
            "beta": float(beta),
            "noise_rate": float(noise_rate),
            "num_noisy": int(noisy_severity.size),
            "mean_severity_noisy": float(noisy_severity.mean()),
            "mean_severity_non_noisy_fit": float(non_noisy_severity.mean()),
            "severity_gap": float(noisy_severity.mean() - non_noisy_severity.mean()),
            "transition_matrix": transition_total.tolist(),
        }
        (regime_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries[beta_token] = summary

    beta0 = summaries.get("beta0")
    beta4 = summaries.get("beta4")
    if beta0 is not None and beta4 is not None:
        if beta4["mean_severity_noisy"] <= beta0["mean_severity_noisy"]:
            raise RuntimeError("beta=4 did not increase noisy-sample severity")
        if beta4["transition_matrix"] != beta0["transition_matrix"]:
            raise RuntimeError("global flip transition matrix differs between beta=0 and beta=4")

    manifest = {
        "protocol": "rahfl_corruption_label_coupling_phase_a1a",
        "data_root": str(data_root.resolve()),
        "metadata_root": str(metadata_root.resolve()),
        "num_clients": int(num_clients),
        "samples_per_client": int(samples_per_client),
        "audit_ratio": float(audit_ratio),
        "noise_rate": float(noise_rate),
        "betas": [float(beta) for beta in betas],
        "partition_seed": int(partition_seed),
        "split_seed": int(split_seed),
        "noise_seed": int(noise_seed),
        "corruption_names": corruption_names,
        "checksums": {
            "train_images": _sha256(image_path),
            "train_clean_labels": _sha256(train_root / "labels.npy"),
            "corruption_type": _sha256(metadata_root / "train" / "corruption_type.npy"),
            "corruption_severity": _sha256(metadata_root / "train" / "corruption_severity.npy"),
            "partition": _sha256(artifact_root / "partition_disjoint_iid.npz"),
            "fit_audit_split": _sha256(artifact_root / "fit_audit_split.npz"),
        },
        "regimes": summaries,
    }
    (artifact_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


class CouplingClientDataset(data.Dataset):
    def __init__(
        self,
        *,
        images_path: str | Path,
        global_indices: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        self.images = np.load(Path(images_path), mmap_mode="r")
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        if self.global_indices.size != self.labels.size:
            raise ValueError("global indices and local labels differ in length")

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int):
        image = np.array(self.images[int(self.global_indices[index])], copy=True)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class CouplingTestDataset(data.Dataset):
    def __init__(self, *, data_root: str | Path, transform=None):
        root = Path(data_root) / "test"
        self.images = np.load(root / "random_corrupt_1.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy").astype(np.int64, copy=False)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int):
        image = np.array(self.images[index], copy=True)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


@dataclass(frozen=True)
class CouplingLoaders:
    fit_augmix: list[data.DataLoader]
    pretrain: list[data.DataLoader]
    audit: dict[int, data.DataLoader]
    test: data.DataLoader
    class_counts: dict[int, torch.Tensor]
    contract: dict[str, object]


def build_coupling_loaders(
    *,
    data_root: str | Path,
    artifact_root: str | Path,
    beta: float,
    num_clients: int,
    train_batch_size: int,
    test_batch_size: int,
    audit_batch_size: int,
    num_workers: int,
    augmix_module: str = "jsd",
    expected_noise_rate: float | None = None,
    expected_audit_ratio: float | None = None,
) -> CouplingLoaders:
    add_vendor_paths()
    from Dataset.dataaug import AugMixDataset

    data_root = Path(data_root)
    artifact_root = Path(artifact_root)
    experiment_manifest = json.loads(
        (artifact_root / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    if int(experiment_manifest["num_clients"]) != int(num_clients):
        raise ValueError("configured client count differs from the frozen coupling artifact")
    if expected_noise_rate is not None and not np.isclose(
        float(experiment_manifest["noise_rate"]), float(expected_noise_rate)
    ):
        raise ValueError("configured noise rate differs from the frozen coupling artifact")
    if expected_audit_ratio is not None and not np.isclose(
        float(experiment_manifest["audit_ratio"]), float(expected_audit_ratio)
    ):
        raise ValueError("configured audit ratio differs from the frozen coupling artifact")
    beta_token = f"beta{int(beta) if float(beta).is_integer() else beta}"
    regime_root = artifact_root / beta_token
    partition_archive = np.load(artifact_root / "partition_disjoint_iid.npz")
    split_archive = np.load(artifact_root / "fit_audit_split.npz")
    clean_global = np.load(data_root / "train" / "labels.npy").astype(np.int64, copy=False)
    images_path = data_root / "train" / "random_corrupt_1.npy"
    base, weak, preprocess = _rahfl_augmix_view_transforms("cifar10")
    fit_loaders: list[data.DataLoader] = []
    pretrain_loaders: list[data.DataLoader] = []
    audit_loaders: dict[int, data.DataLoader] = {}
    class_counts: dict[int, torch.Tensor] = {}
    contract_clients: list[dict[str, object]] = []

    for client_id in range(int(num_clients)):
        global_indices = np.asarray(partition_archive[f"client_{client_id}_global"], dtype=np.int64)
        fit = np.asarray(split_archive[f"client_{client_id}_fit"], dtype=np.int64)
        audit = np.asarray(split_archive[f"client_{client_id}_audit"], dtype=np.int64)
        noisy_labels = np.load(regime_root / f"client_{client_id}_labels.npy").astype(
            np.int64, copy=False
        )
        noisy_mask = np.load(regime_root / f"client_{client_id}_noisy_mask.npy").astype(
            np.bool_, copy=False
        )
        clean_labels = clean_global[global_indices]
        if noisy_labels.size != global_indices.size or noisy_mask.size != global_indices.size:
            raise ValueError(f"client {client_id} coupling artifact length mismatch")
        if bool(noisy_mask[audit].any()) or not np.array_equal(
            noisy_labels[audit], clean_labels[audit]
        ):
            raise ValueError(f"client {client_id} trusted audit is not clean")

        noisy_augmix_base = CouplingClientDataset(
            images_path=images_path,
            global_indices=global_indices,
            labels=noisy_labels,
            transform=TwoViewTransform(base, weak),
        )
        fit_augmix = AugMixDataset(
            data.Subset(noisy_augmix_base, fit.tolist()),
            preprocess,
            jsd_or_nojsd=augmix_module,
        )
        fit_loaders.append(data.DataLoader(
            fit_augmix,
            batch_size=int(train_batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
        ))

        noisy_pretrain = CouplingClientDataset(
            images_path=images_path,
            global_indices=global_indices,
            labels=noisy_labels,
            transform=_private_train_transform(raw_for_prime=False),
        )
        pretrain_loaders.append(data.DataLoader(
            data.Subset(noisy_pretrain, fit.tolist()),
            batch_size=int(train_batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
        ))

        clean_audit = CouplingClientDataset(
            images_path=images_path,
            global_indices=global_indices,
            labels=clean_labels,
            transform=_private_test_transform("cifar10"),
        )
        audit_loaders[client_id] = data.DataLoader(
            data.Subset(clean_audit, audit.tolist()),
            batch_size=int(audit_batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
        )
        class_counts[client_id] = torch.bincount(
            torch.as_tensor(noisy_labels[fit], dtype=torch.long), minlength=10
        ).float()
        label_path = regime_root / f"client_{client_id}_labels.npy"
        contract_clients.append({
            "client_id": client_id,
            "fit_samples": int(fit.size),
            "audit_samples": int(audit.size),
            "noisy_fit_samples": int(noisy_mask[fit].sum()),
            "label_manifest_sha256": _sha256(label_path),
            "pretrain_label_source": str(label_path.resolve()),
            "local_ce_dcl_label_source": str(label_path.resolve()),
            "audit_label_source": "frozen clean train labels restricted to client audit indices",
        })

    test_loader = data.DataLoader(
        CouplingTestDataset(data_root=data_root, transform=_private_test_transform("cifar10")),
        batch_size=int(test_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return CouplingLoaders(
        fit_augmix=fit_loaders,
        pretrain=pretrain_loaders,
        audit=audit_loaders,
        test=test_loader,
        class_counts=class_counts,
        contract={
            "beta": float(beta),
            "artifact_protocol": experiment_manifest["protocol"],
            "noise_rate": float(experiment_manifest["noise_rate"]),
            "audit_ratio": float(experiment_manifest["audit_ratio"]),
            "fit_role": "gradient-bearing pretrain and AugMix/JSD/DCL local updates",
            "audit_role": "clean routing evaluation only; never passed to an optimizer",
            "test_role": "clean final reporting only",
            "clients": contract_clients,
        },
    )
