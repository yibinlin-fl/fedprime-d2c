from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from fedprime.data.loaders import (
    CorruptionSkewClientDataset,
    TwoViewTransform,
    _private_test_transform,
    _rahfl_augmix_view_transforms,
)
from fedprime.utils.env import add_vendor_paths


class EnvironmentAnnotatedDataset(data.Dataset):
    """Attach the generator-provided base corruption group to AugMix samples."""

    def __init__(
        self,
        dataset: data.Dataset,
        environment_ids,
        environment_features=None,
        confidence=None,
    ) -> None:
        if len(dataset) != len(environment_ids):
            raise ValueError("dataset and environment_ids must have the same length")
        self.dataset = dataset
        self.environment_ids = np.asarray(environment_ids, dtype=np.int64)
        self.environment_features = (
            None if environment_features is None else np.asarray(environment_features, dtype=np.float32)
        )
        self.confidence = None if confidence is None else np.asarray(confidence, dtype=np.float32)
        if self.environment_features is not None and len(self.environment_features) != len(dataset):
            raise ValueError("environment_features and dataset must have the same length")
        if self.confidence is not None and len(self.confidence) != len(dataset):
            raise ValueError("confidence and dataset must have the same length")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        images, label = self.dataset[index]
        environment_id = int(self.environment_ids[index])
        if self.environment_features is None:
            return images, label, environment_id
        return (
            images,
            label,
            environment_id,
            torch.as_tensor(self.environment_features[index], dtype=torch.float32),
            float(self.confidence[index]) if self.confidence is not None else 1.0,
        )


def build_fedease_oracle_augmix_loaders(
    root: str | Path,
    num_clients: int,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    augmix_module: str = "jsd",
    environment_annotations: dict[int, dict[str, np.ndarray]] | None = None,
):
    """Build CLE-HFL AugMix loaders with oracle or PEW environment annotations."""

    add_vendor_paths()
    from Dataset.dataaug import AugMixDataset

    base, weak, preprocess = _rahfl_augmix_view_transforms()
    train_loaders = []
    train_datasets = []
    for client_id in range(num_clients):
        base_dataset = CorruptionSkewClientDataset(
            root=root,
            client_id=client_id,
            train=True,
            transform=TwoViewTransform(base, weak),
            return_corruption=False,
        )
        augmix_dataset = AugMixDataset(base_dataset, preprocess, jsd_or_nojsd=augmix_module)
        annotation = (environment_annotations or {}).get(client_id)
        train_dataset = EnvironmentAnnotatedDataset(
            augmix_dataset,
            base_dataset.corruption_ids if annotation is None else annotation["environment_ids"],
            environment_features=None if annotation is None else annotation.get("embedding"),
            confidence=None if annotation is None else annotation.get("confidence"),
        )
        train_loaders.append(data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ))
        train_datasets.append(train_dataset)

    test_dataset = CorruptionSkewClientDataset(
        root=root,
        train=False,
        transform=_private_test_transform(),
        return_corruption=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loaders, test_loader, train_datasets, test_dataset


def build_fedease_fit_augmix_loaders(
    root: str | Path,
    *,
    client_splits,
    train_batch_size: int,
    num_workers: int,
    augmix_module: str = "jsd",
    environment_annotations: dict[int, dict[str, np.ndarray]] | None = None,
) -> list[data.DataLoader]:
    """Build FedEASE loaders restricted to a persisted client-local fit split."""

    add_vendor_paths()
    from Dataset.dataaug import AugMixDataset

    root = Path(root)
    base, weak, preprocess = _rahfl_augmix_view_transforms()
    loaders: list[data.DataLoader] = []
    for client_id, split in sorted(client_splits.items()):
        base_dataset = CorruptionSkewClientDataset(
            root=root,
            client_id=int(client_id),
            train=True,
            transform=TwoViewTransform(base, weak),
            return_corruption=False,
        )
        augmix_dataset = AugMixDataset(base_dataset, preprocess, jsd_or_nojsd=augmix_module)
        annotation = (environment_annotations or {}).get(int(client_id))
        annotated = EnvironmentAnnotatedDataset(
            augmix_dataset,
            base_dataset.corruption_ids if annotation is None else annotation["environment_ids"],
            environment_features=None if annotation is None else annotation.get("embedding"),
            confidence=None if annotation is None else annotation.get("confidence"),
        )
        fit_dataset = data.Subset(annotated, split.fit_indices.tolist())
        loaders.append(data.DataLoader(
            fit_dataset,
            batch_size=int(train_batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
        ))
    return loaders


def load_client_class_environment_counts(
    root: str | Path,
    num_clients: int,
    num_classes: int,
    num_environments: int,
    environment_annotations: dict[int, dict[str, np.ndarray]] | None = None,
    fit_indices: dict[int, np.ndarray] | None = None,
) -> dict[int, torch.Tensor]:
    """Load exact client-local group counts without exposing them to the server."""

    root = Path(root)
    counts = {}
    for client_id in range(num_clients):
        client_root = root / f"client_{client_id}"
        labels = np.load(client_root / "train_labels.npy").astype(np.int64)
        annotation = (environment_annotations or {}).get(client_id)
        environment_ids = (
            np.load(client_root / "train_corruption_ids.npy").astype(np.int64)
            if annotation is None
            else np.asarray(annotation["environment_ids"], dtype=np.int64)
        )
        selected = None if fit_indices is None else np.asarray(fit_indices[client_id], dtype=np.int64)
        if selected is not None:
            labels = labels[selected]
            environment_ids = environment_ids[selected]
        if labels.shape != environment_ids.shape:
            raise ValueError(f"client {client_id} labels/environment IDs have different shapes")
        if ((environment_ids < 0) | (environment_ids >= num_environments)).any():
            raise ValueError(f"client {client_id} has an out-of-range environment ID")
        flat = labels * num_environments + environment_ids
        matrix = np.bincount(
            flat,
            minlength=num_classes * num_environments,
        ).reshape(num_classes, num_environments)
        counts[client_id] = torch.as_tensor(matrix, dtype=torch.float32)
    return counts


class FedEASEEvaluationDataset(data.Dataset):
    """Evaluation split stored as NumPy images, labels, and optional environment IDs."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.images = np.load(directory / "test_images.npy")
        self.labels = np.load(directory / "test_labels.npy").astype(np.int64)
        environment_path = directory / "test_corruption_ids.npy"
        self.environment_ids = (
            np.load(environment_path).astype(np.int64)
            if environment_path.exists()
            else np.full(len(self.labels), -1, dtype=np.int64)
        )
        self.transform = _private_test_transform()

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int):
        return self.transform(self.images[index]), int(self.labels[index]), int(self.environment_ids[index])


def build_fedease_evaluation_loaders(
    root: str | Path,
    *,
    num_clients: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, data.DataLoader | dict[int, data.DataLoader]]:
    """Build every available clean/same/random/swapped/unseen evaluation split."""

    root = Path(root)
    result: dict[str, data.DataLoader | dict[int, data.DataLoader]] = {}
    common = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    shared_directories = {
        "clean": root / "test_clean",
        "random": root / "test_balanced",
        "unseen": root / "test_unseen",
    }
    for split, directory in shared_directories.items():
        if (directory / "test_images.npy").is_file():
            result[split] = data.DataLoader(FedEASEEvaluationDataset(directory), **common)

    for split in ("same", "swapped"):
        split_root = root / f"test_{split}"
        client_loaders = {}
        for client_id in range(num_clients):
            directory = split_root / f"client_{client_id}"
            if (directory / "test_images.npy").is_file():
                client_loaders[client_id] = data.DataLoader(FedEASEEvaluationDataset(directory), **common)
        if len(client_loaders) == num_clients:
            result[split] = client_loaders
    return result
