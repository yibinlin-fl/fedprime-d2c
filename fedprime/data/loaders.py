from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch.autograd import Variable
from torchvision import datasets
from PIL import ImageFilter

from fedprime.utils.env import add_vendor_paths


CIFAR10_MEAN = [125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0]
CIFAR10_STD = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]
CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


@dataclass(frozen=True)
class DatasetStats:
    mean: list[float]
    std: list[float]


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class TwoViewTransform:
    def __init__(self, transform, weak_transform):
        self.transform = transform
        self.weak_transform = weak_transform

    def __call__(self, x):
        return self.transform(x), self.weak_transform(x)


def dataset_stats(name: str) -> DatasetStats:
    if name.lower() == "cifar10":
        return DatasetStats(CIFAR10_MEAN, CIFAR10_STD)
    if name.lower() == "cifar100":
        return DatasetStats(CIFAR100_MEAN, CIFAR100_STD)
    raise ValueError(f"Unsupported dataset stats: {name}")


def normalize_batch(images: torch.Tensor, stats: DatasetStats) -> torch.Tensor:
    mean = torch.tensor(stats.mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(stats.std, device=images.device).view(1, -1, 1, 1)
    return (images - mean) / std


def _private_train_transform(raw_for_prime: bool):
    if raw_for_prime:
        return T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4),
                mode="reflect",
            ).data.squeeze()),
            T.ToPILImage(),
            T.ColorJitter(),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4),
            mode="reflect",
        ).data.squeeze()),
        T.ToPILImage(),
        T.ColorJitter(),
        T.RandomCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _prime_dcl_train_transform():
    base = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4),
            mode="reflect",
        ).data.squeeze()),
        T.ToPILImage(),
        T.ColorJitter(),
        T.RandomCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    weak = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4),
            mode="reflect",
        ).data.squeeze()),
        T.ToPILImage(),
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur()], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    return TwoViewTransform(base, weak)


def _private_test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def load_private_labels(cifar10c_root: str | Path, corrupt_rate: float | int) -> np.ndarray:
    add_vendor_paths()
    from Dataset.init_dataset import CIFAR_C

    ds = CIFAR_C(str(cifar10c_root), train=True, transform=None, corrupt_rate=corrupt_rate)
    return np.asarray(ds.target)


def partition_private_data(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    partition: str,
    dirichlet_alpha: float,
    max_samples_per_client: int | None = None,
) -> dict[int, list[int]]:
    add_vendor_paths()
    from Dataset.sampling import iid_sampling, non_iid_dirichlet_sampling

    if partition == "iid":
        mapping = iid_sampling(labels, num_clients)
    elif partition == "dirichlet":
        mapping = non_iid_dirichlet_sampling(labels, num_classes, num_clients, dirichlet_alpha)
    else:
        raise ValueError(f"Unknown partition: {partition}")

    if max_samples_per_client is not None:
        mapping = {
            client_id: list(indices[:max_samples_per_client])
            for client_id, indices in mapping.items()
        }
    return mapping


def build_private_loaders(
    cifar10c_root: str | Path,
    dataidx_map: dict[int, list[int]],
    train_batch_size: int,
    test_batch_size: int,
    corrupt_rate: float | int,
    test_corrupt_rate: float | int,
    num_workers: int,
    raw_for_prime: bool = True,
):
    add_vendor_paths()
    from Dataset.init_dataset import CIFAR_C

    train_loaders = []
    for client_id in sorted(dataidx_map):
        train_ds = CIFAR_C(
            str(cifar10c_root),
            dataidxs=dataidx_map[client_id],
            train=True,
            transform=_private_train_transform(raw_for_prime),
            corrupt_rate=corrupt_rate,
        )
        train_loaders.append(data.DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ))

    test_ds = CIFAR_C(
        str(cifar10c_root),
        train=False,
        transform=_private_test_transform(),
        corrupt_rate=test_corrupt_rate,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loaders, test_loader


def build_prime_dcl_private_loaders(
    cifar10c_root: str | Path,
    dataidx_map: dict[int, list[int]],
    train_batch_size: int,
    test_batch_size: int,
    corrupt_rate: float | int,
    test_corrupt_rate: float | int,
    num_workers: int,
):
    add_vendor_paths()
    from Dataset.init_dataset import CIFAR_C

    train_loaders = []
    for client_id in sorted(dataidx_map):
        train_ds = CIFAR_C(
            str(cifar10c_root),
            dataidxs=dataidx_map[client_id],
            train=True,
            transform=_prime_dcl_train_transform(),
            corrupt_rate=corrupt_rate,
        )
        train_loaders.append(data.DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ))

    test_ds = CIFAR_C(
        str(cifar10c_root),
        train=False,
        transform=_private_test_transform(),
        corrupt_rate=test_corrupt_rate,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loaders, test_loader


def build_augmix_private_loaders(
    cifar10c_root: str | Path,
    dataidx_map: dict[int, list[int]],
    train_batch_size: int,
    test_batch_size: int,
    corrupt_rate: float | int,
    test_corrupt_rate: float | int,
    num_workers: int,
    augmix_module: str = "jsd",
):
    add_vendor_paths()
    from Dataset.utils import get_dataloader

    train_loaders = []
    train_datasets = []
    for client_id in sorted(dataidx_map):
        train_dl, _, train_ds, _ = get_dataloader(
            dataset="cifar10",
            datadir=str(cifar10c_root),
            train_bs=train_batch_size,
            test_bs=test_batch_size,
            dataidxs=dataidx_map[client_id],
            corrupt_rate=corrupt_rate,
            test_corrupt_rate=test_corrupt_rate,
            augmix_module=augmix_module,
        )
        train_loaders.append(train_dl)
        train_datasets.append(train_ds)

    _, test_loader, _, test_ds = get_dataloader(
        dataset="cifar10",
        datadir=str(cifar10c_root),
        train_bs=train_batch_size,
        test_bs=test_batch_size,
        corrupt_rate=corrupt_rate,
        test_corrupt_rate=test_corrupt_rate,
        augmix_module=None,
    )
    return train_loaders, test_loader, train_datasets, test_ds


def build_public_loader(
    cifar100_root: str | Path,
    public_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    download: bool,
):
    transform = T.Compose([T.ToTensor()])
    ds = datasets.CIFAR100(str(cifar100_root), train=True, transform=transform, download=download)
    rng = np.random.default_rng(seed)
    indices = rng.choice(np.arange(len(ds)), size=min(public_size, len(ds)), replace=False)
    subset = data.Subset(ds, indices.tolist())
    loader = data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader
