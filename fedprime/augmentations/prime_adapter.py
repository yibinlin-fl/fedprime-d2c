from __future__ import annotations

import torch

from fedprime.data.loaders import DatasetStats
from fedprime.utils.env import add_vendor_paths


class NormalizeLayer(torch.nn.Module):
    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def build_prime_module(stats: DatasetStats, cfg: dict) -> torch.nn.Module:
    add_vendor_paths()
    from utils.color_jitter import RandomSmoothColor
    from utils.diffeomorphism import Diffeo
    from utils.prime import GeneralizedPRIMEModule, PRIMEAugModule
    from utils.rand_filter import RandomFilter

    augmentations = []
    enabled = cfg.get("enabled_primitives", ["diffeo", "color", "filter"])

    if "diffeo" in enabled:
        augmentations.append(Diffeo(
            sT=cfg.get("diffeo", {}).get("sT", 1.0),
            rT=cfg.get("diffeo", {}).get("rT", 1.0),
            scut=cfg.get("diffeo", {}).get("scut", 1.0),
            rcut=cfg.get("diffeo", {}).get("rcut", 1.0),
            cutmin=cfg.get("diffeo", {}).get("cutmin", 2),
            cutmax=cfg.get("diffeo", {}).get("cutmax", 100),
            alpha=cfg.get("diffeo", {}).get("alpha", 1.0),
            stochastic=True,
        ))

    if "color" in enabled:
        augmentations.append(RandomSmoothColor(
            cut=cfg.get("color", {}).get("cut", 100),
            T=cfg.get("color", {}).get("T", 0.01),
            freq_bandwidth=cfg.get("color", {}).get("max_freqs", None),
            stochastic=True,
        ))

    if "filter" in enabled:
        augmentations.append(RandomFilter(
            kernel_size=cfg.get("filter", {}).get("kernel_size", 3),
            sigma=cfg.get("filter", {}).get("sigma", 4.0),
            stochastic=True,
        ))

    if not augmentations:
        raise ValueError("PRIME requires at least one primitive augmentation.")

    return GeneralizedPRIMEModule(
        preprocess=NormalizeLayer(stats.mean, stats.std),
        aug_module=PRIMEAugModule(augmentations),
        mixture_width=cfg.get("mixture_width", 3),
        mixture_depth=cfg.get("mixture_depth", -1),
        no_jsd=cfg.get("no_jsd", False),
        max_depth=cfg.get("max_depth", 3),
    )

