from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F


DEFAULT_OPERATORS = (
    "identity",
    "gaussian_noise",
    "blur",
    "brightness",
    "contrast",
    "pixelate",
    "haze",
)


def _seeded_generator(images: torch.Tensor, seed: int) -> torch.Generator:
    device = images.device if images.is_cuda else torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _apply_operator(
    images: torch.Tensor,
    operator: str,
    config: dict,
    seed: int,
) -> torch.Tensor:
    x = images.detach()
    operator = operator.lower()
    if operator == "identity":
        out = x
    elif operator == "gaussian_noise":
        generator = _seeded_generator(x, seed)
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        out = x + float(config.get("noise_std", 0.08)) * noise
    elif operator == "blur":
        kernel = max(1, int(config.get("blur_kernel", 3)))
        if kernel % 2 == 0:
            kernel += 1
        out = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    elif operator == "brightness":
        strength = float(config.get("brightness_strength", 0.20))
        factor = 1.0 - strength if seed % 2 == 0 else 1.0 + strength
        out = x * factor
    elif operator == "contrast":
        strength = float(config.get("contrast_strength", 0.30))
        factor = max(0.05, 1.0 - strength)
        mean = x.mean(dim=(2, 3), keepdim=True)
        out = (x - mean) * factor + mean
    elif operator == "pixelate":
        size = max(2, min(int(config.get("pixelate_size", 16)), x.shape[-1]))
        small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        out = F.interpolate(small, size=x.shape[-2:], mode="nearest")
    elif operator == "haze":
        strength = float(config.get("haze_strength", 0.20))
        out = (1.0 - strength) * x + strength * torch.ones_like(x) * 0.85
    else:
        raise ValueError(f"Unknown counterfactual operator: {operator}")
    return out.clamp(0.0, 1.0)


def select_operators(
    operators: Sequence[str] | None,
    num_views: int,
    seed: int,
) -> list[str]:
    if num_views < 1:
        raise ValueError(f"num_views must be positive, got {num_views}")
    pool = [str(name).lower() for name in (operators or DEFAULT_OPERATORS)]
    if not pool:
        raise ValueError("At least one counterfactual operator is required.")

    non_identity = [name for name in pool if name != "identity"]
    selected = ["identity"]
    if num_views == 1:
        return selected
    if not non_identity:
        return selected * num_views

    rng = random.Random(int(seed))
    while len(selected) < num_views:
        cycle = list(non_identity)
        rng.shuffle(cycle)
        selected.extend(cycle)
    return selected[:num_views]


def build_counterfactual_views(
    images: torch.Tensor,
    config: dict | None = None,
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[str]]:
    """Build deterministic, label-independent counterfactual views for one batch."""

    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got shape {tuple(images.shape)}")
    config = config or {}
    num_views = int(config.get("num_views", 3))
    names = select_operators(config.get("operators"), num_views, seed)
    views = [
        _apply_operator(images, name, config, seed + view_idx * 104729)
        for view_idx, name in enumerate(names)
    ]
    return views, names
