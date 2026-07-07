"""Dataset corruption helpers for RAHFL-style and corruption-skew protocols."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


CORRUPTION_GROUPS: dict[str, tuple[str, ...]] = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "spatter"),
    "digital": ("contrast", "brightness", "jpeg_compression", "pixelate"),
}

GROUP_TO_ID = {name: idx for idx, name in enumerate(CORRUPTION_GROUPS)}
ID_TO_GROUP = {idx: name for name, idx in GROUP_TO_ID.items()}


def apply_corruption(image: np.ndarray | Image.Image, name: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Apply a lightweight CIFAR-size corruption and return uint8 HWC image.

    The goal is not to exactly reproduce CIFAR-C internals; it is to create a
    reproducible client-specific corruption-skew protocol with the same broad
    corruption families used by CIFAR-C: noise, blur, weather, and digital.
    """

    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)
    severity = int(np.clip(severity, 1, 5))

    corruption = _CORRUPTION_FNS.get(name)
    if corruption is None:
        raise ValueError(f"Unknown corruption: {name}")
    return corruption(arr, severity, rng).astype(np.uint8)


def sample_corruption_from_group(group: str, rng: np.random.Generator) -> str:
    methods = CORRUPTION_GROUPS[group]
    return str(rng.choice(methods))


def build_client_group_profiles(
    num_clients: int,
    skew_ratio: float,
    group_names: list[str] | None = None,
) -> dict[int, dict[str, float]]:
    """Return per-client corruption-group distributions.

    Client i gets group_names[i % len(group_names)] as its dominant group.
    """

    groups = group_names or list(CORRUPTION_GROUPS)
    if not groups:
        raise ValueError("At least one corruption group is required.")
    skew_ratio = float(skew_ratio)
    if not 0.0 <= skew_ratio <= 1.0:
        raise ValueError(f"skew_ratio must be in [0, 1], got {skew_ratio}")

    profiles: dict[int, dict[str, float]] = {}
    for client_id in range(num_clients):
        dominant = groups[client_id % len(groups)]
        if len(groups) == 1:
            profiles[client_id] = {dominant: 1.0}
            continue
        rest = (1.0 - skew_ratio) / float(len(groups) - 1)
        profile = {group: rest for group in groups}
        profile[dominant] = skew_ratio
        profiles[client_id] = profile
    return profiles


def _to_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _gaussian_noise(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    scale = [0.04, 0.06, 0.08, 0.10, 0.12][severity - 1]
    x = _to_float(arr)
    return _to_uint8(x + rng.normal(0.0, scale, size=x.shape).astype(np.float32))


def _shot_noise(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    vals = [80, 60, 40, 25, 15][severity - 1]
    x = _to_float(arr)
    return _to_uint8(rng.poisson(x * vals).astype(np.float32) / float(vals))


def _impulse_noise(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    prob = [0.01, 0.02, 0.035, 0.05, 0.07][severity - 1]
    x = arr.copy()
    mask = rng.random(arr.shape[:2])
    salt = mask < prob / 2
    pepper = (mask >= prob / 2) & (mask < prob)
    x[salt] = 255
    x[pepper] = 0
    return x


def _speckle_noise(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    scale = [0.08, 0.12, 0.16, 0.20, 0.25][severity - 1]
    x = _to_float(arr)
    return _to_uint8(x + x * rng.normal(0.0, scale, size=x.shape).astype(np.float32))


def _pil_blur(arr: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.uint8)


def _defocus_blur(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    return _pil_blur(arr, [0.4, 0.6, 0.8, 1.0, 1.2][severity - 1])


def _glass_blur(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    blurred = _pil_blur(arr, [0.3, 0.5, 0.7, 0.9, 1.1][severity - 1])
    shifts = int(severity)
    out = blurred.copy()
    for _ in range(shifts):
        dx = int(rng.integers(-1, 2))
        dy = int(rng.integers(-1, 2))
        out = np.roll(out, shift=(dy, dx), axis=(0, 1))
    return out


def _motion_blur(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    radius = [0.3, 0.5, 0.7, 1.0, 1.2][severity - 1]
    blurred = _pil_blur(arr, radius)
    direction = int(rng.integers(0, 2))
    shifted = np.roll(blurred, shift=severity, axis=direction)
    return ((blurred.astype(np.float32) + shifted.astype(np.float32)) / 2.0).astype(np.uint8)


def _zoom_blur(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    image = Image.fromarray(arr)
    scales = [1.05, 1.08, 1.12, 1.16, 1.20]
    scale = scales[severity - 1]
    new_size = max(33, int(round(32 * scale)))
    zoomed = image.resize((new_size, new_size), Image.BILINEAR)
    left = (new_size - 32) // 2
    cropped = zoomed.crop((left, left, left + 32, left + 32))
    return ((arr.astype(np.float32) + np.asarray(cropped, dtype=np.float32)) / 2.0).astype(np.uint8)


def _fog(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    x = _to_float(arr)
    alpha = [0.10, 0.16, 0.22, 0.30, 0.38][severity - 1]
    haze = rng.normal(0.78, 0.08, size=x.shape).astype(np.float32)
    return _to_uint8((1.0 - alpha) * x + alpha * np.clip(haze, 0.0, 1.0))


def _snow(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    x = _to_float(arr)
    prob = [0.015, 0.025, 0.04, 0.055, 0.075][severity - 1]
    snow = (rng.random(arr.shape[:2]) < prob).astype(np.float32)
    snow = np.repeat(snow[:, :, None], 3, axis=2)
    return _to_uint8(np.maximum(x, snow))


def _frost(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    x = _to_float(arr)
    strength = [0.08, 0.12, 0.18, 0.24, 0.30][severity - 1]
    frost = rng.uniform(0.65, 1.0, size=x.shape).astype(np.float32)
    frost[:, :, 0] *= 0.75
    return _to_uint8((1.0 - strength) * x + strength * frost)


def _spatter(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    x = _to_float(arr)
    prob = [0.01, 0.02, 0.035, 0.05, 0.07][severity - 1]
    drops = (rng.random(arr.shape[:2]) < prob).astype(np.float32)
    drops = np.repeat(drops[:, :, None], 3, axis=2)
    color = np.array([0.55, 0.45, 0.30], dtype=np.float32).reshape(1, 1, 3)
    return _to_uint8(x * (1.0 - drops * 0.6) + color * drops * 0.6)


def _contrast(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    factor = [0.75, 0.60, 0.45, 0.35, 0.25][severity - 1]
    image = Image.fromarray(arr)
    return np.asarray(ImageEnhance.Contrast(image).enhance(factor), dtype=np.uint8)


def _brightness(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    factor = [0.80, 0.70, 0.60, 0.50, 0.40][severity - 1]
    image = Image.fromarray(arr)
    return np.asarray(ImageEnhance.Brightness(image).enhance(factor), dtype=np.uint8)


def _jpeg_compression(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    from io import BytesIO

    quality = [75, 60, 45, 35, 25][severity - 1]
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)


def _pixelate(arr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    sizes = [24, 20, 16, 12, 8]
    size = sizes[severity - 1]
    image = Image.fromarray(arr)
    small = image.resize((size, size), Image.BILINEAR)
    return np.asarray(small.resize((32, 32), Image.NEAREST), dtype=np.uint8)


_CORRUPTION_FNS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "impulse_noise": _impulse_noise,
    "speckle_noise": _speckle_noise,
    "defocus_blur": _defocus_blur,
    "glass_blur": _glass_blur,
    "motion_blur": _motion_blur,
    "zoom_blur": _zoom_blur,
    "snow": _snow,
    "frost": _frost,
    "fog": _fog,
    "spatter": _spatter,
    "contrast": _contrast,
    "brightness": _brightness,
    "jpeg_compression": _jpeg_compression,
    "pixelate": _pixelate,
}
