from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PRIMITIVES = ("diffeo", "color", "filter")
MIXTURE_WIDTH = 3
MAX_DEPTH = 3
IMAGE_SIZE = 32
CHANNELS = 3


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_recipe_hash(recipe: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(f"recipe_id={int(recipe['recipe_id'])}\n".encode("ascii"))
    digest.update(f"mixing={float(recipe['mixing']):.17g}\n".encode("ascii"))
    digest.update(f"weights={_sha256_array(recipe['weights'])}\n".encode("ascii"))
    for chain_id, chain in enumerate(recipe["chains"]):
        digest.update(f"chain={chain_id};depth={len(chain)}\n".encode("ascii"))
        for step_id, step in enumerate(chain):
            digest.update(
                json.dumps(
                    {
                        "chain": chain_id,
                        "step": step_id,
                        "primitive": step["primitive"],
                        "scalars": step["scalars"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            for name in sorted(step["arrays"]):
                digest.update(name.encode("utf-8"))
                digest.update(_sha256_array(step["arrays"][name]).encode("ascii"))
    return digest.hexdigest().upper()


def _temperature_range(size: int, cut: int) -> tuple[float, float]:
    adjusted = float(cut) + 1.0e-6
    log_cut = math.log(adjusted)
    low = 1.0 / (math.pi * float(size) ** 2 * log_cut)
    high = 4.0 / (math.pi**3 * adjusted**2 * log_cut)
    return low, max(low, high)


def _sample_diffeomorphism(rng: np.random.Generator) -> dict[str, Any]:
    cut_min = 2
    cut_max = int(rng.integers(cut_min + 1, 101))
    cut = int(rng.beta(0.5, 0.5) * (cut_max + 1 - cut_min) + cut_min)
    cut = min(max(cut, cut_min), cut_max)
    low, high = _temperature_range(IMAGE_SIZE, cut)
    temperature = float(rng.beta(0.5, 0.5) * (high - low) + low)

    axis = np.linspace(0.0, 1.0, IMAGE_SIZE, dtype=np.float64)
    frequencies = np.arange(1, cut + 1, dtype=np.float64)
    ii, jj = np.meshgrid(frequencies, frequencies, indexing="ij")
    radius = np.sqrt(ii**2 + jj**2)
    energy = (radius < cut + 0.5).astype(np.float64) / radius
    basis = np.sin(math.pi * axis[:, None] * frequencies[None, :])
    coeff_u = rng.standard_normal((cut, cut)) * energy
    coeff_v = rng.standard_normal((cut, cut)) * energy
    # Equivalent to einsum("ij,xi,yj->yx", ...). Explicit reductions avoid
    # platform-specific BLAS loading failures observed on the local Windows env.
    def synthesize(coefficients: np.ndarray) -> np.ndarray:
        intermediate = np.sum(
            basis[:, :, None] * coefficients[None, :, :], axis=1
        )
        return np.sum(
            intermediate[:, :, None] * basis.T[None, :, :], axis=1
        ).T

    field_u = synthesize(coeff_u)
    field_v = synthesize(coeff_v)
    scale = math.sqrt(temperature) * IMAGE_SIZE
    dx = (scale * field_u).astype(np.float32)
    dy = (scale * field_v).astype(np.float32)
    return {
        "primitive": "diffeo",
        "scalars": {
            "cut_min": cut_min,
            "cut_max": cut_max,
            "cut": cut,
            "temperature": temperature,
        },
        "arrays": {
            "spectral_u": coeff_u.astype(np.float32),
            "spectral_v": coeff_v.astype(np.float32),
            "dx": dx,
            "dy": dy,
        },
    }


def _sample_color(rng: np.random.Generator) -> dict[str, Any]:
    cut = int(rng.integers(1, 101))
    temperature = float(rng.uniform(0.0, 0.01))
    coefficients = (
        rng.standard_normal((CHANNELS, cut)) * math.sqrt(temperature)
    ).astype(np.float32)
    return {
        "primitive": "color",
        "scalars": {"cut": cut, "temperature": temperature},
        "arrays": {"coefficients": coefficients},
    }


def _sample_filter(rng: np.random.Generator) -> dict[str, Any]:
    kernel_size = int(rng.choice(np.asarray([3, 5], dtype=np.int64)))
    sigma = float(rng.uniform(0.0, 4.0))
    kernel = sigma * rng.standard_normal((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] += 1.0
    return {
        "primitive": "filter",
        "scalars": {"kernel_size": kernel_size, "sigma": sigma},
        "arrays": {"kernel": kernel.astype(np.float32)},
    }


def sample_frozen_prime_bank(*, seed: int, count: int = 64) -> dict[str, Any]:
    """Sample complete PRIME recipes once; no state is sampled during application."""

    if int(count) < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(int(seed))
    recipes: list[dict[str, Any]] = []
    samplers = (_sample_diffeomorphism, _sample_color, _sample_filter)
    for recipe_id in range(int(count)):
        weights = rng.dirichlet(np.ones(MIXTURE_WIDTH, dtype=np.float64)).astype(np.float32)
        mixing = float(rng.beta(1.0, 1.0))
        chains: list[list[dict[str, Any]]] = []
        for _chain_id in range(MIXTURE_WIDTH):
            depth = int(rng.integers(1, MAX_DEPTH + 1))
            chain: list[dict[str, Any]] = []
            for _step_id in range(depth):
                primitive_id = int(rng.integers(0, len(PRIMITIVES)))
                chain.append(samplers[primitive_id](rng))
            chains.append(chain)
        recipe = {
            "recipe_id": recipe_id,
            "weights": weights,
            "mixing": mixing,
            "chains": chains,
        }
        recipe["recipe_sha256"] = _canonical_recipe_hash(recipe)
        recipes.append(recipe)

    bank_digest = hashlib.sha256()
    bank_digest.update(
        json.dumps(
            {
                "protocol": "frozen_prime_maxent_v2",
                "seed": int(seed),
                "count": int(count),
                "mixture_width": MIXTURE_WIDTH,
                "max_depth": MAX_DEPTH,
                "primitives": PRIMITIVES,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for recipe in recipes:
        bank_digest.update(recipe["recipe_sha256"].encode("ascii"))
    return {
        "protocol": "frozen_prime_maxent_v2",
        "seed": int(seed),
        "count": int(count),
        "mixture_width": MIXTURE_WIDTH,
        "max_depth": MAX_DEPTH,
        "primitives": list(PRIMITIVES),
        "recipes": recipes,
        "bank_sha256": bank_digest.hexdigest().upper(),
    }


def save_frozen_prime_bank(
    bank: dict[str, Any],
    *,
    state_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    recipe_rows: list[dict[str, Any]] = []
    for recipe in bank["recipes"]:
        recipe_id = int(recipe["recipe_id"])
        weight_key = f"recipe_{recipe_id:03d}_weights"
        arrays[weight_key] = np.asarray(recipe["weights"], dtype=np.float32)
        chain_rows: list[dict[str, Any]] = []
        for chain_id, chain in enumerate(recipe["chains"]):
            step_rows: list[dict[str, Any]] = []
            for step_id, step in enumerate(chain):
                array_rows: dict[str, dict[str, Any]] = {}
                for name, values in sorted(step["arrays"].items()):
                    key = f"recipe_{recipe_id:03d}_chain_{chain_id}_step_{step_id}_{name}"
                    array = np.ascontiguousarray(values)
                    arrays[key] = array
                    array_rows[name] = {
                        "key": key,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": _sha256_array(array),
                    }
                step_rows.append(
                    {
                        "primitive": step["primitive"],
                        "scalars": step["scalars"],
                        "arrays": array_rows,
                    }
                )
            chain_rows.append({"depth": len(chain), "steps": step_rows})
        recipe_rows.append(
            {
                "recipe_id": recipe_id,
                "mixing": float(recipe["mixing"]),
                "weights": {
                    "key": weight_key,
                    "shape": list(arrays[weight_key].shape),
                    "dtype": str(arrays[weight_key].dtype),
                    "sha256": _sha256_array(arrays[weight_key]),
                },
                "chains": chain_rows,
                "recipe_sha256": recipe["recipe_sha256"],
            }
        )
    np.savez_compressed(state_path, **arrays)
    manifest = {
        "protocol": bank["protocol"],
        "seed": int(bank["seed"]),
        "count": int(bank["count"]),
        "mixture_width": int(bank["mixture_width"]),
        "max_depth": int(bank["max_depth"]),
        "primitives": list(bank["primitives"]),
        "bank_sha256": bank["bank_sha256"],
        "state_file": state_path.name,
        "state_file_bytes": int(state_path.stat().st_size),
        "state_file_sha256": _sha256_file(state_path),
        "recipes": recipe_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_frozen_prime_bank(*, state_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "frozen_prime_maxent_v2":
        raise ValueError("unexpected frozen PRIME protocol")
    if state_path.stat().st_size != int(manifest["state_file_bytes"]):
        raise ValueError("frozen PRIME state byte-size mismatch")
    if _sha256_file(state_path) != str(manifest["state_file_sha256"]):
        raise ValueError("frozen PRIME state SHA256 mismatch")
    recipes: list[dict[str, Any]] = []
    with np.load(state_path, allow_pickle=False) as archive:
        for recipe_row in manifest["recipes"]:
            weight_row = recipe_row["weights"]
            weights = np.asarray(archive[str(weight_row["key"])])
            if _sha256_array(weights) != str(weight_row["sha256"]):
                raise ValueError("frozen PRIME mixture-weight hash mismatch")
            chains: list[list[dict[str, Any]]] = []
            for chain_row in recipe_row["chains"]:
                steps: list[dict[str, Any]] = []
                for step_row in chain_row["steps"]:
                    arrays: dict[str, np.ndarray] = {}
                    for name, array_row in step_row["arrays"].items():
                        array = np.asarray(archive[str(array_row["key"])])
                        if list(array.shape) != list(array_row["shape"]):
                            raise ValueError("frozen PRIME state shape mismatch")
                        if str(array.dtype) != str(array_row["dtype"]):
                            raise ValueError("frozen PRIME state dtype mismatch")
                        if _sha256_array(array) != str(array_row["sha256"]):
                            raise ValueError("frozen PRIME state array hash mismatch")
                        arrays[name] = array
                    steps.append(
                        {
                            "primitive": str(step_row["primitive"]),
                            "scalars": dict(step_row["scalars"]),
                            "arrays": arrays,
                        }
                    )
                if len(steps) != int(chain_row["depth"]):
                    raise ValueError("frozen PRIME chain depth mismatch")
                chains.append(steps)
            recipe = {
                "recipe_id": int(recipe_row["recipe_id"]),
                "weights": weights,
                "mixing": float(recipe_row["mixing"]),
                "chains": chains,
            }
            computed = _canonical_recipe_hash(recipe)
            if computed != str(recipe_row["recipe_sha256"]):
                raise ValueError("frozen PRIME recipe hash mismatch")
            recipe["recipe_sha256"] = computed
            recipes.append(recipe)

    bank_digest = hashlib.sha256()
    bank_digest.update(
        json.dumps(
            {
                "protocol": manifest["protocol"],
                "seed": int(manifest["seed"]),
                "count": int(manifest["count"]),
                "mixture_width": int(manifest["mixture_width"]),
                "max_depth": int(manifest["max_depth"]),
                "primitives": tuple(manifest["primitives"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for recipe in recipes:
        bank_digest.update(recipe["recipe_sha256"].encode("ascii"))
    computed_bank_hash = bank_digest.hexdigest().upper()
    if computed_bank_hash != str(manifest["bank_sha256"]):
        raise ValueError("frozen PRIME bank hash mismatch")
    return {
        "protocol": manifest["protocol"],
        "seed": int(manifest["seed"]),
        "count": int(manifest["count"]),
        "mixture_width": int(manifest["mixture_width"]),
        "max_depth": int(manifest["max_depth"]),
        "primitives": list(manifest["primitives"]),
        "recipes": recipes,
        "bank_sha256": computed_bank_hash,
    }


def _apply_diffeomorphism(images: torch.Tensor, step: dict[str, Any]) -> torch.Tensor:
    dx = torch.as_tensor(step["arrays"]["dx"], device=images.device, dtype=images.dtype)
    dy = torch.as_tensor(step["arrays"]["dy"], device=images.device, dtype=images.dtype)
    height, width = images.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=images.dtype),
        torch.arange(width, device=images.device, dtype=images.dtype),
        indexing="ij",
    )
    xn = (xx - dx).clamp(0, width - 1)
    yn = (yy - dy).clamp(0, height - 1)
    xf, yf = xn.floor().long(), yn.floor().long()
    xc, yc = xn.ceil().long(), yn.ceil().long()
    xv, yv = xn - xf, yn - yf
    return (
        (1 - yv) * (1 - xv) * images[..., yf, xf]
        + (1 - yv) * xv * images[..., yf, xc]
        + yv * (1 - xv) * images[..., yc, xf]
        + yv * xv * images[..., yc, xc]
    )


def _apply_color(images: torch.Tensor, step: dict[str, Any]) -> torch.Tensor:
    coefficients = torch.as_tensor(
        step["arrays"]["coefficients"], device=images.device, dtype=images.dtype
    )
    delta = torch.zeros_like(images)
    for start in range(0, coefficients.shape[1], 16):
        stop = min(start + 16, coefficients.shape[1])
        frequencies = torch.arange(
            start + 1, stop + 1, device=images.device, dtype=images.dtype
        )
        basis = torch.sin(images.unsqueeze(-1) * frequencies * math.pi)
        delta += torch.einsum("bchwk,ck->bchw", basis, coefficients[:, start:stop])
    return torch.clamp(images + delta, 0.0, 1.0)


def _apply_filter(images: torch.Tensor, step: dict[str, Any]) -> torch.Tensor:
    kernel = torch.as_tensor(
        step["arrays"]["kernel"], device=images.device, dtype=images.dtype
    )
    weight = kernel[None, None].repeat(images.shape[1], 1, 1, 1)
    return torch.clamp(F.conv2d(images, weight, padding="same", groups=images.shape[1]), 0.0, 1.0)


@torch.no_grad()
def apply_frozen_prime_recipe(images: torch.Tensor, recipe: dict[str, Any]) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1:] != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("images must have shape [batch,3,32,32]")
    if not torch.isfinite(images).all():
        raise ValueError("images contain non-finite values")
    chain_outputs: list[torch.Tensor] = []
    for chain in recipe["chains"]:
        transformed = images
        for step in chain:
            primitive = step["primitive"]
            if primitive == "diffeo":
                transformed = _apply_diffeomorphism(transformed, step)
            elif primitive == "color":
                transformed = _apply_color(transformed, step)
            elif primitive == "filter":
                transformed = _apply_filter(transformed, step)
            else:  # pragma: no cover - protected by the frozen sampler.
                raise ValueError(f"unknown PRIME primitive: {primitive}")
        chain_outputs.append(transformed)
    weights = torch.as_tensor(recipe["weights"], device=images.device, dtype=images.dtype)
    mixture = sum(weight * value for weight, value in zip(weights, chain_outputs))
    mixing = float(recipe["mixing"])
    return torch.clamp((1.0 - mixing) * images + mixing * mixture, 0.0, 1.0)


__all__ = [
    "PRIMITIVES",
    "apply_frozen_prime_recipe",
    "load_frozen_prime_bank",
    "sample_frozen_prime_bank",
    "save_frozen_prime_bank",
]
