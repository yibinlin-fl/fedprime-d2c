from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fedprime.data.corruptions import CORRUPTION_GROUPS, apply_corruption


PHASE_A0_SEED = 20260830
PHASE_A0_SEVERITY = 3
FAMILY_NAMES = tuple(CORRUPTION_GROUPS)
OPERATOR_NAMES = tuple(operator for family in FAMILY_NAMES for operator in CORRUPTION_GROUPS[family])
OPERATOR_FAMILY_IDS = np.asarray(
    [family_id for family_id, family in enumerate(FAMILY_NAMES) for _ in CORRUPTION_GROUPS[family]],
    dtype=np.int64,
)


@dataclass(frozen=True)
class DSAResult:
    family_effects: np.ndarray
    client_family: np.ndarray
    client: np.ndarray
    pooled: float


def historical_family_binding(num_clients: int = 4, num_classes: int = 10) -> np.ndarray:
    """Return the exact cyclic class-to-family map used by CLE v1."""

    clients = np.arange(int(num_clients), dtype=np.int64)[:, None]
    classes = np.arange(int(num_classes), dtype=np.int64)[None, :]
    return (clients + classes) % len(FAMILY_NAMES)


def deterministic_corruption_grid(
    clean_images: np.ndarray,
    *,
    severity: int = PHASE_A0_SEVERITY,
    seed: int = PHASE_A0_SEED,
    operator_names: Iterable[str] = OPERATOR_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate source-major paired interventions with one fixed severity."""

    images = np.asarray(clean_images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("clean_images must have shape [N,H,W,3]")
    if images.dtype != np.uint8:
        raise ValueError("clean_images must use uint8 pixels")
    if not 1 <= int(severity) <= 5:
        raise ValueError("severity must be in [1,5]")
    names = tuple(str(name) for name in operator_names)
    if not names:
        raise ValueError("operator_names cannot be empty")

    grid = np.empty((images.shape[0], len(names), *images.shape[1:]), dtype=np.uint8)
    for source_id, image in enumerate(images):
        for operator_id, operator in enumerate(names):
            sequence = np.random.SeedSequence([int(seed), int(source_id), int(operator_id)])
            grid[source_id, operator_id] = apply_corruption(
                image,
                operator,
                int(severity),
                np.random.default_rng(sequence),
            )
    return grid, np.full((images.shape[0], len(names)), int(severity), dtype=np.uint8)


def validate_probability_tensor(
    probabilities: np.ndarray,
    labels: np.ndarray,
    operator_family_ids: np.ndarray = OPERATOR_FAMILY_IDS,
) -> None:
    probs = np.asarray(probabilities)
    labels = np.asarray(labels)
    family_ids = np.asarray(operator_family_ids)
    if probs.ndim != 5:
        raise ValueError("probabilities must have shape [condition,client,source,operator,class]")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[2]:
        raise ValueError("labels must have shape [source]")
    if family_ids.ndim != 1 or family_ids.shape[0] != probs.shape[3]:
        raise ValueError("operator_family_ids must have shape [operator]")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(probs < -1.0e-7):
        raise ValueError("probabilities contain negative values")
    if not np.allclose(probs.sum(axis=-1), 1.0, atol=1.0e-5):
        raise ValueError("probabilities do not sum to one")
    expected = np.arange(int(family_ids.max()) + 1)
    if not np.array_equal(np.unique(family_ids), expected):
        raise ValueError("family ids must be contiguous from zero")


def family_class_contrast(
    probabilities: np.ndarray,
    operator_family_ids: np.ndarray = OPERATOR_FAMILY_IDS,
) -> np.ndarray:
    """Return [client,family,source,class] own-family minus other-family response."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 4:
        raise ValueError("condition probabilities must have shape [client,source,operator,class]")
    family_ids = np.asarray(operator_family_ids, dtype=np.int64)
    if family_ids.shape != (probs.shape[2],):
        raise ValueError("operator_family_ids length does not match operator axis")
    num_families = int(family_ids.max()) + 1
    family_means = np.stack(
        [probs[:, :, family_ids == family_id, :].mean(axis=2) for family_id in range(num_families)],
        axis=1,
    )
    other_means = (family_means.sum(axis=1, keepdims=True) - family_means) / max(num_families - 1, 1)
    return family_means - other_means


def dsa_from_contrast(
    contrast: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
) -> DSAResult:
    contrast = np.asarray(contrast, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    binding = np.asarray(binding, dtype=np.int64)
    if contrast.ndim != 4:
        raise ValueError("contrast must have shape [client,family,source,class]")
    num_clients, num_families, num_sources, num_classes = contrast.shape
    if labels.shape != (num_sources,):
        raise ValueError("labels shape does not match source axis")
    if binding.shape != (num_clients, num_classes):
        raise ValueError("binding must have shape [client,class]")

    effects = np.full((num_clients, num_families, num_sources), np.nan, dtype=np.float64)
    client_family = np.full((num_clients, num_families), np.nan, dtype=np.float64)
    for client_id in range(num_clients):
        for family_id in range(num_families):
            bound_classes = np.flatnonzero(binding[client_id] == family_id)
            if bound_classes.size == 0:
                continue
            valid = ~np.isin(labels, bound_classes)
            values = contrast[client_id, family_id][:, bound_classes].sum(axis=1)
            effects[client_id, family_id, valid] = values[valid]
            if bool(valid.any()):
                client_family[client_id, family_id] = float(values[valid].mean())
    client = np.nanmean(client_family, axis=1)
    return DSAResult(
        family_effects=effects,
        client_family=client_family,
        client=client,
        pooled=float(np.nanmean(client)),
    )


def compute_dsa(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
    operator_family_ids: np.ndarray = OPERATOR_FAMILY_IDS,
) -> DSAResult:
    return dsa_from_contrast(
        family_class_contrast(probabilities, operator_family_ids),
        labels,
        binding,
    )


def paired_bootstrap_delta(
    gamma0: DSAResult,
    gamma09: DSAResult,
    *,
    samples: int = 2000,
    seed: int = PHASE_A0_SEED,
) -> np.ndarray:
    left = np.asarray(gamma0.family_effects, dtype=np.float64)
    right = np.asarray(gamma09.family_effects, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("gamma condition family effects must have identical shapes")
    num_sources = left.shape[-1]
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(samples), dtype=np.float64)
    for bootstrap_id in range(int(samples)):
        indices = rng.integers(0, num_sources, size=num_sources)
        left_client = np.nanmean(np.nanmean(left[..., indices], axis=-1), axis=-1)
        right_client = np.nanmean(np.nanmean(right[..., indices], axis=-1), axis=-1)
        deltas[bootstrap_id] = float(np.nanmean(right_client - left_client))
    return deltas


def shuffled_binding_test(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
    *,
    permutations: int = 1000,
    seed: int = PHASE_A0_SEED,
    operator_family_ids: np.ndarray = OPERATOR_FAMILY_IDS,
) -> dict[str, np.ndarray | float]:
    contrast = family_class_contrast(probabilities, operator_family_ids)
    observed = dsa_from_contrast(contrast, labels, binding)
    rng = np.random.default_rng(int(seed))
    null_client = np.empty((int(permutations), binding.shape[0]), dtype=np.float64)
    null_pooled = np.empty(int(permutations), dtype=np.float64)
    for permutation_id in range(int(permutations)):
        shuffled = np.stack(
            [row[rng.permutation(row.size)] for row in np.asarray(binding, dtype=np.int64)],
            axis=0,
        )
        result = dsa_from_contrast(contrast, labels, shuffled)
        null_client[permutation_id] = result.client
        null_pooled[permutation_id] = result.pooled
    client_p = (1.0 + (null_client >= observed.client[None, :]).sum(axis=0)) / (permutations + 1.0)
    pooled_p = float((1.0 + (null_pooled >= observed.pooled).sum()) / (permutations + 1.0))
    return {
        "observed_client": observed.client,
        "observed_pooled": observed.pooled,
        "null_client": null_client,
        "null_pooled": null_pooled,
        "client_p": client_p,
        "pooled_p": pooled_p,
    }


def secondary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
    operator_family_ids: np.ndarray = OPERATOR_FAMILY_IDS,
) -> dict[str, float | list[float]]:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probs.argmax(axis=-1)
    client_acc = 100.0 * (predictions == labels[None, :, None]).mean(axis=(1, 2))
    num_classes = probs.shape[-1]
    num_families = int(np.max(operator_family_ids)) + 1

    cell_acc = np.full((num_classes, num_families), np.nan, dtype=np.float64)
    for class_id in range(num_classes):
        source_mask = labels == class_id
        for family_id in range(num_families):
            operator_mask = np.asarray(operator_family_ids) == family_id
            cell = predictions[:, source_mask][:, :, operator_mask]
            if cell.size:
                cell_acc[class_id, family_id] = 100.0 * (cell == class_id).mean()
    cfg = float(np.nanmean(np.nanmax(cell_acc, axis=1) - np.nanmin(cell_acc, axis=1)))
    wcca = float(np.nanmin(cell_acc))

    operator_count = predictions.shape[2]
    pair_denominator = max(operator_count * (operator_count - 1), 1)
    agreements = np.zeros(predictions.shape[:2], dtype=np.float64)
    for class_id in range(num_classes):
        counts = (predictions == class_id).sum(axis=2)
        agreements += counts * (counts - 1)
    paired_flip_rate = float(np.mean(1.0 - agreements / pair_denominator))

    top1 = np.eye(num_classes, dtype=np.float64)[predictions]
    top1_dsa = compute_dsa(top1, labels, binding, operator_family_ids)
    entropy = float(np.mean(-np.sum(probs * np.log(np.clip(probs, 1.0e-12, 1.0)), axis=-1)))
    return {
        "avg_acc": float(client_acc.mean()),
        "worst_acc": float(client_acc.min()),
        "client_acc": client_acc.tolist(),
        "wcca": wcca,
        "cfg": cfg,
        "paired_prediction_flip_rate": paired_flip_rate,
        "family_bound_top1_bias": top1_dsa.pooled,
        "mean_entropy": entropy,
    }


def decide_phase_a0(
    gamma0: DSAResult,
    gamma09: DSAResult,
    bootstrap_delta: np.ndarray,
    shuffled: dict[str, np.ndarray | float],
    *,
    minimum_delta: float = 0.020,
) -> dict[str, object]:
    delta_client = gamma09.client - gamma0.client
    delta_pooled = float(gamma09.pooled - gamma0.pooled)
    ci_low, ci_high = np.quantile(np.asarray(bootstrap_delta), [0.025, 0.975]).tolist()
    null_pooled = np.asarray(shuffled["null_pooled"], dtype=np.float64)
    client_p = np.asarray(shuffled["client_p"], dtype=np.float64)
    gates = {
        "G1_minimum_delta": {"value": delta_pooled, "threshold": minimum_delta, "pass": delta_pooled >= minimum_delta},
        "G2_bootstrap_ci": {"low": ci_low, "high": ci_high, "pass": ci_low > 0.0},
        "G3_client_direction": {"positive_clients": int((delta_client > 0.0).sum()), "threshold": 3, "pass": int((delta_client > 0.0).sum()) >= 3},
        "G4_shuffled_pooled": {
            "observed": float(shuffled["observed_pooled"]),
            "null_p95": float(np.quantile(null_pooled, 0.95)),
            "pass": float(shuffled["observed_pooled"]) > float(np.quantile(null_pooled, 0.95)),
        },
        "G5_shuffled_clients": {"passing_clients": int((client_p < 0.05).sum()), "threshold": 3, "pass": int((client_p < 0.05).sum()) >= 3},
    }
    passed = all(bool(gate["pass"]) for gate in gates.values())
    return {
        "verdict": "GO_TO_MATCHED_PARTITION_DESIGN" if passed else "NO_GO_DIRECTIONAL_SHORTCUT",
        "delta_dsa_pooled": delta_pooled,
        "delta_dsa_client": delta_client.tolist(),
        "bootstrap_ci95": [ci_low, ci_high],
        "gates": gates,
    }
