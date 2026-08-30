from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DirectionalWithdrawal:
    matrix: np.ndarray
    class_mean: np.ndarray
    client: np.ndarray
    pooled: float


def _validate_probability_pair(
    original: np.ndarray,
    bridged: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original_array = np.asarray(original, dtype=np.float64)
    bridged_array = np.asarray(bridged, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if original_array.ndim != 4:
        raise ValueError("probabilities must have shape [client,source,probe,class]")
    if bridged_array.shape != original_array.shape:
        raise ValueError("original and bridged probabilities must have identical shapes")
    if label_array.shape != (original_array.shape[1],):
        raise ValueError("labels must have shape [source]")
    for name, values in (("original", original_array), ("bridged", bridged_array)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} probabilities contain non-finite values")
        if np.any(values < -1.0e-7) or not np.allclose(
            values.sum(axis=-1), 1.0, atol=1.0e-5
        ):
            raise ValueError(f"{name} probabilities must be normalized non-negative softmax values")
    return original_array, bridged_array, label_array


def directional_withdrawal(
    original: np.ndarray,
    bridged: np.ndarray,
    labels: np.ndarray,
) -> DirectionalWithdrawal:
    """Estimate class-directed evidence removed by an intervention bridge.

    The estimator accepts no corruption identities and no class--corruption
    binding. Each target class excludes sources carrying that true task label.
    The probe axis is retained only so an oracle audit can stratify the already
    estimated matrix after the fact.
    """

    original_array, bridged_array, label_array = _validate_probability_pair(
        original, bridged, labels
    )
    _, _, num_probes, num_classes = original_array.shape
    differences = original_array - bridged_array
    matrix = np.empty(
        (original_array.shape[0], num_probes, num_classes), dtype=np.float64
    )
    for class_id in range(num_classes):
        valid = label_array != class_id
        if not bool(valid.any()):
            raise ValueError(f"no off-target sources are available for class {class_id}")
        class_differences = differences[..., class_id]
        matrix[:, :, class_id] = class_differences[:, valid, :].mean(axis=1)
    class_mean = matrix.mean(axis=1)
    client = np.square(np.maximum(class_mean, 0.0)).mean(axis=1)
    return DirectionalWithdrawal(
        matrix=matrix,
        class_mean=class_mean,
        client=client,
        pooled=float(client.mean()),
    )


def family_aggregated_withdrawal(
    withdrawal_matrix: np.ndarray,
    probe_family_ids: np.ndarray,
) -> np.ndarray:
    """Aggregate an already-estimated probe matrix for oracle-only scoring."""

    matrix = np.asarray(withdrawal_matrix, dtype=np.float64)
    family_ids = np.asarray(probe_family_ids, dtype=np.int64)
    if matrix.ndim != 3:
        raise ValueError("withdrawal_matrix must have shape [client,probe,class]")
    if family_ids.shape != (matrix.shape[1],):
        raise ValueError("probe_family_ids must match the probe axis")
    unique = np.unique(family_ids)
    if not np.array_equal(unique, np.arange(unique.size, dtype=np.int64)):
        raise ValueError("probe family ids must be contiguous from zero")
    return np.stack(
        [matrix[:, family_ids == family_id].mean(axis=1) for family_id in unique],
        axis=1,
    )


def confidence_calibrated_scdw_loss(
    original_probabilities: torch.Tensor,
    canonical_probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    z_alpha: float = 1.645,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable minibatch SCDW objective with a stopped LCB threshold."""

    if original_probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch,class]")
    if canonical_probabilities.shape != original_probabilities.shape:
        raise ValueError("original/canonical probabilities must have identical shapes")
    if labels.shape != (original_probabilities.shape[0],):
        raise ValueError("labels must have shape [batch]")
    if original_probabilities.shape[0] < 2:
        raise ValueError("at least two samples are required")

    class_means = []
    lower_bounds = []
    for class_id in range(original_probabilities.shape[1]):
        valid = labels != class_id
        count = int(valid.sum().item())
        if count < 2:
            raise ValueError(f"class {class_id} has fewer than two off-target samples")
        differences = (
            original_probabilities[valid, class_id]
            - canonical_probabilities[valid, class_id].detach()
        )
        mean = differences.mean()
        standard_error = differences.std(unbiased=False) / float(count) ** 0.5
        lower = mean - (float(z_alpha) * standard_error).detach()
        class_means.append(mean)
        lower_bounds.append(lower)
    means = torch.stack(class_means)
    lower = torch.stack(lower_bounds)
    return torch.relu(lower).square().mean(), means, lower


def within_source_variance(images: np.ndarray) -> float:
    array = np.asarray(images, dtype=np.float64)
    if array.ndim != 5:
        raise ValueError("images must have shape [source,probe,height,width,channel]")
    normalized = array / 255.0 if array.max(initial=0.0) > 1.5 else array
    centered = normalized - normalized.mean(axis=1, keepdims=True)
    return float(np.square(centered).mean())


def within_source_variance_contraction(
    original_images: np.ndarray,
    bridged_images: np.ndarray,
) -> float:
    if np.asarray(original_images).shape != np.asarray(bridged_images).shape:
        raise ValueError("original and bridged images must have identical shapes")
    original_variance = within_source_variance(original_images)
    if original_variance <= 0.0:
        raise ValueError("original within-source variance must be positive")
    return float(1.0 - within_source_variance(bridged_images) / original_variance)


def _residual_signatures(images: np.ndarray, clean_sources: np.ndarray) -> np.ndarray:
    array = np.asarray(images, dtype=np.float64)
    clean = np.asarray(clean_sources, dtype=np.float64)
    if array.ndim != 5 or clean.shape != (array.shape[0], *array.shape[2:]):
        raise ValueError("clean_sources must match the source and image axes")
    if array.shape[2:4] != (32, 32) or array.shape[-1] != 3:
        raise ValueError("the Phase-B0 signature currently expects 32x32 RGB images")
    residual = (array - clean[:, None]) / 255.0
    pooled = residual.reshape(
        residual.shape[0], residual.shape[1], 4, 8, 4, 8, 3
    ).mean(axis=(3, 5))
    means = residual.mean(axis=(2, 3))
    stds = residual.std(axis=(2, 3))
    absolute = np.abs(residual).mean(axis=(2, 3))
    grad_x = np.abs(np.diff(residual, axis=3)).mean(axis=(2, 3))
    grad_y = np.abs(np.diff(residual, axis=2)).mean(axis=(2, 3))
    return np.concatenate(
        [
            pooled.reshape(residual.shape[0], residual.shape[1], -1),
            means,
            stds,
            absolute,
            grad_x,
            grad_y,
        ],
        axis=2,
    )


def family_separability_accuracy(
    images: np.ndarray,
    clean_sources: np.ndarray,
    probe_family_ids: np.ndarray,
    *,
    folds: int = 5,
) -> float:
    """Oracle-only source-folded nearest-centroid degradation separability."""

    signatures = _residual_signatures(images, clean_sources)
    family_ids = np.asarray(probe_family_ids, dtype=np.int64)
    if family_ids.shape != (signatures.shape[1],):
        raise ValueError("probe_family_ids must match the probe axis")
    if folds < 2 or signatures.shape[0] < folds:
        raise ValueError("folds must be between two and the number of sources")
    predictions = np.empty((signatures.shape[0], signatures.shape[1]), dtype=np.int64)
    unique = np.unique(family_ids)
    source_ids = np.arange(signatures.shape[0])
    for fold in range(int(folds)):
        test_sources = source_ids % int(folds) == fold
        train = signatures[~test_sources].reshape(-1, signatures.shape[-1])
        train_labels = np.tile(family_ids, int((~test_sources).sum()))
        centroids = np.stack([train[train_labels == family].mean(axis=0) for family in unique])
        test = signatures[test_sources].reshape(-1, signatures.shape[-1])
        distances = np.square(test[:, None, :] - centroids[None, :, :]).mean(axis=2)
        predictions[test_sources] = unique[np.argmin(distances, axis=1)].reshape(
            int(test_sources.sum()), signatures.shape[1]
        )
    truth = np.broadcast_to(family_ids[None], predictions.shape)
    return float(np.mean(predictions == truth))


def peak_signal_to_noise_ratio(images: np.ndarray, clean_sources: np.ndarray) -> float:
    array = np.asarray(images, dtype=np.float64)
    clean = np.asarray(clean_sources, dtype=np.float64)
    if array.ndim != 5 or clean.shape != (array.shape[0], *array.shape[2:]):
        raise ValueError("clean_sources must match the source and image axes")
    normalized = array / 255.0 if array.max(initial=0.0) > 1.5 else array
    clean_normalized = clean / 255.0 if clean.max(initial=0.0) > 1.5 else clean
    mse = np.square(normalized - clean_normalized[:, None]).mean(axis=(2, 3, 4))
    return float(np.mean(-10.0 * np.log10(np.maximum(mse, 1.0e-12))))


def decide_bridge_only_gate(
    metrics: dict[str, float | int],
    *,
    minimum_semantic_accuracy_delta: float = -0.01,
    minimum_variance_contraction: float = 0.25,
    minimum_separability_reduction: float = 0.30,
    minimum_gamma9_map: float = 0.65,
    minimum_map_delta: float = 0.20,
    minimum_hit_rate: float = 0.70,
    minimum_positive_clients: int = 3,
    minimum_overlay_margin: float = 0.10,
    maximum_clean_scdw: float = 0.03,
) -> dict[str, object]:
    required = {
        "semantic_accuracy_delta_min",
        "variance_contraction",
        "separability_relative_reduction",
        "hfl_gamma9_map",
        "hfl_map_delta",
        "hfl_hit_rate",
        "hfl_positive_clients",
        "local_gamma9_map",
        "local_map_delta",
        "local_hit_rate",
        "local_positive_clients",
        "canonical_vs_overlay_contraction_margin",
        "clean_scdw_max",
    }
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"missing bridge-gate metrics: {sorted(missing)}")
    gates = {
        "G1_semantic_preservation": float(metrics["semantic_accuracy_delta_min"])
        >= minimum_semantic_accuracy_delta,
        "G2_old_nuisance_contraction": float(metrics["variance_contraction"])
        >= minimum_variance_contraction,
        "G3_family_separability_reduction": float(
            metrics["separability_relative_reduction"]
        )
        >= minimum_separability_reduction,
        "G4_hfl_directional_retrieval": (
            float(metrics["hfl_gamma9_map"]) >= minimum_gamma9_map
            and float(metrics["hfl_map_delta"]) >= minimum_map_delta
            and float(metrics["hfl_hit_rate"]) >= minimum_hit_rate
            and int(metrics["hfl_positive_clients"]) >= minimum_positive_clients
        ),
        "G5_local_directional_retrieval": (
            float(metrics["local_gamma9_map"]) >= minimum_gamma9_map
            and float(metrics["local_map_delta"]) >= minimum_map_delta
            and float(metrics["local_hit_rate"]) >= minimum_hit_rate
            and int(metrics["local_positive_clients"]) >= minimum_positive_clients
        ),
        "G6_better_than_overlay": float(metrics["canonical_vs_overlay_contraction_margin"])
        >= minimum_overlay_margin,
        "G7_clean_artifact_null": float(metrics["clean_scdw_max"]) <= maximum_clean_scdw,
    }
    passed = all(gates.values())
    return {
        "verdict": "GO_TO_12ROUND_ABC" if passed else "NO_GO_PNCB_BRIDGE",
        "gates": gates,
        "values": dict(metrics),
        "thresholds": {
            "minimum_semantic_accuracy_delta": minimum_semantic_accuracy_delta,
            "minimum_variance_contraction": minimum_variance_contraction,
            "minimum_separability_reduction": minimum_separability_reduction,
            "minimum_gamma9_map": minimum_gamma9_map,
            "minimum_map_delta": minimum_map_delta,
            "minimum_hit_rate": minimum_hit_rate,
            "minimum_positive_clients": minimum_positive_clients,
            "minimum_overlay_margin": minimum_overlay_margin,
            "maximum_clean_scdw": maximum_clean_scdw,
        },
    }
