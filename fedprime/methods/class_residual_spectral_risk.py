from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _torch_leading_eigenpair(
    covariance: torch.Tensor,
    *,
    iterations: int = 12,
    epsilon: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable power iteration without a platform LAPACK dependency."""

    index = int(covariance.detach().diagonal().argmax().item())
    direction = F.one_hot(
        torch.tensor(index, device=covariance.device), num_classes=covariance.shape[0]
    ).to(covariance.dtype)
    for _ in range(int(iterations)):
        updated = covariance.matmul(direction)
        norm = torch.sqrt(updated.square().sum()).clamp_min(float(epsilon))
        direction = updated / norm
    eigenvalue = direction.dot(covariance.matmul(direction)).clamp_min(0.0)
    return eigenvalue, direction


def _numpy_leading_eigenpair(
    covariance: np.ndarray,
    *,
    iterations: int = 64,
    epsilon: float = 1.0e-15,
) -> tuple[float, np.ndarray]:
    """Deterministic power iteration for small audit covariance matrices."""

    direction = np.zeros(covariance.shape[0], dtype=np.float64)
    direction[int(np.argmax(np.diag(covariance)))] = 1.0
    for _ in range(int(iterations)):
        updated = np.sum(covariance * direction[None, :], axis=1)
        norm = float(np.sqrt(np.sum(updated * updated)))
        if norm <= float(epsilon):
            return 0.0, direction
        direction = updated / norm
    projected = np.sum(covariance * direction[None, :], axis=1)
    eigenvalue = float(max(np.sum(direction * projected), 0.0))
    return eigenvalue, direction


def prediction_residuals(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return probability residuals ``softmax(logits) - one_hot(labels)``."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if logits.shape[1] < 2:
        raise ValueError("at least two classes are required")
    labels = labels.long()
    if bool((labels < 0).any()) or bool((labels >= logits.shape[1]).any()):
        raise ValueError("labels are outside the logit class range")
    probabilities = F.softmax(logits, dim=1)
    targets = F.one_hot(labels, num_classes=logits.shape[1]).to(probabilities.dtype)
    return probabilities - targets


def class_conditional_residual_spectral_risk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    spectral_weight: float = 2.0,
    min_class_count: int = 2,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-balanced CE plus within-class residual spectral dispersion.

    The spectral term is the square root of the largest eigenvalue of the
    centered probability-residual covariance. No environment annotations,
    group assignments, public examples, or corruption metadata are consumed.
    """

    if spectral_weight < 0.0:
        raise ValueError("spectral_weight must be non-negative")
    if min_class_count < 2:
        raise ValueError("min_class_count must be at least two")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    residuals = prediction_residuals(logits, labels)
    sample_ce = F.cross_entropy(logits, labels.long(), reduction="none")
    class_ce: list[torch.Tensor] = []
    radii: list[torch.Tensor] = []
    top_shares: list[torch.Tensor] = []

    for class_id in torch.unique(labels.long(), sorted=True):
        selected = labels.long() == class_id
        class_ce.append(sample_ce[selected].mean())
        if int(selected.sum().item()) < int(min_class_count):
            continue
        class_residuals = residuals[selected]
        centered = class_residuals - class_residuals.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1).matmul(centered) / float(centered.shape[0])
        top, _direction = _torch_leading_eigenpair(covariance)
        trace = covariance.diagonal().sum().clamp_min(0.0)
        radii.append(torch.sqrt(top + float(epsilon)))
        top_shares.append(top / trace.clamp_min(float(epsilon)))

    if not class_ce:
        raise ValueError("the batch contains no classes")
    balanced_ce = torch.stack(class_ce).mean()
    if radii:
        spectral_radius = torch.stack(radii).mean()
        top_share = torch.stack(top_shares).mean()
    else:
        spectral_radius = balanced_ce.new_zeros(())
        top_share = balanced_ce.new_zeros(())
    loss = balanced_ce + float(spectral_weight) * spectral_radius
    return loss, {
        "balanced_ce": balanced_ce,
        "spectral_radius": spectral_radius,
        "spectral_penalty": float(spectral_weight) * spectral_radius,
        "valid_spectral_classes": balanced_ce.new_tensor(float(len(radii))),
        "mean_top_eigenvalue_share": top_share,
    }


@dataclass(frozen=True)
class ResidualSpectralStatistics:
    mean: np.ndarray
    covariance: np.ndarray
    direction: np.ndarray
    top_eigenvalue: float
    trace: float
    support: int


def _validate_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or labels.shape != (probabilities.shape[0],):
        raise ValueError("probabilities and labels must have shapes [sample, class] and [sample]")
    if probabilities.shape[1] < 2 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite with at least two classes")
    if np.any(probabilities < -1.0e-8) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-5):
        raise ValueError("rows must be probability distributions")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels are outside the probability class range")
    return probabilities, labels


def fit_residual_spectral_statistics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    min_class_support: int = 16,
) -> dict[int, ResidualSpectralStatistics]:
    """Fit class-wise residual means and leading covariance directions."""

    probabilities, labels = _validate_probabilities(probabilities, labels)
    if min_class_support < 2:
        raise ValueError("min_class_support must be at least two")
    residuals = probabilities - np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    result: dict[int, ResidualSpectralStatistics] = {}
    for class_id in np.unique(labels):
        selected = labels == int(class_id)
        support = int(selected.sum())
        if support < int(min_class_support):
            continue
        values = residuals[selected]
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = np.einsum("ni,nj->ij", centered, centered, optimize=False) / float(support)
        top_eigenvalue, direction = _numpy_leading_eigenpair(covariance)
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction = -direction
        result[int(class_id)] = ResidualSpectralStatistics(
            mean=mean,
            covariance=covariance,
            direction=direction,
            top_eigenvalue=top_eigenvalue,
            trace=float(max(np.trace(covariance), 0.0)),
            support=support,
        )
    return result


def deterministic_random_directions(
    num_classes: int,
    class_ids: Sequence[int],
    *,
    seed: int,
) -> dict[int, np.ndarray]:
    """Build fixed random directions in the probability-simplex tangent space."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    result: dict[int, np.ndarray] = {}
    ones = np.ones(num_classes, dtype=np.float64)
    for class_id in class_ids:
        rng = np.random.default_rng(int(seed) * 1009 + int(class_id) * 9176 + 13)
        direction = rng.normal(size=num_classes)
        direction = direction - ones * float(direction.mean())
        norm = float(np.sqrt(np.sum(direction * direction)))
        if norm <= 1.0e-12:
            raise RuntimeError("failed to construct a random tangent direction")
        result[int(class_id)] = direction / norm
    return result


def score_residual_directions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    statistics: Mapping[int, ResidualSpectralStatistics],
    *,
    directions: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Score squared centered residual energy along class-wise directions."""

    probabilities, labels = _validate_probabilities(probabilities, labels)
    residuals = probabilities - np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    scores = np.full(labels.shape, np.nan, dtype=np.float64)
    for class_id, stats in statistics.items():
        selected = labels == int(class_id)
        if not selected.any():
            continue
        direction = stats.direction if directions is None else np.asarray(directions[int(class_id)])
        if direction.shape != (probabilities.shape[1],):
            raise ValueError("direction has an incompatible shape")
        projections = np.sum((residuals[selected] - stats.mean) * direction[None, :], axis=1)
        scores[selected] = projections * projections
    return scores


def cross_split_spectral_metrics(
    source: Mapping[int, ResidualSpectralStatistics],
    target: Mapping[int, ResidualSpectralStatistics],
    random_directions: Mapping[int, np.ndarray],
) -> dict[str, float]:
    """Measure direction stability and held-out covariance transfer."""

    cosines: list[float] = []
    transfer_shares: list[float] = []
    random_shares: list[float] = []
    source_top_shares: list[float] = []
    for class_id in sorted(set(source) & set(target)):
        left = source[class_id]
        right = target[class_id]
        if left.trace <= 0.0 or right.trace <= 0.0:
            continue
        cosine = float(np.sum(left.direction * right.direction))
        cosines.append(abs(cosine))
        left_projected = np.sum(right.covariance * left.direction[None, :], axis=1)
        random_direction = random_directions[class_id]
        random_projected = np.sum(right.covariance * random_direction[None, :], axis=1)
        transfer_shares.append(
            float(np.sum(left.direction * left_projected) / right.trace)
        )
        random_shares.append(
            float(np.sum(random_direction * random_projected) / right.trace)
        )
        source_top_shares.append(float(left.top_eigenvalue / left.trace))
    median = lambda values: float(np.median(values)) if values else float("nan")
    transfer = median(transfer_shares)
    random = median(random_shares)
    return {
        "valid_classes": float(len(cosines)),
        "median_direction_cosine": median(cosines),
        "median_source_top_share": median(source_top_shares),
        "median_transfer_share": transfer,
        "median_random_share": random,
        "median_transfer_advantage": transfer - random,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float("nan")
    left = _rankdata(left[valid])
    right = _rankdata(right[valid])
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(
        np.sqrt(np.sum(left * left)) * np.sqrt(np.sum(right * right))
    )
    return float(np.sum(left * right) / denominator) if denominator > 0.0 else float("nan")


def class_balanced_spearman(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    *,
    min_class_support: int = 8,
) -> float:
    correlations: list[float] = []
    labels = np.asarray(labels, dtype=np.int64)
    for class_id in np.unique(labels):
        selected = labels == int(class_id)
        if int(selected.sum()) < int(min_class_support):
            continue
        value = spearman_correlation(np.asarray(left)[selected], np.asarray(right)[selected])
        if np.isfinite(value):
            correlations.append(value)
    return float(np.median(correlations)) if correlations else float("nan")


def class_operator_cell_correlation(
    scores: np.ndarray,
    errors: np.ndarray,
    labels: np.ndarray,
    operator_ids: np.ndarray,
    *,
    min_cell_support: int = 4,
) -> tuple[float, int]:
    """Post-hoc correlation; operator IDs must never influence fitted scores."""

    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    operator_ids = np.asarray(operator_ids, dtype=np.int64)
    cell_scores: list[float] = []
    cell_errors: list[float] = []
    for class_id, operator_id in np.unique(np.stack([labels, operator_ids], axis=1), axis=0):
        selected = (
            (labels == int(class_id))
            & (operator_ids == int(operator_id))
            & np.isfinite(scores)
        )
        if int(selected.sum()) < int(min_cell_support):
            continue
        cell_scores.append(float(scores[selected].mean()))
        cell_errors.append(float(errors[selected].mean()))
    return spearman_correlation(np.asarray(cell_scores), np.asarray(cell_errors)), len(cell_scores)


def decide_crsr_audit0(
    client_metrics: Sequence[Mapping[str, float]],
    one_step_metrics: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, object]:
    """Apply frozen signal and optional one-step promotion gates."""

    def median(key: str) -> float:
        values = np.asarray([float(row[key]) for row in client_metrics], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else float("nan")

    valid_cells = int(sum(int(row["valid_cells"]) for row in client_metrics))
    g0 = all(float(row["base_accuracy"]) >= 0.20 and float(row["valid_classes"]) >= 6 for row in client_metrics)
    g1 = median("source_top_share") >= 0.18
    g2 = median("direction_cosine") >= 0.35 and median("transfer_advantage") >= 0.03
    g3 = median("spectral_ce_abs_correlation") <= 0.95 and median("spectral_brier_abs_correlation") <= 0.95
    baseline_cell = np.asarray(
        [
            max(
                float(row["ce_cell_correlation"]),
                float(row["brier_cell_correlation"]),
                float(row["random_cell_correlation"]),
            )
            for row in client_metrics
        ],
        dtype=np.float64,
    )
    spectral_cell = np.asarray(
        [float(row["spectral_cell_correlation"]) for row in client_metrics], dtype=np.float64
    )
    advantages = spectral_cell - baseline_cell
    finite_advantages = advantages[np.isfinite(advantages)]
    g4 = (
        valid_cells >= 20
        and median("spectral_cell_correlation") >= 0.25
        and finite_advantages.size > 0
        and float(np.median(finite_advantages)) >= 0.02
    )

    gates: dict[str, dict[str, object]] = {
        "G0_validity": {"pass": g0},
        "G1_spectral_activity": {"pass": g1, "median_top_share": median("source_top_share")},
        "G2_cross_split_stability": {
            "pass": g2,
            "median_direction_cosine": median("direction_cosine"),
            "median_transfer_advantage": median("transfer_advantage"),
        },
        "G3_nonredundancy": {
            "pass": g3,
            "median_abs_corr_ce": median("spectral_ce_abs_correlation"),
            "median_abs_corr_brier": median("spectral_brier_abs_correlation"),
        },
        "G4_cell_relevance": {
            "pass": g4,
            "valid_cells": valid_cells,
            "median_spectral_correlation": median("spectral_cell_correlation"),
            "median_baseline_advantage": (
                float(np.median(finite_advantages)) if finite_advantages.size else float("nan")
            ),
        },
    }

    if one_step_metrics is not None:
        mean_delta = np.asarray([float(row["mean_ce_delta"]) for row in one_step_metrics])
        worst_delta = np.asarray([float(row["worst_cell_ce_delta"]) for row in one_step_metrics])
        gap_delta = np.asarray([float(row["cell_gap_ce_delta"]) for row in one_step_metrics])
        g5 = bool(np.all(mean_delta <= 1.0e-4))
        g6 = bool(np.all(worst_delta < -1.0e-5) and np.all(gap_delta <= 0.0))
        gates["G5_one_step_mean_noninferiority"] = {
            "pass": g5,
            "deltas": mean_delta.tolist(),
        }
        gates["G6_one_step_weak_cell"] = {
            "pass": g6,
            "worst_cell_deltas": worst_delta.tolist(),
            "cell_gap_deltas": gap_delta.tolist(),
        }

    verdict = "GO" if gates and all(bool(gate["pass"]) for gate in gates.values()) else "NO-GO"
    if not g0:
        verdict = "INVALID_PROBE"
    return {"verdict": verdict, "gates": gates}
