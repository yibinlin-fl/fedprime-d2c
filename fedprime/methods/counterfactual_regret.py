from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SIGNAL_NAMES = ("regret", "ce", "jsd")


def correct_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return true-class logit minus the largest competing logit."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, class]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if logits.shape[1] < 2:
        raise ValueError("at least two classes are required")
    labels = labels.long()
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - competitors.max(dim=1).values


def counterfactual_regret(
    base_margin: torch.Tensor,
    augmented_margins: torch.Tensor,
) -> torch.Tensor:
    """One-sided margin drop from a base view to the worst augmented view."""

    if base_margin.ndim != 1:
        raise ValueError("base_margin must have shape [batch]")
    if augmented_margins.ndim != 2 or augmented_margins.shape[0] != base_margin.shape[0]:
        raise ValueError("augmented_margins must have shape [batch, views]")
    if augmented_margins.shape[1] < 1:
        raise ValueError("at least one augmented view is required")
    return torch.relu(base_margin - augmented_margins.min(dim=1).values)


def per_sample_jsd(logits_views: Sequence[torch.Tensor]) -> torch.Tensor:
    """Per-sample Jensen-Shannon divergence across matching logit views."""

    if len(logits_views) < 2:
        raise ValueError("at least two logit views are required")
    shape = logits_views[0].shape
    if len(shape) != 2 or any(view.shape != shape for view in logits_views):
        raise ValueError("all logit views must share shape [batch, class]")
    probabilities = [F.softmax(view, dim=1).clamp_min(1.0e-12) for view in logits_views]
    mixture = torch.stack(probabilities, dim=0).mean(dim=0).clamp_min(1.0e-12)
    return torch.stack(
        [(probability * (probability.log() - mixture.log())).sum(dim=1) for probability in probabilities],
        dim=0,
    ).mean(dim=0)


def split_fit_internal_probe(
    fit_indices: np.ndarray,
    labels: np.ndarray,
    *,
    ratio: float = 0.15,
    min_class_count: int = 32,
    min_probe: int = 16,
    max_probe: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic class-stratified probe strictly inside fit."""

    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if fit_indices.ndim != 1 or labels.ndim != 1:
        raise ValueError("fit_indices and labels must be one-dimensional")
    if np.any(fit_indices < 0) or np.any(fit_indices >= labels.size):
        raise ValueError("fit_indices contains an out-of-range index")
    if not 0.0 < float(ratio) < 1.0:
        raise ValueError("ratio must be between zero and one")
    if min_class_count < 2 or min_probe < 1 or max_probe < min_probe:
        raise ValueError("invalid probe support settings")

    rng = np.random.default_rng(int(seed))
    probe_parts: list[np.ndarray] = []
    for class_id in np.unique(labels[fit_indices]):
        class_indices = fit_indices[labels[fit_indices] == int(class_id)]
        if class_indices.size < int(min_class_count):
            continue
        proposed = max(int(min_probe), int(round(float(ratio) * class_indices.size)))
        probe_count = min(int(max_probe), proposed, int(class_indices.size) - 2)
        shuffled = rng.permutation(class_indices)
        probe_parts.append(shuffled[:probe_count])

    if not probe_parts:
        raise ValueError("no class has enough support for a fit-internal probe")
    probe = np.sort(np.concatenate(probe_parts)).astype(np.int64, copy=False)
    train = np.setdiff1d(fit_indices, probe, assume_unique=False).astype(np.int64, copy=False)
    if np.intersect1d(train, probe).size or train.size + probe.size != np.unique(fit_indices).size:
        raise AssertionError("fit-internal train/probe split is not disjoint and exhaustive")
    return train, probe


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
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
    if valid.sum() < 2:
        return float("nan")
    left_rank = _rankdata(left[valid])
    right_rank = _rankdata(right[valid])
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def binary_auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & np.isin(targets, [0, 1])
    scores = scores[valid]
    targets = targets[valid]
    positives = int((targets == 1).sum())
    negatives = int((targets == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    positive_rank_sum = _rankdata(scores)[targets == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def class_percentile_ranks(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    min_class_support: int = 8,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.shape != labels.shape:
        raise ValueError("values and labels must have matching shape")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for class_id in np.unique(labels):
        selected = np.flatnonzero(labels == int(class_id))
        if selected.size < int(min_class_support):
            continue
        ranks = _rankdata(values[selected])
        result[selected] = (ranks - 0.5) / float(selected.size)
    return result


def class_balanced_spearman(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    *,
    min_class_support: int = 8,
) -> float:
    correlations = []
    for class_id in np.unique(labels):
        selected = np.flatnonzero(labels == int(class_id))
        if selected.size < int(min_class_support):
            continue
        correlation = spearman_correlation(left[selected], right[selected])
        if np.isfinite(correlation):
            correlations.append(correlation)
    return float(np.mean(correlations)) if correlations else float("nan")


def top_fraction_enrichment(
    scores: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    *,
    fraction: float = 0.25,
    min_class_support: int = 8,
) -> float:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ranked = class_percentile_ranks(scores, labels, min_class_support=min_class_support)
    valid = np.isfinite(ranked)
    if not valid.any():
        return float("nan")
    targets = np.asarray(targets, dtype=np.float64)
    prevalence = float(targets[valid].mean())
    if prevalence <= 0.0:
        return float("nan")
    selected = valid & (ranked >= 1.0 - float(fraction))
    if not selected.any():
        return float("nan")
    return float(targets[selected].mean() / prevalence)


def cell_relevance(
    scores: np.ndarray,
    robust_errors: np.ndarray,
    labels: np.ndarray,
    corruption_ids: np.ndarray,
    *,
    min_class_support: int = 8,
    min_cell_support: int = 8,
) -> tuple[float, int]:
    ranked = class_percentile_ranks(scores, labels, min_class_support=min_class_support)
    labels = np.asarray(labels, dtype=np.int64)
    corruption_ids = np.asarray(corruption_ids, dtype=np.int64)
    robust_errors = np.asarray(robust_errors, dtype=np.float64)
    cell_scores = []
    cell_errors = []
    for class_id, corruption_id in np.unique(np.stack([labels, corruption_ids], axis=1), axis=0):
        selected = (labels == int(class_id)) & (corruption_ids == int(corruption_id)) & np.isfinite(ranked)
        if selected.sum() < int(min_cell_support):
            continue
        cell_scores.append(float(ranked[selected].mean()))
        cell_errors.append(float(robust_errors[selected].mean()))
    return spearman_correlation(np.asarray(cell_scores), np.asarray(cell_errors)), len(cell_scores)


@dataclass(frozen=True)
class SeedSignals:
    sample_indices: np.ndarray
    labels: np.ndarray
    corruption_ids: np.ndarray
    regret: np.ndarray
    ce: np.ndarray
    jsd: np.ndarray
    robust_error: np.ndarray
    flip_error: np.ndarray
    base_accuracy: float

    def validate(self) -> None:
        size = self.sample_indices.size
        arrays = (
            self.labels,
            self.corruption_ids,
            self.regret,
            self.ce,
            self.jsd,
            self.robust_error,
            self.flip_error,
        )
        if any(np.asarray(array).shape != (size,) for array in arrays):
            raise ValueError("all SeedSignals arrays must be one-dimensional and aligned")
        if np.unique(self.sample_indices).size != size:
            raise ValueError("sample_indices must be unique")


def evaluate_directed_pair(
    source: SeedSignals,
    target: SeedSignals,
    *,
    min_class_support: int = 8,
    min_cell_support: int = 8,
    top_fraction: float = 0.25,
) -> dict[str, object]:
    source.validate()
    target.validate()
    if not np.array_equal(source.sample_indices, target.sample_indices):
        raise ValueError("source and target samples must be identically ordered")
    if not np.array_equal(source.labels, target.labels) or not np.array_equal(
        source.corruption_ids, target.corruption_ids
    ):
        raise ValueError("source and target metadata must match")

    metrics: dict[str, object] = {
        "flip_prevalence": float(np.mean(target.flip_error)),
        "robust_error_prevalence": float(np.mean(target.robust_error)),
        "regret_persistence": class_balanced_spearman(
            source.regret,
            target.regret,
            source.labels,
            min_class_support=min_class_support,
        ),
        "signals": {},
    }
    for name in SIGNAL_NAMES:
        values = np.asarray(getattr(source, name), dtype=np.float64)
        ranked = class_percentile_ranks(values, source.labels, min_class_support=min_class_support)
        valid = np.isfinite(ranked)
        cell_corr, valid_cells = cell_relevance(
            values,
            target.robust_error,
            source.labels,
            source.corruption_ids,
            min_class_support=min_class_support,
            min_cell_support=min_cell_support,
        )
        metrics["signals"][name] = {
            "flip_auroc": binary_auroc(ranked[valid], target.flip_error[valid]),
            "top_fraction_enrichment": top_fraction_enrichment(
                values,
                target.flip_error,
                source.labels,
                fraction=top_fraction,
                min_class_support=min_class_support,
            ),
            "cell_correlation": cell_corr,
            "valid_cells": int(valid_cells),
        }
    return metrics


def _median_finite(values: Sequence[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.median(finite)) if finite.size else float("nan")


def decide_audit0(
    clients: Mapping[int, Mapping[str, object]],
    pair_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the pre-registered G0--G6 gates without tuning from results."""

    client_validity = {}
    client_activity = {}
    for client_id, payload in clients.items():
        accuracies = [float(value) for value in payload["base_accuracies"]]
        positive = [float(value) for value in payload["regret_positive_fractions"]]
        p90 = [float(value) for value in payload["regret_p90"]]
        client_validity[int(client_id)] = float(np.mean(accuracies)) >= 0.20
        client_activity[int(client_id)] = _median_finite(positive) >= 0.20 and _median_finite(p90) >= 0.25

    flip_prevalences = [float(pair["flip_prevalence"]) for pair in pair_metrics]
    g0 = all(client_validity.values()) and all(0.02 <= value <= 0.80 for value in flip_prevalences)
    g1 = all(client_activity.values())

    persistence = [float(pair["regret_persistence"]) for pair in pair_metrics]
    regret_auc = [float(pair["signals"]["regret"]["flip_auroc"]) for pair in pair_metrics]
    baseline_auc = [
        max(float(pair["signals"]["ce"]["flip_auroc"]), float(pair["signals"]["jsd"]["flip_auroc"]))
        for pair in pair_metrics
    ]
    auc_advantage = [candidate - baseline for candidate, baseline in zip(regret_auc, baseline_auc)]
    regret_enrichment = [
        float(pair["signals"]["regret"]["top_fraction_enrichment"]) for pair in pair_metrics
    ]
    baseline_enrichment = [
        max(
            float(pair["signals"]["ce"]["top_fraction_enrichment"]),
            float(pair["signals"]["jsd"]["top_fraction_enrichment"]),
        )
        for pair in pair_metrics
    ]
    enrichment_advantage = [
        candidate - baseline for candidate, baseline in zip(regret_enrichment, baseline_enrichment)
    ]
    cell_correlations = [
        float(pair["signals"]["regret"]["cell_correlation"]) for pair in pair_metrics
    ]
    cells_by_client: dict[int, list[int]] = {}
    wins_by_client: dict[int, int] = {}
    total_wins = 0
    for pair, candidate, baseline in zip(pair_metrics, regret_auc, baseline_auc):
        client_id = int(pair["client_id"])
        cells_by_client.setdefault(client_id, []).append(int(pair["signals"]["regret"]["valid_cells"]))
        won = int(np.isfinite(candidate) and np.isfinite(baseline) and candidate > baseline)
        total_wins += won
        wins_by_client[client_id] = wins_by_client.get(client_id, 0) + won

    valid_cell_total = sum(min(values) for values in cells_by_client.values())
    g2 = _median_finite(persistence) >= 0.25
    g3 = _median_finite(regret_auc) >= 0.60 and _median_finite(auc_advantage) >= 0.02
    g4 = _median_finite(regret_enrichment) >= 1.30 and _median_finite(enrichment_advantage) >= 0.0
    g5 = valid_cell_total >= 20 and _median_finite(cell_correlations) >= 0.30
    g6 = total_wins >= 4 and all(wins_by_client.get(int(client_id), 0) >= 1 for client_id in clients)

    gates = {
        "G0_validity": {"pass": g0, "client_validity": client_validity, "flip_prevalences": flip_prevalences},
        "G1_activity": {"pass": g1, "client_activity": client_activity},
        "G2_persistence": {"pass": g2, "median": _median_finite(persistence)},
        "G3_predictive_auroc": {
            "pass": g3,
            "regret_median": _median_finite(regret_auc),
            "advantage_median": _median_finite(auc_advantage),
        },
        "G4_tail_enrichment": {
            "pass": g4,
            "regret_median": _median_finite(regret_enrichment),
            "advantage_median": _median_finite(enrichment_advantage),
        },
        "G5_cell_relevance": {
            "pass": g5,
            "valid_cell_total": int(valid_cell_total),
            "correlation_median": _median_finite(cell_correlations),
        },
        "G6_cross_client": {"pass": g6, "total_wins": int(total_wins), "wins_by_client": wins_by_client},
    }
    if not g0:
        verdict = "INVALID_PROBE"
    else:
        verdict = "GO" if all(bool(gate["pass"]) for gate in gates.values()) else "NO-GO"
    return {"verdict": verdict, "gates": gates}
