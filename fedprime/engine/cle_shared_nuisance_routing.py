from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1.0e-12
SVD_RELATIVE_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class SharednessStatistics:
    mean_a: np.ndarray
    mean_b: np.ndarray
    energy_a: np.ndarray
    energy_b: np.ndarray
    sharedness: np.ndarray


@dataclass(frozen=True)
class ProbeMatching:
    high_probe_ids: np.ndarray
    matched_probe_ids: np.ndarray
    high_weights: np.ndarray


@dataclass(frozen=True)
class Subspace:
    basis: np.ndarray
    singular_values: np.ndarray
    rank: int


def _as_float64(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = _as_float64(values)
    weights = _as_float64(weights)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("values and weights must be matching one-dimensional arrays")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return float(np.dot(values, weights / total))


def sharedness_statistics(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> SharednessStatistics:
    """Carrier cross-fit sharedness for probe-wise representation responses.

    ``delta_a`` and ``delta_b`` have shape ``[carrier, probe, feature]``.
    """

    delta_a = _as_float64(delta_a)
    delta_b = _as_float64(delta_b)
    if delta_a.ndim != 3 or delta_b.ndim != 3:
        raise ValueError("representation deltas must have shape [carrier, probe, feature]")
    if delta_a.shape[1:] != delta_b.shape[1:]:
        raise ValueError("carrier halves must share probe and feature dimensions")
    mean_a = delta_a.mean(axis=0)
    mean_b = delta_b.mean(axis=0)
    energy_a = np.square(delta_a).sum(axis=-1).mean(axis=0)
    energy_b = np.square(delta_b).sum(axis=-1).mean(axis=0)
    numerator = np.maximum(np.einsum("pd,pd->p", mean_a, mean_b), 0.0)
    denominator = np.sqrt(energy_a * energy_b) + float(epsilon)
    return SharednessStatistics(
        mean_a=mean_a,
        mean_b=mean_b,
        energy_a=energy_a,
        energy_b=energy_b,
        sharedness=numerator / denominator,
    )


def match_low_energy_probes(
    high_probe_ids: np.ndarray,
    active_probe_ids: np.ndarray,
    representation_energy: np.ndarray,
    high_weights: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> ProbeMatching:
    """Match each high-risk probe to a distinct active low-risk energy peer.

    Matching minimizes absolute log representation-energy distance. Probe id is
    the deterministic tie breaker. High probes are processed in their frozen
    manifest order and the paired low probe inherits the high probe's weight.
    """

    high = np.asarray(high_probe_ids, dtype=np.int64)
    active = np.asarray(active_probe_ids, dtype=np.int64)
    energy = _as_float64(representation_energy)
    weights = _as_float64(high_weights)
    if high.ndim != 1 or active.ndim != 1 or energy.ndim != 1:
        raise ValueError("probe ids and energy must be one-dimensional")
    if weights.shape != high.shape:
        raise ValueError("one frozen weight is required per high-risk probe")
    if np.unique(high).size != high.size:
        raise ValueError("high-risk probe ids must be unique")
    candidates = np.setdiff1d(active, high, assume_unique=False)
    if candidates.size < high.size:
        raise ValueError("not enough active non-high probes for one-to-one matching")
    log_energy = np.log(np.maximum(energy, float(epsilon)))
    used: set[int] = set()
    matched: list[int] = []
    for probe_id in high.tolist():
        available = np.asarray([item for item in candidates.tolist() if item not in used], dtype=np.int64)
        distances = np.abs(log_energy[available] - log_energy[int(probe_id)])
        order = np.lexsort((available, distances))
        chosen = int(available[order[0]])
        used.add(chosen)
        matched.append(chosen)
    normalized = weights / weights.sum()
    return ProbeMatching(high, np.asarray(matched, dtype=np.int64), normalized)


def weighted_response_subspace(
    probe_means: np.ndarray,
    weights: np.ndarray,
    *,
    relative_tolerance: float = SVD_RELATIVE_TOLERANCE,
) -> Subspace:
    """Return the numerical column span of sqrt(w_q) * mean(delta h_q)."""

    means = _as_float64(probe_means)
    weights = _as_float64(weights)
    if means.ndim != 2 or weights.shape != (means.shape[0],):
        raise ValueError("probe_means must be [probe, feature] with matching weights")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("weights must be non-negative with positive sum")
    matrix = (means * np.sqrt(weights / weights.sum())[:, None]).T
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
        rank = 0
    else:
        rank = int(np.sum(singular_values > float(relative_tolerance) * singular_values[0]))
    return Subspace(left[:, :rank], singular_values, rank)


def projection_fraction(vector: np.ndarray, basis: np.ndarray, *, epsilon: float = EPSILON) -> float:
    vector = _as_float64(vector)
    basis = _as_float64(basis)
    if vector.ndim != 1 or basis.ndim != 2 or basis.shape[0] != vector.size:
        raise ValueError("basis must have shape [feature, rank]")
    denominator = float(np.dot(vector, vector)) + float(epsilon)
    if basis.shape[1] == 0:
        return 0.0
    projected = basis.T @ vector
    return float(np.dot(projected, projected) / denominator)


def aggregate_transfer(
    target_probe_means: np.ndarray,
    target_weights: np.ndarray,
    basis: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> tuple[float, np.ndarray]:
    means = _as_float64(target_probe_means)
    if means.ndim != 2:
        raise ValueError("target_probe_means must have shape [probe, feature]")
    fractions = np.asarray(
        [projection_fraction(vector, basis, epsilon=epsilon) for vector in means],
        dtype=np.float64,
    )
    return weighted_mean(fractions, target_weights), fractions


def paired_random_subspace_bases(
    feature_dimension: int,
    rank_left: int,
    rank_right: int,
    *,
    draws: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rank-matched Haar-style Gaussian QR null with a paired base stream."""

    if min(feature_dimension, rank_left, rank_right, draws) < 0:
        raise ValueError("dimensions, ranks and draws must be non-negative")
    maximum_rank = max(int(rank_left), int(rank_right))
    if maximum_rank > int(feature_dimension):
        raise ValueError("subspace rank cannot exceed feature dimension")
    rng = np.random.default_rng(int(seed))
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for _ in range(int(draws)):
        if maximum_rank == 0:
            base = np.empty((int(feature_dimension), 0), dtype=np.float64)
        else:
            gaussian = rng.standard_normal((int(feature_dimension), maximum_rank))
            base, _ = np.linalg.qr(gaussian, mode="reduced")
        left.append(base[:, : int(rank_left)].copy())
        right.append(base[:, : int(rank_right)].copy())
    return left, right


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = _as_float64(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    tail = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def bootstrap_index_matrix(*, carriers: int, replicates: int, seed: int) -> np.ndarray:
    if carriers <= 0 or replicates <= 0:
        raise ValueError("carriers and replicates must be positive")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(carriers), size=(int(replicates), int(carriers)), dtype=np.int64)


def _bootstrap_count_matrix(indices: np.ndarray, carriers: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 2:
        raise ValueError("bootstrap indices must have shape [replicate, draw]")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= int(carriers)):
        raise ValueError("bootstrap index is outside the carrier range")
    counts = np.zeros((indices.shape[0], int(carriers)), dtype=np.float64)
    for replicate in range(indices.shape[0]):
        np.add.at(counts[replicate], indices[replicate], 1.0)
    return counts


def _bootstrap_means_and_energy(
    delta: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = _as_float64(delta)
    if delta.ndim != 3:
        raise ValueError("representation deltas must have shape [carrier, probe, feature]")
    counts = _bootstrap_count_matrix(indices, delta.shape[0])
    draws = float(np.asarray(indices).shape[1])
    means = (counts @ delta.reshape(delta.shape[0], -1) / draws).reshape(
        counts.shape[0], delta.shape[1], delta.shape[2]
    )
    squared_norm = np.square(delta).sum(axis=-1)
    energy = counts @ squared_norm / draws
    return means, energy


def bootstrap_sharedness(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    weights: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Recompute aggregate sharedness for every carrier bootstrap replicate."""

    delta_a = _as_float64(delta_a)
    delta_b = _as_float64(delta_b)
    indices_a = np.asarray(indices_a, dtype=np.int64)
    indices_b = np.asarray(indices_b, dtype=np.int64)
    if indices_a.ndim != 2 or indices_b.ndim != 2 or indices_a.shape[0] != indices_b.shape[0]:
        raise ValueError("paired bootstrap index matrices are required")
    mean_a, energy_a = _bootstrap_means_and_energy(delta_a, indices_a)
    mean_b, energy_b = _bootstrap_means_and_energy(delta_b, indices_b)
    numerator = np.maximum(np.einsum("bpd,bpd->bp", mean_a, mean_b), 0.0)
    sharedness = numerator / (np.sqrt(energy_a * energy_b) + float(epsilon))
    normalized = _as_float64(weights)
    normalized = normalized / normalized.sum()
    return sharedness @ normalized


def bootstrap_cross_bank_transfer(
    source_delta: np.ndarray,
    target_delta: np.ndarray,
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    *,
    relative_tolerance: float = SVD_RELATIVE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the source span and transfer score in every bootstrap replicate."""

    source_delta = _as_float64(source_delta)
    target_delta = _as_float64(target_delta)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if source_indices.shape[0] != target_indices.shape[0]:
        raise ValueError("source and target bootstrap replicates must be paired")
    source_means, _ = _bootstrap_means_and_energy(source_delta, source_indices)
    target_means, _ = _bootstrap_means_and_energy(target_delta, target_indices)
    scores = np.empty(source_indices.shape[0], dtype=np.float64)
    ranks = np.empty(source_indices.shape[0], dtype=np.int64)
    for replicate in range(source_indices.shape[0]):
        subspace = weighted_response_subspace(
            source_means[replicate],
            source_weights,
            relative_tolerance=relative_tolerance,
        )
        scores[replicate], _ = aggregate_transfer(
            target_means[replicate], target_weights, subspace.basis
        )
        ranks[replicate] = subspace.rank
    return scores, ranks


__all__ = [
    "EPSILON",
    "SVD_RELATIVE_TOLERANCE",
    "ProbeMatching",
    "SharednessStatistics",
    "Subspace",
    "aggregate_transfer",
    "bootstrap_cross_bank_transfer",
    "bootstrap_index_matrix",
    "bootstrap_sharedness",
    "match_low_energy_probes",
    "paired_random_subspace_bases",
    "percentile_interval",
    "projection_fraction",
    "sharedness_statistics",
    "weighted_mean",
    "weighted_response_subspace",
]
