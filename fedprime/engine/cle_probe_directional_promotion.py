from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeDirectionalPromotion:
    matrix: np.ndarray
    client: np.ndarray
    pooled: float


def probe_directional_promotion(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> ProbeDirectionalPromotion:
    """Estimate probe-to-wrong-class promotion without any binding metadata.

    Args:
        probabilities: Softmax tensor with shape [client, source, probe, class].
        labels: True task labels with shape [source].

    The estimator intentionally accepts neither corruption-family ids nor a
    class-to-family binding map. For each candidate target class, sources of
    that true class are excluded before contrasting one probe against all
    other probes.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 4:
        raise ValueError("probabilities must have shape [client,source,probe,class]")
    if labels.shape != (probs.shape[1],):
        raise ValueError("labels must have shape [source]")
    if probs.shape[2] < 2:
        raise ValueError("at least two distinguishable probes are required")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(probs < -1.0e-7) or not np.allclose(probs.sum(axis=-1), 1.0, atol=1.0e-5):
        raise ValueError("probabilities must be normalized non-negative softmax values")

    num_clients, _, num_probes, num_classes = probs.shape
    probe_class_means = np.empty((num_clients, num_probes, num_classes), dtype=np.float64)
    for class_id in range(num_classes):
        valid = labels != class_id
        if not bool(valid.any()):
            raise ValueError(f"no off-target sources are available for class {class_id}")
        class_probabilities = probs[..., class_id]
        probe_class_means[:, :, class_id] = class_probabilities[:, valid, :].mean(axis=1)

    other_probe_mean = (
        probe_class_means.sum(axis=1, keepdims=True) - probe_class_means
    ) / float(num_probes - 1)
    matrix = probe_class_means - other_probe_mean
    client = np.maximum(matrix, 0.0).sum(axis=(1, 2)) / float(num_probes)
    return ProbeDirectionalPromotion(
        matrix=matrix,
        client=client,
        pooled=float(client.mean()),
    )


def binding_truth(
    binding: np.ndarray,
    probe_family_ids: np.ndarray,
) -> np.ndarray:
    binding = np.asarray(binding, dtype=np.int64)
    family_ids = np.asarray(probe_family_ids, dtype=np.int64)
    if binding.ndim != 2:
        raise ValueError("binding must have shape [client,class]")
    if family_ids.ndim != 1:
        raise ValueError("probe_family_ids must have shape [probe]")
    return binding[:, None, :] == family_ids[None, :, None]


def _average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = truth[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked].sum() / positives)


def _roc_auc(scores: np.ndarray, truth: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    positive = scores[truth]
    negative = scores[~truth]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size)


def score_binding_retrieval(
    promotion_matrix: np.ndarray,
    binding: np.ndarray,
    probe_family_ids: np.ndarray,
) -> dict[str, object]:
    matrix = np.asarray(promotion_matrix, dtype=np.float64)
    truth = binding_truth(binding, probe_family_ids)
    if matrix.shape != truth.shape:
        raise ValueError("promotion matrix and binding truth must have identical shapes")

    client_map = np.asarray(
        [np.mean([_average_precision(matrix[i, j], truth[i, j]) for j in range(matrix.shape[1])]) for i in range(matrix.shape[0])],
        dtype=np.float64,
    )
    client_auc = np.asarray(
        [_roc_auc(matrix[i].reshape(-1), truth[i].reshape(-1)) for i in range(matrix.shape[0])],
        dtype=np.float64,
    )
    predicted_positive = matrix > 0.0
    true_positive = np.logical_and(predicted_positive, truth).sum(axis=(1, 2))
    predicted_count = predicted_positive.sum(axis=(1, 2))
    truth_count = truth.sum(axis=(1, 2))
    client_precision = true_positive / np.maximum(predicted_count, 1)
    client_recall = true_positive / np.maximum(truth_count, 1)

    family_ids = np.asarray(probe_family_ids, dtype=np.int64)
    binding_array = np.asarray(binding, dtype=np.int64)
    best_probe = matrix.argmax(axis=1)
    client_hit = np.mean(
        family_ids[best_probe] == binding_array,
        axis=1,
    )
    pidr_cell = np.maximum(matrix, 0.0).sum(axis=2)
    aligned_cell = np.where(truth, matrix, 0.0).sum(axis=2)
    if np.std(pidr_cell) > 0.0 and np.std(aligned_cell) > 0.0:
        pidr_centered = pidr_cell.reshape(-1) - pidr_cell.mean()
        aligned_centered = aligned_cell.reshape(-1) - aligned_cell.mean()
        denominator = np.linalg.norm(pidr_centered) * np.linalg.norm(aligned_centered)
        alignment_correlation = float(np.dot(pidr_centered, aligned_centered) / denominator)
    else:
        alignment_correlation = float("nan")

    probe_ap = np.asarray(
        [[_average_precision(matrix[i, j], truth[i, j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])],
        dtype=np.float64,
    )
    return {
        "mean_average_precision": float(client_map.mean()),
        "roc_auc": float(client_auc.mean()),
        "positive_precision": float(client_precision.mean()),
        "positive_recall": float(client_recall.mean()),
        "class_to_probe_family_hit_rate": float(client_hit.mean()),
        "client_mean_average_precision": client_map.tolist(),
        "client_roc_auc": client_auc.tolist(),
        "client_positive_precision": client_precision.tolist(),
        "client_positive_recall": client_recall.tolist(),
        "client_class_to_probe_family_hit_rate": client_hit.tolist(),
        "probe_average_precision_min": float(probe_ap.min()),
        "probe_average_precision_median": float(np.median(probe_ap)),
        "probe_average_precision": probe_ap.tolist(),
        "oracle_aligned_promotion": float(aligned_cell.mean()),
        "pidr_oracle_alignment_correlation": alignment_correlation,
    }


def shuffled_retrieval_nulls(
    promotion_matrix: np.ndarray,
    binding: np.ndarray,
    probe_family_ids: np.ndarray,
    *,
    permutations: int = 1000,
    seed: int = 20260830,
) -> dict[str, object]:
    matrix = np.asarray(promotion_matrix, dtype=np.float64)
    binding_array = np.asarray(binding, dtype=np.int64)
    family_ids = np.asarray(probe_family_ids, dtype=np.int64)
    observed = score_binding_retrieval(matrix, binding_array, family_ids)
    rng = np.random.default_rng(int(seed))
    metric_names = (
        "mean_average_precision",
        "roc_auc",
        "class_to_probe_family_hit_rate",
    )
    class_null = {name: np.empty(int(permutations), dtype=np.float64) for name in metric_names}
    probe_null = {name: np.empty(int(permutations), dtype=np.float64) for name in metric_names}
    for permutation_id in range(int(permutations)):
        shuffled_binding = np.stack(
            [row[rng.permutation(row.size)] for row in binding_array],
            axis=0,
        )
        shuffled_families = family_ids[rng.permutation(family_ids.size)]
        class_metrics = score_binding_retrieval(matrix, shuffled_binding, family_ids)
        probe_metrics = score_binding_retrieval(matrix, binding_array, shuffled_families)
        for name in metric_names:
            class_null[name][permutation_id] = float(class_metrics[name])
            probe_null[name][permutation_id] = float(probe_metrics[name])

    def summarize(values: np.ndarray, observed_value: float) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p95": float(np.quantile(values, 0.95)),
            "one_sided_p": float((1.0 + np.count_nonzero(values >= observed_value)) / (values.size + 1.0)),
        }

    return {
        "observed": {name: float(observed[name]) for name in metric_names},
        "class_map_null": {
            name: summarize(class_null[name], float(observed[name])) for name in metric_names
        },
        "probe_identity_null": {
            name: summarize(probe_null[name], float(observed[name])) for name in metric_names
        },
    }


def decide_zero_training_gate(
    arm_results: dict[str, dict[str, object]],
    *,
    minimum_gamma9_map: float = 0.60,
    minimum_map_delta: float = 0.20,
    minimum_hit_rate: float = 0.70,
    maximum_null_p: float = 0.01,
) -> dict[str, object]:
    required = {"h0", "h9", "l0", "l9"}
    if set(arm_results) != required:
        raise ValueError(f"arm_results must contain exactly {sorted(required)}")

    gates: dict[str, object] = {}
    all_pass = True
    for system, zero_arm, nine_arm in (("hfl", "h0", "h9"), ("local", "l0", "l9")):
        zero_metrics = arm_results[zero_arm]["retrieval"]
        nine_metrics = arm_results[nine_arm]["retrieval"]
        nine_nulls = arm_results[nine_arm]["nulls"]
        delta = float(nine_metrics["mean_average_precision"] - zero_metrics["mean_average_precision"])
        client_delta = np.asarray(nine_metrics["client_mean_average_precision"]) - np.asarray(
            zero_metrics["client_mean_average_precision"]
        )
        values = {
            "gamma9_map": float(nine_metrics["mean_average_precision"]),
            "map_delta": delta,
            "positive_clients": int(np.count_nonzero(client_delta > 0.0)),
            "gamma9_hit_rate": float(nine_metrics["class_to_probe_family_hit_rate"]),
            "class_map_p": float(nine_nulls["class_map_null"]["mean_average_precision"]["one_sided_p"]),
            "probe_identity_p": float(nine_nulls["probe_identity_null"]["mean_average_precision"]["one_sided_p"]),
            "client_map_delta": client_delta.tolist(),
        }
        system_gates = {
            "G1_gamma9_map": values["gamma9_map"] >= minimum_gamma9_map,
            "G2_gamma_map_delta": values["map_delta"] >= minimum_map_delta,
            "G3_client_direction": values["positive_clients"] >= 3,
            "G4_hit_rate": values["gamma9_hit_rate"] >= minimum_hit_rate,
            "G5_class_map_null": values["class_map_p"] <= maximum_null_p,
            "G6_probe_identity_null": values["probe_identity_p"] <= maximum_null_p,
        }
        passed = all(system_gates.values())
        all_pass = all_pass and passed
        gates[system] = {"values": values, "gates": system_gates, "pass": passed}
    return {
        "verdict": "GO_TO_INTERVENTION_BRIDGE_DESIGN" if all_pass else "NO_GO_PIDR_OBSERVABILITY",
        "thresholds": {
            "minimum_gamma9_map": minimum_gamma9_map,
            "minimum_map_delta": minimum_map_delta,
            "minimum_hit_rate": minimum_hit_rate,
            "maximum_null_p": maximum_null_p,
        },
        "systems": gates,
    }
