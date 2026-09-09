from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _nanmean(array: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    valid = np.isfinite(values)
    counts = valid.sum(axis=axis)
    totals = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(
        totals,
        counts,
        out=np.full_like(totals, np.nan, dtype=np.float64),
        where=counts > 0,
    )


@dataclass(frozen=True)
class OperatorDSAResult:
    source_effects: np.ndarray
    client_operator: np.ndarray
    client: np.ndarray
    pooled: float


def validate_operator_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
) -> None:
    probs = np.asarray(probabilities)
    labels = np.asarray(labels)
    binding = np.asarray(binding)
    if probs.ndim != 4:
        raise ValueError("probabilities must have shape [client,source,operator,class]")
    clients, sources, operators, classes = probs.shape
    if labels.shape != (sources,):
        raise ValueError("labels must have shape [source]")
    if binding.shape != (clients, classes):
        raise ValueError("binding must have shape [client,class]")
    if np.any(binding < 0) or np.any(binding >= operators):
        raise ValueError("binding contains an out-of-range operator id")
    if not np.isfinite(probs).all() or np.any(probs < -1.0e-7):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(probs.sum(axis=-1), 1.0, atol=1.0e-5):
        raise ValueError("probabilities do not sum to one")


def compute_operator_dsa(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
) -> OperatorDSAResult:
    """Compute concrete-operator directional shortcut alignment for CLE-HFL v2.

    For every client and bound operator, probability mass assigned to the
    operator-bound classes under that operator is contrasted with the same
    source under all other operators. Sources whose true label is one of the
    bound classes are excluded.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    binding = np.asarray(binding, dtype=np.int64)
    validate_operator_inputs(probs, labels, binding)
    clients, sources, operators, _ = probs.shape
    effects = np.full((clients, operators, sources), np.nan, dtype=np.float64)
    for client_id in range(clients):
        for operator_id in range(operators):
            bound_classes = np.flatnonzero(binding[client_id] == operator_id)
            if bound_classes.size == 0:
                continue
            valid = ~np.isin(labels, bound_classes)
            if not bool(valid.any()):
                continue
            mass = np.take(probs[client_id], bound_classes, axis=-1).sum(axis=-1)
            target = mass[:, operator_id]
            reference = (mass.sum(axis=1) - target) / max(operators - 1, 1)
            effects[client_id, operator_id, valid] = target[valid] - reference[valid]
    client_operator = _nanmean(effects, axis=-1)
    client = _nanmean(client_operator, axis=-1)
    return OperatorDSAResult(
        source_effects=effects,
        client_operator=client_operator,
        client=client,
        pooled=float(np.nanmean(client)),
    )


def paired_bootstrap_estimand(
    terms: dict[str, OperatorDSAResult],
    coefficients: dict[str, float],
    *,
    samples: int = 2000,
    seed: int = 20260909,
) -> np.ndarray:
    if set(terms) != set(coefficients):
        raise ValueError("terms and coefficients must have identical keys")
    shapes = {result.source_effects.shape for result in terms.values()}
    if len(shapes) != 1:
        raise ValueError("all DSA terms must have identical shapes")
    source_count = next(iter(shapes))[-1]
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(samples), dtype=np.float64)
    for sample_id in range(int(samples)):
        indices = rng.integers(0, source_count, size=source_count)
        value = 0.0
        for name, result in terms.items():
            per_client = _nanmean(
                _nanmean(result.source_effects[..., indices], axis=-1), axis=-1
            )
            value += float(coefficients[name]) * float(np.nanmean(per_client))
        output[sample_id] = value
    return output


def shuffled_binding_null(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
    *,
    permutations: int = 1000,
    seed: int = 20260909,
) -> dict[str, object]:
    observed = compute_operator_dsa(probabilities, labels, binding)
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(permutations), dtype=np.float64)
    for permutation_id in range(int(permutations)):
        shuffled = np.stack(
            [row[rng.permutation(row.size)] for row in np.asarray(binding, dtype=np.int64)]
        )
        null[permutation_id] = compute_operator_dsa(
            probabilities, labels, shuffled
        ).pooled
    return {
        "observed": observed.pooled,
        "null": null,
        "p_value": float((1 + (null >= observed.pooled).sum()) / (len(null) + 1)),
        "null_p95": float(np.quantile(null, 0.95)),
    }


def factorial_estimands(results: dict[str, OperatorDSAResult]) -> dict[str, object]:
    required = {"h0_b", "h9_b", "l0_b", "l9_b", "h0_p", "h9_p", "l0_p", "l9_p"}
    missing = sorted(required - set(results))
    if missing:
        raise ValueError(f"missing factorial arms: {missing}")

    pooled = {name: float(results[name].pooled) for name in required}
    client = {name: np.asarray(results[name].client) for name in required}

    def contrast(values: dict[str, float | np.ndarray], coefficients: dict[str, float]):
        total = None
        for name, coefficient in coefficients.items():
            term = float(coefficient) * values[name]
            total = term if total is None else total + term
        return total

    definitions = {
        "hfl_cle_effect_base": {"h9_b": 1.0, "h0_b": -1.0},
        "local_cle_effect_base": {"l9_b": 1.0, "l0_b": -1.0},
        "communication_amplification_base": {
            "h9_b": 1.0, "h0_b": -1.0, "l9_b": -1.0, "l0_b": 1.0
        },
        "hfl_plugin_mitigation": {
            "h9_b": 1.0, "h0_b": -1.0, "h9_p": -1.0, "h0_p": 1.0
        },
        "local_plugin_mitigation": {
            "l9_b": 1.0, "l0_b": -1.0, "l9_p": -1.0, "l0_p": 1.0
        },
        "communication_amplification_plugin": {
            "h9_p": 1.0, "h0_p": -1.0, "l9_p": -1.0, "l0_p": 1.0
        },
    }
    return {
        "pooled_dsa": pooled,
        "estimands": {
            name: float(contrast(pooled, coefficients))
            for name, coefficients in definitions.items()
        },
        "client_estimands": {
            name: np.asarray(contrast(client, coefficients)).tolist()
            for name, coefficients in definitions.items()
        },
        "definitions": definitions,
    }
