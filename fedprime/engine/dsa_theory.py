from __future__ import annotations

import math

import numpy as np


def operator_exchangeable_projection(probabilities: np.ndarray) -> np.ndarray:
    """Project [client, source, operator, class] predictions onto operator exchangeability."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("probabilities must have shape [client,source,operator,class]")
    mean = values.mean(axis=2, keepdims=True)
    return np.broadcast_to(mean, values.shape).copy()


def probability_mixture(
    semantic: np.ndarray, shortcut: np.ndarray, shortcut_weight: float
) -> np.ndarray:
    """Return the convex probability mixture used by the DSA linearity proposition."""

    left = np.asarray(semantic, dtype=np.float64)
    right = np.asarray(shortcut, dtype=np.float64)
    weight = float(shortcut_weight)
    if left.shape != right.shape:
        raise ValueError("mixture endpoints must have identical shapes")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("shortcut_weight must be in [0, 1]")
    return (1.0 - weight) * left + weight * right


def categorical_jsd(view_probabilities: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence over a leading view axis, retaining other axes."""

    values = np.asarray(view_probabilities, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("view_probabilities must contain view and class axes")
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError("view probabilities must be finite and nonnegative")
    if not np.allclose(values.sum(axis=-1), 1.0, atol=1.0e-7):
        raise ValueError("view probabilities must sum to one")
    mixture = values.mean(axis=0)
    terms = np.where(
        values > 0.0,
        values * (np.log(np.maximum(values, 1.0e-300)) - np.log(np.maximum(mixture, 1.0e-300))),
        0.0,
    )
    return terms.sum(axis=-1).mean(axis=0)


def binding_aligned_distribution(
    binding: np.ndarray,
    *,
    sources: int,
    operators: int,
    classes: int,
) -> np.ndarray:
    """Construct a controlled probability response aligned with a CLE binding map.

    Operators with bound classes place all probability mass uniformly on those
    classes. Operators absent from the training binding use a uniform fallback.
    The construction is evaluation-only and never consumes private images.
    """

    binding_values = np.asarray(binding, dtype=np.int64)
    if binding_values.ndim != 2 or binding_values.shape[1] != int(classes):
        raise ValueError("binding must have shape [client,class]")
    if np.any(binding_values < 0) or np.any(binding_values >= int(operators)):
        raise ValueError("binding contains an out-of-range operator id")
    output = np.full(
        (binding_values.shape[0], int(sources), int(operators), int(classes)),
        1.0 / float(classes),
        dtype=np.float64,
    )
    for client_id in range(binding_values.shape[0]):
        for operator_id in range(int(operators)):
            bound = np.flatnonzero(binding_values[client_id] == operator_id)
            if bound.size:
                output[client_id, :, operator_id, :] = 0.0
                output[client_id, :, operator_id, bound] = 1.0 / float(bound.size)
    return output


def add_operator_invariant_probability_shift(
    probabilities: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """Add one zero-sum class shift per client/source to every operator.

    This realizes the nuisance-cancellation assumption in probability space.
    It intentionally rejects shifts that leave the probability simplex.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    offsets = np.asarray(shift, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("probabilities must have shape [client,source,operator,class]")
    expected = (values.shape[0], values.shape[1], values.shape[3])
    if offsets.shape != expected:
        raise ValueError(f"shift must have shape {expected}")
    if not np.allclose(offsets.sum(axis=-1), 0.0, atol=1.0e-12):
        raise ValueError("operator-invariant class shifts must sum to zero")
    shifted = values + offsets[:, :, None, :]
    if np.any(shifted < -1.0e-12) or np.any(shifted > 1.0 + 1.0e-12):
        raise ValueError("shift leaves the probability simplex")
    if not np.allclose(shifted.sum(axis=-1), 1.0, atol=1.0e-12):
        raise ValueError("shifted probabilities do not sum to one")
    return shifted


def source_level_hoeffding_radius(
    source_count: int,
    *,
    delta: float = 0.05,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Two-sided Hoeffding radius for independent bounded source effects."""

    count = int(source_count)
    if count <= 0:
        raise ValueError("source_count must be positive")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not float(lower) < float(upper):
        raise ValueError("lower must be smaller than upper")
    return float(upper - lower) * math.sqrt(
        math.log(2.0 / float(delta)) / (2.0 * float(count))
    )
