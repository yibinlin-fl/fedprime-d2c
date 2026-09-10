from __future__ import annotations

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
