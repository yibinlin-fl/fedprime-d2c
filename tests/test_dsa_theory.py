from __future__ import annotations

import numpy as np
import pytest

from fedprime.engine.cle_v2_factorial import compute_operator_dsa
from fedprime.engine.dsa_theory import (
    add_operator_invariant_probability_shift,
    binding_aligned_distribution,
    categorical_jsd,
    operator_exchangeable_projection,
    probability_mixture,
    source_level_hoeffding_radius,
)


def _fixture():
    probabilities = np.asarray(
        [[[[0.8, 0.2], [0.2, 0.8]], [[0.7, 0.3], [0.3, 0.7]]]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1], dtype=np.int64)
    binding = np.asarray([[0, 1]], dtype=np.int64)
    return probabilities, labels, binding


def test_exchangeable_projection_has_zero_dsa() -> None:
    probabilities, labels, binding = _fixture()
    projected = operator_exchangeable_projection(probabilities)
    assert compute_operator_dsa(projected, labels, binding).pooled == pytest.approx(0.0, abs=1e-15)


def test_dsa_is_affine_under_probability_mixture() -> None:
    shortcut, labels, binding = _fixture()
    semantic = operator_exchangeable_projection(shortcut)
    left = compute_operator_dsa(semantic, labels, binding).pooled
    right = compute_operator_dsa(shortcut, labels, binding).pooled
    for weight in (0.0, 0.2, 0.5, 1.0):
        observed = compute_operator_dsa(
            probability_mixture(semantic, shortcut, weight), labels, binding
        ).pooled
        assert observed == pytest.approx((1.0 - weight) * left + weight * right, abs=1e-15)


def test_identical_views_have_zero_jsd_without_restricting_dsa() -> None:
    probabilities, labels, binding = _fixture()
    views = np.stack([probabilities, probabilities, probabilities], axis=0)
    assert float(categorical_jsd(views).max()) == pytest.approx(0.0, abs=1e-15)
    assert compute_operator_dsa(probabilities, labels, binding).pooled > 0.0


def test_probability_mixture_rejects_invalid_weight() -> None:
    probabilities, _, _ = _fixture()
    with pytest.raises(ValueError, match="shortcut_weight"):
        probability_mixture(probabilities, probabilities, 1.1)


def test_binding_aligned_injection_is_recovered_linearly() -> None:
    _, labels, binding = _fixture()
    semantic = np.full((1, 2, 2, 2), 0.5, dtype=np.float64)
    aligned = binding_aligned_distribution(binding, sources=2, operators=2, classes=2)
    endpoint = compute_operator_dsa(aligned, labels, binding).pooled
    for strength in (0.0, 0.25, 0.75, 1.0):
        injected = probability_mixture(semantic, aligned, strength)
        assert compute_operator_dsa(injected, labels, binding).pooled == pytest.approx(
            strength * endpoint, abs=1e-15
        )


def test_operator_invariant_probability_shift_cancels_from_dsa() -> None:
    probabilities, labels, binding = _fixture()
    shift = np.asarray([[[0.01, -0.01], [-0.02, 0.02]]], dtype=np.float64)
    shifted = add_operator_invariant_probability_shift(probabilities, shift)
    assert compute_operator_dsa(shifted, labels, binding).pooled == pytest.approx(
        compute_operator_dsa(probabilities, labels, binding).pooled, abs=1e-15
    )


def test_source_level_hoeffding_radius_contract() -> None:
    radius = source_level_hoeffding_radius(1000, delta=0.05)
    assert 0.0 < radius < 0.1
    with pytest.raises(ValueError, match="source_count"):
        source_level_hoeffding_radius(0)
