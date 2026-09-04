from __future__ import annotations

import numpy as np

from scripts.analyze_post_no_go_p4_routing_targetability_gap import (
    centered_response,
    destructive_score,
    harmful_objects,
)


def test_centered_response_does_not_mutate_clean_logits() -> None:
    rng = np.random.default_rng(2)
    clean = rng.normal(size=(7, 10))
    original = clean.copy()
    corrupt = rng.normal(size=(7, 16, 10))
    corrupt -= corrupt.max(axis=-1, keepdims=True)
    probabilities = np.exp(corrupt)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    result = centered_response(probabilities, clean)
    assert result.shape == (7, 16, 10)
    assert np.array_equal(clean, original)
    assert np.allclose(result.sum(axis=-1), 0.0, atol=1.0e-12)


def test_harmful_matrix_uses_dsa_valid_source_rule() -> None:
    response = np.zeros((6, 4, 10), dtype=np.float64)
    labels = np.asarray([0, 1, 2, 3, 4, 5])
    binding = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    family_ids = np.arange(4)
    response[:, 0, 0] = np.arange(6, dtype=np.float64)
    matrix, positive, support = harmful_objects(response, labels, binding, family_ids)
    valid = ~np.isin(labels, np.flatnonzero(binding == 0))
    assert matrix[0, 0] == np.arange(6, dtype=np.float64)[valid].mean()
    assert positive[0] == np.maximum(np.arange(6, dtype=np.float64)[valid], 0.0).mean()
    assert support[0] == int(valid.sum())


def test_destructive_score_is_zero_for_identity() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(10, 10))
    result = destructive_score(matrix, np.arange(10))
    assert result["signed_destructive_score"] == 0.0
    assert result["positive_energy_destructive_score"] == 0.0
    assert result["positive_energy_reduction_fraction"] == 0.0
