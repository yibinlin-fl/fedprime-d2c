from __future__ import annotations

import numpy as np

from fedprime.engine.cle_shortcut_alignment import compute_dsa
from scripts.analyze_post_no_go_p3a_routing_identity import (
    dsa_values,
    generate_unique_derangements,
    rank_reversal,
    response_invariance,
)


def test_rank_reversal_is_deterministic_derangement() -> None:
    profile = np.asarray([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0])
    ranking, permutation = rank_reversal(profile)
    assert np.array_equal(ranking, np.asarray([0, 2, 4, 6, 8, 7, 5, 3, 1, 9]))
    assert np.array_equal(np.sort(permutation), np.arange(10))
    assert np.all(permutation != np.arange(10))
    assert np.array_equal(permutation[permutation], np.arange(10))


def test_unique_random_derangements_are_valid() -> None:
    first = generate_unique_derangements(np.random.default_rng(20260904), 100)
    second = generate_unique_derangements(np.random.default_rng(20260904), 100)
    assert np.array_equal(first, second)
    assert len({tuple(row) for row in first}) == 100
    assert np.all(np.sort(first, axis=1) == np.arange(10)[None])
    assert np.all(first != np.arange(10)[None])


def test_coordinate_permutation_preserves_response_geometry() -> None:
    rng = np.random.default_rng(7)
    response = rng.normal(size=(23, 5, 10))
    permutation = generate_unique_derangements(np.random.default_rng(8), 1)[0]
    metrics = response_invariance(response, permutation)
    assert max(abs(value) for value in metrics.values()) <= 1.0e-8


def test_local_dsa_matches_project_oracle() -> None:
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(4, 11, 16, 10))
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    labels = rng.integers(0, 10, size=11)
    binding = np.stack([np.roll(np.arange(10) % 4, client) for client in range(4)])
    family_ids = np.repeat(np.arange(4), 4)
    clients, pooled = dsa_values(probabilities, labels, binding, family_ids)
    reference = compute_dsa(probabilities, labels, binding, family_ids)
    assert np.allclose(clients, reference.client, atol=1.0e-12)
    assert abs(pooled - reference.pooled) <= 1.0e-12
