from __future__ import annotations

import numpy as np

from fedprime.engine.cle_shared_nuisance_routing import (
    aggregate_transfer,
    bootstrap_cross_bank_transfer,
    bootstrap_index_matrix,
    bootstrap_sharedness,
    match_low_energy_probes,
    paired_random_subspace_bases,
    projection_fraction,
    sharedness_statistics,
    weighted_response_subspace,
)


def test_sharedness_distinguishes_stable_from_cancelled_response() -> None:
    stable_a = np.ones((4, 1, 3), dtype=np.float64)
    stable_b = np.ones((4, 1, 3), dtype=np.float64)
    cancelled_b = -stable_b
    assert sharedness_statistics(stable_a, stable_b).sharedness[0] > 0.99
    assert sharedness_statistics(stable_a, cancelled_b).sharedness[0] == 0.0


def test_low_matching_is_energy_nearest_unique_and_deterministic() -> None:
    result = match_low_energy_probes(
        np.asarray([4, 1]),
        np.asarray([0, 1, 2, 3, 4, 5]),
        np.asarray([1.0, 10.0, 9.0, 90.0, 100.0, 1000.0]),
        np.asarray([3.0, 1.0]),
    )
    np.testing.assert_array_equal(result.matched_probe_ids, np.asarray([3, 2]))
    np.testing.assert_allclose(result.high_weights, np.asarray([0.75, 0.25]))


def test_weighted_subspace_and_transfer_recover_known_plane() -> None:
    source = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    subspace = weighted_response_subspace(source, np.asarray([0.5, 0.5]))
    assert subspace.rank == 2
    assert projection_fraction(np.asarray([3.0, 4.0, 0.0]), subspace.basis) > 0.999999
    assert projection_fraction(np.asarray([0.0, 0.0, 1.0]), subspace.basis) < 1.0e-10
    score, per_probe = aggregate_transfer(
        np.asarray([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.asarray([0.75, 0.25]),
        subspace.basis,
    )
    np.testing.assert_allclose(per_probe, np.asarray([1.0, 0.0]), atol=1.0e-10)
    np.testing.assert_allclose(score, 0.75, atol=1.0e-10)


def test_paired_random_subspaces_share_prefix_when_ranks_differ() -> None:
    left, right = paired_random_subspace_bases(8, 2, 4, draws=3, seed=7)
    assert len(left) == len(right) == 3
    for small, large in zip(left, right):
        np.testing.assert_allclose(small, large[:, :2])
        np.testing.assert_allclose(large.T @ large, np.eye(4), atol=1.0e-10)


def test_bootstrap_is_deterministic_and_rebuilds_subspace() -> None:
    rng = np.random.default_rng(3)
    delta_a = rng.normal(size=(12, 2, 4))
    delta_b = delta_a + 0.01 * rng.normal(size=(12, 2, 4))
    index_a = bootstrap_index_matrix(carriers=12, replicates=5, seed=11)
    index_b = bootstrap_index_matrix(carriers=12, replicates=5, seed=12)
    shared_1 = bootstrap_sharedness(delta_a, delta_b, np.asarray([0.4, 0.6]), index_a, index_b)
    shared_2 = bootstrap_sharedness(delta_a, delta_b, np.asarray([0.4, 0.6]), index_a, index_b)
    np.testing.assert_array_equal(shared_1, shared_2)
    transfer, ranks = bootstrap_cross_bank_transfer(
        delta_a,
        delta_b,
        np.asarray([0.4, 0.6]),
        np.asarray([0.4, 0.6]),
        index_a,
        index_b,
    )
    assert transfer.shape == ranks.shape == (5,)
    assert np.all((transfer >= 0.0) & (transfer <= 1.0 + 1.0e-10))
    assert np.all(ranks > 0)
