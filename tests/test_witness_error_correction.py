from __future__ import annotations

import numpy as np

from fedprime.methods.witness_error_correction import (
    channel_diagnostics,
    select_public_regularization,
    solve_nonnegative_ridge,
    solve_simplex_ridge,
)


def test_identity_channel_returns_observed() -> None:
    observed = np.asarray([0.1, 0.2, 0.3, 0.15, 0.25])
    corrected = solve_simplex_ridge(np.eye(5), observed, 0.0)
    np.testing.assert_allclose(corrected, observed, atol=1.0e-10)


def test_known_full_rank_channel_recovers_truth() -> None:
    channel = np.asarray(
        [
            [0.80, 0.05, 0.05],
            [0.10, 0.85, 0.10],
            [0.05, 0.05, 0.75],
            [0.05, 0.05, 0.10],
        ]
    )
    truth = np.asarray([0.2, 0.5, 0.3])
    corrected = solve_simplex_ridge(channel, channel @ truth, 0.0)
    np.testing.assert_allclose(corrected, truth, atol=1.0e-10)


def test_ill_conditioned_channel_fails_frozen_gate() -> None:
    channel = np.asarray(
        [
            [0.50, 0.500001, 0.10],
            [0.30, 0.299999, 0.20],
            [0.20, 0.200000, 0.70],
        ]
    )
    diagnostics = channel_diagnostics(channel)
    assert diagnostics["min_singular_value"] < 0.05
    assert diagnostics["condition_number"] > 20.0


def test_solvers_are_nonnegative_and_simplex_is_normalized() -> None:
    channel = np.asarray([[0.8, 0.1], [0.1, 0.7], [0.1, 0.2]])
    probability = solve_simplex_ridge(channel, np.asarray([0.2, 0.7, 0.1]), 0.01)
    mass = solve_nonnegative_ridge(channel, np.asarray([0.2, 0.7, 0.1]), 0.01)
    assert np.all(probability >= 0)
    assert np.all(mass >= 0)
    np.testing.assert_allclose(probability.sum(), 1.0, atol=1.0e-12)


def test_loss_mass_scale_consistency_without_ridge() -> None:
    channel = np.asarray([[0.8, 0.1], [0.1, 0.7], [0.1, 0.2]])
    mass = np.asarray([0.2, 0.7, 0.1])
    first = solve_nonnegative_ridge(channel, mass, 0.0)
    second = solve_nonnegative_ridge(channel, 7.0 * mass, 0.0)
    np.testing.assert_allclose(second, 7.0 * first, atol=1.0e-10)


def test_lambda_selector_accepts_public_folds_only_and_is_deterministic() -> None:
    channel = np.asarray([[0.8, 0.1], [0.1, 0.7], [0.1, 0.2]])
    truth = np.asarray([1.0, 0.0])
    folds = [{"channel": channel, "observed": channel @ truth, "truth": truth}]
    first = select_public_regularization(folds, (0.0, 1.0e-3, 1.0e-2))
    second = select_public_regularization(folds, (0.0, 1.0e-3, 1.0e-2))
    assert first == second


def test_rectangular_channel_keeps_unknown_observation_row() -> None:
    channel = np.asarray(
        [
            [0.8, 0.1],
            [0.1, 0.7],
            [0.1, 0.2],  # observable unknown, not a latent environment
        ]
    )
    truth = np.asarray([0.3, 0.7])
    corrected = solve_simplex_ridge(channel, channel @ truth, 0.0)
    np.testing.assert_allclose(corrected, truth, atol=1.0e-10)
