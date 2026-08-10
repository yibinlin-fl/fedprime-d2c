import numpy as np
import pytest
import torch

from fedprime.methods.class_residual_spectral_risk import (
    class_balanced_spearman,
    class_conditional_residual_spectral_risk,
    class_operator_cell_correlation,
    cross_split_spectral_metrics,
    decide_crsr_audit0,
    deterministic_random_directions,
    fit_residual_spectral_statistics,
    prediction_residuals,
    score_residual_directions,
)


def test_prediction_residuals_are_simplex_tangent_vectors():
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    labels = torch.tensor([0, 2])
    residuals = prediction_residuals(logits, labels)
    assert residuals.shape == logits.shape
    assert torch.allclose(residuals.sum(dim=1), torch.zeros(2), atol=1.0e-7)
    assert torch.all(residuals[torch.arange(2), labels] < 0)


def test_crsr_is_class_balanced_and_has_finite_gradients():
    logits = torch.tensor(
        [[3.0, 0.0], [2.0, 0.0], [0.1, 1.0], [0.2, 1.0], [0.3, 1.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1, 1])
    loss, stats = class_conditional_residual_spectral_risk(
        logits, labels, spectral_weight=2.0, min_class_count=2
    )
    assert torch.isfinite(loss)
    assert stats["valid_spectral_classes"].item() == 2
    assert stats["spectral_radius"].item() > 0.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_crsr_rejects_invalid_shapes_and_weights():
    with pytest.raises(ValueError):
        prediction_residuals(torch.zeros(2), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError):
        class_conditional_residual_spectral_risk(
            torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), spectral_weight=-1.0
        )


def _probabilities(seed: int, samples_per_class: int = 40):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), samples_per_class)
    probabilities = np.zeros((labels.size, 3), dtype=np.float64)
    for index, label in enumerate(labels):
        nuisance = rng.normal()
        raw = np.full(3, 0.05)
        raw[label] = 0.82 - 0.12 * nuisance
        raw[(label + 1) % 3] += 0.12 * nuisance
        raw = np.clip(raw, 0.01, None)
        probabilities[index] = raw / raw.sum()
    return probabilities, labels


def test_spectral_statistics_score_and_transfer_across_splits():
    left_p, left_y = _probabilities(1)
    right_p, right_y = _probabilities(2)
    left = fit_residual_spectral_statistics(left_p, left_y, min_class_support=16)
    right = fit_residual_spectral_statistics(right_p, right_y, min_class_support=16)
    random = deterministic_random_directions(3, left.keys(), seed=9)
    scores = score_residual_directions(right_p, right_y, left)
    assert np.isfinite(scores).all()
    metrics = cross_split_spectral_metrics(left, right, random)
    assert metrics["valid_classes"] == 3
    assert metrics["median_direction_cosine"] > 0.9
    assert metrics["median_transfer_share"] > metrics["median_random_share"]


def test_class_balanced_and_cell_correlations_are_finite():
    scores = np.tile(np.arange(8, dtype=np.float64), 2)
    errors = (scores > 3).astype(np.float64)
    labels = np.repeat([0, 1], 8)
    operators = np.tile(np.repeat([0, 1], 4), 2)
    assert class_balanced_spearman(scores, errors, labels, min_class_support=4) > 0.8
    correlation, cells = class_operator_cell_correlation(
        scores, errors, labels, operators, min_cell_support=4
    )
    assert correlation == pytest.approx(1.0)
    assert cells == 4


def _passing_client_metrics():
    return {
        "base_accuracy": 0.6,
        "valid_classes": 10,
        "source_top_share": 0.4,
        "direction_cosine": 0.8,
        "transfer_advantage": 0.2,
        "spectral_ce_abs_correlation": 0.5,
        "spectral_brier_abs_correlation": 0.6,
        "spectral_cell_correlation": 0.7,
        "ce_cell_correlation": 0.4,
        "brier_cell_correlation": 0.3,
        "random_cell_correlation": 0.2,
        "valid_cells": 12,
    }


def test_decision_requires_signal_and_one_step_gates():
    clients = [_passing_client_metrics(), _passing_client_metrics()]
    one_step = [
        {"mean_ce_delta": -0.001, "worst_cell_ce_delta": -0.002, "cell_gap_ce_delta": -0.001},
        {"mean_ce_delta": -0.001, "worst_cell_ce_delta": -0.002, "cell_gap_ce_delta": -0.001},
    ]
    decision = decide_crsr_audit0(clients, one_step)
    assert decision["verdict"] == "GO"
    assert all(gate["pass"] for gate in decision["gates"].values())


def test_decision_marks_invalid_low_accuracy_probe():
    client = _passing_client_metrics()
    client["base_accuracy"] = 0.1
    assert decide_crsr_audit0([client])["verdict"] == "INVALID_PROBE"
