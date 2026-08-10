import numpy as np
import pytest
import torch

from fedprime.methods.counterfactual_regret import (
    SeedSignals,
    binary_auroc,
    class_percentile_ranks,
    correct_class_margin,
    counterfactual_regret,
    decide_audit0,
    evaluate_directed_pair,
    per_sample_jsd,
    split_fit_internal_probe,
)


def test_correct_class_margin_and_regret():
    logits = torch.tensor([[4.0, 1.0, 2.0], [2.0, 5.0, 1.0]])
    labels = torch.tensor([0, 1])
    margin = correct_class_margin(logits, labels)
    assert torch.allclose(margin, torch.tensor([2.0, 3.0]))
    augmented = torch.tensor([[1.0, -0.5], [3.5, 2.0]])
    assert torch.allclose(counterfactual_regret(margin, augmented), torch.tensor([2.5, 1.0]))


def test_per_sample_jsd_is_zero_for_matching_views_and_positive_otherwise():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    assert torch.allclose(per_sample_jsd([logits, logits, logits]), torch.zeros(2), atol=1.0e-7)
    changed = logits.flip(1)
    assert torch.all(per_sample_jsd([logits, changed]) > 0)


def test_fit_internal_probe_is_disjoint_deterministic_and_class_stratified():
    labels = np.repeat(np.arange(3), [40, 20, 50])
    fit = np.arange(labels.size)
    train_a, probe_a = split_fit_internal_probe(
        fit, labels, ratio=0.2, min_class_count=32, min_probe=5, max_probe=8, seed=7
    )
    train_b, probe_b = split_fit_internal_probe(
        fit, labels, ratio=0.2, min_class_count=32, min_probe=5, max_probe=8, seed=7
    )
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(probe_a, probe_b)
    assert np.intersect1d(train_a, probe_a).size == 0
    assert np.array_equal(np.sort(np.concatenate([train_a, probe_a])), fit)
    assert np.bincount(labels[probe_a], minlength=3).tolist() == [8, 0, 8]


def test_binary_auroc_and_class_percentile_ranks():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    targets = np.array([0, 0, 1, 1])
    assert binary_auroc(scores, targets) == pytest.approx(1.0)
    ranked = class_percentile_ranks(scores, np.array([0, 0, 1, 1]), min_class_support=2)
    assert ranked.tolist() == pytest.approx([0.25, 0.75, 0.25, 0.75])


def _signals(seed_shift: float = 0.0) -> SeedSignals:
    labels = np.repeat(np.arange(2), 8)
    base = np.tile(np.linspace(0.0, 1.0, 8), 2) + seed_shift
    flips = (base > 0.55).astype(np.int64)
    return SeedSignals(
        sample_indices=np.arange(16),
        labels=labels,
        corruption_ids=np.tile(np.repeat([0, 1], 4), 2),
        regret=base,
        ce=base[::-1],
        jsd=np.full(16, 0.1),
        robust_error=flips,
        flip_error=flips,
        base_accuracy=0.75,
    )


def test_directed_pair_uses_cross_seed_targets_and_reports_all_signals():
    result = evaluate_directed_pair(_signals(), _signals(0.01), min_class_support=4, min_cell_support=4)
    assert result["regret_persistence"] == pytest.approx(1.0)
    assert set(result["signals"]) == {"regret", "ce", "jsd"}
    assert result["signals"]["regret"]["flip_auroc"] == pytest.approx(1.0)
    assert result["signals"]["regret"]["valid_cells"] == 4


def test_decision_returns_go_for_metrics_above_every_frozen_gate():
    clients = {
        1: {
            "base_accuracies": [0.6, 0.62, 0.61],
            "regret_positive_fractions": [0.7, 0.72, 0.71],
            "regret_p90": [1.0, 1.1, 1.2],
        },
        3: {
            "base_accuracies": [0.55, 0.57, 0.56],
            "regret_positive_fractions": [0.65, 0.67, 0.66],
            "regret_p90": [0.9, 1.0, 1.1],
        },
    }
    pairs = []
    for client_id in (1, 3):
        for _ in range(3):
            pairs.append({
                "client_id": client_id,
                "flip_prevalence": 0.2,
                "regret_persistence": 0.5,
                "signals": {
                    "regret": {
                        "flip_auroc": 0.75,
                        "top_fraction_enrichment": 2.0,
                        "cell_correlation": 0.6,
                        "valid_cells": 12,
                    },
                    "ce": {
                        "flip_auroc": 0.65,
                        "top_fraction_enrichment": 1.5,
                        "cell_correlation": 0.4,
                        "valid_cells": 12,
                    },
                    "jsd": {
                        "flip_auroc": 0.60,
                        "top_fraction_enrichment": 1.4,
                        "cell_correlation": 0.3,
                        "valid_cells": 12,
                    },
                },
            })
    decision = decide_audit0(clients, pairs)
    assert decision["verdict"] == "GO"
    assert all(gate["pass"] for gate in decision["gates"].values())


def test_decision_marks_low_accuracy_as_invalid_probe():
    clients = {
        1: {
            "base_accuracies": [0.1, 0.1, 0.1],
            "regret_positive_fractions": [0.7, 0.7, 0.7],
            "regret_p90": [1.0, 1.0, 1.0],
        }
    }
    pair = {
        "client_id": 1,
        "flip_prevalence": 0.2,
        "regret_persistence": 0.5,
        "signals": {
            name: {
                "flip_auroc": 0.7,
                "top_fraction_enrichment": 1.5,
                "cell_correlation": 0.5,
                "valid_cells": 20,
            }
            for name in ("regret", "ce", "jsd")
        },
    }
    assert decide_audit0(clients, [pair])["verdict"] == "INVALID_PROBE"
