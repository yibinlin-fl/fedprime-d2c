import numpy as np

from fedprime.engine.cle_shortcut_alignment import (
    OPERATOR_FAMILY_IDS,
    compute_dsa,
    decide_phase_a0,
    deterministic_corruption_grid,
    historical_family_binding,
    paired_bootstrap_delta,
    shuffled_binding_test,
    validate_probability_tensor,
)


def _synthetic_probabilities(strength: float, sources_per_class: int = 20):
    labels = np.repeat(np.arange(10), sources_per_class)
    binding = historical_family_binding()
    probabilities = np.empty((4, labels.size, 16, 10), dtype=np.float64)
    for client_id in range(4):
        for source_id, label in enumerate(labels):
            for operator_id, family_id in enumerate(OPERATOR_FAMILY_IDS):
                raw = np.full(10, 0.04, dtype=np.float64)
                raw[label] = 0.64
                bound = np.flatnonzero(binding[client_id] == family_id)
                raw[bound] *= 1.0 + strength
                probabilities[client_id, source_id, operator_id] = raw / raw.sum()
    return probabilities, labels, binding


def test_deterministic_grid_is_source_operator_paired_and_reproducible():
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    images[1] = 127
    left, severities = deterministic_corruption_grid(images)
    right, right_severities = deterministic_corruption_grid(images)
    assert left.shape == (2, 16, 32, 32, 3)
    assert np.array_equal(left, right)
    assert np.array_equal(severities, right_severities)
    assert np.all(severities == 3)


def test_aligned_gamma09_passes_directional_shortcut_gates():
    gamma0_probs, labels, binding = _synthetic_probabilities(0.0)
    gamma09_probs, _, _ = _synthetic_probabilities(3.0)
    stacked = np.stack([gamma0_probs, gamma09_probs])
    validate_probability_tensor(stacked, labels)
    gamma0 = compute_dsa(gamma0_probs, labels, binding)
    gamma09 = compute_dsa(gamma09_probs, labels, binding)
    bootstrap = paired_bootstrap_delta(gamma0, gamma09, samples=200, seed=7)
    shuffled = shuffled_binding_test(
        gamma09_probs,
        labels,
        binding,
        permutations=200,
        seed=7,
    )
    decision = decide_phase_a0(gamma0, gamma09, bootstrap, shuffled)
    assert gamma09.pooled - gamma0.pooled > 0.02
    assert decision["verdict"] == "GO_TO_MATCHED_PARTITION_DESIGN"
    assert all(gate["pass"] for gate in decision["gates"].values())


def test_identical_conditions_fail_minimum_delta_gate():
    probabilities, labels, binding = _synthetic_probabilities(0.0)
    result = compute_dsa(probabilities, labels, binding)
    bootstrap = paired_bootstrap_delta(result, result, samples=50, seed=3)
    shuffled = shuffled_binding_test(
        probabilities,
        labels,
        binding,
        permutations=50,
        seed=3,
    )
    decision = decide_phase_a0(result, result, bootstrap, shuffled)
    assert decision["verdict"] == "NO_GO_DIRECTIONAL_SHORTCUT"
    assert not decision["gates"]["G1_minimum_delta"]["pass"]
