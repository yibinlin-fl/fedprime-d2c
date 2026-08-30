import numpy as np

from fedprime.engine.cle_probe_directional_promotion import (
    decide_zero_training_gate,
    probe_directional_promotion,
    score_binding_retrieval,
    shuffled_retrieval_nulls,
)
from fedprime.engine.cle_shortcut_alignment import OPERATOR_FAMILY_IDS, historical_family_binding


def _synthetic_probabilities(strength: float, sources_per_class: int = 20):
    labels = np.repeat(np.arange(10), sources_per_class)
    binding = historical_family_binding()
    probabilities = np.empty((4, labels.size, 16, 10), dtype=np.float64)
    for client_id in range(4):
        for source_id, label in enumerate(labels):
            for probe_id, family_id in enumerate(OPERATOR_FAMILY_IDS):
                raw = np.full(10, 0.03, dtype=np.float64)
                raw[label] = 0.70
                bound = np.flatnonzero(binding[client_id] == family_id)
                raw[bound] *= 1.0 + strength
                probabilities[client_id, source_id, probe_id] = raw / raw.sum()
    return probabilities, labels, binding


def test_estimator_recovers_direction_without_accepting_binding_metadata():
    probabilities, labels, binding = _synthetic_probabilities(6.0)
    result = probe_directional_promotion(probabilities, labels)
    metrics = score_binding_retrieval(result.matrix, binding, OPERATOR_FAMILY_IDS)
    assert result.matrix.shape == (4, 16, 10)
    assert metrics["mean_average_precision"] > 0.99
    assert metrics["roc_auc"] > 0.99
    assert metrics["class_to_probe_family_hit_rate"] == 1.0


def test_iid_probe_responses_have_zero_pidr():
    probabilities, labels, _ = _synthetic_probabilities(0.0)
    result = probe_directional_promotion(probabilities, labels)
    assert np.allclose(result.matrix, 0.0, atol=1.0e-12)
    assert np.isclose(result.pooled, 0.0)


def test_real_binding_exceeds_class_and_probe_permutation_nulls():
    probabilities, labels, binding = _synthetic_probabilities(6.0)
    result = probe_directional_promotion(probabilities, labels)
    nulls = shuffled_retrieval_nulls(
        result.matrix,
        binding,
        OPERATOR_FAMILY_IDS,
        permutations=200,
        seed=7,
    )
    observed = nulls["observed"]["mean_average_precision"]
    assert observed > nulls["class_map_null"]["mean_average_precision"]["p95"]
    assert observed > nulls["probe_identity_null"]["mean_average_precision"]["p95"]


def test_four_arm_decision_requires_both_hfl_and_local_recovery():
    gamma0, labels, binding = _synthetic_probabilities(0.0)
    gamma9, _, _ = _synthetic_probabilities(6.0)
    arms = {}
    for index, (name, probabilities) in enumerate(
        (("h0", gamma0), ("h9", gamma9), ("l0", gamma0), ("l9", gamma9))
    ):
        promotion = probe_directional_promotion(probabilities, labels)
        arms[name] = {
            "retrieval": score_binding_retrieval(promotion.matrix, binding, OPERATOR_FAMILY_IDS),
            "nulls": shuffled_retrieval_nulls(
                promotion.matrix,
                binding,
                OPERATOR_FAMILY_IDS,
                permutations=200,
                seed=11 + index,
            ),
        }
    decision = decide_zero_training_gate(arms, maximum_null_p=0.01)
    assert decision["verdict"] == "GO_TO_INTERVENTION_BRIDGE_DESIGN"
    assert decision["systems"]["hfl"]["pass"]
    assert decision["systems"]["local"]["pass"]
