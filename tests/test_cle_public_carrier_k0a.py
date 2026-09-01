from __future__ import annotations

import numpy as np

from fedprime.engine.cle_public_carrier_moment import (
    class_vs_rest_evidence,
    decide_public_carrier_gate,
    directional_moment,
    paired_bootstrap_moment_delta,
    public_carrier_responses,
)


def test_class_vs_rest_evidence_ignores_common_logit_offset() -> None:
    logits = np.asarray([[1.0, 2.0, -1.0], [0.2, -0.3, 0.7]])
    assert np.allclose(
        class_vs_rest_evidence(logits),
        class_vs_rest_evidence(logits + 17.0),
    )


def test_public_carrier_response_is_class_centered() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=(2, 7, 3))
    probes = rng.normal(size=(2, 7, 5, 3))
    result = public_carrier_responses(base, probes)
    assert result.centered_response.shape == (2, 7, 5, 3)
    assert np.allclose(result.centered_response.mean(axis=-1), 0.0, atol=1.0e-12)
    assert np.allclose(result.probability_response.sum(axis=-1), 0.0, atol=1.0e-12)


def test_directional_moment_separates_coherent_and_canceling_responses() -> None:
    coherent = np.zeros((1, 20, 2, 3), dtype=np.float64)
    coherent[:, :, 0] = np.asarray([1.0, -0.5, -0.5])
    coherent[:, :, 1] = np.asarray([-0.5, 1.0, -0.5])
    canceling = coherent.copy()
    canceling[:, 10:] *= -1.0

    coherent_result = directional_moment(coherent)
    canceling_result = directional_moment(canceling)
    assert coherent_result.coherence_client[0] > 0.99
    assert coherent_result.split_cosine_client[0] > 0.99
    assert canceling_result.coherence_client[0] < 1.0e-12
    assert canceling_result.split_cosine_client[0] < -0.99


def test_paired_bootstrap_detects_stronger_directional_moment() -> None:
    rng = np.random.default_rng(8)
    zero = rng.normal(scale=0.1, size=(2, 80, 3, 4))
    zero -= zero.mean(axis=-1, keepdims=True)
    nine = zero.copy()
    nine[:, :, 0] += np.asarray([0.6, -0.2, -0.2, -0.2])
    result = paired_bootstrap_moment_delta(zero, nine, samples=200, seed=9)
    assert result["directional_strength_delta"]["ci95"][0] > 0.0
    assert result["coherence_delta"]["ci95"][0] > 0.0


def _arm_result(map_value: float, split: float, null_p: float = 0.001) -> dict[str, object]:
    return {
        "retrieval": {
            "mean_average_precision": map_value,
            "class_to_probe_family_hit_rate": 0.8,
            "client_mean_average_precision": [map_value] * 4,
        },
        "split_cosine_client": [split] * 4,
        "nulls": {
            "class_map_null": {"mean_average_precision": {"one_sided_p": null_p}},
            "probe_identity_null": {"mean_average_precision": {"one_sided_p": null_p}},
        },
    }


def test_public_carrier_gate_requires_all_hfl_and_local_conditions() -> None:
    arms = {
        "h0": _arm_result(0.40, 0.1),
        "h9": _arm_result(0.75, 0.6),
        "l0": _arm_result(0.42, 0.1),
        "l9": _arm_result(0.76, 0.65),
    }
    bootstrap_system = {
        "directional_strength_delta": {"ci95": [0.1, 0.2]},
        "coherence_delta": {"ci95": [0.05, 0.1]},
    }
    decision = decide_public_carrier_gate(
        arms,
        {"hfl": bootstrap_system, "local": bootstrap_system},
    )
    assert decision["verdict"] == "GO_TO_K0_B"

    arms["l9"] = _arm_result(0.60, 0.65)
    decision = decide_public_carrier_gate(
        arms,
        {"hfl": bootstrap_system, "local": bootstrap_system},
    )
    assert decision["verdict"] == "NO_GO_PUBLIC_CARRIER_ROUTE"
