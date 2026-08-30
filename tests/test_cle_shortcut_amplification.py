from __future__ import annotations

import numpy as np

from fedprime.engine.cle_shortcut_alignment import DSAResult
from fedprime.engine.cle_shortcut_amplification import (
    compute_amplification,
    decide_phase_a1a,
    paired_bootstrap_amplification,
)


def result(client_values: list[float], source_offset: float = 0.0) -> DSAResult:
    client = np.asarray(client_values, dtype=np.float64)
    effects = np.repeat(client[:, None, None], 4, axis=1)
    effects = np.repeat(effects, 20, axis=2)
    effects[:, :, :10] += float(source_offset)
    return DSAResult(
        family_effects=effects,
        client_family=np.repeat(client[:, None], 4, axis=1),
        client=client,
        pooled=float(client.mean()),
    )


def test_difference_in_differences_direction() -> None:
    results = {
        "h0": result([0.00, 0.01, 0.02, 0.03]),
        "h9": result([0.09, 0.11, 0.13, 0.15], 0.01),
        "l0": result([0.00, 0.01, 0.02, 0.03]),
        "l9": result([0.04, 0.05, 0.07, 0.08]),
    }
    amplification = compute_amplification(results)
    np.testing.assert_allclose(amplification.client, [0.05, 0.06, 0.06, 0.07])
    assert np.isclose(amplification.pooled, 0.06)


def test_paired_bootstrap_and_gate_pass() -> None:
    results = {
        "h0": result([0.00, 0.00, 0.00, 0.00]),
        "h9": result([0.10, 0.10, 0.10, 0.10], 0.01),
        "l0": result([0.00, 0.00, 0.00, 0.00]),
        "l9": result([0.03, 0.03, 0.03, 0.03]),
    }
    amplification = compute_amplification(results)
    bootstrap = paired_bootstrap_amplification(results, samples=50, seed=3)
    decision = decide_phase_a1a(
        amplification,
        bootstrap,
        top1_amplification=0.02,
        h9_observed_dsa=0.10,
        h9_shuffled_p95=0.04,
    )
    assert decision["verdict"] == "GO_TO_CENTRALIZED_HOMOGENEOUS_ATTRIBUTION"
    assert all(gate["pass"] for gate in decision["gates"].values())


def test_gate_rejects_non_fl_specific_effect() -> None:
    results = {
        "h0": result([0.00, 0.00, 0.00, 0.00]),
        "h9": result([0.05, 0.05, 0.05, 0.05]),
        "l0": result([0.00, 0.00, 0.00, 0.00]),
        "l9": result([0.05, 0.05, 0.05, 0.05]),
    }
    amplification = compute_amplification(results)
    bootstrap = paired_bootstrap_amplification(results, samples=20, seed=2)
    decision = decide_phase_a1a(
        amplification,
        bootstrap,
        top1_amplification=0.0,
        h9_observed_dsa=0.05,
        h9_shuffled_p95=0.04,
    )
    assert decision["verdict"] == "NO_GO_FL_SPECIFIC_AMPLIFICATION"
    assert not decision["gates"]["G1_minimum_amplification"]["pass"]
