from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedprime.engine.cle_shortcut_alignment import DSAResult


ARM_NAMES = ("h0", "h9", "l0", "l9")


@dataclass(frozen=True)
class AmplificationResult:
    hfl_effect: np.ndarray
    local_effect: np.ndarray
    client: np.ndarray
    pooled: float


def compute_amplification(results: dict[str, DSAResult]) -> AmplificationResult:
    missing = set(ARM_NAMES) - set(results)
    if missing:
        raise KeyError(f"Missing Phase-A1a arms: {sorted(missing)}")
    shapes = {np.asarray(results[name].client).shape for name in ARM_NAMES}
    if len(shapes) != 1:
        raise ValueError("All Phase-A1a arms must have the same client shape")
    hfl_effect = np.asarray(results["h9"].client) - np.asarray(results["h0"].client)
    local_effect = np.asarray(results["l9"].client) - np.asarray(results["l0"].client)
    client = hfl_effect - local_effect
    return AmplificationResult(
        hfl_effect=hfl_effect,
        local_effect=local_effect,
        client=client,
        pooled=float(np.nanmean(client)),
    )


def paired_bootstrap_amplification(
    results: dict[str, DSAResult],
    *,
    samples: int = 2000,
    seed: int = 20260830,
) -> np.ndarray:
    missing = set(ARM_NAMES) - set(results)
    if missing:
        raise KeyError(f"Missing Phase-A1a arms: {sorted(missing)}")
    effects = {
        name: np.asarray(results[name].family_effects, dtype=np.float64)
        for name in ARM_NAMES
    }
    shapes = {value.shape for value in effects.values()}
    if len(shapes) != 1:
        raise ValueError("All Phase-A1a family-effect tensors must have identical shapes")
    num_sources = next(iter(effects.values())).shape[-1]
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(samples), dtype=np.float64)
    for bootstrap_id in range(int(samples)):
        indices = rng.integers(0, num_sources, size=num_sources)
        pooled = {}
        for name, value in effects.items():
            per_client = np.nanmean(np.nanmean(value[..., indices], axis=-1), axis=-1)
            pooled[name] = per_client
        amplification = (pooled["h9"] - pooled["h0"]) - (
            pooled["l9"] - pooled["l0"]
        )
        output[bootstrap_id] = float(np.nanmean(amplification))
    return output


def decide_phase_a1a(
    amplification: AmplificationResult,
    bootstrap: np.ndarray,
    *,
    top1_amplification: float,
    h9_observed_dsa: float,
    h9_shuffled_p95: float,
    minimum_amplification: float = 0.020,
) -> dict[str, object]:
    ci_low, ci_high = np.quantile(np.asarray(bootstrap, dtype=np.float64), [0.025, 0.975])
    positive_clients = int(np.sum(np.asarray(amplification.client) > 0.0))
    gates = {
        "G1_minimum_amplification": {
            "value": amplification.pooled,
            "threshold": minimum_amplification,
            "pass": amplification.pooled >= minimum_amplification,
        },
        "G2_bootstrap_ci": {
            "low": float(ci_low),
            "high": float(ci_high),
            "pass": float(ci_low) > 0.0,
        },
        "G3_client_direction": {
            "positive_clients": positive_clients,
            "threshold": 3,
            "pass": positive_clients >= 3,
        },
        "G4_top1_amplification": {
            "value": float(top1_amplification),
            "threshold": 0.0,
            "pass": float(top1_amplification) > 0.0,
        },
        "G5_h9_shuffled_binding": {
            "observed": float(h9_observed_dsa),
            "null_p95": float(h9_shuffled_p95),
            "pass": float(h9_observed_dsa) > float(h9_shuffled_p95),
        },
    }
    passed = all(bool(value["pass"]) for value in gates.values())
    return {
        "verdict": (
            "GO_TO_CENTRALIZED_HOMOGENEOUS_ATTRIBUTION"
            if passed
            else "NO_GO_FL_SPECIFIC_AMPLIFICATION"
        ),
        "amplification_pooled": amplification.pooled,
        "amplification_client": amplification.client.tolist(),
        "hfl_cle_effect_client": amplification.hfl_effect.tolist(),
        "local_cle_effect_client": amplification.local_effect.tolist(),
        "bootstrap_ci95": [float(ci_low), float(ci_high)],
        "gates": gates,
    }
