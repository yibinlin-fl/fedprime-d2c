from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GenericProbeStatistics:
    mu_a: np.ndarray
    mu_b: np.ndarray
    energy_a: np.ndarray
    energy_b: np.ndarray
    energy: np.ndarray
    cross_dot_positive: np.ndarray
    kappa_cf: np.ndarray
    selectivity: np.ndarray
    rho: np.ndarray
    active: np.ndarray
    S: np.ndarray
    Dcf: np.ndarray
    K: np.ndarray
    R: np.ndarray


def _top_fraction_mean(values: np.ndarray, active: np.ndarray, fraction: float) -> np.ndarray:
    result = np.zeros(values.shape[0], dtype=np.float64)
    for client_id in range(values.shape[0]):
        selected = np.asarray(values[client_id][active[client_id]], dtype=np.float64)
        if selected.size == 0:
            raise ValueError("every client must have at least one active probe")
        count = max(1, int(np.ceil(float(fraction) * selected.size)))
        result[client_id] = float(np.partition(selected, selected.size - count)[-count:].mean())
    return result


def generic_probe_statistics(
    centered_response: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
    top_fraction: float = 0.20,
) -> GenericProbeStatistics:
    """Measure carrier-stable and class-selective directions without taxonomy metadata."""

    response = np.asarray(centered_response, dtype=np.float64)
    if response.ndim != 4:
        raise ValueError("centered_response must have shape [client,source,probe,class]")
    if response.shape[1] < 2 or response.shape[1] % 2:
        raise ValueError("an even number of at least two carriers is required")
    if response.shape[2] < 2 or response.shape[3] < 2:
        raise ValueError("at least two probes and two classes are required")
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0,1]")
    if not np.isfinite(response).all():
        raise ValueError("centered_response contains non-finite values")

    split = response.shape[1] // 2
    left, right = response[:, :split], response[:, split:]
    mu_a = left.mean(axis=1)
    mu_b = right.mean(axis=1)
    energy_a = np.mean(np.sum(np.square(left), axis=-1), axis=1)
    energy_b = np.mean(np.sum(np.square(right), axis=-1), axis=1)
    energy = 0.5 * (energy_a + energy_b)
    cross_dot = np.sum(mu_a * mu_b, axis=-1)
    cross_dot_positive = np.maximum(cross_dot, 0.0)
    kappa_cf = cross_dot_positive / (
        np.sqrt(np.maximum(energy_a * energy_b, 0.0)) + float(epsilon)
    )

    mu_bar = 0.5 * (mu_a + mu_b)
    top_two = np.partition(mu_bar, kth=mu_bar.shape[-1] - 2, axis=-1)[..., -2:]
    top_two.sort(axis=-1)
    selectivity = (top_two[..., 1] - top_two[..., 0]) / (
        np.linalg.norm(mu_bar, axis=-1) + float(epsilon)
    )
    rho = kappa_cf * np.maximum(selectivity, 0.0)
    active_threshold = np.median(energy, axis=1, keepdims=True)
    active = energy >= active_threshold

    return GenericProbeStatistics(
        mu_a=mu_a,
        mu_b=mu_b,
        energy_a=energy_a,
        energy_b=energy_b,
        energy=energy,
        cross_dot_positive=cross_dot_positive,
        kappa_cf=kappa_cf,
        selectivity=selectivity,
        rho=rho,
        active=active,
        S=energy.mean(axis=1),
        Dcf=cross_dot_positive.mean(axis=1),
        K=kappa_cf.mean(axis=1),
        R=_top_fraction_mean(rho, active, float(top_fraction)),
    )


def _bootstrap_arm_metrics(
    response: np.ndarray,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    *,
    epsilon: float,
    top_fraction: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values = np.asarray(response, dtype=np.float64)
    split = values.shape[1] // 2
    left, right = values[:, :split], values[:, split:]
    def weighted_mean(counts: np.ndarray, samples: np.ndarray) -> np.ndarray:
        sample_first = np.moveaxis(samples, 1, 0)
        flat = np.ascontiguousarray(sample_first.reshape(sample_first.shape[0], -1))
        counts_tensor = torch.from_numpy(np.ascontiguousarray(counts, dtype=np.float32)).to(device)
        sample_tensor = torch.from_numpy(np.ascontiguousarray(flat, dtype=np.float32)).to(device)
        product = counts_tensor.matmul(sample_tensor).cpu().numpy()
        return product.reshape((counts.shape[0],) + samples.shape[:1] + samples.shape[2:]) / split

    mean_a = weighted_mean(counts_a, left)
    mean_b = weighted_mean(counts_b, right)
    left_energy = np.sum(np.square(left), axis=-1)
    right_energy = np.sum(np.square(right), axis=-1)
    energy_a = weighted_mean(counts_a, left_energy)
    energy_b = weighted_mean(counts_b, right_energy)
    energy = 0.5 * (energy_a + energy_b)
    cross_dot = np.maximum(np.sum(mean_a * mean_b, axis=-1), 0.0)
    kappa = cross_dot / (np.sqrt(np.maximum(energy_a * energy_b, 0.0)) + float(epsilon))
    mean_bar = 0.5 * (mean_a + mean_b)
    top_two = np.partition(mean_bar, kth=mean_bar.shape[-1] - 2, axis=-1)[..., -2:]
    top_two.sort(axis=-1)
    selectivity = (top_two[..., 1] - top_two[..., 0]) / (
        np.linalg.norm(mean_bar, axis=-1) + float(epsilon)
    )
    rho = kappa * np.maximum(selectivity, 0.0)
    active = energy >= np.median(energy, axis=2, keepdims=True)
    risk = np.zeros(rho.shape[:2], dtype=np.float64)
    for bootstrap_id in range(rho.shape[0]):
        for client_id in range(rho.shape[1]):
            selected = rho[bootstrap_id, client_id, active[bootstrap_id, client_id]]
            count = max(1, int(np.ceil(float(top_fraction) * selected.size)))
            risk[bootstrap_id, client_id] = np.partition(
                selected, selected.size - count
            )[-count:].mean()
    return {
        "S": energy.mean(axis=2),
        "Dcf": cross_dot.mean(axis=2),
        "K": kappa.mean(axis=2),
        "R": risk,
    }


def paired_bootstrap_generic_deltas(
    gamma0_response: np.ndarray,
    gamma9_response: np.ndarray,
    *,
    samples: int = 1000,
    seed: int = 20260904,
    epsilon: float = 1.0e-12,
    top_fraction: float = 0.20,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    zero = np.asarray(gamma0_response, dtype=np.float64)
    nine = np.asarray(gamma9_response, dtype=np.float64)
    if zero.shape != nine.shape or zero.ndim != 4:
        raise ValueError("paired responses must share [client,source,probe,class] shape")
    if zero.shape[1] % 2:
        raise ValueError("paired responses require equal disjoint carrier halves")
    if int(samples) < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    half = zero.shape[1] // 2
    rng = np.random.default_rng(int(seed))
    probabilities = np.full(half, 1.0 / half, dtype=np.float64)
    counts_a = rng.multinomial(half, probabilities, size=int(samples)).astype(np.float64)
    counts_b = rng.multinomial(half, probabilities, size=int(samples)).astype(np.float64)
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA bootstrap requested but unavailable")
    zero_metrics = _bootstrap_arm_metrics(
        zero,
        counts_a,
        counts_b,
        epsilon=epsilon,
        top_fraction=top_fraction,
        device=compute_device,
    )
    nine_metrics = _bootstrap_arm_metrics(
        nine,
        counts_a,
        counts_b,
        epsilon=epsilon,
        top_fraction=top_fraction,
        device=compute_device,
    )

    result: dict[str, object] = {"samples": int(samples), "seed": int(seed)}
    for metric in ("S", "Dcf", "K", "R"):
        delta = nine_metrics[metric] - zero_metrics[metric]
        pooled = delta.mean(axis=1)
        result[f"{metric}_delta"] = {
            "pooled_mean": float(pooled.mean()),
            "ci95": [float(np.quantile(pooled, 0.025)), float(np.quantile(pooled, 0.975))],
            "client_mean": delta.mean(axis=0).tolist(),
            "positive_clients": int(np.count_nonzero(delta.mean(axis=0) > 0.0)),
        }
    return result


def _arm_summary(statistics: GenericProbeStatistics) -> dict[str, object]:
    return {
        "S": float(statistics.S.mean()),
        "Dcf": float(statistics.Dcf.mean()),
        "K": float(statistics.K.mean()),
        "R": float(statistics.R.mean()),
        "active_probe_count_client": statistics.active.sum(axis=1).astype(int).tolist(),
        "S_client": statistics.S.tolist(),
        "Dcf_client": statistics.Dcf.tolist(),
        "K_client": statistics.K.tolist(),
        "R_client": statistics.R.tolist(),
    }


def summarize_generic_probe_arm(
    combined: GenericProbeStatistics,
    bank_a: GenericProbeStatistics,
    bank_b: GenericProbeStatistics,
) -> dict[str, object]:
    return {
        "combined": _arm_summary(combined),
        "bank_a": _arm_summary(bank_a),
        "bank_b": _arm_summary(bank_b),
    }


def decide_generic_probe_gate(
    arm_results: dict[str, dict[str, object]],
    bootstrap: dict[str, dict[str, object]],
) -> dict[str, object]:
    if set(arm_results) != {"h0", "h9", "l0", "l9"}:
        raise ValueError("arm_results must contain h0, h9, l0 and l9")
    if set(bootstrap) != {"hfl", "local"}:
        raise ValueError("bootstrap must contain hfl and local")

    systems: dict[str, object] = {}
    all_pass = True
    any_fragility_kill = False
    for system, zero_arm, nine_arm in (("hfl", "h0", "h9"), ("local", "l0", "l9")):
        zero = arm_results[zero_arm]
        nine = arm_results[nine_arm]
        zero_combined = zero["combined"]
        nine_combined = nine["combined"]
        boot = bootstrap[system]
        r_delta_client = np.asarray(nine_combined["R_client"]) - np.asarray(
            zero_combined["R_client"]
        )

        def ratio(numerator: float, denominator: float) -> float:
            if denominator == 0.0:
                return float("inf") if numerator > 0.0 else 1.0
            return float(numerator / denominator)

        values = {
            "Dcf_delta": float(nine_combined["Dcf"] - zero_combined["Dcf"]),
            "Dcf_delta_ci95": boot["Dcf_delta"]["ci95"],
            "K_delta": float(nine_combined["K"] - zero_combined["K"]),
            "K_delta_ci95": boot["K_delta"]["ci95"],
            "R_ratio_combined": ratio(float(nine_combined["R"]), float(zero_combined["R"])),
            "R_delta_ci95": boot["R_delta"]["ci95"],
            "R_positive_clients": int(np.count_nonzero(r_delta_client > 0.0)),
            "R_delta_client": r_delta_client.tolist(),
            "R_ratio_bank_a": ratio(float(nine["bank_a"]["R"]), float(zero["bank_a"]["R"])),
            "R_ratio_bank_b": ratio(float(nine["bank_b"]["R"]), float(zero["bank_b"]["R"])),
            "S_delta": float(nine_combined["S"] - zero_combined["S"]),
            "S_delta_ci95": boot["S_delta"]["ci95"],
        }
        gates = {
            "G1_Dcf_bootstrap": float(values["Dcf_delta_ci95"][0]) > 0.0,
            "G2_K_effect_size": values["K_delta"] >= 0.03,
            "G3_K_bootstrap": float(values["K_delta_ci95"][0]) > 0.0,
            "G4_R_combined_ratio": values["R_ratio_combined"] >= 1.20,
            "G5_R_bootstrap": float(values["R_delta_ci95"][0]) > 0.0,
            "G6_R_positive_clients": values["R_positive_clients"] >= 3,
            "G7_bank_a_replication": values["R_ratio_bank_a"] >= 1.10,
            "G8_bank_b_replication": values["R_ratio_bank_b"] >= 1.10,
        }
        fragility_significant = float(values["S_delta_ci95"][0]) > 0.0
        k_or_r_failed = not all(gates[key] for key in (
            "G2_K_effect_size",
            "G3_K_bootstrap",
            "G4_R_combined_ratio",
            "G5_R_bootstrap",
            "G6_R_positive_clients",
        ))
        generic_fragility_kill = bool(fragility_significant and k_or_r_failed)
        passed = all(gates.values())
        systems[system] = {
            "values": values,
            "gates": gates,
            "generic_fragility_kill": generic_fragility_kill,
            "pass": passed,
        }
        all_pass = all_pass and passed
        any_fragility_kill = any_fragility_kill or generic_fragility_kill

    return {
        "verdict": (
            "GO_TO_K1_CHECKPOINT_SURGERY"
            if all_pass
            else "NO_GO_GENERIC_DIRECTIONAL_SIGNAL"
        ),
        "thresholds": {
            "Dcf_bootstrap_ci_lower": 0.0,
            "minimum_K_delta": 0.03,
            "K_bootstrap_ci_lower": 0.0,
            "minimum_combined_R_ratio": 1.20,
            "R_bootstrap_ci_lower": 0.0,
            "minimum_positive_R_clients": 3,
            "minimum_each_bank_R_ratio": 1.10,
        },
        "generic_fragility_kill_triggered": any_fragility_kill,
        "systems": systems,
    }


__all__ = [
    "GenericProbeStatistics",
    "decide_generic_probe_gate",
    "generic_probe_statistics",
    "paired_bootstrap_generic_deltas",
    "summarize_generic_probe_arm",
]
