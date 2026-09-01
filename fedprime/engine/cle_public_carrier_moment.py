from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PublicCarrierResponses:
    class_vs_rest_delta: np.ndarray
    centered_response: np.ndarray
    centered_raw_logit_response: np.ndarray
    probability_response: np.ndarray


@dataclass(frozen=True)
class DirectionalMoment:
    mean_response: np.ndarray
    second_moment: np.ndarray
    coherence: np.ndarray
    directional_strength_client: np.ndarray
    coherence_client: np.ndarray
    split_cosine: np.ndarray
    split_cosine_client: np.ndarray


def _validate_logits(base_logits: np.ndarray, probe_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_logits, dtype=np.float64)
    probes = np.asarray(probe_logits, dtype=np.float64)
    if base.ndim != 3:
        raise ValueError("base_logits must have shape [client,source,class]")
    if probes.ndim != 4:
        raise ValueError("probe_logits must have shape [client,source,probe,class]")
    if probes.shape[:2] != base.shape[:2] or probes.shape[-1] != base.shape[-1]:
        raise ValueError("base/probe logits have incompatible shapes")
    if base.shape[-1] < 2 or probes.shape[2] < 1:
        raise ValueError("at least two classes and one probe are required")
    if not np.isfinite(base).all() or not np.isfinite(probes).all():
        raise ValueError("logits contain non-finite values")
    return base, probes


def class_vs_rest_evidence(logits: np.ndarray) -> np.ndarray:
    """Return z_c - logsumexp(z_not_c), preserving architecture-local logit scale."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 2:
        raise ValueError("logits must have a final class axis with at least two classes")
    if not np.isfinite(values).all():
        raise ValueError("logits contain non-finite values")
    result = np.empty_like(values, dtype=np.float64)
    for class_id in range(values.shape[-1]):
        other = np.delete(values, class_id, axis=-1)
        maximum = other.max(axis=-1, keepdims=True)
        logsumexp = maximum[..., 0] + np.log(np.exp(other - maximum).sum(axis=-1))
        result[..., class_id] = values[..., class_id] - logsumexp
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def public_carrier_responses(
    base_logits: np.ndarray,
    probe_logits: np.ndarray,
) -> PublicCarrierResponses:
    """Compute blind sample-level responses without family or binding metadata."""

    base, probes = _validate_logits(base_logits, probe_logits)
    evidence_delta = class_vs_rest_evidence(probes) - class_vs_rest_evidence(base)[:, :, None, :]
    centered = evidence_delta - evidence_delta.mean(axis=-1, keepdims=True)
    raw_delta = probes - base[:, :, None, :]
    centered_raw = raw_delta - raw_delta.mean(axis=-1, keepdims=True)
    probability_delta = _softmax(probes) - _softmax(base)[:, :, None, :]
    return PublicCarrierResponses(
        class_vs_rest_delta=evidence_delta,
        centered_response=centered,
        centered_raw_logit_response=centered_raw,
        probability_response=probability_delta,
    )


def directional_moment(
    centered_response: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> DirectionalMoment:
    response = np.asarray(centered_response, dtype=np.float64)
    if response.ndim != 4:
        raise ValueError("centered_response must have shape [client,source,probe,class]")
    if response.shape[1] < 2:
        raise ValueError("at least two carriers are required")
    if not np.isfinite(response).all():
        raise ValueError("centered_response contains non-finite values")

    mean_response = response.mean(axis=1)
    mean_norm_sq = np.sum(np.square(mean_response), axis=-1)
    second_moment = np.mean(np.sum(np.square(response), axis=-1), axis=1)
    coherence = mean_norm_sq / np.maximum(second_moment, float(epsilon))

    split = response.shape[1] // 2
    left = response[:, :split].mean(axis=1)
    right = response[:, split:].mean(axis=1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    split_cosine = np.divide(
        np.sum(left * right, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > float(epsilon),
    )
    return DirectionalMoment(
        mean_response=mean_response,
        second_moment=second_moment,
        coherence=coherence,
        directional_strength_client=mean_norm_sq.mean(axis=1),
        coherence_client=coherence.mean(axis=1),
        split_cosine=split_cosine,
        split_cosine_client=split_cosine.mean(axis=1),
    )


def paired_bootstrap_moment_delta(
    gamma0_response: np.ndarray,
    gamma9_response: np.ndarray,
    *,
    samples: int = 1000,
    seed: int = 20260901,
    epsilon: float = 1.0e-12,
) -> dict[str, object]:
    zero = np.asarray(gamma0_response, dtype=np.float64)
    nine = np.asarray(gamma9_response, dtype=np.float64)
    if zero.shape != nine.shape or zero.ndim != 4:
        raise ValueError("paired responses must share [client,source,probe,class] shape")
    if int(samples) < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    rng = np.random.default_rng(int(seed))
    strength_delta = np.zeros((int(samples), zero.shape[0]), dtype=np.float64)
    coherence_delta = np.zeros_like(strength_delta)

    for bootstrap_id in range(int(samples)):
        indices = rng.integers(0, zero.shape[1], size=zero.shape[1])
        for values, target_strength, target_coherence, sign in (
            (zero[:, indices], strength_delta, coherence_delta, -1.0),
            (nine[:, indices], strength_delta, coherence_delta, 1.0),
        ):
            mean = values.mean(axis=1)
            mean_norm_sq = np.sum(np.square(mean), axis=-1)
            second = np.mean(np.sum(np.square(values), axis=-1), axis=1)
            target_strength[bootstrap_id] += sign * mean_norm_sq.mean(axis=1)
            target_coherence[bootstrap_id] += sign * (
                mean_norm_sq / np.maximum(second, float(epsilon))
            ).mean(axis=1)

    def summarize(values: np.ndarray) -> dict[str, object]:
        pooled = values.mean(axis=1)
        return {
            "pooled_mean": float(pooled.mean()),
            "ci95": [float(np.quantile(pooled, 0.025)), float(np.quantile(pooled, 0.975))],
            "client_mean": values.mean(axis=0).tolist(),
            "positive_clients": int(np.count_nonzero(values.mean(axis=0) > 0.0)),
        }

    return {
        "samples": int(samples),
        "seed": int(seed),
        "directional_strength_delta": summarize(strength_delta),
        "coherence_delta": summarize(coherence_delta),
    }


def decide_public_carrier_gate(
    arm_results: dict[str, dict[str, object]],
    bootstrap: dict[str, dict[str, object]],
    *,
    minimum_gamma9_map: float = 0.65,
    minimum_map_delta: float = 0.20,
    minimum_hit_rate: float = 0.70,
    maximum_null_p: float = 0.01,
) -> dict[str, object]:
    if set(arm_results) != {"h0", "h9", "l0", "l9"}:
        raise ValueError("arm_results must contain h0, h9, l0 and l9")
    if set(bootstrap) != {"hfl", "local"}:
        raise ValueError("bootstrap must contain hfl and local")

    systems: dict[str, object] = {}
    all_pass = True
    for system, zero_arm, nine_arm in (("hfl", "h0", "h9"), ("local", "l0", "l9")):
        zero = arm_results[zero_arm]
        nine = arm_results[nine_arm]
        zero_retrieval = zero["retrieval"]
        nine_retrieval = nine["retrieval"]
        client_map_delta = np.asarray(nine_retrieval["client_mean_average_precision"]) - np.asarray(
            zero_retrieval["client_mean_average_precision"]
        )
        split_delta = np.asarray(nine["split_cosine_client"]) - np.asarray(
            zero["split_cosine_client"]
        )
        strength_ci = bootstrap[system]["directional_strength_delta"]["ci95"]
        coherence_ci = bootstrap[system]["coherence_delta"]["ci95"]
        class_p = float(nine["nulls"]["class_map_null"]["mean_average_precision"]["one_sided_p"])
        probe_p = float(nine["nulls"]["probe_identity_null"]["mean_average_precision"]["one_sided_p"])
        values = {
            "gamma9_map": float(nine_retrieval["mean_average_precision"]),
            "map_delta": float(
                nine_retrieval["mean_average_precision"] - zero_retrieval["mean_average_precision"]
            ),
            "positive_map_clients": int(np.count_nonzero(client_map_delta > 0.0)),
            "gamma9_hit_rate": float(nine_retrieval["class_to_probe_family_hit_rate"]),
            "class_map_p": class_p,
            "probe_identity_p": probe_p,
            "directional_strength_ci95": strength_ci,
            "coherence_ci95": coherence_ci,
            "gamma9_split_cosine": float(np.mean(nine["split_cosine_client"])),
            "split_cosine_delta": float(np.mean(split_delta)),
            "positive_split_clients": int(np.count_nonzero(split_delta > 0.0)),
            "client_map_delta": client_map_delta.tolist(),
            "client_split_cosine_delta": split_delta.tolist(),
        }
        gates = {
            "G1_gamma9_map": values["gamma9_map"] >= minimum_gamma9_map,
            "G2_map_delta": values["map_delta"] >= minimum_map_delta,
            "G3_positive_map_clients": values["positive_map_clients"] >= 3,
            "G4_hit_rate": values["gamma9_hit_rate"] >= minimum_hit_rate,
            "G5_binding_null": max(class_p, probe_p) <= maximum_null_p,
            "G6_directional_strength_ci": float(strength_ci[0]) > 0.0,
            "G7_coherence_ci": float(coherence_ci[0]) > 0.0,
            "G8_gamma9_split_reproducibility": values["gamma9_split_cosine"] > 0.0,
            "G9_split_reproducibility_delta": values["split_cosine_delta"] > 0.0,
            "G10_positive_split_clients": values["positive_split_clients"] >= 3,
        }
        passed = all(gates.values())
        systems[system] = {"values": values, "gates": gates, "pass": passed}
        all_pass = all_pass and passed

    return {
        "verdict": "GO_TO_K0_B" if all_pass else "NO_GO_PUBLIC_CARRIER_ROUTE",
        "thresholds": {
            "minimum_gamma9_map": minimum_gamma9_map,
            "minimum_map_delta": minimum_map_delta,
            "minimum_hit_rate": minimum_hit_rate,
            "maximum_null_p": maximum_null_p,
            "minimum_positive_clients": 3,
            "bootstrap_ci_lower_bound": 0.0,
            "split_cosine_lower_bound": 0.0,
        },
        "systems": systems,
    }
