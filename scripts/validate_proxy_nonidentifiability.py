from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def dependence_tv(labels: np.ndarray, environments: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    environments = np.asarray(environments, dtype=np.int64).reshape(-1)
    if labels.shape != environments.shape or labels.size == 0:
        raise ValueError("labels and environments must be non-empty aligned vectors")
    y_values, y_inverse = np.unique(labels, return_inverse=True)
    e_values, e_inverse = np.unique(environments, return_inverse=True)
    joint = np.zeros((y_values.size, e_values.size), dtype=np.float64)
    np.add.at(joint, (y_inverse, e_inverse), 1.0)
    joint /= float(labels.size)
    product = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    return float(0.5 * np.abs(joint - product).sum())


def balanced_proxy(labels: np.ndarray, environments: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if environments < 2:
        raise ValueError("at least two proxy environments are required")
    proxy = np.empty_like(labels)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        proxy[indices] = np.arange(indices.size, dtype=np.int64) % int(environments)
    return proxy


def nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    output = np.full(np.asarray(total).shape, np.nan, dtype=np.float64)
    return np.divide(total, count, out=output, where=count > 0)


def operator_dsa(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    binding = np.asarray(binding, dtype=np.int64)
    clients, sources, operators, _ = probs.shape
    if labels.shape != (sources,) or binding.shape[0] != clients:
        raise ValueError("unexpected DSA cache contract")
    effects = np.full((clients, operators, sources), np.nan, dtype=np.float64)
    for client_id in range(clients):
        for operator_id in range(operators):
            bound = np.flatnonzero(binding[client_id] == operator_id)
            if bound.size == 0:
                continue
            valid = ~np.isin(labels, bound)
            mass = np.take(probs[client_id], bound, axis=-1).sum(axis=-1)
            target = mass[:, operator_id]
            reference = (mass.sum(axis=1) - target) / float(operators - 1)
            effects[client_id, operator_id, valid] = target[valid] - reference[valid]
    return float(nanmean(nanmean(nanmean(effects, axis=-1), axis=-1), axis=-1))


def identical_view_jsd(probabilities: np.ndarray) -> float:
    values = np.stack([probabilities, probabilities, probabilities], axis=0).astype(
        np.float64
    )
    mixture = values.mean(axis=0)
    terms = np.where(
        values > 0.0,
        values
        * (
            np.log(np.maximum(values, 1.0e-300))
            - np.log(np.maximum(mixture, 1.0e-300))
        ),
        0.0,
    )
    return float(terms.sum(axis=-1).mean(axis=0).max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate proxy non-identifiability and target-aligned DSA necessity."
    )
    parser.add_argument("--stage1-predictions", type=Path, required=True)
    parser.add_argument("--cvrs-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = args.stage1_predictions.resolve()
    cvrs_path = args.cvrs_result.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(prediction_path, allow_pickle=False)
    probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    arms = tuple(str(value) for value in archive["arms"])
    labels = np.asarray(archive["labels"], dtype=np.int64)
    binding = np.asarray(archive["binding"], dtype=np.int64)
    if probabilities.shape != (4, 4, 1000, 15, 10):
        raise ValueError("unexpected Stage-1 probability shape")
    if arms != ("h0_b", "h9_b", "l0_b", "l9_b"):
        raise ValueError("unexpected Stage-1 arms")

    proxy = balanced_proxy(labels, environments=5)
    latent_world_independent = proxy.copy()
    latent_world_bound = labels % 5
    observable_tv = dependence_tv(labels, proxy)
    world_independent_tv = dependence_tv(labels, latent_world_independent)
    world_bound_tv = dependence_tv(labels, latent_world_bound)

    h9 = probabilities[arms.index("h9_b")]
    cached_dsa = operator_dsa(h9, labels, binding)
    cached_identical_view_jsd = identical_view_jsd(h9)

    cvrs = json.loads(cvrs_path.read_text(encoding="utf-8"))
    mobile_rows = {
        str(row["arm"]): row
        for row in cvrs["rows"]
        if str(row["architecture"]).lower() == "mobilenetv2"
    }
    jsd_proxy = float(mobile_rows["jsd"]["routing_strength"])
    cvrs_proxy = float(mobile_rows["cvrs"]["routing_strength"])
    jsd_dsa = float(mobile_rows["jsd"]["oracle"]["dsa"])
    cvrs_dsa = float(mobile_rows["cvrs"]["oracle"]["dsa"])
    proxy_change = cvrs_proxy - jsd_proxy
    dsa_change = cvrs_dsa - jsd_dsa

    gates = {
        "N0_balanced_observable_proxy": observable_tv <= 1.0e-12,
        "N1_same_observable_world_with_zero_true_dependence": world_independent_tv
        <= 1.0e-12,
        "N2_same_observable_world_with_strong_true_dependence": world_bound_tv >= 0.75,
        "N3_cached_zero_jsd_positive_dsa": cached_identical_view_jsd <= 1.0e-12
        and cached_dsa >= 0.10,
        "N4_cvrs_proxy_down_true_dsa_up": proxy_change < 0.0 and dsa_change > 0.0,
    }

    summary = {
        "protocol": "cle_proxy_nonidentifiability_cache_validation_v1",
        "training_or_inference_run": False,
        "inputs": {
            "stage1_predictions": str(prediction_path),
            "stage1_sha256": sha256(prediction_path),
            "cvrs_result": str(cvrs_path),
            "cvrs_result_sha256": sha256(cvrs_path),
        },
        "constructive_nonidentifiability": {
            "source_count": int(labels.size),
            "classes": int(np.unique(labels).size),
            "proxy_environments": 5,
            "same_observable_y_proxy": True,
            "observable_proxy_dependence_tv": observable_tv,
            "world_A_true_dependence_tv": world_independent_tv,
            "world_B_true_dependence_tv": world_bound_tv,
            "world_B_closed_form": "1-sum_e P(E=e)^2",
            "interpretation": (
                "The same observed (Y, proxy) distribution admits a latent world with "
                "Y independent of E and another with E deterministically bound to Y."
            ),
        },
        "cached_jsd_counterexample": {
            "identical_view_jsd": cached_identical_view_jsd,
            "h9_operator_dsa": cached_dsa,
        },
        "cvrs_formal_counterexample": {
            "architecture": "Mobilenetv2",
            "public_jsd_proxy": jsd_proxy,
            "cvrs_proxy": cvrs_proxy,
            "cvrs_minus_jsd_proxy": proxy_change,
            "public_jsd_dsa": jsd_dsa,
            "cvrs_dsa": cvrs_dsa,
            "cvrs_minus_jsd_dsa": dsa_change,
        },
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "claim_boundary": (
            "Proxy-only improvement cannot certify true CLE routing without an additional "
            "identifiability condition. DSA is the target-aligned validation endpoint in "
            "this protocol, not a universally unique shortcut metric."
        ),
    }
    (output_dir / "PROXY_NONIDENTIFIABILITY_VALIDATION.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    latent_bars = axes[0].bar(
        ["Observed\nY-proxy", "Latent world A\nY-E", "Latent world B\nY-E"],
        [observable_tv, world_independent_tv, world_bound_tv],
        color=["#6B8EBD", "#65A87A", "#D96B5F"],
    )
    axes[0].set_ylim(0.0, 0.9)
    axes[0].set_ylabel("Dependence TV")
    axes[0].set_title("Same observables, incompatible latent harm", fontweight="bold")
    axes[0].text(1.0, 0.84, "same P(Y, proxy)", ha="center", fontsize=9)
    for bar, value in zip(
        latent_bars, [observable_tv, world_independent_tv, world_bound_tv]
    ):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            max(value + 0.025, 0.025),
            f"{value:.1f}",
            ha="center",
        )

    axes[1].bar(
        ["Within-operator\nJSD", "Cross-operator\nDSA"],
        [cached_identical_view_jsd, cached_dsa],
        color=["#65A87A", "#D96B5F"],
    )
    axes[1].set_ylim(0.0, max(0.14, cached_dsa * 1.2))
    axes[1].set_title("Frozen Stage-1 cache counterexample", fontweight="bold")
    axes[1].text(0, 0.006, f"{cached_identical_view_jsd:.1e}", ha="center")
    axes[1].text(1, cached_dsa + 0.005, f"{cached_dsa:.6f}", ha="center")

    x = np.arange(2)
    width = 0.34
    axes[2].bar(
        x - width / 2,
        [jsd_proxy, jsd_dsa],
        width,
        label="Public-JSD",
        color="#6B8EBD",
    )
    axes[2].bar(
        x + width / 2,
        [cvrs_proxy, cvrs_dsa],
        width,
        label="CVRS",
        color="#D96B5F",
    )
    axes[2].set_xticks(x, ["Generic proxy", "True DSA"])
    axes[2].set_ylim(0.0, 0.28)
    axes[2].set_title("CVRS Formal: proxy down, DSA up", fontweight="bold")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "Why proxy balancing requires target-aligned CLE validation",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "PROXY_NONIDENTIFIABILITY_THEORY.png", bbox_inches="tight")
    fig.savefig(output_dir / "PROXY_NONIDENTIFIABILITY_THEORY.pdf", bbox_inches="tight")
    plt.close(fig)

    if not summary["all_gates_pass"]:
        raise RuntimeError(f"one or more frozen validation gates failed: {gates}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
