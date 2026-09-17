from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_v2_factorial import (  # noqa: E402
    compute_operator_dsa,
    paired_bootstrap_estimand,
    shuffled_binding_null,
)
from scripts.analyze_cle_v2_factorial import infer_arm, load_grid, resolve_device  # noqa: E402
from scripts.analyze_cle_v2_plugin_stage2 import load_trace  # noqa: E402


ARMS = ("h0_b", "h9_b", "l0_b", "l9_b")
DEFINITIONS = {
    "hfl_cle_effect_base": {"h9_b": 1.0, "h0_b": -1.0},
    "local_cle_effect_base": {"l9_b": 1.0, "l0_b": -1.0},
    "communication_amplification_base": {
        "h9_b": 1.0,
        "h0_b": -1.0,
        "l9_b": -1.0,
        "l0_b": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the four-arm CLE-v2 local-first factorial.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--mode", choices=("benchmark", "formal"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def linear_contrast(values: dict[str, float | np.ndarray], coefficients: dict[str, float]):
    terms = [float(weight) * values[name] for name, weight in coefficients.items()]
    return sum(terms[1:], terms[0])


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Formal local-first analysis requires --confirm-formal")
    package_root = args.package_root.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operator_names, binding = load_grid(package_root)
    device = resolve_device(args.device)
    predictions, results, traces = {}, {}, {}
    for arm in ARMS:
        root = outputs_root / f"cle_v2_factorial_{arm}_trainseed{args.train_seed}"
        predictions[arm] = infer_arm(root / "checkpoints", grid, device, int(args.batch_size))
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        traces[arm] = load_trace(root / "local_batch_trace.jsonl")

    pooled = {arm: float(results[arm].pooled) for arm in ARMS}
    client = {arm: np.asarray(results[arm].client) for arm in ARMS}
    estimands = {
        name: float(linear_contrast(pooled, coefficients))
        for name, coefficients in DEFINITIONS.items()
    }
    client_estimands = {
        name: np.asarray(linear_contrast(client, coefficients)).tolist()
        for name, coefficients in DEFINITIONS.items()
    }
    bootstrap = {}
    for name, coefficients in DEFINITIONS.items():
        values = paired_bootstrap_estimand(
            {arm: results[arm] for arm in coefficients},
            coefficients,
            samples=int(args.bootstrap_samples),
            seed=20260909 + int(args.train_seed),
        )
        bootstrap[name] = {
            "mean": float(values.mean()),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        }
    shuffled = shuffled_binding_null(
        predictions["h9_b"],
        labels,
        binding,
        permutations=int(args.permutations),
        seed=20260909 + int(args.train_seed),
    )
    trace_match = {
        "gamma00_hfl_vs_local": traces["h0_b"] == traces["l0_b"],
        "gamma09_hfl_vs_local": traces["h9_b"] == traces["l9_b"],
    }
    local_share = estimands["local_cle_effect_base"] / max(
        abs(estimands["hfl_cle_effect_base"]), 1.0e-12
    )
    gates = {
        "I0_matched_local_traces": all(trace_match.values()),
        "M1_hfl_directional_shortcut": bool(
            estimands["hfl_cle_effect_base"] >= 0.10
            and bootstrap["hfl_cle_effect_base"]["ci95"][0] > 0.0
            and min(client_estimands["hfl_cle_effect_base"]) > 0.0
        ),
        "M2_binding_specificity": bool(
            shuffled["observed"] > shuffled["null_p95"] and shuffled["p_value"] <= 0.05
        ),
        "M3_local_first_presence": bool(
            estimands["local_cle_effect_base"] >= 0.05
            and bootstrap["local_cle_effect_base"]["ci95"][0] > 0.0
            and min(client_estimands["local_cle_effect_base"]) > 0.0
            and local_share >= 0.50
        ),
    }
    summary = {
        "protocol": "cle_v2_local_first_multiseed_v1",
        "mode": args.mode,
        "train_seed": int(args.train_seed),
        "arms": list(ARMS),
        "pooled_dsa": pooled,
        "estimands": estimands,
        "client_estimands": client_estimands,
        "definitions": DEFINITIONS,
        "bootstrap": bootstrap,
        "h9_base_shuffled_binding": {
            "observed": float(shuffled["observed"]),
            "null_p95": float(shuffled["null_p95"]),
            "p_value": float(shuffled["p_value"]),
        },
        "paired_local_traces": trace_match,
        "local_first_share_descriptive_ratio": float(local_share),
        "ratio_warning": "Descriptive ratio, not a sample-level causal decomposition.",
        "frozen_gates": gates,
        "scientific_verdict": (
            "GO_LOCAL_FIRST_SEED" if args.mode == "formal" and all(gates.values())
            else "NO_GO_LOCAL_FIRST_SEED" if args.mode == "formal"
            else "BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION"
        ),
        "scientific_evidence": args.mode == "formal",
    }
    np.savez_compressed(
        output_dir / "LOCAL_FIRST_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    (output_dir / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
