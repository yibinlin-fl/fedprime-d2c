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
from scripts.analyze_cle_v2_factorial import (  # noqa: E402
    infer_arm,
    load_grid,
    resolve_device,
    trailing_metrics,
)
from scripts.analyze_cle_v2_mechanism_stage1 import balanced_source_indices, grid_accuracy  # noqa: E402
from scripts.analyze_cle_v2_plugin_stage2 import load_trace  # noqa: E402
from scripts.run_cle_v2_kt_fccl_plugin import ARMS, BASES  # noqa: E402


PAIRS = {"kt": ("kt_b", "kt_p"), "fc": ("fc_b", "fc_p")}
UTILITY_METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CLE-v2 KT-pFL/FCCL plugin pairs.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "benchmark", "formal"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--null-permutations", type=int, default=1000)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def audit_pair(outputs_root: Path, pair: tuple[str, str]) -> dict[str, object]:
    traces = {
        arm: load_trace(
            outputs_root
            / f"cle_v2_kt_fccl_plugin_{arm}_trainseed0"
            / "local_batch_trace.jsonl"
        )
        for arm in pair
    }
    return {"matched": traces[pair[0]] == traces[pair[1]], "rows": len(traces[pair[0]])}


def evaluate_pair_gates(
    *,
    base_dsa: float,
    null: dict[str, float],
    reduction: float,
    client_reduction: np.ndarray,
    confidence_interval: list[float],
    accuracies: dict[str, dict[str, object]],
    utility_delta: dict[str, float],
    trace_audit: dict[str, object],
    pair: tuple[str, str],
) -> dict[str, bool]:
    base, plugin = pair
    return {
        "I0_paired_integrity": bool(trace_audit["matched"]),
        "L0_learning_floor": all(float(accuracies[arm]["pooled"]) >= 20.0 for arm in pair),
        "C1_directional_shortcut": bool(
            base_dsa >= 0.05
            and base_dsa > float(null["null_p95"])
            and float(null["p_value"]) <= 0.01
        ),
        "C2_dsa_reduction": bool(
            reduction >= 0.02
            and confidence_interval[0] > 0.0
            and float(np.min(client_reduction)) > 0.0
        ),
        "U0_mean_utility": bool(
            utility_delta["avg_acc"] >= 0.0
            and float(accuracies[plugin]["pooled"])
            >= float(accuracies[base]["pooled"]) - 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("KT/FCCL plugin Formal analysis requires --confirm-formal")
    package_root = args.package_root.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operator_names, binding = load_grid(package_root)
    if args.max_sources is not None:
        selected = balanced_source_indices(labels, int(args.max_sources))
        grid, labels = grid[selected], labels[selected]
    device = resolve_device(args.device)
    predictions, results, accuracies, metrics = {}, {}, {}, {}
    for arm in ARMS:
        experiment_root = outputs_root / f"cle_v2_kt_fccl_plugin_{arm}_trainseed0"
        predictions[arm] = infer_arm(
            experiment_root / "checkpoints", grid, device, int(args.batch_size)
        )
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        accuracies[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = trailing_metrics(experiment_root / "metrics.csv", window=5)

    pair_results = {}
    for pair_index, (key, pair) in enumerate(PAIRS.items()):
        base, plugin = pair
        reduction = float(results[base].pooled - results[plugin].pooled)
        client_reduction = np.asarray(results[base].client) - np.asarray(results[plugin].client)
        bootstrap = paired_bootstrap_estimand(
            {arm: results[arm] for arm in pair},
            {base: 1.0, plugin: -1.0},
            samples=int(args.bootstrap_samples),
            seed=20260914 + pair_index,
        )
        confidence_interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        null_raw = shuffled_binding_null(
            predictions[base],
            labels,
            binding,
            permutations=int(args.null_permutations),
            seed=20260914 + pair_index,
        )
        null = {
            "observed": float(null_raw["observed"]),
            "null_p95": float(null_raw["null_p95"]),
            "p_value": float(null_raw["p_value"]),
        }
        utility_delta = {
            name: metrics[plugin].get(name, float("nan"))
            - metrics[base].get(name, float("nan"))
            for name in UTILITY_METRICS
        }
        trace_audit = audit_pair(outputs_root, pair)
        gates = evaluate_pair_gates(
            base_dsa=float(results[base].pooled),
            null=null,
            reduction=reduction,
            client_reduction=client_reduction,
            confidence_interval=confidence_interval,
            accuracies=accuracies,
            utility_delta=utility_delta,
            trace_audit=trace_audit,
            pair=pair,
        )
        if args.mode != "formal":
            verdict = f"{args.mode.upper()}_ONLY_NO_SCIENTIFIC_DECISION"
        elif all(gates.values()):
            verdict = f"GO_PEW_BER_{key.upper()}_PLUGIN_SEED0"
        else:
            verdict = f"NO_GO_PEW_BER_{key.upper()}_PLUGIN_SEED0"
        pair_results[key] = {
            "adapter_scope": BASES[key]["claim_name"],
            "pooled_dsa": {arm: float(results[arm].pooled) for arm in pair},
            "plugin_dsa_reduction": reduction,
            "client_plugin_dsa_reduction": client_reduction.tolist(),
            "source_paired_bootstrap_ci95": confidence_interval,
            "base_shuffled_binding_null": null,
            "operator_grid_accuracy": {arm: accuracies[arm] for arm in pair},
            "reporting_metrics_last5": {arm: metrics[arm] for arm in pair},
            "plugin_minus_base_last5": utility_delta,
            "paired_local_traces": trace_audit,
            "frozen_gates": gates,
            "scientific_verdict": verdict,
        }

    np.savez_compressed(
        output_dir / "KT_FCCL_PLUGIN_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    summary = {
        "protocol": "cle_v2_kt_fccl_plugin_analysis_v1",
        "mode": args.mode,
        "train_seed": 0,
        "pairs": pair_results,
        "scientific_evidence": args.mode == "formal",
        "baseline_communication_implementations_modified": False,
        "cdep_used": False,
    }
    (output_dir / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
