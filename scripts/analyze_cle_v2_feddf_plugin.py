from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_v2_factorial import compute_operator_dsa, paired_bootstrap_estimand  # noqa: E402
from scripts.analyze_cle_v2_factorial import infer_arm, load_grid, resolve_device, trailing_metrics  # noqa: E402
from scripts.analyze_cle_v2_mechanism_stage1 import balanced_source_indices, grid_accuracy  # noqa: E402
from scripts.analyze_cle_v2_plugin_stage2 import load_trace  # noqa: E402


ARMS = ("fd_b", "fd_p")
UTILITY_METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen CLE-v2 FedDF PEW+BER A/B.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "benchmark", "formal"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def audit_pair(outputs_root: Path) -> dict[str, object]:
    traces = {
        arm: load_trace(
            outputs_root / f"cle_v2_feddf_plugin_{arm}_trainseed0" / "local_batch_trace.jsonl"
        )
        for arm in ARMS
    }
    return {"matched": traces["fd_b"] == traces["fd_p"], "rows": len(traces["fd_b"])}


def evaluate_gates(
    *,
    reduction: float,
    client_reduction: np.ndarray,
    confidence_interval: list[float],
    accuracies: dict[str, dict[str, object]],
    utility_delta: dict[str, float],
    trace_audit: dict[str, object],
) -> dict[str, bool]:
    return {
        "I0_paired_integrity": bool(trace_audit["matched"]),
        "L0_learning_floor": all(float(accuracies[arm]["pooled"]) >= 20.0 for arm in ARMS),
        "P1_dsa_reduction": bool(
            reduction >= 0.02
            and confidence_interval[0] > 0.0
            and float(np.min(client_reduction)) > 0.0
        ),
        "P2_mean_utility": bool(
            utility_delta["avg_acc"] >= 0.0
            and float(accuracies["fd_p"]["pooled"])
            >= float(accuracies["fd_b"]["pooled"]) - 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("FedDF plugin Formal analysis requires --confirm-formal")
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
        experiment_root = outputs_root / f"cle_v2_feddf_plugin_{arm}_trainseed0"
        predictions[arm] = infer_arm(
            experiment_root / "checkpoints", grid, device, int(args.batch_size)
        )
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        accuracies[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = trailing_metrics(experiment_root / "metrics.csv", window=5)
    reduction = float(results["fd_b"].pooled - results["fd_p"].pooled)
    client_reduction = np.asarray(results["fd_b"].client) - np.asarray(results["fd_p"].client)
    bootstrap = paired_bootstrap_estimand(
        results,
        {"fd_b": 1.0, "fd_p": -1.0},
        samples=int(args.bootstrap_samples),
        seed=20260913,
    )
    confidence_interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    utility_delta = {
        name: metrics["fd_p"].get(name, float("nan"))
        - metrics["fd_b"].get(name, float("nan"))
        for name in UTILITY_METRICS
    }
    trace_audit = audit_pair(outputs_root)
    gates = evaluate_gates(
        reduction=reduction,
        client_reduction=client_reduction,
        confidence_interval=confidence_interval,
        accuracies=accuracies,
        utility_delta=utility_delta,
        trace_audit=trace_audit,
    )
    if args.mode != "formal":
        verdict = f"{args.mode.upper()}_ONLY_NO_SCIENTIFIC_DECISION"
    elif all(gates.values()):
        verdict = "GO_PEW_BER_FEDDF_PLUGIN_SEED0"
    else:
        verdict = "NO_GO_PEW_BER_FEDDF_PLUGIN_SEED0"
    np.savez_compressed(
        output_dir / "FEDDF_PLUGIN_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    summary = {
        "protocol": "cle_v2_feddf_plugin_analysis_v1",
        "mode": args.mode,
        "train_seed": 0,
        "pooled_dsa": {arm: float(results[arm].pooled) for arm in ARMS},
        "plugin_dsa_reduction": reduction,
        "client_plugin_dsa_reduction": client_reduction.tolist(),
        "source_paired_bootstrap_ci95": confidence_interval,
        "operator_grid_accuracy": accuracies,
        "reporting_metrics_last5": metrics,
        "plugin_minus_base_last5": utility_delta,
        "paired_local_traces": trace_audit,
        "frozen_gates": gates,
        "scientific_verdict": verdict,
        "claim_scope": "PEW+BER portability to matched FedDF-fidelity communication",
        "architecture_uniform_utility_claim": False,
        "cdep_used": False,
    }
    (output_dir / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
