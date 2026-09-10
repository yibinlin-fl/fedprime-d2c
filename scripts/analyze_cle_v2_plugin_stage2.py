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
)
from scripts.analyze_cle_v2_factorial import (  # noqa: E402
    infer_arm,
    load_grid,
    resolve_device,
    trailing_metrics,
)
from scripts.analyze_cle_v2_mechanism_stage1 import (  # noqa: E402
    balanced_source_indices,
    grid_accuracy,
)


ARMS = ("h9_b", "h9_p")
UTILITY_METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen CLE-v2 PEW+BER Stage-2 A/B.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def load_trace(path: Path) -> list[tuple[int, int, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append((int(item["round"]), int(item["client"]), str(item["sha256"])))
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    return rows


def audit_pair(outputs_root: Path) -> dict[str, object]:
    traces = {
        arm: load_trace(
            outputs_root
            / f"cle_v2_plugin_stage2_{arm}_trainseed0"
            / "local_batch_trace.jsonl"
        )
        for arm in ARMS
    }
    return {"matched": traces["h9_b"] == traces["h9_p"], "rows": len(traces["h9_b"])}


def evaluate_gates(
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
            and float(confidence_interval[0]) > 0.0
            and float(np.min(client_reduction)) > 0.0
        ),
        "P2_utility_last5": bool(
            utility_delta["avg_acc"] >= 1.5
            and utility_delta["worst_acc"] >= 1.0
            and utility_delta["wcca"] >= 0.0
            and utility_delta["cfg"] <= -1.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Stage-2 Formal analysis requires --confirm-formal")
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
        experiment_root = outputs_root / f"cle_v2_plugin_stage2_{arm}_trainseed0"
        predictions[arm] = infer_arm(
            experiment_root / "checkpoints", grid, device, int(args.batch_size)
        )
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        accuracies[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = trailing_metrics(experiment_root / "metrics.csv", window=5)
    reduction = float(results["h9_b"].pooled - results["h9_p"].pooled)
    client_reduction = np.asarray(results["h9_b"].client) - np.asarray(results["h9_p"].client)
    bootstrap_values = paired_bootstrap_estimand(
        results,
        {"h9_b": 1.0, "h9_p": -1.0},
        samples=int(args.bootstrap_samples),
        seed=20260910,
    )
    confidence_interval = np.quantile(bootstrap_values, [0.025, 0.975]).tolist()
    utility_delta = {
        name: metrics["h9_p"].get(name, float("nan"))
        - metrics["h9_b"].get(name, float("nan"))
        for name in UTILITY_METRICS
    }
    trace_audit = audit_pair(outputs_root)
    gates = evaluate_gates(
        reduction,
        client_reduction,
        confidence_interval,
        accuracies,
        utility_delta,
        trace_audit,
    )
    if args.mode != "formal":
        verdict = "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
    elif all(gates.values()):
        verdict = "GO_PEW_BER_STAGE2_SEED0"
    else:
        verdict = "NO_GO_PEW_BER_STAGE2_SEED0"
    np.savez_compressed(
        output_dir / "STAGE2_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    summary = {
        "protocol": "cle_v2_plugin_stage2_analysis_v1",
        "mode": args.mode,
        "train_seed": 0,
        "rounds": 1 if args.mode == "smoke" else 12,
        "grid": {"sources": int(grid.shape[0]), "operators": int(grid.shape[1]), "severity": 3},
        "pooled_dsa": {arm: float(results[arm].pooled) for arm in ARMS},
        "plugin_dsa_reduction": reduction,
        "client_plugin_dsa_reduction": client_reduction.tolist(),
        "source_paired_bootstrap": {
            "mean": float(bootstrap_values.mean()),
            "ci95": confidence_interval,
        },
        "operator_grid_accuracy": accuracies,
        "reporting_metrics_last5": metrics,
        "plugin_minus_base_last5": utility_delta,
        "paired_local_traces": trace_audit,
        "frozen_gates": gates,
        "scientific_verdict": verdict,
        "cdep_used": False,
        "granularity_ablation_in_scope": False,
        "multiseed_or_long_training_authorized": False,
    }
    path = output_dir / "RESULT_SUMMARY.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
