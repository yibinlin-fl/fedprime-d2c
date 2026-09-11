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
from scripts.analyze_cle_v2_mechanism_stage1 import (  # noqa: E402
    balanced_source_indices,
    grid_accuracy,
)
from scripts.run_cle_v2_cross_scenario import ARMS, MAP_SEEDS  # noqa: E402


UTILITY_METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a cross-binding-map PEW+BER A/B.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map-seed", type=int, choices=MAP_SEEDS, required=True)
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--null-permutations", type=int, default=1000)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def load_trace(path: Path) -> list[tuple[int, int, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append((int(item["round"]), int(item["client"]), str(item["sha256"])))
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    return rows


def audit_pair(outputs_root: Path, map_seed: int) -> dict[str, object]:
    traces = {
        arm: load_trace(
            outputs_root
            / f"cle_v2_cross_map{map_seed}_{arm}_trainseed0"
            / "local_batch_trace.jsonl"
        )
        for arm in ARMS
    }
    return {"matched": traces["h9_b"] == traces["h9_p"], "rows": len(traces["h9_b"])}


def evaluate_cross_gates(
    base_dsa: float,
    null: dict[str, object],
    reduction: float,
    client_reduction: np.ndarray,
    confidence_interval: list[float],
    accuracies: dict[str, dict[str, object]],
    trace_audit: dict[str, object],
) -> dict[str, bool]:
    return {
        "I0_paired_integrity": bool(trace_audit["matched"]),
        "L0_learning_floor": all(float(accuracies[arm]["pooled"]) >= 20.0 for arm in ARMS),
        "C1_directional_shortcut": bool(
            base_dsa >= 0.05
            and base_dsa > float(null["null_p95"])
            and float(null["p_value"]) <= 0.01
        ),
        "C2_cross_map_mitigation": bool(
            reduction >= 0.02
            and float(confidence_interval[0]) > 0.0
            and float(np.min(client_reduction)) > 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Cross-scenario Formal analysis requires --confirm-formal")
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
        experiment_root = outputs_root / f"cle_v2_cross_map{args.map_seed}_{arm}_trainseed0"
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
        seed=20260911 + int(args.map_seed),
    )
    confidence_interval = np.quantile(bootstrap_values, [0.025, 0.975]).tolist()
    null = shuffled_binding_null(
        predictions["h9_b"],
        labels,
        binding,
        permutations=int(args.null_permutations),
        seed=20260911 + int(args.map_seed),
    )
    null_summary = {
        "observed": float(null["observed"]),
        "null_p95": float(null["null_p95"]),
        "p_value": float(null["p_value"]),
    }
    utility_delta = {
        name: metrics["h9_p"].get(name, float("nan"))
        - metrics["h9_b"].get(name, float("nan"))
        for name in UTILITY_METRICS
    }
    trace_audit = audit_pair(outputs_root, int(args.map_seed))
    gates = evaluate_cross_gates(
        float(results["h9_b"].pooled),
        null_summary,
        reduction,
        client_reduction,
        confidence_interval,
        accuracies,
        trace_audit,
    )
    if args.mode != "formal":
        verdict = f"{args.mode.upper()}_ONLY_NO_SCIENTIFIC_DECISION"
    elif all(gates.values()):
        verdict = f"GO_PEW_BER_CROSS_MAP{args.map_seed}"
    else:
        verdict = f"NO_GO_PEW_BER_CROSS_MAP{args.map_seed}"
    np.savez_compressed(
        output_dir / f"CROSS_MAP{args.map_seed}_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    summary = {
        "protocol": "cle_v2_cross_binding_map_analysis_v1",
        "mode": args.mode,
        "scenario_id": f"cle_hfl_v2_cross_map{args.map_seed}_seed0_split0",
        "partition_seed": 0,
        "binding_map_seed": int(args.map_seed),
        "train_seed": 0,
        "grid": {
            "sources": int(grid.shape[0]),
            "operators": int(grid.shape[1]),
            "severity": 3,
        },
        "pooled_dsa": {arm: float(results[arm].pooled) for arm in ARMS},
        "plugin_dsa_reduction": reduction,
        "client_plugin_dsa_reduction": client_reduction.tolist(),
        "source_paired_bootstrap": {
            "mean": float(bootstrap_values.mean()),
            "ci95": confidence_interval,
        },
        "base_shuffled_binding_null": null_summary,
        "operator_grid_accuracy": accuracies,
        "reporting_metrics_last5": metrics,
        "plugin_minus_base_last5": utility_delta,
        "paired_local_traces": trace_audit,
        "frozen_gates": gates,
        "scientific_verdict": verdict,
        "utility_is_reporting_not_primary_gate": True,
        "cdep_used": False,
        "scientific_evidence": args.mode == "formal",
    }
    path = output_dir / "RESULT_SUMMARY.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
