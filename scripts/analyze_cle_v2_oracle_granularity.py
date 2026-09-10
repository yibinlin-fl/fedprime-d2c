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


ARMS = ("h9_of", "h9_oo", "h9_ro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen CLE-v2 oracle granularity test.")
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


def load_trace(path: Path) -> list[tuple[int, int, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append((int(item["round"]), int(item["client"]), str(item["sha256"])))
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    return rows


def paired_trace_audit(outputs_root: Path) -> dict[str, object]:
    traces = {
        arm: load_trace(
            outputs_root
            / f"cle_v2_oracle_granularity_{arm}_trainseed0"
            / "local_batch_trace.jsonl"
        )
        for arm in ARMS
    }
    first = traces[ARMS[0]]
    return {"matched": all(traces[arm] == first for arm in ARMS[1:]), "rows": len(first)}


def zero_recall_counts(probabilities: np.ndarray, labels: np.ndarray) -> list[int]:
    predictions = probabilities.argmax(axis=-1)
    counts = []
    for client in range(predictions.shape[0]):
        zero = 0
        for class_id in range(probabilities.shape[-1]):
            mask = labels == class_id
            if float((predictions[client, mask] == class_id).mean()) == 0.0:
                zero += 1
        counts.append(zero)
    return counts


def evaluate_gates(
    *,
    dsa: dict[str, float],
    confidence_intervals: dict[str, list[float]],
    accuracies: dict[str, dict[str, object]],
    metrics: dict[str, dict[str, float]],
    zero_recall: dict[str, list[int]],
    traces: dict[str, object],
) -> dict[str, bool]:
    return {
        "I0_paired_integrity": bool(traces["matched"]),
        "L0_learning_floor": all(float(accuracies[arm]["pooled"]) >= 20.0 for arm in ARMS),
        "G1_operator_association": bool(
            dsa["h9_ro"] - dsa["h9_oo"] >= 0.02
            and confidence_intervals["random_minus_operator"][0] > 0.0
        ),
        "G2_operator_granularity_gap": bool(
            dsa["h9_of"] - dsa["h9_oo"] >= 0.02
            and confidence_intervals["family_minus_operator"][0] > 0.0
        ),
        "G3_operator_utility_noninferiority": bool(
            float(accuracies["h9_oo"]["pooled"])
            >= float(accuracies["h9_of"]["pooled"]) - 1.0
            and metrics["h9_oo"].get("avg_acc", float("-inf"))
            >= metrics["h9_of"].get("avg_acc", float("inf")) - 1.0
            and sum(zero_recall["h9_oo"]) <= sum(zero_recall["h9_of"])
        ),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Oracle granularity Formal analysis requires --confirm-formal")
    package_root = args.package_root.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operator_names, binding = load_grid(package_root)
    if args.max_sources is not None:
        selected = balanced_source_indices(labels, int(args.max_sources))
        grid, labels = grid[selected], labels[selected]
    device = resolve_device(args.device)
    predictions, results, accuracies, metrics, zero_recall = {}, {}, {}, {}, {}
    for arm in ARMS:
        experiment_root = outputs_root / f"cle_v2_oracle_granularity_{arm}_trainseed0"
        predictions[arm] = infer_arm(
            experiment_root / "checkpoints", grid, device, int(args.batch_size)
        )
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        accuracies[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = trailing_metrics(experiment_root / "metrics.csv", window=5)
        zero_recall[arm] = zero_recall_counts(predictions[arm], labels)
    dsa = {arm: float(results[arm].pooled) for arm in ARMS}
    bootstrap = {
        "family_minus_operator": paired_bootstrap_estimand(
            {"h9_of": results["h9_of"], "h9_oo": results["h9_oo"]},
            {"h9_of": 1.0, "h9_oo": -1.0},
            samples=int(args.bootstrap_samples),
            seed=20260911,
        ),
        "random_minus_operator": paired_bootstrap_estimand(
            {"h9_ro": results["h9_ro"], "h9_oo": results["h9_oo"]},
            {"h9_ro": 1.0, "h9_oo": -1.0},
            samples=int(args.bootstrap_samples),
            seed=20260912,
        ),
    }
    confidence_intervals = {
        name: np.quantile(values, [0.025, 0.975]).tolist()
        for name, values in bootstrap.items()
    }
    traces = paired_trace_audit(outputs_root)
    gates = evaluate_gates(
        dsa=dsa,
        confidence_intervals=confidence_intervals,
        accuracies=accuracies,
        metrics=metrics,
        zero_recall=zero_recall,
        traces=traces,
    )
    if args.mode != "formal":
        verdict = f"{args.mode.upper()}_ONLY_NO_SCIENTIFIC_DECISION"
    elif all(gates.values()):
        verdict = "GO_OPERATOR_GRANULARITY_GAP"
    else:
        verdict = "NO_GO_OPERATOR_GRANULARITY_GAP"
    np.savez_compressed(
        output_dir / "ORACLE_GRANULARITY_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    summary = {
        "protocol": "cle_v2_oracle_granularity_analysis_v1",
        "mode": args.mode,
        "pooled_dsa": dsa,
        "contrasts": {
            "family_minus_operator": dsa["h9_of"] - dsa["h9_oo"],
            "random_minus_operator": dsa["h9_ro"] - dsa["h9_oo"],
        },
        "source_paired_bootstrap_ci95": confidence_intervals,
        "operator_grid_accuracy": accuracies,
        "reporting_metrics_last5": metrics,
        "zero_recall_classes_per_client": zero_recall,
        "paired_local_traces": traces,
        "frozen_gates": gates,
        "scientific_verdict": verdict,
        "oracle_metadata_training_used": True,
        "deployable_method_claim": False,
        "hierarchical_pew_training_authorized": bool(
            args.mode == "formal" and all(gates.values())
        ),
    }
    (output_dir / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
