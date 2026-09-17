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
from scripts.analyze_cle_v2_mechanism_stage1 import grid_accuracy  # noqa: E402
from scripts.analyze_cle_v2_plugin_stage2 import load_trace  # noqa: E402
from scripts.prepare_cle_v2_factorial_pew import oracle_family_ids  # noqa: E402
from scripts.run_cle_taxonomy_stress import ARMS, HELDOUT_OPERATOR, PEW_ASSET  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze bounded PEW taxonomy stress.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "formal"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def witness_heldout_diagnostic(package_root: Path) -> dict:
    data_root = package_root / "data/gamma09"
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    heldout_id = int(metadata["operator_to_id"][HELDOUT_OPERATOR])
    correct = unknown = total = 0
    for client_id in range(4):
        client_root = data_root / f"client_{client_id}"
        operator_ids = np.load(client_root / "train_corruption_ids.npy", allow_pickle=False)
        annotation = np.load(
            package_root / PEW_ASSET / "annotations/gamma09" / f"client_{client_id}.npz",
            allow_pickle=False,
        )["environment_ids"]
        oracle = oracle_family_ids(operator_ids, metadata)
        mask = operator_ids == heldout_id
        total += int(mask.sum())
        correct += int((annotation[mask] == oracle[mask]).sum())
        unknown += int((annotation[mask] == 5).sum())
    return {
        "operator": HELDOUT_OPERATOR,
        "private_samples": total,
        "family_accuracy": 100.0 * correct / max(total, 1),
        "unknown_rate": unknown / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Formal taxonomy-stress analysis requires --confirm-formal")
    package_root, outputs_root = args.package_root.resolve(), args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operators, binding = load_grid(package_root)
    operator_id = operators.index(HELDOUT_OPERATOR)
    device = resolve_device(args.device)
    predictions, results, accuracy, metrics, traces = {}, {}, {}, {}, {}
    for arm in ARMS:
        root = outputs_root / f"cle_taxonomy_stress_motion_blur_{arm}_trainseed0"
        predictions[arm] = infer_arm(root / "checkpoints", grid, device, 256)
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        accuracy[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = trailing_metrics(root / "metrics.csv", 10)
        traces[arm] = load_trace(root / "local_batch_trace.jsonl")
    dsa = {arm: float(results[arm].pooled) for arm in ARMS}
    heldout_dsa = {arm: float(np.nanmean(results[arm].client_operator[:, operator_id])) for arm in ARMS}
    values = paired_bootstrap_estimand(
        {"erm": results["erm"], "pew_ber_loo": results["pew_ber_loo"]},
        {"erm": 1.0, "pew_ber_loo": -1.0},
        samples=int(args.bootstrap_samples),
        seed=20260917,
    )
    trace_match = all(traces[arm] == traces["erm"] for arm in ARMS[1:])
    gates = {
        "I0_local_traces_matched": trace_match,
        "P1_overall_dsa_reduction_positive": dsa["erm"] > dsa["pew_ber_loo"],
        "P2_overall_dsa_ci_excludes_zero": float(np.quantile(values, 0.025)) > 0.0,
        "P3_heldout_operator_dsa_reduction_positive": heldout_dsa["erm"] > heldout_dsa["pew_ber_loo"],
    }
    summary = {
        "protocol": "cle_taxonomy_stress_motion_blur_analysis_v1",
        "mode": args.mode,
        "heldout_operator": HELDOUT_OPERATOR,
        "pew_heldout_operator_diagnostic": witness_heldout_diagnostic(package_root),
        "pooled_dsa": dsa,
        "heldout_operator_dsa": heldout_dsa,
        "operator_grid_accuracy": accuracy,
        "reporting_metrics_last10": metrics,
        "erm_minus_pew_ber_loo_dsa": {
            "mean": float(values.mean()),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        },
        "frozen_gates": gates,
        "scientific_verdict": (
            "BOUNDED_STRESS_PASS" if args.mode == "formal" and all(gates.values())
            else "BOUNDED_STRESS_FAIL" if args.mode == "formal"
            else "BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION"
        ),
        "scope_warning": "One pre-registered held-out operator; not an open-world or compound-corruption claim.",
    }
    np.savez_compressed(
        output_dir / "TAXONOMY_STRESS_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operators),
    )
    (output_dir / "RESULT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
