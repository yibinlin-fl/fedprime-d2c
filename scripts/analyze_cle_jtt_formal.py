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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched ERM vs JTT Formal.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "formal"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confirm-formal", action="store_true")
    args = parser.parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Formal JTT analysis requires --confirm-formal")
    grid, labels, operators, binding = load_grid(args.package_root.resolve())
    device = resolve_device(args.device)
    results, predictions, accuracy, metrics, traces = {}, {}, {}, {}, {}
    for arm in ("erm", "jtt"):
        root = args.outputs_root.resolve() / f"cle_jtt_formal_{arm}_trainseed0"
        predictions[arm] = infer_arm(root / "checkpoints", grid, device, 256)
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        accuracy[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = trailing_metrics(root / "metrics.csv", 10)
        traces[arm] = load_trace(root / "local_batch_trace.jsonl")
    bootstrap = paired_bootstrap_estimand(
        results, {"erm": 1.0, "jtt": -1.0}, samples=int(args.bootstrap_samples), seed=20260917
    )
    summary = {
        "protocol": "cle_jtt_formal_analysis_v1",
        "mode": args.mode,
        "pooled_dsa": {arm: float(results[arm].pooled) for arm in results},
        "client_dsa": {arm: list(results[arm].client) for arm in results},
        "operator_grid_accuracy": accuracy,
        "reporting_metrics_last10": metrics,
        "erm_minus_jtt_dsa": {"mean": float(bootstrap.mean()), "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist()},
        "local_batch_traces_matched": traces["erm"] == traces["jtt"],
        "jtt_budget_warning": "JTT uses a full ERM stage plus a second retraining stage; compute is not equal to one-stage ERM.",
        "scientific_evidence": args.mode == "formal",
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "JTT_FORMAL_PREDICTIONS.npz", probabilities=np.stack([predictions["erm"], predictions["jtt"]]), arms=np.asarray(["erm", "jtt"]), labels=labels, binding=binding, operator_names=np.asarray(operators))
    (output_dir / "RESULT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
