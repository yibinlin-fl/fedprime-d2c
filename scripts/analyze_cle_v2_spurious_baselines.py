from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_v2_factorial import compute_operator_dsa  # noqa: E402
from scripts.analyze_cle_v2_factorial import (  # noqa: E402
    infer_arm,
    load_grid,
    resolve_device,
    trailing_metrics,
)
from scripts.analyze_cle_v2_mechanism_stage1 import balanced_source_indices, grid_accuracy  # noqa: E402
from scripts.analyze_cle_v2_plugin_stage2 import load_trace  # noqa: E402
from scripts.run_cle_v2_spurious_baselines import ARMS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CLE-v2 spurious-correlation baselines.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "benchmark", "screen"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-sources", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operator_names, binding = load_grid(package_root)
    if args.max_sources is not None:
        selected = balanced_source_indices(labels, int(args.max_sources))
        grid, labels = grid[selected], labels[selected]
    device = resolve_device(args.device)
    predictions, dsa, accuracy, metrics, traces = {}, {}, {}, {}, {}
    for arm in ARMS:
        experiment = outputs_root / f"cle_v2_spurious_{arm}_trainseed0"
        predictions[arm] = infer_arm(experiment / "checkpoints", grid, device, int(args.batch_size))
        result = compute_operator_dsa(predictions[arm], labels, binding)
        dsa[arm] = {"pooled": float(result.pooled), "client": list(result.client)}
        accuracy[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = trailing_metrics(experiment / "metrics.csv", window=5)
        traces[arm] = load_trace(experiment / "local_batch_trace.jsonl")
    erm_dsa = dsa["erm"]["pooled"]
    summary = {
        "protocol": "cle_v2_spurious_baseline_analysis_v1",
        "mode": args.mode,
        "dsa": dsa,
        "dsa_reduction_vs_erm": {arm: erm_dsa - dsa[arm]["pooled"] for arm in ARMS},
        "operator_grid_accuracy": accuracy,
        "reporting_metrics_last5": metrics,
        "all_local_batch_traces_matched": all(traces[arm] == traces["erm"] for arm in ARMS[1:]),
        "jtt_is_two_stage": True,
        "jtt_and_cvar_pew_free": True,
        "pew_groupdro_uses_ber": False,
        "scientific_evidence": False,
        "verdict": (
            "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
            if args.mode == "smoke"
            else "BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION"
            if args.mode == "benchmark"
            else "SCREEN_ONLY_NOT_FINAL_PAPER_EVIDENCE"
        ),
    }
    np.savez_compressed(
        output_dir / "SPURIOUS_BASELINE_PREDICTIONS.npz",
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
