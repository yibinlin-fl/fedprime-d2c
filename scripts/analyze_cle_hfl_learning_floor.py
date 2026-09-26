from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_v2_factorial import compute_operator_dsa, shuffled_binding_null  # noqa: E402
from scripts.analyze_cle_v2_factorial import infer_arm, load_grid, resolve_device, trailing_metrics  # noqa: E402
from scripts.analyze_cle_v2_mechanism_stage1 import grid_accuracy  # noqa: E402
from scripts.run_cle_hfl_learning_floor import BUDGETS, ROUND_BUDGET  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a CLE-HFL Local learning-floor run.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, required=True)
    parser.add_argument("--local-batches", type=int, choices=BUDGETS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confirm-formal", action="store_true")
    args = parser.parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Learning-floor Formal analysis requires --confirm-formal")
    grid, labels, operators, binding = load_grid(args.package_root.resolve())
    device = resolve_device(args.device)
    root = (
        args.outputs_root.resolve()
        / f"cle_hfl_learning_floor_b{args.local_batches}_trainseed0"
    )
    probabilities = infer_arm(root / "checkpoints", grid, device, 256)
    dsa = compute_operator_dsa(probabilities, labels, binding)
    null = shuffled_binding_null(
        probabilities, labels, binding, permutations=1000, seed=20260927
    )
    summary = {
        "protocol": "cle_hfl_learning_floor_analysis_v1",
        "mode": args.mode,
        "local_batches_per_client_round": args.local_batches,
        "rounds": ROUND_BUDGET[args.mode],
        "pooled_dsa": float(dsa.pooled),
        "client_dsa": list(dsa.client),
        "operator_grid_accuracy": grid_accuracy(probabilities, labels),
        "trailing_metrics": trailing_metrics(
            root / "metrics.csv", min(10, ROUND_BUDGET[args.mode])
        ),
        "shuffled_binding_null": {
            "observed": float(null["observed"]),
            "null_p95": float(null["null_p95"]),
            "p_value": float(null["p_value"]),
        },
        "scientific_evidence": args.mode == "formal",
        "claim_boundary": (
            "This run calibrates the Local/ERM optimization floor only; it does not compare HFL methods."
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "LEARNING_FLOOR_PREDICTIONS.npz",
        probabilities=probabilities,
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operators),
    )
    (output_dir / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
