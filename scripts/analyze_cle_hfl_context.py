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
from scripts.run_cle_hfl_context import ARMS, fidelity_manifest  # noqa: E402


def audit_local_batch_pairing(outputs_root: Path) -> dict:
    traces = {}
    for arm in ARMS:
        path = outputs_root / f"cle_hfl_context_{arm}_trainseed0" / "local_batch_trace.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        traces[arm] = {
            (int(row["round"]), int(row["client"]), int(row["batch"])): str(row["sha256"])
            for row in rows
        }
    reference = traces["local_erm"]
    arm_matches = {arm: trace == reference for arm, trace in traces.items()}
    return {
        "reference_arm": "local_erm",
        "trace_rows": {arm: len(trace) for arm, trace in traces.items()},
        "arm_matches": arm_matches,
        "all_arms_match": all(arm_matches.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the HFL context table.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pairing", "benchmark", "formal"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confirm-formal", action="store_true")
    args = parser.parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Formal HFL context analysis requires --confirm-formal")
    grid, labels, operators, binding = load_grid(args.package_root.resolve())
    device = resolve_device(args.device)
    rows = {}
    probabilities = []
    for arm in ARMS:
        root = args.outputs_root.resolve() / f"cle_hfl_context_{arm}_trainseed0"
        prediction = infer_arm(root / "checkpoints", grid, device, 256)
        probabilities.append(prediction)
        dsa = compute_operator_dsa(prediction, labels, binding)
        rows[arm] = {
            "pooled_dsa": float(dsa.pooled),
            "client_dsa": list(dsa.client),
            "operator_grid_accuracy": grid_accuracy(prediction, labels),
            "last10": trailing_metrics(root / "metrics.csv", 10),
        }
    rahfl_null = shuffled_binding_null(probabilities[-1], labels, binding, permutations=1000, seed=20260917)
    pairing_audit = audit_local_batch_pairing(args.outputs_root.resolve())
    if args.mode == "pairing" and not pairing_audit["all_arms_match"]:
        raise RuntimeError(f"Local-batch pairing audit failed: {pairing_audit}")
    summary = {
        "protocol": "cle_hfl_context_table_analysis_v2",
        "mode": args.mode,
        "rows": rows,
        "fidelity_manifest": fidelity_manifest(),
        "rahfl_shuffled_binding_null": {
            "observed": float(rahfl_null["observed"]),
            "null_p95": float(rahfl_null["null_p95"]),
            "p_value": float(rahfl_null["p_value"]),
        },
        "local_batch_pairing": pairing_audit,
        "scientific_evidence": args.mode == "formal",
        "claim_boundary": "Context table only; it does not identify BER's causal contribution.",
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "HFL_CONTEXT_PREDICTIONS.npz",
        probabilities=np.stack(probabilities),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operators),
    )
    (output_dir / "RESULT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
