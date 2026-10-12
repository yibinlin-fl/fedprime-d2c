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
from scripts.run_cle_hfl_context import (  # noqa: E402
    SHARDS,
    fidelity_manifest,
    local_loader_mode_for_arm,
    selected_arms,
)


def audit_local_batch_pairing(outputs_root: Path, arms: tuple[str, ...]) -> dict:
    full_traces = {}
    label_traces = {}
    for arm in arms:
        path = outputs_root / f"cle_hfl_context_{arm}_trainseed0" / "local_batch_trace.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        full_traces[arm] = {
            (int(row["round"]), int(row["client"]), int(row["batch"])): str(row["sha256"])
            for row in rows
        }
        label_traces[arm] = {
            (int(row["round"]), int(row["client"]), int(row["batch"])): str(row["label_sha256"])
            for row in rows
        }
    reference_arm = "local_erm" if "local_erm" in full_traces else arms[0]
    label_reference = label_traces[reference_arm]
    label_matches = {arm: trace == label_reference for arm, trace in label_traces.items()}
    loader_modes = {arm: local_loader_mode_for_arm(arm) for arm in arms}
    exact_matches = {}
    for mode in sorted(set(loader_modes.values())):
        mode_arms = [arm for arm in arms if loader_modes[arm] == mode]
        mode_reference = mode_arms[0]
        exact_matches[mode] = {
            arm: full_traces[arm] == full_traces[mode_reference]
            for arm in mode_arms
        }
    exact_all = all(all(matches.values()) for matches in exact_matches.values())
    all_match = all(label_matches.values()) and exact_all
    return {
        "reference_arm": reference_arm,
        "trace_rows": {arm: len(trace) for arm, trace in full_traces.items()},
        "loader_modes": loader_modes,
        "label_trace_matches": label_matches,
        "exact_trace_matches_within_loader": exact_matches,
        "all_arms_match": all_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the HFL context table.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pairing", "benchmark", "formal"), required=True)
    parser.add_argument("--shard", choices=SHARDS, default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confirm-formal", action="store_true")
    args = parser.parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Formal HFL context analysis requires --confirm-formal")
    grid, labels, operators, binding = load_grid(args.package_root.resolve())
    device = resolve_device(args.device)
    arms = selected_arms(args.shard, args.mode)
    rows = {}
    probabilities = []
    for arm in arms:
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
    null_reference_index = arms.index("rahfl_fidelity") if "rahfl_fidelity" in arms else len(arms) - 1
    null_reference_arm = arms[null_reference_index]
    reference_null = shuffled_binding_null(
        probabilities[null_reference_index], labels, binding, permutations=1000, seed=20260917
    )
    pairing_audit = audit_local_batch_pairing(args.outputs_root.resolve(), arms)
    if args.mode == "pairing" and not pairing_audit["all_arms_match"]:
        raise RuntimeError(f"Local-batch pairing audit failed: {pairing_audit}")
    null_payload = {
        "reference_arm": null_reference_arm,
        "observed": float(reference_null["observed"]),
        "null_p95": float(reference_null["null_p95"]),
        "p_value": float(reference_null["p_value"]),
    }
    summary = {
        "protocol": "cle_hfl_context_table_budget64_analysis_v5",
        "mode": args.mode,
        "execution_shard": args.shard,
        "selected_arms": list(arms),
        "rows": rows,
        "fidelity_manifest": fidelity_manifest(),
        "reference_shuffled_binding_null": null_payload,
        "local_batch_pairing": pairing_audit,
        "scientific_evidence": args.mode == "formal",
        "claim_boundary": "Context table only; it does not identify BER's causal contribution.",
    }
    if null_reference_arm == "rahfl_fidelity":
        summary["rahfl_shuffled_binding_null"] = {
            key: value for key, value in null_payload.items() if key != "reference_arm"
        }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "HFL_CONTEXT_PREDICTIONS.npz",
        probabilities=np.stack(probabilities),
        arms=np.asarray(arms),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operators),
    )
    (output_dir / "RESULT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
