from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_cle_hfl_context_shards import trace_digest, trace_map  # noqa: E402
from scripts.run_cle_hfl_context import ARMS as CONTEXT_ARMS  # noqa: E402


ASYM_ROWS = ("asymhfl_erm", "asymhfl_pew_ber")
FINAL_ROWS = (*CONTEXT_ARMS, *ASYM_ROWS)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def merge_domain_table(context_root: Path, asym_root: Path, output_dir: Path) -> dict:
    context = read_json(context_root / "RESULT_SUMMARY_MERGED.json")
    asym = read_json(asym_root / "analysis" / "RESULT_SUMMARY.json")
    expected = {
        "mode": "formal",
        "scenario_id": "cle_hfl_v2_cross_map2_seed0_split0",
        "train_seed": 0,
        "rounds": 40,
    }
    for label, payload in (("context", context), ("asym", asym)):
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"{label} {key} mismatch: {payload.get(key)!r} != {value!r}")
        if payload.get("scientific_evidence") is not True:
            raise ValueError(f"{label} is not Formal scientific evidence")
    if context.get("binding_map_seed") != 2 or context.get("partition_seed") != 0:
        raise ValueError("HFL context table is not the frozen held-out map2 scenario")

    asym_contract = read_json(
        asym_root / "configs" / "SPURIOUS_FINAL_MAP2_CONTRACT_TRAINSEED0.json"
    )
    for key, value in (
        ("scenario_id", expected["scenario_id"]),
        ("binding_map_seed", 2),
        ("partition_seed", 0),
        ("evaluation_seed", 20260909),
        ("rounds", 40),
        ("train_seed", 0),
    ):
        if asym_contract.get(key) != value:
            raise ValueError(f"AsymHFL contract {key} mismatch")

    asym_trace = trace_map(
        asym_root
        / "outputs"
        / "cle_v2_spurious_final_map2_erm_trainseed0"
        / "local_batch_trace.jsonl"
    )
    asym_trace_sha = trace_digest(asym_trace)
    context_trace_sha = context["cross_shard_local_batch_pairing"]["reference_trace_sha256"]
    if asym_trace_sha != context_trace_sha:
        raise ValueError("AsymHFL and HFL-context local batch traces do not match")

    rows = dict(context["rows"])
    for source_arm, destination_arm in (("erm", "asymhfl_erm"), ("pew_ber", "asymhfl_pew_ber")):
        rows[destination_arm] = {
            "pooled_dsa": asym["pooled_dsa"][source_arm],
            "client_dsa": asym["client_dsa"][source_arm],
            "operator_grid_accuracy": asym["operator_grid_accuracy"][source_arm],
            "last10": asym["reporting_metrics"][source_arm]["last10"],
        }

    with np.load(context_root / "HFL_CONTEXT_PREDICTIONS_MERGED.npz", allow_pickle=False) as payload:
        context_arms = tuple(str(value) for value in payload["arms"].tolist())
        if context_arms != CONTEXT_ARMS:
            raise ValueError("Unexpected context prediction row order")
        context_probabilities = payload["probabilities"]
        common = {key: payload[key] for key in ("labels", "binding", "operator_names")}
    with np.load(asym_root / "analysis" / "SPURIOUS_FINAL_MAP2_PREDICTIONS.npz", allow_pickle=False) as payload:
        asym_arms = tuple(str(value) for value in payload["arms"].tolist())
        if asym_arms != ("erm", "cvar_dro", "pew_groupdro", "pew_ber"):
            raise ValueError("Unexpected AsymHFL prediction row order")
        if any(not np.array_equal(common[key], payload[key]) for key in common):
            raise ValueError("AsymHFL and HFL-context evaluation grids differ")
        asym_probabilities = payload["probabilities"][[0, 3]]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "CLE_HFL_DOMAIN_TABLE_PREDICTIONS.npz",
        probabilities=np.concatenate([context_probabilities, asym_probabilities], axis=0),
        arms=np.asarray(FINAL_ROWS),
        **common,
    )
    merged = {
        "protocol": "cle_hfl_domain_table_map2_v1",
        **expected,
        "partition_seed": 0,
        "binding_map_seed": 2,
        "evaluation_seed": 20260909,
        "rows": {arm: rows[arm] for arm in FINAL_ROWS},
        "local_batch_pairing": {
            "matched": True,
            "reference_trace_sha256": context_trace_sha,
            "trace_rows": len(asym_trace),
        },
        "scientific_evidence": True,
        "claim_boundary": "Field-position context on held-out map2; not a universal plug-in comparison.",
    }
    (output_dir / "RESULT_SUMMARY_DOMAIN_TABLE.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ten HFL baselines with two matched AsymHFL rows.")
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--asym-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = merge_domain_table(
        args.context_root.resolve(), args.asym_root.resolve(), args.output_dir.resolve()
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
