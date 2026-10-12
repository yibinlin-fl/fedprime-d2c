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
    shuffled_binding_null,
)
from scripts.analyze_cle_v2_spurious_final import bootstrap_contrast  # noqa: E402
from scripts.merge_cle_hfl_context_shards import (  # noqa: E402
    input_audit_fingerprint,
    read_json,
    trace_digest,
    trace_map,
)
from scripts.prepare_cle_v2_factorial_data import sha256_file  # noqa: E402
from scripts.run_cle_v2_spurious_final import ARMS  # noqa: E402


def parse_arm_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected ARM=PATH, got: {value}")
        arm, raw_path = value.split("=", 1)
        if arm not in ARMS:
            raise ValueError(f"Unknown Axis-I arm: {arm}")
        if arm in roots:
            raise ValueError(f"Duplicate Axis-I arm: {arm}")
        roots[arm] = Path(raw_path).resolve()
    missing = set(ARMS) - set(roots)
    if missing:
        raise ValueError(f"Missing Axis-I arms: {sorted(missing)}")
    return {arm: roots[arm] for arm in ARMS}


def merge_axis1_shards(roots: dict[str, Path], output_dir: Path, train_seed: int = 0) -> dict:
    contracts: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    traces: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    common: dict[str, np.ndarray] | None = None

    for arm in ARMS:
        root = roots[arm]
        contract = read_json(
            root / "configs" / f"SPURIOUS_FINAL_MAP2_CONTRACT_TRAINSEED{train_seed}_{arm}.json"
        )
        completion = read_json(
            root / "configs" / f"SPURIOUS_FINAL_MAP2_COMPLETION_TRAINSEED{train_seed}_{arm}.json"
        )
        summary = read_json(root / "analysis" / "RESULT_SUMMARY.json")
        audit = read_json(root / "outputs" / "INPUT_AUDIT.json")
        if contract.get("execution_arm") != arm or contract.get("selected_arms") != [arm]:
            raise ValueError(f"Contract arm mismatch: {arm}")
        if completion.get("completed_arms") != [arm] or completion.get("complete") is not True:
            raise ValueError(f"Incomplete Axis-I arm: {arm}")
        if summary.get("execution_arm") != arm or summary.get("arms") != [arm]:
            raise ValueError(f"Analysis arm mismatch: {arm}")
        if summary.get("mode") != "formal" or summary.get("scientific_evidence") is not True:
            raise ValueError(f"Axis-I arm is not Formal evidence: {arm}")
        config_path = root / "configs" / f"map2_{arm}_trainseed{train_seed}.json"
        if sha256_file(config_path) != contract["arms"][arm]["sha256"]:
            raise ValueError(f"Config hash mismatch: {arm}")
        traces[arm] = trace_map(
            root
            / "outputs"
            / f"cle_v2_spurious_final_map2_{arm}_trainseed{train_seed}"
            / "local_batch_trace.jsonl"
        )
        with np.load(
            root / "analysis" / f"SPURIOUS_FINAL_MAP2_PREDICTIONS_{arm}.npz",
            allow_pickle=False,
        ) as payload:
            if tuple(str(value) for value in payload["arms"].tolist()) != (arm,):
                raise ValueError(f"Prediction arm mismatch: {arm}")
            predictions[arm] = payload["probabilities"][0]
            shared = {key: payload[key] for key in ("labels", "binding", "operator_names")}
            if common is None:
                common = shared
            elif any(not np.array_equal(common[key], value) for key, value in shared.items()):
                raise ValueError(f"Evaluation grid mismatch: {arm}")
        contracts[arm], summaries[arm], audits[arm] = contract, summary, audit

    protocol_keys = (
        "protocol",
        "mode",
        "scenario_id",
        "partition_seed",
        "binding_map_seed",
        "evaluation_seed",
        "train_seed",
        "rounds",
        "local_training",
        "max_local_batches",
        "batch_size",
        "public_batch_size",
        "public_batches_per_round",
        "pretrain_epochs",
        "common_communication",
    )
    reference = contracts[ARMS[0]]
    frozen = {
        "mode": "formal",
        "scenario_id": "cle_hfl_v2_cross_map2_seed0_split0",
        "partition_seed": 0,
        "binding_map_seed": 2,
        "evaluation_seed": 20260909,
        "train_seed": int(train_seed),
        "rounds": 40,
        "local_training": "64_batches_per_client_round",
        "max_local_batches": 64,
        "batch_size": 64,
        "public_batch_size": 128,
        "public_batches_per_round": 4,
        "pretrain_epochs": 0,
        "common_communication": "strict AsymHFL-val",
    }
    for key, value in frozen.items():
        if reference.get(key) != value:
            raise ValueError(f"Frozen Axis-I contract mismatch at {key}")
    for arm, contract in contracts.items():
        if any(contract.get(key) != reference.get(key) for key in protocol_keys):
            raise ValueError(f"Protocol mismatch: {arm}")
    reference_audit = input_audit_fingerprint(audits[ARMS[0]])
    if reference_audit.get("status") != "PASS":
        raise ValueError("Reference input audit did not pass")
    if any(input_audit_fingerprint(audit) != reference_audit for audit in audits.values()):
        raise ValueError("Input audit mismatch across Axis-I arms")
    trace_matches = {arm: traces[arm] == traces["erm"] for arm in ARMS}
    if not all(trace_matches.values()):
        raise ValueError(f"Axis-I local batch pairing failed: {trace_matches}")
    assert common is not None

    results = {
        arm: compute_operator_dsa(predictions[arm], common["labels"], common["binding"])
        for arm in ARMS
    }
    dsa = {arm: float(results[arm].pooled) for arm in ARMS}
    client_dsa = {arm: list(results[arm].client) for arm in ARMS}
    accuracy = {arm: summaries[arm]["operator_grid_accuracy"][arm] for arm in ARMS}
    metrics = {arm: summaries[arm]["reporting_metrics"][arm] for arm in ARMS}
    contrasts = {
        "erm_minus_pew_ber": bootstrap_contrast(results, "erm", "pew_ber", 5000),
        "pew_groupdro_minus_pew_ber": bootstrap_contrast(results, "pew_groupdro", "pew_ber", 5000),
        "pew_ber_minus_cvar_dro": bootstrap_contrast(results, "pew_ber", "cvar_dro", 5000),
    }
    null = shuffled_binding_null(
        predictions["erm"], common["labels"], common["binding"], permutations=1000, seed=20260913
    )
    client_reduction = np.asarray(results["erm"].client) - np.asarray(results["pew_ber"].client)
    ber_minus_cvar_grid = float(accuracy["pew_ber"]["pooled"] - accuracy["cvar_dro"]["pooled"])
    ber_minus_erm_grid = float(accuracy["pew_ber"]["pooled"] - accuracy["erm"]["pooled"])
    ber_minus_erm_avg10 = float(
        metrics["pew_ber"]["last10"]["avg_acc"] - metrics["erm"]["last10"]["avg_acc"]
    )
    gates = {
        "I0_all_local_batch_traces_matched": True,
        "S0_heldout_erm_directional_shortcut": bool(
            dsa["erm"] >= 0.08 and dsa["erm"] > float(null["null_p95"]) and float(null["p_value"]) <= 0.01
        ),
        "P1_ber_reduces_erm_dsa": bool(
            dsa["erm"] - dsa["pew_ber"] >= 0.05
            and contrasts["erm_minus_pew_ber"]["ci95"][0] > 0.0
            and float(client_reduction.min()) > 0.0
        ),
        "P2_ber_beats_matched_groupdro_dsa": bool(
            dsa["pew_groupdro"] - dsa["pew_ber"] >= 0.02
            and contrasts["pew_groupdro_minus_pew_ber"]["ci95"][0] > 0.0
        ),
        "P3_ber_dsa_within_0p02_of_cvar": bool(dsa["pew_ber"] - dsa["cvar_dro"] <= 0.02),
        "P4_ber_operator_accuracy_beats_cvar_by_1pp": bool(ber_minus_cvar_grid >= 1.0),
        "P5_ber_utility_not_below_erm": bool(ber_minus_erm_grid >= 0.0 and ber_minus_erm_avg10 >= 0.0),
    }
    verdict = "GO_FOUR_ARM_HELDOUT_MAP2" if all(gates.values()) else "NO_GO_FOUR_ARM_HELDOUT_MAP2"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)
    (output_dir / "analysis").mkdir(exist_ok=True)
    np.savez_compressed(
        output_dir / "analysis" / "SPURIOUS_FINAL_MAP2_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        **common,
    )
    summary = {
        "protocol": "cle_v2_spurious_final_map2_budget64_merged_analysis_v2",
        "mode": "formal",
        "scenario_id": frozen["scenario_id"],
        "train_seed": int(train_seed),
        "rounds": 40,
        "common_communication": frozen["common_communication"],
        "arms": list(ARMS),
        "pooled_dsa": dsa,
        "client_dsa": client_dsa,
        "operator_grid_accuracy": accuracy,
        "reporting_metrics": metrics,
        "source_paired_bootstrap_contrasts": contrasts,
        "erm_shuffled_binding_null": {
            "observed": float(null["observed"]),
            "null_p95": float(null["null_p95"]),
            "p_value": float(null["p_value"]),
        },
        "key_deltas": {
            "erm_minus_pew_ber_dsa": dsa["erm"] - dsa["pew_ber"],
            "pew_groupdro_minus_pew_ber_dsa": dsa["pew_groupdro"] - dsa["pew_ber"],
            "pew_ber_minus_cvar_dsa": dsa["pew_ber"] - dsa["cvar_dro"],
            "pew_ber_minus_cvar_operator_accuracy_pp": ber_minus_cvar_grid,
            "pew_ber_minus_erm_operator_accuracy_pp": ber_minus_erm_grid,
            "pew_ber_minus_erm_last10_avg_pp": ber_minus_erm_avg10,
        },
        "paired_local_traces": {
            "matched": True,
            "arm_matches": trace_matches,
            "rows": len(traces["erm"]),
            "reference_trace_sha256": trace_digest(traces["erm"]),
        },
        "frozen_gates": gates,
        "scientific_verdict": verdict,
        "scientific_evidence": True,
        "cdep_used": False,
        "source_arm_roots": {arm: str(roots[arm]) for arm in ARMS},
    }
    (output_dir / "analysis" / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    combined_contract = dict(reference)
    combined_contract.update(
        {
            "execution_arm": "merged",
            "selected_arms": list(ARMS),
            "arms": {arm: contracts[arm]["arms"][arm] for arm in ARMS},
            "source_arm_roots": {arm: str(roots[arm]) for arm in ARMS},
        }
    )
    (output_dir / "configs" / f"SPURIOUS_FINAL_MAP2_CONTRACT_TRAINSEED{train_seed}.json").write_text(
        json.dumps(combined_contract, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge four independent Axis-I Formal arms.")
    parser.add_argument("--arm-root", action="append", required=True)
    parser.add_argument("--train-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = merge_axis1_shards(
        parse_arm_roots(args.arm_root), args.output_dir.resolve(), train_seed=int(args.train_seed)
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
