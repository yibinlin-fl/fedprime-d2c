from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_cle_v2_factorial_data import sha256_file  # noqa: E402
from scripts.run_cle_hfl_context import ARMS, SHARDS  # noqa: E402


REQUIRED_SHARDS = ("cheap_a", "cheap_b", "fedtgp", "rhfl")


def parse_shard_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected SHARD=PATH, got: {value}")
        shard, raw_path = value.split("=", 1)
        if shard not in REQUIRED_SHARDS:
            raise ValueError(f"Unknown or non-mergeable shard: {shard}")
        if shard in roots:
            raise ValueError(f"Duplicate shard: {shard}")
        roots[shard] = Path(raw_path).resolve()
    missing = set(REQUIRED_SHARDS) - set(roots)
    if missing:
        raise ValueError(f"Missing shards: {sorted(missing)}")
    return roots


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def trace_map(path: Path) -> dict[tuple[int, int, int], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        (int(row["round"]), int(row["client"]), int(row["batch"])): str(row["sha256"])
        for row in rows
    }


def trace_digest(trace: dict[tuple[int, int, int], str]) -> str:
    payload = "\n".join(
        f"{round_idx},{client},{batch},{digest}"
        for (round_idx, client, batch), digest in sorted(trace.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def merge_shards(roots: dict[str, Path], output_dir: Path) -> dict:
    contracts: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    input_audits: dict[str, dict] = {}
    rows: dict[str, dict] = {}
    traces: dict[str, dict] = {}
    prediction_by_arm: dict[str, np.ndarray] = {}
    common_arrays: dict[str, np.ndarray] | None = None

    for shard in REQUIRED_SHARDS:
        root = roots[shard]
        contract = read_json(root / "configs" / f"CONTRACT_{shard}.json")
        completion = read_json(root / "configs" / f"COMPLETION_{shard}.json")
        summary = read_json(root / "analysis" / "RESULT_SUMMARY.json")
        audit = read_json(root / "outputs" / "INPUT_AUDIT.json")
        expected_arms = list(SHARDS[shard])

        if contract.get("execution_shard") != shard or contract.get("selected_arms") != expected_arms:
            raise ValueError(f"Contract shard/arm mismatch for {shard}")
        if completion.get("completed_arms") != expected_arms or completion.get("complete") is not True:
            raise ValueError(f"Incomplete shard: {shard}")
        if summary.get("execution_shard") != shard or summary.get("selected_arms") != expected_arms:
            raise ValueError(f"Analysis shard/arm mismatch for {shard}")
        if summary.get("scientific_evidence") is not True or summary.get("mode") != "formal":
            raise ValueError(f"Shard is not Formal scientific evidence: {shard}")
        if set(summary.get("rows", {})) != set(expected_arms):
            raise ValueError(f"Unexpected result rows for {shard}")

        for arm, record in contract["arms"].items():
            config_path = root / "configs" / f"{arm}.json"
            if sha256_file(config_path) != record["sha256"]:
                raise ValueError(f"Config hash mismatch: {shard}/{arm}")
            if arm in rows:
                raise ValueError(f"Arm appears in multiple shards: {arm}")
            rows[arm] = summary["rows"][arm]
            traces[arm] = trace_map(
                root / "outputs" / f"cle_hfl_context_{arm}_trainseed0" / "local_batch_trace.jsonl"
            )

        prediction_path = root / "analysis" / "HFL_CONTEXT_PREDICTIONS.npz"
        with np.load(prediction_path, allow_pickle=False) as payload:
            shard_arms = tuple(str(value) for value in payload["arms"].tolist())
            if shard_arms != tuple(expected_arms):
                raise ValueError(f"Prediction arm order mismatch for {shard}")
            predictions = payload["probabilities"]
            if predictions.shape[0] != len(expected_arms):
                raise ValueError(f"Prediction count mismatch for {shard}")
            for index, arm in enumerate(expected_arms):
                prediction_by_arm[arm] = predictions[index]
            shared = {
                "labels": payload["labels"],
                "binding": payload["binding"],
                "operator_names": payload["operator_names"],
            }
            if common_arrays is None:
                common_arrays = shared
            elif any(not np.array_equal(common_arrays[key], value) for key, value in shared.items()):
                raise ValueError(f"Evaluation grid mismatch for {shard}")

        contracts[shard] = contract
        summaries[shard] = summary
        input_audits[shard] = audit

    protocol_keys = (
        "protocol",
        "mode",
        "scenario_id",
        "partition_seed",
        "binding_map_seed",
        "evaluation_seed",
        "rounds",
        "train_seed",
    )
    reference_contract = contracts[REQUIRED_SHARDS[0]]
    for shard, contract in contracts.items():
        if any(contract.get(key) != reference_contract.get(key) for key in protocol_keys):
            raise ValueError(f"Protocol mismatch for {shard}")
    reference_audit = input_audits[REQUIRED_SHARDS[0]]
    if any(audit != reference_audit for audit in input_audits.values()):
        raise ValueError("Input audits differ across shards")
    if set(rows) != set(ARMS):
        raise ValueError(f"Merged arm coverage mismatch: {sorted(rows)}")

    reference_trace = traces["local_erm"]
    pairing_matches = {arm: trace == reference_trace for arm, trace in traces.items()}
    if not all(pairing_matches.values()):
        raise ValueError(f"Cross-shard local-batch pairing failed: {pairing_matches}")
    assert common_arrays is not None

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "HFL_CONTEXT_PREDICTIONS_MERGED.npz",
        probabilities=np.stack([prediction_by_arm[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        **common_arrays,
    )
    merged = {
        "protocol": "cle_hfl_context_table_merged_v2",
        "source_protocol": reference_contract["protocol"],
        "mode": "formal",
        "train_seed": reference_contract["train_seed"],
        "rounds": reference_contract["rounds"],
        "scenario_id": reference_contract["scenario_id"],
        "partition_seed": reference_contract["partition_seed"],
        "binding_map_seed": reference_contract["binding_map_seed"],
        "evaluation_seed": reference_contract["evaluation_seed"],
        "rows": {arm: rows[arm] for arm in ARMS},
        "source_shards": {shard: str(roots[shard]) for shard in REQUIRED_SHARDS},
        "cross_shard_local_batch_pairing": {
            "reference_arm": "local_erm",
            "arm_matches": pairing_matches,
            "all_arms_match": True,
            "reference_trace_sha256": trace_digest(reference_trace),
            "reference_trace_rows": len(reference_trace),
        },
        "scientific_evidence": True,
        "claim_boundary": "Protocol-matched HFL context table; not a plugin attribution experiment.",
    }
    (output_dir / "RESULT_SUMMARY_MERGED.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and merge four S2 HFL context shards.")
    parser.add_argument(
        "--shard-root",
        action="append",
        default=[],
        metavar="SHARD=PATH",
        help="Extracted shard archive root; provide cheap_a, cheap_b, fedtgp, and rhfl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_shards(parse_shard_roots(args.shard_root), args.output_dir.resolve())
    print(json.dumps(merged, indent=2), flush=True)


if __name__ == "__main__":
    main()
