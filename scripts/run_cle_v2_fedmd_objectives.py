from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cle_v2_cross_scenario import sha256_file, verify_cross_scenario  # noqa: E402
from scripts.run_cle_v2_spurious_baselines import spurious_arm_config  # noqa: E402


ARMS = ("erm", "cvar_dro", "pew_groupdro", "pew_ber")
ROUND_BUDGET = {"smoke": 1, "benchmark": 1, "formal": 40}
LOCAL_BATCH_BUDGET = {"smoke": 1, "benchmark": 8, "formal": 16}
MAP_SEED = 2
EXPERIMENT_PREFIX = "cle_v2_fedmd_objectives_map2"


def fedmd_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    device: str,
    output_root: Path,
    train_seed: int = 0,
) -> dict:
    source_mode = "smoke" if mode == "smoke" else ("benchmark" if mode == "benchmark" else "screen")
    config = spurious_arm_config(
        arm,
        package_root=package_root,
        mode=source_mode,
        device=device,
        output_root=output_root,
        jtt_annotation_root=output_root / "unused_jtt_annotations",
        train_seed=int(train_seed),
    )
    config["experiment_name"] = f"{EXPERIMENT_PREFIX}_{arm}_trainseed{train_seed}"
    config["data"]["scenario"] = "cle_hfl_v2"
    config["data"]["scenario_id"] = "cle_hfl_v2_cross_map2_seed0_split0"
    config["method"]["communication"] = "fedmd"
    config["train"]["rounds"] = ROUND_BUDGET[mode]
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode != "formal" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = 1 if mode != "formal" else None
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out map2 FedMD four-objective replication.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("FedMD four-objective Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_cross_scenario(package_root, MAP_SEED)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = fedmd_arm_config(
            arm,
            package_root=package_root,
            mode=args.mode,
            device=args.device,
            output_root=output_root,
            train_seed=int(args.train_seed),
        )
        path = config_root / f"map2_fedmd_{arm}_trainseed{args.train_seed}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
    contract = {
        "protocol": "cle_v2_fedmd_four_objective_map2_v1",
        "mode": args.mode,
        "scenario_id": manifest["scenario_id"],
        "partition_seed": 0,
        "binding_map_seed": MAP_SEED,
        "evaluation_seed": 20260909,
        "train_seed": int(args.train_seed),
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "arms": records,
        "common_communication": "FedMD symmetric public-logit exchange",
        "common_augmentation": "none",
        "cdep_used": False,
        "purpose": "Cross-communication replication of four local objectives; not a universal plug-in claim.",
        "formal_requires_explicit_confirmation": True,
    }
    contract_path = config_root / f"FEDMD_OBJECTIVES_MAP2_CONTRACT_TRAINSEED{args.train_seed}.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    for arm in ARMS:
        path = config_root / f"map2_fedmd_{arm}_trainseed{args.train_seed}.json"
        subprocess.check_call(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)], cwd=ROOT
        )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
