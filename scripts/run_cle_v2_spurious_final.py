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
from scripts.run_cle_hfl_context import (  # noqa: E402
    FORMAL_LOCAL_BATCHES,
    fixed_local_budget_audit,
    strict_fit_full_epoch_budget,
)
from scripts.run_cle_v2_spurious_baselines import spurious_arm_config  # noqa: E402


ARMS = ("erm", "cvar_dro", "pew_groupdro", "pew_ber")
ARM_CHOICES = ("all", *ARMS)
ROUND_BUDGET = {"smoke": 1, "benchmark": 2, "formal": 40}
LOCAL_BATCH_CAP = {
    "smoke": 1,
    "benchmark": FORMAL_LOCAL_BATCHES,
    "formal": FORMAL_LOCAL_BATCHES,
}
MAP_SEED = 2


def selected_arms(arm: str) -> tuple[str, ...]:
    if arm == "all":
        return ARMS
    if arm not in ARMS:
        raise ValueError(f"Unknown Axis-I arm: {arm}")
    return (arm,)


def final_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    device: str,
    output_root: Path,
    train_seed: int = 0,
) -> dict:
    # Reuse the frozen screen objective definitions, then change only the
    # pre-registered duration and held-out scenario identity.
    config = spurious_arm_config(
        arm,
        package_root=package_root,
        mode="smoke" if mode == "smoke" else "screen",
        device=device,
        output_root=output_root,
        jtt_annotation_root=output_root / "unused_jtt_annotations",
        train_seed=int(train_seed),
    )
    config["experiment_name"] = f"cle_v2_spurious_final_map2_{arm}_trainseed{train_seed}"
    config["data"]["scenario"] = "cle_hfl_v2"
    config["data"]["scenario_id"] = "cle_hfl_v2_cross_map2_seed0_split0"
    config["train"]["rounds"] = ROUND_BUDGET[mode]
    config["train"].update(
        {
            "pretrain_epochs": 0,
            "local_epochs": 1,
            "batch_size": 64,
            "public_batch_size": 128,
            "public_batches_per_round": 4,
            "max_local_batches": LOCAL_BATCH_CAP[mode],
            "skip_nonfinite": False,
        }
    )
    config["train"]["max_test_batches"] = 1 if mode == "smoke" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = (
        1 if mode == "smoke" else None
    )
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    config["method"]["record_local_batch_trace"] = True
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out map2 40-round four-arm Formal.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--arm", choices=ARM_CHOICES, default="all")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("40-round four-arm Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_cross_scenario(package_root, MAP_SEED)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    arms = selected_arms(args.arm)
    records = {}
    configs = {}
    for arm in arms:
        config = final_arm_config(
            arm,
            package_root=package_root,
            mode=args.mode,
            device=args.device,
            output_root=output_root,
            train_seed=int(args.train_seed),
        )
        configs[arm] = config
        path = config_root / f"map2_{arm}_trainseed{args.train_seed}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
    contract = {
        "protocol": "cle_v2_spurious_final_map2_budget64_v3",
        "mode": args.mode,
        "scenario_id": manifest["scenario_id"],
        "partition_seed": 0,
        "binding_map_seed": MAP_SEED,
        "evaluation_seed": 20260909,
        "train_seed": int(args.train_seed),
        "execution_arm": args.arm,
        "selected_arms": list(arms),
        "rounds": ROUND_BUDGET[args.mode],
        "local_training": "1_batch_smoke" if args.mode == "smoke" else "64_batches_per_client_round",
        "max_local_batches": LOCAL_BATCH_CAP[args.mode],
        "batch_size": 64,
        "public_batch_size": 128,
        "public_batches_per_round": 4,
        "pretrain_epochs": 0,
        "full_epoch_budget_audit": strict_fit_full_epoch_budget(package_root),
        "effective_budget_audit": fixed_local_budget_audit(
            package_root,
            rounds=ROUND_BUDGET[args.mode],
            max_local_batches=LOCAL_BATCH_CAP[args.mode],
        ),
        "arms": records,
        "common_communication": "strict AsymHFL-val",
        "common_augmentation": "none",
        "cdep_used": False,
        "map2_was_not_used_for_five_arm_selection": True,
        "formal_requires_explicit_confirmation": True,
    }
    (config_root / f"SPURIOUS_FINAL_MAP2_CONTRACT_TRAINSEED{args.train_seed}_{args.arm}.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    completed = []
    completion_path = config_root / f"SPURIOUS_FINAL_MAP2_COMPLETION_TRAINSEED{args.train_seed}_{args.arm}.json"
    for arm in arms:
        path = config_root / f"map2_{arm}_trainseed{args.train_seed}.json"
        print(f"[spurious-final] arm={arm} config={path}", flush=True)
        subprocess.check_call(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)], cwd=ROOT
        )
        completed.append(arm)
        completion_path.write_text(
            json.dumps(
                {
                    "protocol": contract["protocol"],
                    "execution_arm": args.arm,
                    "selected_arms": list(arms),
                    "completed_arms": completed,
                    "complete": completed == list(arms),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
