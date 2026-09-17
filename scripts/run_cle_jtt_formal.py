from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.utils.env import resolve_device  # noqa: E402
from scripts.prepare_cle_v2_factorial_data import sha256_file  # noqa: E402
from scripts.run_cle_v2_plugin_stage2 import verify_stage2_assets  # noqa: E402
from scripts.run_cle_v2_spurious_baselines import generate_jtt_error_masks, spurious_arm_config  # noqa: E402
ARMS = ("erm", "jtt")
ROUND_BUDGET = {"benchmark": 1, "formal": 40}
LOCAL_BATCH_BUDGET = {"benchmark": 8, "formal": 16}


def jtt_config(arm: str, *, package_root: Path, mode: str, output_root: Path, config_root: Path, device: str) -> dict:
    config = spurious_arm_config(
        arm,
        package_root=package_root,
        mode="benchmark" if mode == "benchmark" else "screen",
        device=device,
        output_root=output_root,
        jtt_annotation_root=config_root / "jtt_stage1_fit_errors",
        train_seed=0,
    )
    config["experiment_name"] = f"cle_jtt_formal_{arm}_trainseed0"
    config["train"]["rounds"] = ROUND_BUDGET[mode]
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode == "benchmark" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = 1 if mode == "benchmark" else None
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched 40-round ERM vs JTT Formal.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    args = parser.parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("JTT Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    verify_stage2_assets(package_root)
    output_root, config_root = args.output_root.resolve(), args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    configs, records = {}, {}
    for arm in ARMS:
        config = jtt_config(
            arm, package_root=package_root, mode=args.mode, output_root=output_root, config_root=config_root, device=args.device
        )
        configs[arm] = config
        path = config_root / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
    contract = {
        "protocol": "cle_jtt_formal_v1",
        "mode": args.mode,
        "rounds_per_stage": ROUND_BUDGET[args.mode],
        "jtt_two_stage_total_training_budget_disclosed": True,
        "jtt_selection_source": "stage-1 ERM fit errors only",
        "arms": records,
    }
    (config_root / "CONTRACT.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    subprocess.check_call([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_root / "erm.json")], cwd=ROOT)
    generate_jtt_error_masks(
        erm_config=configs["erm"],
        erm_checkpoint_root=output_root / configs["erm"]["experiment_name"] / "checkpoints",
        destination=config_root / "jtt_stage1_fit_errors",
        device=resolve_device(args.device),
    )
    subprocess.check_call([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_root / "jtt.json")], cwd=ROOT)
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
