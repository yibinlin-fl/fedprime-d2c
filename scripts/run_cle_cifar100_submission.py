from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_cle_v2_factorial_data import sha256_file  # noqa: E402
from scripts.run_cle_v2_spurious_baselines import spurious_arm_config  # noqa: E402


PROTOCOL = "cle_hfl_v2_cifar100_factorial_seed0_split0_v1"
ARMS = ("erm_gamma0", "erm", "cvar_dro", "pew_ber")
ROUND_BUDGET = {"benchmark": 1, "formal": 40}
LOCAL_BATCH_BUDGET = {"benchmark": 8, "formal": 16}


def verify_package(package_root: Path) -> dict:
    manifest_path = package_root / "manifest.json"
    pew_manifest = package_root / "pew_standard/manifest.json"
    if not manifest_path.is_file() or not pew_manifest.is_file():
        raise FileNotFoundError("CIFAR-100 package or frozen PEW manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected CIFAR-100 submission protocol")
    for path in (
        package_root / "splits/strict_cle_cifar100_seed0_split0.npz",
        package_root / "evaluation/paired_dsa/test_images.npy",
        package_root / "pew_standard/pew_standard.pt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    return manifest


def submission_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    train_seed: int,
    device: str,
    output_root: Path,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    base_arm = "erm" if arm == "erm_gamma0" else arm
    config = spurious_arm_config(
        base_arm,
        package_root=package_root,
        mode="benchmark" if mode == "benchmark" else "screen",
        device=device,
        output_root=output_root,
        jtt_annotation_root=output_root / "unused_jtt_annotations",
        train_seed=int(train_seed),
    )
    condition = "gamma00" if arm == "erm_gamma0" else "gamma09"
    config["experiment_name"] = f"cle_cifar100_submission_{arm}_trainseed{train_seed}"
    config["data"].update(
        {
            "scenario": "cle_hfl_v2_cifar100_submission",
            "scenario_id": "cle_hfl_v2_cifar100_seed0_split0",
            "private_dataset": "cifar100",
            "private_root": str(package_root / "data" / condition),
            "public_dataset": "cifar10",
            "public_root": str(package_root / "public"),
            "num_classes": 100,
            "public_size": 5000,
        }
    )
    config["train"]["rounds"] = ROUND_BUDGET[mode]
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode == "benchmark" else None
    config["method"]["strict_fit_audit"].update(
        {
            "split_path": str(package_root / "splits/strict_cle_cifar100_seed0_split0.npz"),
            "max_audit_batches": 1 if mode == "benchmark" else None,
            "loader_seed": 20260917 + int(train_seed),
        }
    )
    if arm == "pew_ber":
        config["method"]["fedease"]["pew"].update(
            {
                "annotation_root": str(package_root / "pew_standard/annotations/gamma09"),
                "checkpoint": str(package_root / "pew_standard/pew_standard.pt"),
            }
        )
    config["checkpoints"].update(
        {
            "load_dir": str(package_root / "initial_states"),
            "require_all": True,
            "strict": True,
            "save_rounds": [],
            "save_final": True,
        }
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIFAR-100 submission CLE protocol.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="benchmark")
    parser.add_argument("--train-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("CIFAR-100 Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_package(package_root)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = submission_arm_config(
            arm,
            package_root=package_root,
            mode=args.mode,
            train_seed=int(args.train_seed),
            device=args.device,
            output_root=output_root,
        )
        path = config_root / f"{arm}_trainseed{args.train_seed}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
    contract = {
        "protocol": "cle_cifar100_submission_run_v1",
        "data_protocol": manifest["protocol"],
        "mode": args.mode,
        "train_seed": int(args.train_seed),
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "arms": records,
        "private_dataset": "cifar100",
        "public_communication_dataset": "cifar10",
        "pew_trained_once_on_cifar10_public_then_frozen": True,
        "pew_checkpoint_reused_from_main_cifar10_experiment": False,
        "cdep_used": False,
    }
    contract_path = config_root / f"CONTRACT_TRAINSEED{args.train_seed}.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    for arm in ARMS:
        path = config_root / f"{arm}_trainseed{args.train_seed}.json"
        subprocess.check_call(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)], cwd=ROOT
        )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
