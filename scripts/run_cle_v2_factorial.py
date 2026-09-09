from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARMS = {
    "h0_b": ("gamma00", "asymhfl_val", False),
    "h9_b": ("gamma09", "asymhfl_val", False),
    "l0_b": ("gamma00", "none", False),
    "l9_b": ("gamma09", "none", False),
    "h0_p": ("gamma00", "asymhfl_val", True),
    "h9_p": ("gamma09", "asymhfl_val", True),
    "l0_p": ("gamma00", "none", True),
    "l9_p": ("gamma09", "none", True),
}
MODES = {
    "mechanism": ("h0_b", "h9_b", "l0_b", "l9_b"),
    "plugin": ("h0_p", "h9_p", "l0_p", "l9_p"),
    "all": tuple(ARMS),
}
MODEL_NAMES = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]
DATA_PROTOCOL = "cle_hfl_v2_paired_factorial_seed0_split0_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the matched CLE-v2 eight-arm factorial.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="all")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--config-root", type=Path, default=ROOT / "local_runs/cle_v2_factorial_configs")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_package(package_root: Path) -> dict:
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("Unexpected CLE-v2 factorial package protocol")
    required = [
        package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz",
        package_root / "initial_states/client_0.pt",
        package_root / "evaluation/paired_dsa/test_images.npy",
        package_root / "pew_standard/pew_standard.pt",
        package_root / "pew_standard/manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    return manifest


def arm_config(
    arm: str,
    *,
    package_root: Path,
    train_seed: int,
    rounds: int,
    device: str,
    output_root: Path,
    smoke: bool,
    benchmark: bool = False,
) -> dict:
    if smoke and benchmark:
        raise ValueError("smoke and benchmark are mutually exclusive")
    condition, communication, plugin = ARMS[arm]
    train = {
        "pretrain_epochs": 0,
        "rounds": int(rounds),
        "local_epochs": 1,
        "batch_size": 16 if smoke else 64,
        "test_batch_size": 128 if smoke else 512,
        "public_batch_size": 32 if smoke else 128,
        "public_batches_per_round": 1 if (smoke or benchmark) else 4,
        "max_grad_norm": 5.0,
        "skip_nonfinite": True,
        "local_log_interval": 50 if not smoke else 1,
        "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
    }
    if smoke:
        train.update({"max_local_batches": 1, "max_test_batches": 1})
    elif benchmark:
        train.update({"max_local_batches": 8, "max_test_batches": 1})
    method = {
        "use_prime": False,
        "augmix_module": "jsd",
        "cl_module": "fedease" if plugin else "dcl",
        "lambda_jsd": 12.0,
        "communication": communication,
        "record_local_batch_trace": True,
        "paired_local_rng": {
            "enabled": True,
            "base_seed": 20260909,
            "protocol": "round_client_epoch_reset_v1",
        },
        "strict_fit_audit": {
            "enabled": True,
            "split_path": str(
                package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz"
            ),
            "audit_ratio": 0.15,
            "min_audit_per_class": 5,
            "min_fit_per_class": 2,
            "audit_batch_size": 64 if smoke else 256,
            "max_audit_batches": 1 if (smoke or benchmark) else None,
            "seed": 0,
            "loader_seed": 20260909 + int(train_seed),
        },
    }
    if plugin:
        method["fedease"] = {
            "environment_mode": "learned",
            "num_environments": 6,
            "preserve_dcl": True,
            "pew": {
                "annotation_root": str(
                    package_root / "pew_standard/annotations" / condition
                ),
                "checkpoint": str(package_root / "pew_standard/pew_standard.pt"),
                "unknown_threshold": "frozen_in_annotation_manifest",
            },
            "ber": {
                "enabled": True,
                "support_gamma": 0.5,
                "count_cap": 32,
                "min_group_count": 2,
            },
        }
    return {
        "experiment_name": f"cle_v2_factorial_{arm}_trainseed{train_seed}",
        "method_name": "fedease" if plugin else "rahfl",
        "seed": int(train_seed),
        "device": device,
        "num_workers": 0 if smoke else 2,
        "output_root": str(output_root),
        "data": {
            "scenario": "cle_hfl_v2",
            "private_dataset": "cifar10",
            "private_root": str(package_root / "data" / condition),
            "public_root": str(package_root / "public"),
            "num_classes": 10,
            "public_size": 5000,
            "download_public": False,
        },
        "models": {"names": MODEL_NAMES},
        "train": train,
        "method": method,
        "checkpoints": {
            "load_dir": str(package_root / "initial_states"),
            "require_all": True,
            "strict": True,
            "save_rounds": [] if rounds <= 1 else [12],
            "save_final": True,
        },
    }


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    verify_package(package_root)
    if args.smoke and args.benchmark:
        raise ValueError("--smoke and --benchmark are mutually exclusive")
    rounds = 1 if (args.smoke or args.benchmark) else int(args.rounds)
    if not args.prepare_only and not args.smoke and not args.benchmark and not args.confirm_formal:
        raise PermissionError("Formal execution requires --confirm-formal")
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root.resolve()
    selected = MODES[args.mode]
    records = {}
    for arm in selected:
        config = arm_config(
            arm,
            package_root=package_root,
            train_seed=int(args.train_seed),
            rounds=rounds,
            device=args.device,
            output_root=output_root,
            smoke=bool(args.smoke),
            benchmark=bool(args.benchmark),
        )
        path = config_root / f"{arm}_trainseed{args.train_seed}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
        if not args.prepare_only:
            print(f"[run] arm={arm} config={path}", flush=True)
            subprocess.check_call(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)],
                cwd=ROOT,
            )
    contract = {
        "protocol": "cle_v2_factorial_run_contract_v1",
        "data_manifest_sha256": sha256_file(package_root / "manifest.json"),
        "pew_manifest_sha256": sha256_file(package_root / "pew_standard/manifest.json"),
        "mode": args.mode,
        "train_seed": int(args.train_seed),
        "rounds": rounds,
        "smoke": bool(args.smoke),
        "benchmark": bool(args.benchmark),
        "arms": records,
    }
    contract_path = config_root / f"contract_{args.mode}_trainseed{args.train_seed}.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
