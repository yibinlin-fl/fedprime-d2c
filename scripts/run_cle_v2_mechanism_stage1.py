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

from scripts.run_cle_v2_factorial import arm_config  # noqa: E402


ARMS = ("h0_b", "h9_b", "l0_b", "l9_b")
DATA_PROTOCOL = "cle_hfl_v2_paired_factorial_seed0_split0_v1"
ROUND_BUDGET = {"smoke": 1, "benchmark": 1, "formal": 12}
LOCAL_BATCH_BUDGET = {"smoke": 1, "benchmark": 16, "formal": 16}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CLE-v2 mechanism Stage-1.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def verify_mechanism_assets(package_root: Path) -> dict:
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("Unexpected CLE-v2 factorial package protocol")
    required = [
        package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz",
        package_root / "public" / manifest["public"]["file"],
    ]
    required.extend(package_root / "initial_states" / f"client_{client}.pt" for client in range(4))
    required.extend(
        package_root / "evaluation/paired_dsa" / f"{name}.npy"
        for name in (
            "test_images",
            "test_labels",
            "test_corruption_ids",
            "test_severity_ids",
            "test_source_ids",
        )
    )
    required.extend(
        package_root / "data" / condition / f"client_{client}"
        for condition in ("gamma00", "gamma09")
        for client in range(4)
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    return manifest


def stage1_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    device: str,
    output_root: Path,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Stage-1 does not allow arm {arm}")
    smoke = mode == "smoke"
    config = arm_config(
        arm,
        package_root=package_root,
        train_seed=0,
        rounds=ROUND_BUDGET[mode],
        device=device,
        output_root=output_root,
        smoke=smoke,
        benchmark=False,
    )
    config["experiment_name"] = f"cle_v2_mechanism_stage1_{arm}_trainseed0"
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1
    config["method"]["strict_fit_audit"]["max_audit_batches"] = 1 if smoke else None
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    if config["method_name"] != "rahfl" or "fedease" in config["method"]:
        raise ValueError("Stage-1 must remain baseline-only and PEW-free")
    return config


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Stage-1 Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    verify_mechanism_assets(package_root)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = stage1_arm_config(
            arm,
            package_root=package_root,
            mode=args.mode,
            device=args.device,
            output_root=output_root,
        )
        path = config_root / f"{arm}_trainseed0.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
        if not args.prepare_only:
            print(f"[stage1] arm={arm} config={path}", flush=True)
            subprocess.check_call(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)],
                cwd=ROOT,
            )
    contract = {
        "protocol": "cle_v2_mechanism_stage1_contract_v1",
        "data_manifest_sha256": sha256_file(package_root / "manifest.json"),
        "mode": args.mode,
        "train_seed": 0,
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "batch_size": 16 if args.mode == "smoke" else 64,
        "paired_dsa": "final_checkpoint_only",
        "pew_used": False,
        "ber_used": False,
        "arms": records,
    }
    path = config_root / "STAGE1_CONTRACT.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
