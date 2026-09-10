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

from scripts.run_cle_v2_factorial import arm_config, verify_package  # noqa: E402


ARMS = ("h9_b", "h9_p")
ROUND_BUDGET = {"smoke": 1, "formal": 12}
LOCAL_BATCH_BUDGET = {"smoke": 1, "formal": 16}
ALLOWED_FEDEASE_KEYS = {
    "environment_mode",
    "num_environments",
    "preserve_dcl",
    "pew",
    "ber",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CLE-v2 PEW+BER Stage-2 A/B.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def verify_stage2_assets(package_root: Path) -> dict:
    manifest = verify_package(package_root)
    required = [
        package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz",
        package_root / "pew_standard/pew_standard.pt",
        package_root / "pew_standard/manifest.json",
    ]
    required.extend(package_root / "initial_states" / f"client_{client}.pt" for client in range(4))
    required.extend(package_root / "data/gamma09" / f"client_{client}" for client in range(4))
    required.extend(
        package_root / "pew_standard/annotations/gamma09" / f"client_{client}.npz"
        for client in range(4)
    )
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
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    return manifest


def stage2_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    device: str,
    output_root: Path,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Stage-2 does not allow arm {arm}")
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
    config["experiment_name"] = f"cle_v2_plugin_stage2_{arm}_trainseed0"
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    # Smoke limits reporting to one batch, but Formal must use the complete
    # reporting split because P2 is a scientific last-five utility gate.
    config["train"]["max_test_batches"] = 1 if smoke else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = 1 if smoke else None
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    if arm == "h9_b":
        if config["method_name"] != "rahfl" or "fedease" in config["method"]:
            raise ValueError("Stage-2 baseline must be PEW/BER-free")
    else:
        fedease = config["method"].get("fedease", {})
        if config["method_name"] != "fedease" or config["method"].get("cl_module") != "fedease":
            raise ValueError("Stage-2 plugin must use the PEW+BER local trainer")
        if set(fedease) != ALLOWED_FEDEASE_KEYS:
            raise ValueError(f"Unexpected plugin fields: {sorted(set(fedease) - ALLOWED_FEDEASE_KEYS)}")
        if fedease.get("environment_mode") != "learned" or not fedease.get("ber", {}).get("enabled"):
            raise ValueError("Stage-2 plugin must use learned PEW and enabled BER")
        if "cdep" in json.dumps(config).lower():
            raise ValueError("CDep is forbidden in the pure PEW+BER Stage-2 arm")
    return config


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Stage-2 Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_stage2_assets(package_root)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = stage2_arm_config(
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
            print(f"[stage2] arm={arm} config={path}", flush=True)
            subprocess.check_call(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)],
                cwd=ROOT,
            )
    contract = {
        "protocol": "cle_v2_plugin_stage2_contract_v1",
        "data_manifest_sha256": sha256_file(package_root / "manifest.json"),
        "pew_manifest_sha256": sha256_file(package_root / "pew_standard/manifest.json"),
        "mode": args.mode,
        "train_seed": 0,
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "batch_size": 16 if args.mode == "smoke" else 64,
        "scenario": manifest["protocol"],
        "arms": records,
        "baseline": "AugMix/JSD/DCL + strict AsymHFL-val",
        "candidate": "baseline + frozen public PEW + hard BER",
        "cdep_used": False,
    }
    path = config_root / "STAGE2_CONTRACT.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
