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
from scripts.run_cle_v2_plugin_stage2 import (  # noqa: E402
    ALLOWED_FEDEASE_KEYS,
    verify_stage2_assets,
)


ARMS = {"fd_b": "h9_b", "fd_p": "h9_p"}
ROUND_BUDGET = {"smoke": 1, "benchmark": 1, "formal": 12}
LOCAL_BATCH_BUDGET = {"smoke": 1, "benchmark": 8, "formal": 16}
FEDDF_CONFIG = {
    "temperature": 1.0,
    "student_learning_rate": 1.0e-3,
    "server_steps_per_batch": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def feddf_plugin_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    device: str,
    output_root: Path,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"FedDF plugin protocol does not allow arm {arm}")
    smoke, benchmark = mode == "smoke", mode == "benchmark"
    config = arm_config(
        ARMS[arm],
        package_root=package_root,
        train_seed=0,
        rounds=ROUND_BUDGET[mode],
        device=device,
        output_root=output_root,
        smoke=smoke,
        benchmark=benchmark,
    )
    config["experiment_name"] = f"cle_v2_feddf_plugin_{arm}_trainseed0"
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode != "formal" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = (
        1 if mode != "formal" else None
    )
    config["method"]["communication"] = "feddf_fidelity"
    config["method"]["baseline"] = dict(FEDDF_CONFIG)
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True

    if arm == "fd_b":
        if config["method_name"] != "rahfl" or config["method"]["cl_module"] != "dcl":
            raise ValueError("FedDF base must preserve the AugMix/JSD/DCL local trainer")
        if "fedease" in config["method"]:
            raise ValueError("FedDF base must be PEW/BER-free")
    else:
        fedease = config["method"].get("fedease", {})
        if config["method_name"] != "fedease" or config["method"]["cl_module"] != "fedease":
            raise ValueError("FedDF plugin arm must use the PEW+BER local trainer")
        if set(fedease) != ALLOWED_FEDEASE_KEYS:
            raise ValueError("FedDF plugin arm contains unexpected FedEASE fields")
        if fedease.get("environment_mode") != "learned" or not fedease["ber"]["enabled"]:
            raise ValueError("FedDF plugin arm must use frozen learned PEW and hard BER")
        if not fedease.get("preserve_dcl"):
            raise ValueError("FedDF plugin arm must preserve DCL")
        if "cdep" in json.dumps(config).lower():
            raise ValueError("CDep is forbidden in the FedDF plugin attribution")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CLE-v2 FedDF PEW+BER A/B.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("FedDF plugin Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_stage2_assets(package_root)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = feddf_plugin_arm_config(
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
            print(f"[feddf-plugin] arm={arm} config={path}", flush=True)
            subprocess.check_call(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)],
                cwd=ROOT,
            )
    contract = {
        "protocol": "cle_v2_feddf_plugin_contract_v1",
        "mode": args.mode,
        "scenario": manifest["protocol"],
        "train_seed": 0,
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "batch_size": 16 if args.mode == "smoke" else 64,
        "communication_both_arms": "feddf_fidelity",
        "local_backbone_both_arms": "AugMix/JSD/DCL",
        "only_candidate_addition": "frozen coarse PEW + hard BER",
        "cdep_used": False,
        "arms": records,
    }
    path = config_root / "FEDDF_PLUGIN_CONTRACT.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
