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
from scripts.run_cle_v2_plugin_stage2 import verify_stage2_assets  # noqa: E402


ARMS = {
    "h9_of": ("oracle_family", 6),
    "h9_oo": ("oracle_operator", 15),
    "h9_ro": ("oracle_operator_shuffled", 15),
}
ROUND_BUDGET = {"smoke": 1, "benchmark": 1, "formal": 12}
LOCAL_BATCH_BUDGET = {"smoke": 1, "benchmark": 8, "formal": 16}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def oracle_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    device: str,
    output_root: Path,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Oracle granularity protocol does not allow arm {arm}")
    smoke, benchmark = mode == "smoke", mode == "benchmark"
    config = arm_config(
        "h9_p",
        package_root=package_root,
        train_seed=0,
        rounds=ROUND_BUDGET[mode],
        device=device,
        output_root=output_root,
        smoke=smoke,
        benchmark=benchmark,
    )
    environment_mode, num_environments = ARMS[arm]
    config["experiment_name"] = f"cle_v2_oracle_granularity_{arm}_trainseed0"
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode != "formal" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = (
        1 if mode != "formal" else None
    )
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    fedease = config["method"]["fedease"]
    fedease["environment_mode"] = environment_mode
    fedease["num_environments"] = num_environments
    fedease.pop("pew", None)
    if config["method_name"] != "fedease" or config["method"]["cl_module"] != "fedease":
        raise ValueError("Oracle arms must preserve the PEW+BER local trainer")
    if not fedease["ber"]["enabled"] or not fedease["preserve_dcl"]:
        raise ValueError("Oracle arms must preserve hard BER and DCL")
    if "cdep" in json.dumps(config).lower() or "pew" in json.dumps(fedease).lower():
        raise ValueError("Oracle granularity arms forbid CDep and learned PEW")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CLE-v2 oracle granularity arms.")
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
        raise PermissionError("Oracle granularity Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_stage2_assets(package_root)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = oracle_arm_config(
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
            print(f"[oracle-granularity] arm={arm} config={path}", flush=True)
            subprocess.check_call(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)],
                cwd=ROOT,
            )
    contract = {
        "protocol": "cle_v2_oracle_granularity_contract_v1",
        "mode": args.mode,
        "scenario": manifest["protocol"],
        "train_seed": 0,
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "batch_size": 16 if args.mode == "smoke" else 64,
        "arms": records,
        "learned_pew_used": False,
        "private_operator_metadata_used": True,
        "deployable_method_claim": False,
        "cdep_used": False,
    }
    path = config_root / "ORACLE_GRANULARITY_CONTRACT.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
