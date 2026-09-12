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


BASES = {
    "kt": {
        "communication": "kt_pfl_fidelity",
        "baseline": {
            "temperature": 1.0,
            "coefficient_lr": 0.01,
            "uniform_regularization": 0.5,
            "distillation_lr": 0.02,
            "distillation_steps": 1,
            "knowledge_weight": 1.0,
        },
        "claim_name": "equation-oriented KT-pFL fidelity adapter",
    },
    "fc": {
        "communication": "fccl",
        "baseline": {"offdiag_weight": 0.0051, "eps": 1.0e-6},
        "claim_name": "FCCL public cross-correlation core adapter",
    },
}
ARMS = ("kt_b", "kt_p", "fc_b", "fc_p")
ROUND_BUDGET = {"smoke": 1, "benchmark": 1, "formal": 12}
LOCAL_BATCH_BUDGET = {"smoke": 1, "benchmark": 8, "formal": 16}
PLUGIN_KEYS = {
    "environment_mode",
    "num_environments",
    "objective",
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


def base_key(arm: str) -> str:
    key = arm.split("_", maxsplit=1)[0]
    if key not in BASES or arm not in ARMS:
        raise ValueError(f"Unknown KT/FCCL plugin arm: {arm}")
    return key


def kt_fccl_plugin_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    device: str,
    output_root: Path,
) -> dict:
    key = base_key(arm)
    plugin = arm.endswith("_p")
    config = arm_config(
        "h9_p" if plugin else "h9_b",
        package_root=package_root,
        train_seed=0,
        rounds=ROUND_BUDGET[mode],
        device=device,
        output_root=output_root,
        smoke=mode == "smoke",
        benchmark=mode == "benchmark",
    )
    config["experiment_name"] = f"cle_v2_kt_fccl_plugin_{arm}_trainseed0"
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode != "formal" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = (
        1 if mode != "formal" else None
    )
    config["method"]["communication"] = BASES[key]["communication"]
    config["method"]["baseline"] = dict(BASES[key]["baseline"])
    config["method"]["local_loader_mode"] = "standard"
    config["method"]["lambda_jsd"] = 0.0
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True

    if not plugin:
        config["method"]["cl_module"] = "none"
        if config["method_name"] != "rahfl":
            raise ValueError("Base arm must use the standard CE experiment runner")
        if "fedease" in config["method"]:
            raise ValueError("Base arm must remain PEW/BER-free")
    else:
        fedease = config["method"].get("fedease", {})
        fedease["objective"] = "ce_ber"
        fedease["preserve_dcl"] = False
        if config["method_name"] != "fedease" or config["method"]["cl_module"] != "fedease":
            raise ValueError("Plugin arm must use the independent PEW+BER local trainer")
        if set(fedease) != PLUGIN_KEYS:
            raise ValueError("Plugin arm contains unexpected FedEASE fields")
        if fedease.get("environment_mode") != "learned" or not fedease["ber"]["enabled"]:
            raise ValueError("Plugin arm must use frozen learned PEW and hard BER")
        if fedease.get("preserve_dcl") or config["method"]["lambda_jsd"] != 0.0:
            raise ValueError("Native-base attribution forbids AugMix/JSD/DCL")
        if "cdep" in json.dumps(config).lower():
            raise ValueError("CDep is forbidden in the KT/FCCL plugin experiment")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLE-v2 KT-pFL/FCCL PEW+BER pairs.")
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
        raise PermissionError("KT/FCCL plugin Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_stage2_assets(package_root)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = kt_fccl_plugin_arm_config(
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
            print(f"[kt-fccl-plugin] arm={arm} config={path}", flush=True)
            subprocess.check_call(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)],
                cwd=ROOT,
            )
    contract = {
        "protocol": "cle_v2_kt_fccl_plugin_contract_v1",
        "mode": args.mode,
        "scenario": manifest["protocol"],
        "train_seed": 0,
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "pairs": {
            key: {
                "communication_both_arms": value["communication"],
                "adapter_scope": value["claim_name"],
                "base_local_objective": "standard cross entropy",
                "plugin_local_objective": "PEW-grouped hard BER-weighted cross entropy",
            }
            for key, value in BASES.items()
        },
        "baseline_communication_implementations_modified": False,
        "augmentation_or_contrastive_objective_all_arms": "none",
        "cdep_used": False,
        "arms": records,
    }
    path = config_root / "KT_FCCL_PLUGIN_CONTRACT.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
