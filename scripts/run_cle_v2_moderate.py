from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_cle_v2_factorial_data import sha256_file  # noqa: E402
from scripts.run_cle_hfl_context import context_arm_config  # noqa: E402
from scripts.run_cle_v2_spurious_baselines import spurious_arm_config  # noqa: E402


PACKAGE_NAME = "cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0"
PROTOCOL = "cle_hfl_v2_moderate_alpha10_rho05_gamma_grid_seed0_split0_v1"
ROUND_BUDGET = {"smoke": 1, "benchmark": 2, "formal": 40}
LOCAL_BATCH_BUDGET = {"smoke": 1, "benchmark": 8, "formal": 64}

DOSE_ARMS = ("erm_g00", "erm_g05", "erm_g09", "pew_ber_g09")
AXIS2_ARMS = (
    "local_erm",
    "fedmd_adapter",
    "fedproto_adapter",
    "feddf_fidelity",
    "aughfl_fidelity",
    "rahfl_fidelity",
    "asymhfl_erm",
    "asymhfl_pew_ber",
)
SHARDS = {
    "dose_all": DOSE_ARMS,
    "axis2_all": AXIS2_ARMS,
    "axis2_plain": AXIS2_ARMS[:4],
    "axis2_robust": AXIS2_ARMS[4:6],
    "axis2_asym": AXIS2_ARMS[6:],
    **{arm: (arm,) for arm in (*DOSE_ARMS, *AXIS2_ARMS)},
}


def verify_moderate_package(package_root: Path) -> dict:
    package_root = package_root.resolve()
    if package_root.name != PACKAGE_NAME:
        raise ValueError(f"Expected package {PACKAGE_NAME}, got {package_root.name}")
    manifest_path = package_root / "manifest.json"
    audit_path = package_root / "integrity_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL,
        "package": PACKAGE_NAME,
        "binding_map_seed": 2,
        "partition_seed": 0,
        "dirichlet_alpha": 1.0,
        "corruption_rate": 0.5,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest mismatch for {key}: {manifest.get(key)!r} != {value!r}")
    if manifest.get("gammas") != {"gamma00": 0.0, "gamma05": 0.5, "gamma09": 0.9}:
        raise ValueError("Moderate package must contain the frozen gamma grid 0.0/0.5/0.9")
    if audit.get("status") != "PASS":
        raise ValueError("Moderate package integrity audit did not pass")
    required = [
        "splits/strict_cle_v2_factorial_seed0_split0.npz",
        "evaluation/paired_dsa/test_images.npy",
        "evaluation/paired_dsa/test_labels.npy",
        "evaluation/paired_dsa/test_corruption_ids.npy",
        "pew_standard/pew_standard.pt",
    ]
    for condition in ("gamma00", "gamma05", "gamma09"):
        for client_id in range(4):
            required.extend(
                [
                    f"data/{condition}/client_{client_id}/train_images.npy",
                    f"data/{condition}/client_{client_id}/train_labels.npy",
                    f"pew_standard/annotations/{condition}/client_{client_id}.npz",
                ]
            )
        for split in ("test_clean", "test_balanced", "test_seen", "test_unseen"):
            required.extend(
                [
                    f"data/{condition}/{split}/test_images.npy",
                    f"data/{condition}/{split}/test_labels.npy",
                ]
            )
    missing = [item for item in required if not (package_root / item).is_file()]
    if missing:
        raise FileNotFoundError(f"Moderate package is incomplete: {missing[:8]}")
    with np.load(package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz") as splits:
        if any(f"client_{client_id}_fit" not in splits for client_id in range(4)):
            raise ValueError("Strict fit/audit split is incomplete")
    return manifest


def _patch_common(config: dict, *, package_root: Path, condition: str, mode: str, arm: str) -> dict:
    config["experiment_name"] = f"cle_v2_moderate_{arm}_trainseed0"
    # The prepared-data loader dispatches on the stable cle_hfl_v2 scenario
    # key; scenario_id below carries the moderate-protocol identity.
    config["data"]["scenario"] = "cle_hfl_v2"
    config["data"]["scenario_id"] = f"{PACKAGE_NAME}_{condition}"
    config["data"]["private_root"] = str(package_root / "data" / condition)
    config["data"]["private_corrupt_rate"] = 0.5
    config["train"].update(
        {
            "pretrain_epochs": 0,
            "rounds": ROUND_BUDGET[mode],
            "local_epochs": 1,
            "batch_size": 64,
            "public_batch_size": 128,
            "public_batches_per_round": 4,
            "max_local_batches": LOCAL_BATCH_BUDGET[mode],
            "max_test_batches": 1 if mode == "smoke" else None,
            "skip_nonfinite": False,
        }
    )
    config["method"]["strict_fit_audit"]["max_audit_batches"] = 1 if mode == "smoke" else None
    config["method"]["record_local_batch_trace"] = True
    config["checkpoints"].update({"save_rounds": [], "save_final": True})
    fedease = config["method"].get("fedease")
    if fedease:
        fedease["pew"]["annotation_root"] = str(
            package_root / "pew_standard" / "annotations" / condition
        )
    return config


def moderate_arm_config(
    arm: str, *, package_root: Path, mode: str, output_root: Path, device: str
) -> dict:
    if arm in DOSE_ARMS:
        condition = {"erm_g00": "gamma00", "erm_g05": "gamma05"}.get(arm, "gamma09")
        base_arm = "pew_ber" if arm == "pew_ber_g09" else "erm"
        config = spurious_arm_config(
            base_arm,
            package_root=package_root,
            mode="smoke" if mode == "smoke" else "screen",
            device=device,
            output_root=output_root,
            jtt_annotation_root=output_root / "unused_jtt",
            train_seed=0,
        )
        return _patch_common(config, package_root=package_root, condition=condition, mode=mode, arm=arm)
    if arm in AXIS2_ARMS[:6]:
        config = context_arm_config(
            arm, package_root=package_root, mode="pairing" if mode == "smoke" else "formal",
            output_root=output_root, device=device
        )
    elif arm == "asymhfl_erm":
        config = spurious_arm_config(
            "erm", package_root=package_root, mode="smoke" if mode == "smoke" else "screen",
            device=device, output_root=output_root, jtt_annotation_root=output_root / "unused_jtt",
            train_seed=0,
        )
    elif arm == "asymhfl_pew_ber":
        config = spurious_arm_config(
            "pew_ber", package_root=package_root, mode="smoke" if mode == "smoke" else "screen",
            device=device, output_root=output_root, jtt_annotation_root=output_root / "unused_jtt",
            train_seed=0,
        )
    else:
        raise ValueError(f"Unknown moderate arm: {arm}")
    return _patch_common(config, package_root=package_root, condition="gamma09", mode=mode, arm=arm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen moderate CLE dose/Axis-II protocol.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="smoke")
    parser.add_argument("--shard", choices=SHARDS, default="axis2_plain")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Moderate Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_moderate_package(package_root)
    output_root, config_root = args.output_root.resolve(), args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    arms = SHARDS[args.shard]
    records = {}
    for arm in arms:
        config = moderate_arm_config(
            arm, package_root=package_root, mode=args.mode, output_root=output_root, device=args.device
        )
        path = config_root / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
    contract = {
        "protocol": "cle_v2_moderate_submission_v1",
        "source_protocol": manifest["protocol"],
        "mode": args.mode,
        "shard": args.shard,
        "arms": list(arms),
        "rounds": ROUND_BUDGET[args.mode],
        "max_local_batches": LOCAL_BATCH_BUDGET[args.mode],
        "batch_size": 64,
        "public_batches_per_round": 4,
        "train_seed": 0,
        "map_seed": 2,
        "dirichlet_alpha": 1.0,
        "corruption_rate": 0.5,
        "records": records,
        "claim_boundary": "Dose response and fair Axis-II context; strong-protocol DSA evidence remains separate.",
    }
    completion_path = config_root / f"COMPLETION_{args.shard}.json"
    (config_root / f"CONTRACT_{args.shard}.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    completed = []
    for arm in arms:
        subprocess.check_call(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_root / f"{arm}.json")],
            cwd=ROOT,
        )
        completed.append(arm)
        completion_path.write_text(
            json.dumps({"selected": list(arms), "completed": completed, "complete": completed == list(arms)}, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
