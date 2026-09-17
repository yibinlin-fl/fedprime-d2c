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


HELDOUT_OPERATOR = "motion_blur"
PEW_ASSET = "pew_loo_motion_blur"
ARMS = ("erm", "cvar_dro", "pew_ber_loo")
ROUND_BUDGET = {"benchmark": 1, "formal": 40}
LOCAL_BATCH_BUDGET = {"benchmark": 8, "formal": 16}


def verify_package(package_root: Path) -> dict:
    manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
    if HELDOUT_OPERATOR not in manifest.get("seen_operators", []):
        raise ValueError("Held-out PEW operator must occur in private CLE training")
    pew_manifest_path = package_root / PEW_ASSET / "manifest.json"
    pew_manifest = json.loads(pew_manifest_path.read_text(encoding="utf-8"))
    if pew_manifest.get("excluded_public_operators") != [HELDOUT_OPERATOR]:
        raise ValueError("PEW LOO exclusion does not match frozen stress operator")
    return manifest


def stress_arm_config(arm: str, *, package_root: Path, mode: str, device: str, output_root: Path) -> dict:
    base_arm = "pew_ber" if arm == "pew_ber_loo" else arm
    config = spurious_arm_config(
        base_arm,
        package_root=package_root,
        mode="benchmark" if mode == "benchmark" else "screen",
        device=device,
        output_root=output_root,
        jtt_annotation_root=output_root / "unused_jtt_annotations",
        train_seed=0,
    )
    config["experiment_name"] = f"cle_taxonomy_stress_motion_blur_{arm}_trainseed0"
    config["train"]["rounds"] = ROUND_BUDGET[mode]
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode == "benchmark" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = 1 if mode == "benchmark" else None
    if arm == "pew_ber_loo":
        config["method"]["fedease"]["pew"].update(
            {
                "annotation_root": str(package_root / PEW_ASSET / "annotations/gamma09"),
                "checkpoint": str(package_root / PEW_ASSET / "pew_standard.pt"),
            }
        )
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded PEW taxonomy held-out-operator stress.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Taxonomy stress Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    verify_package(package_root)
    output_root, config_root = args.output_root.resolve(), args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = stress_arm_config(
            arm, package_root=package_root, mode=args.mode, device=args.device, output_root=output_root
        )
        path = config_root / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
    contract = {
        "protocol": "cle_taxonomy_stress_motion_blur_v1",
        "mode": args.mode,
        "heldout_from_pew_public_training": HELDOUT_OPERATOR,
        "present_in_private_cle_training": True,
        "rounds": ROUND_BUDGET[args.mode],
        "arms": records,
        "scientific_scope": "bounded held-out-operator stress, not open-world robustness",
    }
    (config_root / "CONTRACT.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    for arm in ARMS:
        subprocess.check_call(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_root / f"{arm}.json")],
            cwd=ROOT,
        )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
