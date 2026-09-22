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
from scripts.run_cle_v2_factorial import arm_config, verify_package  # noqa: E402


ARMS = (
    "local_erm",
    "fedmd_adapter",
    "fedproto_adapter",
    "fedtgp_adapter",
    "feddf_fidelity",
    "kt_pfl_fidelity",
    "fccl_adapter",
    "rhfl_adapter",
    "aughfl_fidelity",
    "rahfl_fidelity",
)
ROUND_BUDGET = {"pairing": 2, "benchmark": 1, "formal": 40}
LOCAL_BATCH_BUDGET = {"pairing": 2, "benchmark": 8, "formal": 16}


def context_arm_config(arm: str, *, package_root: Path, mode: str, output_root: Path, device: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown HFL context arm: {arm}")
    config = arm_config(
        "h9_b",
        package_root=package_root,
        train_seed=0,
        rounds=ROUND_BUDGET[mode],
        device=device,
        output_root=output_root,
        smoke=False,
        benchmark=mode != "formal",
    )
    config["experiment_name"] = f"cle_hfl_context_{arm}_trainseed0"
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode != "formal" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = 1 if mode != "formal" else None
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    if arm == "local_erm":
        config["method"].update({"communication": "none", "cl_module": "none", "lambda_jsd": 0.0})
    elif arm == "fedmd_adapter":
        config["method"].update({"communication": "fedmd", "cl_module": "none", "lambda_jsd": 0.0})
    elif arm == "fedproto_adapter":
        config["method"].update({"communication": "fedproto", "cl_module": "none", "lambda_jsd": 0.0})
        config["method"]["baseline"] = {
            "proto_weight": 1.0,
            "max_proto_batches": 1 if mode == "pairing" else None,
        }
    elif arm == "fedtgp_adapter":
        config["method"].update({"communication": "fedtgp", "cl_module": "none", "lambda_jsd": 0.0})
        config["method"]["baseline"] = {
            "proto_weight": 10.0,
            "server_learning_rate": 0.01,
            "max_proto_batches": 1 if mode == "pairing" else None,
            "server_epochs": 1 if mode == "pairing" else 100,
            "server_batch_size": 10,
            "margin_threshold": 100.0,
        }
    elif arm == "feddf_fidelity":
        config["method"].update({"communication": "feddf_fidelity", "cl_module": "none", "lambda_jsd": 0.0})
        config["method"]["baseline"] = {
            "temperature": 1.0,
            "student_learning_rate": 1.0e-3,
            "server_steps_per_batch": 1,
        }
    elif arm == "kt_pfl_fidelity":
        config["method"].update({"communication": "kt_pfl_fidelity", "cl_module": "none", "lambda_jsd": 0.0})
        config["method"]["baseline"] = {
            "temperature": 1.0,
            "coefficient_lr": 0.01,
            "uniform_regularization": 0.5,
            "distillation_lr": 0.02,
            "distillation_steps": 1,
            "knowledge_weight": 1.0,
        }
    elif arm == "fccl_adapter":
        config["method"].update({"communication": "fccl", "cl_module": "none", "lambda_jsd": 0.0})
        config["method"]["baseline"] = {"offdiag_weight": 0.0051, "eps": 1.0e-6}
    elif arm == "rhfl_adapter":
        config["method"].update({"communication": "rhfl", "cl_module": "rhfl_sce", "lambda_jsd": 0.0})
        config["method"]["baseline"] = {
            "beta": 0.5,
            "max_quality_batches": 1 if mode == "pairing" else None,
        }
    elif arm == "aughfl_fidelity":
        config["method"].update({"communication": "aughfl_fidelity", "cl_module": "none", "lambda_jsd": 12.0})
        config["method"]["baseline"] = {"collaborative_lr": 1.0e-3}
    return config


def fidelity_manifest() -> dict:
    return {
        "scope": "Protocol-matched CLE context table; not untouched official full-recipe runs.",
        "arms": {
            "local_erm": "No communication; ERM local objective.",
            "fedmd_adapter": "Protocol-matched FedMD symmetric public-logit exchange.",
            "fedproto_adapter": "Protocol-matched FedProto class-prototype aggregation.",
            "fedtgp_adapter": "Protocol-matched FedTGP trainable global-prototype core; not an untouched official recipe.",
            "feddf_fidelity": "Post-local frozen-teacher FedDF fidelity adapter.",
            "kt_pfl_fidelity": "Post-local alternating KT-pFL fidelity adapter.",
            "fccl_adapter": "Protocol-matched FCCL cross-correlation adapter; not claimed official fidelity.",
            "rhfl_adapter": "Protocol-matched RHFL SCE and client-confidence reweighting core.",
            "aughfl_fidelity": "Participant-specific public AugMix views and released-style collaboration.",
            "rahfl_fidelity": "Repository RAHFL/AugHFL-JSD/DCL plus strict AsymHFL-val anchor.",
        },
        "purpose": "Contextualize CLE across representative HFL methods; not BER attribution.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the submission HFL context table.")
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
        raise PermissionError("HFL context Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    verify_package(package_root)
    output_root, config_root = args.output_root.resolve(), args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = context_arm_config(
            arm, package_root=package_root, mode=args.mode, output_root=output_root, device=args.device
        )
        path = config_root / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
    (config_root / "FIDELITY_MANIFEST.json").write_text(
        json.dumps(fidelity_manifest(), indent=2), encoding="utf-8"
    )
    contract = {
        "protocol": "cle_hfl_context_table_v2",
        "mode": args.mode,
        "rounds": ROUND_BUDGET[args.mode],
        "train_seed": 0,
        "arms": records,
        "not_a_plugin_attribution_experiment": True,
    }
    (config_root / "CONTRACT.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    for arm in ARMS:
        subprocess.check_call(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_root / f"{arm}.json")], cwd=ROOT
        )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
