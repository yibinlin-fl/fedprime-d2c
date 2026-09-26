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
from scripts.run_cle_v2_cross_scenario import verify_cross_scenario  # noqa: E402
from scripts.run_cle_v2_factorial import arm_config  # noqa: E402


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
PRACTICAL_ARMS = (
    "local_erm",
    "fedmd_adapter",
    "fedproto_adapter",
    "feddf_fidelity",
    "kt_pfl_fidelity",
    "fccl_adapter",
    "aughfl_fidelity",
    "rahfl_fidelity",
)
SHARDS = {
    "all": ARMS,
    "cheap_a": ("local_erm", "fedmd_adapter", "fedproto_adapter", "aughfl_fidelity"),
    "cheap_b": ("feddf_fidelity", "kt_pfl_fidelity", "fccl_adapter", "rahfl_fidelity"),
    "local": ("local_erm",),
    "fedmd": ("fedmd_adapter",),
    "fedproto": ("fedproto_adapter",),
    "feddf": ("feddf_fidelity",),
    "kt_pfl": ("kt_pfl_fidelity",),
    "fccl": ("fccl_adapter",),
    "aughfl": ("aughfl_fidelity",),
    "rahfl": ("rahfl_fidelity",),
    "fedtgp": ("fedtgp_adapter",),
    "rhfl": ("rhfl_adapter",),
}
ROUND_BUDGET = {"pairing": 2, "benchmark": 1, "formal": 40}
LOCAL_BATCH_BUDGET = {"pairing": 2, "benchmark": 8, "formal": 16}
MAP_SEED = 2


def selected_arms(shard: str, mode: str) -> tuple[str, ...]:
    if shard not in SHARDS:
        raise ValueError(f"Unknown HFL context shard: {shard}")
    arms = SHARDS[shard]
    if mode == "pairing" and "local_erm" not in arms:
        return ("local_erm", *arms)
    return tuple(arms)


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
    config["data"]["scenario"] = "cle_hfl_v2"
    config["data"]["scenario_id"] = "cle_hfl_v2_cross_map2_seed0_split0"
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    # Context-table evidence must never silently skip non-finite updates.  A
    # numerical failure invalidates the arm and must stop the task.
    config["train"]["skip_nonfinite"] = False
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
            "prototype_source": "local_batches",
            "max_proto_batches": None,
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
    parser.add_argument("--shard", choices=SHARDS, default="all")
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
    manifest = verify_cross_scenario(package_root, MAP_SEED)
    output_root, config_root = args.output_root.resolve(), args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    arms = selected_arms(args.shard, args.mode)
    records = {}
    for arm in arms:
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
        "protocol": "cle_hfl_context_table_map2_v3",
        "mode": args.mode,
        "scenario_id": manifest["scenario_id"],
        "partition_seed": 0,
        "binding_map_seed": MAP_SEED,
        "evaluation_seed": 20260909,
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "batch_size": 64,
        "public_batch_size": 128,
        "train_seed": 0,
        "execution_shard": args.shard,
        "selected_arms": list(arms),
        "arms": records,
        "not_a_plugin_attribution_experiment": True,
        "map2_was_not_used_for_five_arm_selection": True,
    }
    (config_root / f"CONTRACT_{args.shard}.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    completed = []
    completion_path = config_root / f"COMPLETION_{args.shard}.json"
    for arm in arms:
        subprocess.check_call(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_root / f"{arm}.json")], cwd=ROOT
        )
        completed.append(arm)
        completion_path.write_text(
            json.dumps(
                {
                    "protocol": contract["protocol"],
                    "execution_shard": args.shard,
                    "selected_arms": list(arms),
                    "completed_arms": completed,
                    "complete": completed == list(arms),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
