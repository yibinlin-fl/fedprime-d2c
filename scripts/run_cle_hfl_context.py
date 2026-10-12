from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


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
STANDARD_LOCAL_ARMS = (
    "local_erm",
    "fedmd_adapter",
    "fedproto_adapter",
    "fedtgp_adapter",
    "feddf_fidelity",
    "kt_pfl_fidelity",
    "fccl_adapter",
    "rhfl_adapter",
)
AUGMIX_LOCAL_ARMS = (
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
ROUND_BUDGET = {"pairing": 2, "benchmark": 2, "formal": 40}
# Pairing is deliberately tiny.  The former 16-batch budget yielded only
# about 4.8 local epochs over 40 rounds, while a full 133-batch epoch made a
# single arm cost about 28 V100 hours.  The pre-registered middle budget keeps
# all 40 communication rounds and yields about 19.25 local epoch equivalents.
FORMAL_LOCAL_BATCHES = 64
LOCAL_BATCH_CAP = {
    "pairing": 2,
    "benchmark": FORMAL_LOCAL_BATCHES,
    "formal": FORMAL_LOCAL_BATCHES,
}
MAP_SEED = 2


def strict_fit_full_epoch_budget(package_root: Path, batch_size: int = 64) -> dict:
    split_path = package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    with np.load(split_path, allow_pickle=False) as payload:
        fit_samples = {
            str(client_id): int(len(payload[f"client_{client_id}_fit"]))
            for client_id in range(4)
        }
    batches = {
        client_id: int(math.ceil(count / int(batch_size)))
        for client_id, count in fit_samples.items()
    }
    return {
        "split": str(split_path.resolve()),
        "batch_size": int(batch_size),
        "fit_samples_per_client": fit_samples,
        "expected_batches_per_client_round": batches,
        "fit_samples_all_clients_per_round": int(sum(fit_samples.values())),
        "expected_optimizer_steps_all_clients_per_round": int(sum(batches.values())),
    }


def fixed_local_budget_audit(
    package_root: Path,
    *,
    rounds: int,
    max_local_batches: int,
    batch_size: int = 64,
) -> dict:
    full = strict_fit_full_epoch_budget(package_root, batch_size=batch_size)
    full_batches = full["expected_batches_per_client_round"]
    effective_batches = {
        client_id: min(int(count), int(max_local_batches))
        for client_id, count in full_batches.items()
    }
    epoch_equivalents = {
        client_id: float(rounds) * effective_batches[client_id] / int(full_batches[client_id])
        for client_id in full_batches
    }
    return {
        "rounds": int(rounds),
        "max_local_batches": int(max_local_batches),
        "effective_batches_per_client_round": effective_batches,
        "optimizer_steps_all_clients_per_round": int(sum(effective_batches.values())),
        "local_epoch_equivalents_per_client": epoch_equivalents,
        "mean_local_epoch_equivalents": float(np.mean(list(epoch_equivalents.values()))),
    }
def selected_arms(shard: str, mode: str) -> tuple[str, ...]:
    if shard not in SHARDS:
        raise ValueError(f"Unknown HFL context shard: {shard}")
    arms = SHARDS[shard]
    if mode == "pairing" and "local_erm" not in arms:
        return ("local_erm", *arms)
    return tuple(arms)


def local_loader_mode_for_arm(arm: str) -> str:
    if arm in STANDARD_LOCAL_ARMS:
        return "standard"
    if arm in AUGMIX_LOCAL_ARMS:
        return "augmix"
    raise ValueError(f"Unknown HFL context arm: {arm}")


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
    config["train"].update(
        {
            "pretrain_epochs": 0,
            "local_epochs": 1,
            "batch_size": 64,
            "public_batch_size": 128,
            "public_batches_per_round": 4,
            "max_local_batches": LOCAL_BATCH_CAP[mode],
        }
    )
    # Context-table evidence must never silently skip non-finite updates.  A
    # numerical failure invalidates the arm and must stop the task.
    config["train"]["skip_nonfinite"] = False
    config["train"]["max_test_batches"] = 1 if mode == "pairing" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = 1 if mode == "pairing" else None
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    config["method"]["record_local_batch_trace"] = True
    # Only corruption-robust AugHFL/RAHFL use their native multi-view AugMix
    # pipeline.  Plain HFL baselines must use a true single-view CE loader;
    # setting lambda_jsd=0 on an AugMix loader still incurs three-view forward
    # compute and changes BatchNorm statistics.
    config["method"]["local_loader_mode"] = local_loader_mode_for_arm(arm)
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
            "local_erm": "No communication; single-view standard-loader ERM local objective.",
            "fedmd_adapter": "Single-view CE plus protocol-matched FedMD symmetric public-logit exchange.",
            "fedproto_adapter": "Single-view CE plus protocol-matched FedProto class-prototype aggregation.",
            "fedtgp_adapter": "Single-view CE plus protocol-matched FedTGP trainable global-prototype core; not an untouched official recipe.",
            "feddf_fidelity": "Single-view CE plus post-local frozen-teacher FedDF fidelity adapter.",
            "kt_pfl_fidelity": "Single-view CE plus post-local alternating KT-pFL fidelity adapter.",
            "fccl_adapter": "Single-view CE plus protocol-matched FCCL cross-correlation adapter; not claimed official fidelity.",
            "rhfl_adapter": "Single-view RHFL SCE and client-confidence reweighting core.",
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
        "protocol": "cle_hfl_context_table_map2_budget64_v6",
        "mode": args.mode,
        "scenario_id": manifest["scenario_id"],
        "partition_seed": 0,
        "binding_map_seed": MAP_SEED,
        "evaluation_seed": 20260909,
        "rounds": ROUND_BUDGET[args.mode],
        "local_training": "2_batch_pairing" if args.mode == "pairing" else "64_batches_per_client_round",
        "max_local_batches": LOCAL_BATCH_CAP[args.mode],
        "batch_size": 64,
        "public_batch_size": 128,
        "public_batches_per_round": 4,
        "pretrain_epochs": 0,
        "full_epoch_budget_audit": strict_fit_full_epoch_budget(package_root),
        "effective_budget_audit": fixed_local_budget_audit(
            package_root,
            rounds=ROUND_BUDGET[args.mode],
            max_local_batches=LOCAL_BATCH_CAP[args.mode],
        ),
        "public_batch_size": 128,
        "train_seed": 0,
        "execution_shard": args.shard,
        "selected_arms": list(arms),
        "local_loader_policy": {
            arm: local_loader_mode_for_arm(arm)
            for arm in arms
        },
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
