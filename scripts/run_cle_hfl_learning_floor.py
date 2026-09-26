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
from scripts.run_cle_hfl_context import MAP_SEED, context_arm_config  # noqa: E402
from scripts.run_cle_v2_cross_scenario import verify_cross_scenario  # noqa: E402


BUDGETS = (32, 64)
ROUND_BUDGET = {"smoke": 1, "benchmark": 2, "formal": 40}


def learning_floor_config(
    *,
    package_root: Path,
    mode: str,
    local_batches: int,
    output_root: Path,
    device: str,
) -> dict:
    if local_batches not in BUDGETS:
        raise ValueError(f"Unsupported local-batch budget: {local_batches}")
    base_mode = "formal" if mode == "formal" else "benchmark"
    config = context_arm_config(
        "local_erm",
        package_root=package_root,
        mode=base_mode,
        output_root=output_root,
        device=device,
    )
    config["experiment_name"] = f"cle_hfl_learning_floor_b{local_batches}_trainseed0"
    config["train"]["rounds"] = ROUND_BUDGET[mode]
    config["train"]["max_local_batches"] = 2 if mode == "smoke" else local_batches
    config["train"]["max_test_batches"] = None if mode == "formal" else 1
    config["train"]["skip_nonfinite"] = False
    config["method"]["strict_fit_audit"]["max_audit_batches"] = None if mode == "formal" else 1
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CLE-HFL Local learning-floor test.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="benchmark")
    parser.add_argument("--local-batches", type=int, choices=BUDGETS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Learning-floor Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_cross_scenario(package_root, MAP_SEED)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    config = learning_floor_config(
        package_root=package_root,
        mode=args.mode,
        local_batches=args.local_batches,
        output_root=output_root,
        device=args.device,
    )
    config_path = config_root / f"local_b{args.local_batches}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    contract = {
        "protocol": "cle_hfl_learning_floor_map2_v1",
        "mode": args.mode,
        "scenario_id": manifest["scenario_id"],
        "partition_seed": 0,
        "binding_map_seed": MAP_SEED,
        "evaluation_seed": 20260909,
        "train_seed": 0,
        "rounds": ROUND_BUDGET[args.mode],
        "target_local_batches_per_client_round": args.local_batches,
        "executed_local_batches_per_client_round": config["train"]["max_local_batches"],
        "batch_size": 64,
        "arm": "local_erm",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "reuses_existing_b16_formal": True,
        "scientific_evidence": args.mode == "formal",
    }
    contract_path = config_root / f"CONTRACT_b{args.local_batches}.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return
    subprocess.check_call(
        [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_path)],
        cwd=ROOT,
    )
    completion = {
        "protocol": contract["protocol"],
        "mode": args.mode,
        "local_batches": args.local_batches,
        "complete": True,
    }
    (config_root / f"COMPLETION_b{args.local_batches}.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
