#!/usr/bin/python
# coding=utf-8
"""Run source-faithful baseline repairs without overwriting historical adapters."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.utils.config import load_config
from scripts.openi_strict_pew_asymhfl_entry import (
    candidate_roots,
    find_dataset,
    prepare_c2net,
    run,
)


GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_baseline_fidelity"
ARM_ORDER = ("aughfl_fidelity", "feddf_fidelity", "kt_pfl_fidelity", "rahfl", "pew_ber")
ARCHIVE_NAME = "cle_baseline_fidelity_seed0_12round_outputs.tar.gz"
REPORT_NAME = "cle_baseline_fidelity_seed0_12round.json"


def build_configs() -> dict[str, dict]:
    control = load_config(ROOT / "configs/openi_v100_rahfl_val_cle_v2_probe.yaml")
    candidate = load_config(
        ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml"
    )
    configs: dict[str, dict] = {}
    for arm in ARM_ORDER:
        config = copy.deepcopy(candidate if arm == "pew_ber" else control)
        config["seed"] = 0
        config["experiment_name"] = f"cle_fidelity_{arm}_seed0_12round"
        config["train"]["rounds"] = 12
        config["checkpoints"]["save_final"] = False
        config["method"]["strict_fit_audit"]["split_path"] = (
            "outputs/partitions/strict_cle_v2_seed0_split0.npz"
        )
        if arm == "aughfl_fidelity":
            config["method"]["communication"] = "aughfl_fidelity"
            config["method"]["lambda_jsd"] = 12.0
            config["method"]["cl_module"] = "none"
            config["method"]["baseline"] = {"collaborative_lr": 1.0e-3}
        elif arm == "feddf_fidelity":
            config["method"]["communication"] = "feddf_fidelity"
            config["method"]["lambda_jsd"] = 0.0
            config["method"]["cl_module"] = "none"
            config["method"]["baseline"] = {
                "temperature": 1.0,
                "student_learning_rate": 1.0e-3,
                "server_steps_per_batch": 1,
            }
        elif arm == "kt_pfl_fidelity":
            config["method"]["communication"] = "kt_pfl_fidelity"
            config["method"]["lambda_jsd"] = 0.0
            config["method"]["cl_module"] = "none"
            config["method"]["baseline"] = {
                "temperature": 1.0,
                "coefficient_lr": 0.01,
                "uniform_regularization": 0.5,
                "distillation_lr": 0.02,
                "distillation_steps": 1,
                "knowledge_weight": 1.0,
            }
        elif arm == "pew_ber":
            config["method"]["fedease"]["cdep"] = {"enabled": False}
            config["method"]["fedease"]["pew"]["checkpoint"] = (
                "outputs/pew_checkpoints/cle_baseline_fidelity_seed0.pt"
            )
            config["method"]["fedease"]["pew"]["reuse_checkpoint"] = True
        configs[arm] = config
    return configs


def fidelity_manifest() -> dict:
    return {
        "scope": "CLE-HFL protocol-matched fidelity repair; not an untouched official recipe run",
        "historical_adapters_preserved": ["aughfl", "feddf", "kt_pfl"],
        "repaired_adapters": {
            "aughfl_fidelity": {
                "source": "FangXiuwen/AugHFL HHF/AugHFL.py",
                "repairs": [
                    "participant-specific public AugMix triplets",
                    "released public normalization",
                    "fresh collaborative Adam optimizer",
                ],
            },
            "feddf_fidelity": {
                "source": "epfml/federated-learning-public-code FedDF-code",
                "repairs": [
                    "post-local server fusion",
                    "frozen round teacher snapshots",
                    "average-logit teacher and server Adam/cosine schedule",
                ],
            },
            "kt_pfl_fidelity": {
                "source": "NeurIPS 2021 paper, Algorithm 1 and Eqs. 6-7",
                "repairs": [
                    "post-local personalized distillation",
                    "post-distillation coefficient update",
                    "client-data-weighted coefficient objective",
                    "row-stochastic coefficient diagnostics",
                ],
            },
        },
        "matched_budget": {
            "rounds": 12,
            "local_epochs": 1,
            "public_batches_per_round": 4,
            "note": "This intentionally remains the common CLE screening budget, not each paper's full schedule.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE repaired baseline fidelity screen.")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def select_arms(raw: str) -> list[str]:
    selected = (
        list(ARM_ORDER)
        if raw == "all"
        else [item.strip() for item in raw.split(",") if item.strip()]
    )
    unknown = sorted(set(selected) - set(ARM_ORDER))
    if unknown:
        raise ValueError(f"Unknown fidelity arms: {unknown}")
    return selected


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    if not args.skip_import:
        source = find_dataset(candidate_roots(args, context))
        run(
            [sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."],
            environment,
        )

    configs = build_configs()
    if args.smoke:
        for arm, config in configs.items():
            config["experiment_name"] = f"smoke_cle_fidelity_{arm}"
            config["output_root"] = "local_test_outputs"
            config["num_workers"] = 0
            config["train"]["rounds"] = 1
            config["train"]["batch_size"] = 8
            config["train"]["test_batch_size"] = 8
            config["train"]["public_batch_size"] = 8
            config["train"]["max_local_batches"] = 1
            config["train"]["max_test_batches"] = 1
            config["train"]["public_batches_per_round"] = 1
            config["method"]["strict_fit_audit"]["max_audit_batches"] = 1
    selected = select_arms(args.arms)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for arm, config in configs.items():
        path = GENERATED_ROOT / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        paths[arm] = path
    manifest_path = GENERATED_ROOT / "fidelity_manifest.json"
    manifest_path.write_text(json.dumps(fidelity_manifest(), indent=2), encoding="utf-8")

    for arm in selected:
        run([sys.executable, "scripts/check_environment.py", "--config", str(paths[arm])], environment)
        if not args.skip_train:
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(paths[arm])], environment)

    completed = [
        arm
        for arm in selected
        if (ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").exists()
    ]
    report = ROOT / "outputs" / REPORT_NAME
    if completed and "rahfl" in completed:
        command = [sys.executable, "scripts/analyze_cle_external_baselines.py"]
        for arm in completed:
            command.extend(
                ["--arm", f"{arm}={ROOT / 'outputs' / configs[arm]['experiment_name']}"]
            )
        command.extend(["--output", str(report)])
        run(command, environment)

    archive = ROOT / "outputs" / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        for arm in selected:
            output = ROOT / "outputs" / configs[arm]["experiment_name"]
            if output.exists():
                handle.add(output, arcname=f"outputs/{output.name}")
            handle.add(paths[arm], arcname=f"configs/generated/{arm}.json")
        handle.add(manifest_path, arcname="configs/generated/fidelity_manifest.json")
        if report.exists():
            handle.add(report, arcname=f"outputs/{report.name}")

    if not args.no_upload:
        from c2net.context import upload_output

        upload_output()


if __name__ == "__main__":
    main()
