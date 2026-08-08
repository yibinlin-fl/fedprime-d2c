#!/usr/bin/python
# coding=utf-8
"""Run the RAHFL-table baselines that were absent from the first CLE screen."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
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
    log,
    prepare_c2net,
    run,
)


GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_remaining_baselines"
ARM_ORDER = ("feddf", "kt_pfl", "fccl", "rahfl", "pew_ber")
ARCHIVE_NAME = "cle_remaining_baselines_seed0_12round_outputs.tar.gz"
REPORT_NAME = "cle_remaining_baselines_seed0_12round.json"


def build_configs() -> dict[str, dict]:
    control = load_config(ROOT / "configs/openi_v100_rahfl_val_cle_v2_probe.yaml")
    candidate = load_config(
        ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml"
    )
    configs = {}
    for arm in ARM_ORDER:
        config = copy.deepcopy(candidate if arm == "pew_ber" else control)
        config["seed"] = 0
        config["experiment_name"] = f"cle_remaining_{arm}_seed0_12round"
        config["train"]["rounds"] = 12
        config["checkpoints"]["save_final"] = False
        config["method"]["strict_fit_audit"]["split_path"] = (
            "outputs/partitions/strict_cle_v2_seed0_split0.npz"
        )
        if arm in {"feddf", "kt_pfl", "fccl"}:
            config["method"]["communication"] = arm
            config["method"]["lambda_jsd"] = 0.0
            config["method"]["cl_module"] = "none"
        if arm == "feddf":
            config["method"]["baseline"] = {"temperature": 1.0}
        elif arm == "kt_pfl":
            config["method"]["baseline"] = {
                "temperature": 1.0,
                "coefficient_lr": 0.01,
                "uniform_regularization": 0.5,
            }
        elif arm == "fccl":
            config["method"]["baseline"] = {
                "offdiag_weight": 0.0051,
                "eps": 1.0e-6,
            }
        elif arm == "pew_ber":
            config["method"]["fedease"]["cdep"] = {"enabled": False}
            config["method"]["fedease"]["pew"]["checkpoint"] = (
                "outputs/pew_checkpoints/cle_remaining_baselines_seed0.pt"
            )
            config["method"]["fedease"]["pew"]["reuse_checkpoint"] = True
        configs[arm] = config
    return configs


def select_arms(raw: str) -> list[str]:
    selected = (
        list(ARM_ORDER)
        if raw == "all"
        else [item.strip() for item in raw.split(",") if item.strip()]
    )
    unknown = sorted(set(selected) - set(ARM_ORDER))
    if unknown:
        raise ValueError(f"Unknown remaining-baseline arms: {unknown}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE remaining RAHFL baselines.")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


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
    selected = select_arms(args.arms)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for arm in ARM_ORDER:
        path = GENERATED_ROOT / f"{arm}.json"
        path.write_text(json.dumps(configs[arm], indent=2), encoding="utf-8")
        paths[arm] = path

    for arm in selected:
        run([sys.executable, "scripts/check_environment.py", "--config", str(paths[arm])], environment)
        if not args.skip_train:
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(paths[arm])], environment)

    report = ROOT / "outputs" / REPORT_NAME
    completed = [
        arm
        for arm in selected
        if (ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").is_file()
    ]
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
            handle.add(paths[arm], arcname=f"configs/generated/{arm}.json")
            output = ROOT / "outputs" / configs[arm]["experiment_name"]
            if output.exists():
                handle.add(output, arcname=f"outputs/{output.name}")
        if report.is_file():
            handle.add(report, arcname=f"outputs/{report.name}")
    log(f"Wrote {archive}")
    if not args.no_upload and context is not None:
        from c2net.context import upload_output

        destination = Path(context.output_path)
        destination.mkdir(parents=True, exist_ok=True)
        for source in (archive, report):
            if source.is_file():
                shutil.copy2(source, destination / source.name)
        upload_output()


if __name__ == "__main__":
    main()
