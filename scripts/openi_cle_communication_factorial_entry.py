#!/usr/bin/python
# coding=utf-8
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

from fedprime.utils.config import load_config
from scripts.openi_strict_pew_asymhfl_entry import (
    candidate_roots,
    find_dataset,
    log,
    prepare_c2net,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_communication_factorial"
COMMUNICATIONS = ("none", "hfl", "asymhfl_val")
ARM_ORDER = tuple(
    f"{local}_{communication}"
    for local in ("l0", "l1")
    for communication in COMMUNICATIONS
)


def build_arm_configs() -> dict[str, dict]:
    templates = {
        "l0": load_config(ROOT / "configs/openi_v100_rahfl_val_cle_v2_probe.yaml"),
        "l1": load_config(ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml"),
    }
    configs = {}
    for local, template in templates.items():
        for communication in COMMUNICATIONS:
            name = f"{local}_{communication}"
            config = copy.deepcopy(template)
            config["experiment_name"] = f"cle_comm_factorial_{name}_seed0_12round"
            config["method"]["communication"] = communication
            config["train"]["rounds"] = 12
            config["checkpoints"]["save_final"] = False
            if local == "l1":
                config["method"]["fedease"]["pew"]["checkpoint"] = (
                    "outputs/pew_checkpoints/cle_v2_seed0_public5000.pt"
                )
            configs[name] = config
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI 2x3 local/communication factorial.")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def select_arms(raw: str) -> list[str]:
    arms = list(ARM_ORDER) if raw == "all" else [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(arms) - set(ARM_ORDER))
    if unknown:
        raise ValueError(f"Unknown factorial arms: {unknown}")
    return arms


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    arms = select_arms(args.arms)
    configs = build_arm_configs()
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for arm, config in configs.items():
        path = GENERATED_ROOT / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        paths[arm] = path

    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    if not args.skip_import:
        source = find_dataset(candidate_roots(args, context))
        run([sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."], environment)
    for arm in arms:
        run([sys.executable, "scripts/check_environment.py", "--config", str(paths[arm])], environment)
        if not args.skip_train:
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(paths[arm])], environment)

    result = ROOT / "outputs/cle_communication_factorial_seed0_12round.json"
    if all((ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").exists() for arm in ARM_ORDER):
        run([
            sys.executable,
            "scripts/analyze_cle_communication_factorial.py",
            "--root", "outputs",
            "--output", str(result),
        ], environment)

    archive = ROOT / "outputs/cle_communication_factorial_seed0_12round_outputs.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for arm in arms:
            output = ROOT / "outputs" / configs[arm]["experiment_name"]
            if output.exists():
                handle.add(output, arcname=f"outputs/{output.name}")
            handle.add(paths[arm], arcname=f"configs/generated/{arm}.json")
        if result.exists():
            handle.add(result, arcname=f"outputs/{result.name}")
    log(f"Wrote {archive}")
    if not args.no_upload and context is not None:
        from c2net.context import upload_output

        destination = Path(context.output_path)
        destination.mkdir(parents=True, exist_ok=True)
        for source in (archive, result):
            if source.exists():
                shutil.copy2(source, destination / source.name)
        upload_output()


if __name__ == "__main__":
    main()
