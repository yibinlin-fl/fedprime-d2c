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
from scripts.openi_strict_pew_asymhfl_entry import candidate_roots, find_dataset, log, prepare_c2net, run


ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_external_baselines"
ARM_ORDER = ("local_only", "fedmd", "rhfl", "fedproto", "aughfl", "rahfl", "candidate")


def build_arm_configs() -> dict[str, dict]:
    control = load_config(ROOT / "configs/openi_v100_rahfl_val_cle_v2_probe.yaml")
    candidate = load_config(ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml")
    configs = {}
    for arm in ARM_ORDER:
        config = copy.deepcopy(candidate if arm == "candidate" else control)
        config["experiment_name"] = f"cle_external_{arm}_seed0_12round"
        config["train"]["rounds"] = 12
        config["checkpoints"]["save_final"] = False
        if arm == "local_only":
            config["method"]["communication"] = "none"
        elif arm == "fedmd":
            config["method"]["communication"] = "fedmd"
            config["method"]["lambda_jsd"] = 0.0
            config["method"]["cl_module"] = "none"
        elif arm == "rhfl":
            config["method"]["communication"] = "rhfl"
            config["method"]["lambda_jsd"] = 0.0
            config["method"]["cl_module"] = "rhfl_sce"
            config["method"]["baseline"] = {"beta": 0.5}
        elif arm == "fedproto":
            config["method"]["communication"] = "fedproto"
            config["method"]["lambda_jsd"] = 0.0
            config["method"]["cl_module"] = "none"
            config["method"]["baseline"] = {"proto_weight": 1.0}
        elif arm == "aughfl":
            config["method"]["communication"] = "aughfl"
            config["method"]["lambda_jsd"] = 12.0
            config["method"]["cl_module"] = "none"
        elif arm == "rahfl":
            config["method"]["communication"] = "asymhfl_val"
            config["method"]["lambda_jsd"] = 12.0
            config["method"]["cl_module"] = "dcl"
        else:
            config["method"]["fedease"]["pew"]["checkpoint"] = (
                "outputs/pew_checkpoints/cle_v2_seed0_public5000.pt"
            )
        configs[arm] = config
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE external baseline screen.")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def select_arms(raw: str) -> list[str]:
    arms = list(ARM_ORDER) if raw == "all" else [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(arms) - set(ARM_ORDER))
    if unknown:
        raise ValueError(f"Unknown baseline arms: {unknown}")
    return arms


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    configs = build_arm_configs()
    arms = select_arms(args.arms)
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

    result = ROOT / "outputs/cle_external_baselines_seed0_12round.json"
    completed = [arm for arm in arms if (ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").exists()]
    if completed and "rahfl" in completed:
        command = [sys.executable, "scripts/analyze_cle_external_baselines.py"]
        for arm in completed:
            command.extend(["--arm", f"{arm}={ROOT / 'outputs' / configs[arm]['experiment_name']}"])
        command.extend(["--output", str(result)])
        run(command, environment)

    archive = ROOT / "outputs/cle_external_baselines_seed0_12round_outputs.tar.gz"
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
