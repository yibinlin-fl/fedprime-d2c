#!/usr/bin/python
# coding=utf-8
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.utils.config import load_config
from scripts.openi_strict_pew_asymhfl_entry import (
    DATASET_ARCHIVE,
    candidate_roots,
    find_dataset,
    log,
    prepare_c2net,
    run,
)


CONTROL_TEMPLATE = ROOT / "configs/openi_v100_rahfl_val_cle_v2_probe.yaml"
CANDIDATE_TEMPLATE = ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml"
GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_local_ablation"
SHARED_PEW = "outputs/pew_checkpoints/cle_v2_seed0_public5000.pt"
ARM_ORDER = (
    "a0_rahfl",
    "a1_ber",
    "a2_cdep",
    "a3_full",
    "a4_uncalibrated",
    "a5_shuffled",
    "a6_oracle_family",
)


def build_arm_configs() -> dict[str, dict]:
    control = load_config(CONTROL_TEMPLATE)
    candidate = load_config(CANDIDATE_TEMPLATE)
    configs = {}

    configs["a0_rahfl"] = copy.deepcopy(control)
    configs["a1_ber"] = copy.deepcopy(candidate)
    configs["a1_ber"]["method"]["fedease"]["cdep"]["enabled"] = False
    configs["a2_cdep"] = copy.deepcopy(candidate)
    configs["a2_cdep"]["method"]["fedease"]["ber"]["enabled"] = False
    configs["a3_full"] = copy.deepcopy(candidate)
    configs["a4_uncalibrated"] = copy.deepcopy(candidate)
    configs["a4_uncalibrated"]["method"]["fedease"]["pew"]["unknown_threshold"] = 0.55
    configs["a5_shuffled"] = copy.deepcopy(candidate)
    configs["a5_shuffled"]["method"]["fedease"]["environment_mode"] = "learned_shuffled"
    configs["a6_oracle_family"] = copy.deepcopy(candidate)
    configs["a6_oracle_family"]["method"]["fedease"]["environment_mode"] = "oracle_family"

    for name, config in configs.items():
        config["experiment_name"] = f"cle_local_ablation_{name}_seed0_12round"
        config["train"]["rounds"] = 12
        config["checkpoints"]["save_final"] = False
        if config.get("method_name") == "fedease":
            config["method"]["fedease"]["pew"]["checkpoint"] = SHARED_PEW
    return configs


def write_configs(configs: dict[str, dict]) -> dict[str, Path]:
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, config in configs.items():
        path = GENERATED_ROOT / f"{name}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        paths[name] = path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE local ablation screen.")
    parser.add_argument("--arms", default="all", help="all or comma-separated arm names")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def selected_arms(raw: str) -> list[str]:
    arms = list(ARM_ORDER) if raw == "all" else [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(arms) - set(ARM_ORDER))
    if unknown:
        raise ValueError(f"Unknown ablation arms: {unknown}")
    return arms


def package_outputs(arms: list[str], config_paths: dict[str, Path]) -> Path:
    archive = ROOT / "outputs/cle_local_ablation_12round_seed0_outputs.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    comparison = ROOT / "outputs/cle_local_ablation_12round_seed0.json"
    with tarfile.open(archive, "w:gz") as handle:
        for arm in arms:
            output = ROOT / "outputs" / f"cle_local_ablation_{arm}_seed0_12round"
            if output.exists():
                handle.add(output, arcname=f"outputs/{output.name}")
            handle.add(config_paths[arm], arcname=f"configs/generated/{arm}.json")
        if comparison.exists():
            handle.add(comparison, arcname="outputs/cle_local_ablation_12round_seed0.json")
    return archive


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    arms = selected_arms(args.arms)
    configs = build_arm_configs()
    config_paths = write_configs(configs)
    context = prepare_c2net()

    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    if not args.skip_import:
        source = find_dataset(candidate_roots(args, context))
        log(f"Prepared CLE-HFL v2 dataset: {source}")
        run([sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."], environment)

    for arm in arms:
        config_path = config_paths[arm]
        run([sys.executable, "scripts/check_environment.py", "--config", str(config_path)], environment)
        if not args.skip_train:
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_path)], environment)

    comparison = ROOT / "outputs/cle_local_ablation_12round_seed0.json"
    completed = [arm for arm in arms if (ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").exists()]
    if completed:
        command = [sys.executable, "scripts/analyze_cle_local_ablations.py"]
        for arm in completed:
            command.extend(["--arm", f"{arm}={ROOT / 'outputs' / configs[arm]['experiment_name']}"])
        command.extend(["--output", str(comparison)])
        run(command, environment)

    archive = package_outputs(arms, config_paths)
    log(f"Wrote {archive}")
    if not args.no_upload and context is not None:
        from c2net.context import upload_output

        output_path = Path(context.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for source in (archive, comparison):
            if source.exists():
                shutil.copy2(source, output_path / source.name)
        upload_output()


if __name__ == "__main__":
    main()
