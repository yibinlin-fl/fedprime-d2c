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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.utils.config import load_config
from scripts.openi_strict_pew_asymhfl_entry import candidate_roots, log, prepare_c2net, run


GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_cross_scenario_40round"
SCENARIOS = {
    1: "alpha05_gamma09_seed1_split1",
    2: "alpha05_gamma09_seed2_split2",
}


def dataset_archive_name(seed: int) -> str:
    return f"cle_hfl_v2_prepared_{SCENARIOS[seed]}.tar.gz"


def find_scenario_archive(seed: int, roots: list[Path]) -> Path:
    name = dataset_archive_name(seed)
    for root in roots:
        direct = root / name
        if direct.is_file():
            return direct
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name} under {roots}")


def build_configs(seed: int) -> dict[str, dict]:
    if seed not in SCENARIOS:
        raise ValueError(f"Unsupported cross-scenario seed: {seed}")
    templates = {
        "control": load_config(ROOT / "configs/openi_v100_rahfl_val_cle_v2_40round_probe.yaml"),
        "candidate": load_config(ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_40round_probe.yaml"),
    }
    scenario = SCENARIOS[seed]
    configs = {}
    for arm, template in templates.items():
        config = copy.deepcopy(template)
        config["seed"] = 0
        config["experiment_name"] = f"cross_scenario40_{arm}_{scenario}_trainseed0"
        config["data"]["private_root"] = f"RAHFL-master/Dataset/cifar_10_cle_v2/{scenario}"
        strict = config["method"]["strict_fit_audit"]
        strict["split_path"] = f"outputs/partitions/strict_cle_v2_{scenario}.npz"
        strict["seed"] = 0
        config["train"]["rounds"] = 40
        config["checkpoints"]["save_final"] = False
        if arm == "candidate":
            config["method"]["fedease"]["pew"]["checkpoint"] = (
                f"outputs/pew_checkpoints/cle_v2_scenario{seed}_trainseed0.pt"
            )
        configs[arm] = config
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI strict CLE cross-scenario 40-round A/B.")
    parser.add_argument("--scenario_seed", choices=["1", "2", "all"], default="all")
    parser.add_argument("--mode", choices=["control", "candidate", "both"], default="both")
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
    seeds = [1, 2] if args.scenario_seed == "all" else [int(args.scenario_seed)]
    arms = ["control", "candidate"] if args.mode == "both" else [args.mode]
    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)

    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    roots = candidate_roots(args, context)
    for seed in seeds:
        if not args.skip_import:
            archive = find_scenario_archive(seed, roots)
            run([sys.executable, "scripts/import_cle_data.py", "--source", str(archive), "--destination", "."], environment)
        configs = build_configs(seed)
        paths = {}
        for arm, config in configs.items():
            path = GENERATED_ROOT / f"scenario{seed}_{arm}.json"
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            paths[arm] = path
        for arm in arms:
            run([sys.executable, "scripts/check_environment.py", "--config", str(paths[arm])], environment)
            if not args.skip_train:
                run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(paths[arm])], environment)
        comparison = ROOT / f"outputs/cross_scenario40_seed{seed}_comparison.json"
        if all((ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").exists() for arm in ("control", "candidate")):
            run([
                sys.executable, "scripts/analyze_strict_pew_asymhfl_40round.py",
                "--control", str(ROOT / "outputs" / configs["control"]["experiment_name"]),
                "--candidate", str(ROOT / "outputs" / configs["candidate"]["experiment_name"]),
                "--output", str(comparison),
            ], environment)
        output_archive = ROOT / f"outputs/cross_scenario40_seed{seed}_outputs.tar.gz"
        with tarfile.open(output_archive, "w:gz") as handle:
            for arm in arms:
                output = ROOT / "outputs" / configs[arm]["experiment_name"]
                if output.exists():
                    handle.add(output, arcname=f"outputs/{output.name}")
                handle.add(paths[arm], arcname=f"configs/generated/scenario{seed}_{arm}.json")
            if comparison.exists():
                handle.add(comparison, arcname=f"outputs/{comparison.name}")
        log(f"Wrote {output_archive}")
        if not args.no_upload and context is not None:
            from c2net.context import upload_output

            destination = Path(context.output_path)
            destination.mkdir(parents=True, exist_ok=True)
            for source in (output_archive, comparison):
                if source.exists():
                    shutil.copy2(source, destination / source.name)
            upload_output()


if __name__ == "__main__":
    main()
