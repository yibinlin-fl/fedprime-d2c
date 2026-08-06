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


GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_stress_grid"
GRID = {
    "a01_g05": (0.1, 0.5, "alpha01_gamma05_seed0_split0"),
    "a01_g09": (0.1, 0.9, "alpha01_gamma09_seed0_split0"),
    "a05_g05": (0.5, 0.5, "alpha05_gamma05_seed0_split0"),
    "a05_g09": (0.5, 0.9, "alpha05_gamma09_seed0_split0"),
    "a10_g05": (1.0, 0.5, "alpha10_gamma05_seed0_split0"),
    "a10_g09": (1.0, 0.9, "alpha10_gamma09_seed0_split0"),
}


def dataset_archive_name(cell: str) -> str:
    return f"cle_hfl_v2_prepared_{GRID[cell][2]}.tar.gz"


def find_archive(cell: str, roots: list[Path]) -> Path:
    name = dataset_archive_name(cell)
    for root in roots:
        direct = root / name
        if direct.is_file():
            return direct
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name} under {roots}")


def build_configs(cell: str) -> dict[str, dict]:
    _, _, scenario = GRID[cell]
    templates = {
        "control": load_config(ROOT / "configs/openi_v100_rahfl_val_cle_v2_probe.yaml"),
        "candidate": load_config(ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml"),
    }
    configs = {}
    for arm, template in templates.items():
        config = copy.deepcopy(template)
        config["experiment_name"] = f"cle_stress_{cell}_{arm}_seed0_12round"
        config["data"]["private_root"] = f"RAHFL-master/Dataset/cifar_10_cle_v2/{scenario}"
        config["method"]["strict_fit_audit"]["split_path"] = (
            f"outputs/partitions/strict_cle_v2_{scenario}.npz"
        )
        config["method"]["strict_fit_audit"]["seed"] = 0
        config["train"]["rounds"] = 12
        config["checkpoints"]["save_final"] = False
        if arm == "candidate":
            config["method"]["fedease"]["pew"]["checkpoint"] = (
                "outputs/pew_checkpoints/cle_stress_shared_seed0.pt"
            )
        configs[arm] = config
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI 2x3 alpha/gamma CLE stress grid.")
    parser.add_argument("--cells", default="all")
    parser.add_argument("--mode", choices=["control", "candidate", "both"], default="both")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def select_cells(raw: str) -> list[str]:
    cells = list(GRID) if raw == "all" else [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(cells) - set(GRID))
    if unknown:
        raise ValueError(f"Unknown stress cells: {unknown}")
    return cells


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    cells = select_cells(args.cells)
    arms = ["control", "candidate"] if args.mode == "both" else [args.mode]
    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    roots = candidate_roots(args, context)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    produced = []
    for cell in cells:
        if not args.skip_import:
            run([
                sys.executable, "scripts/import_cle_data.py",
                "--source", str(find_archive(cell, roots)), "--destination", ".",
            ], environment)
        configs = build_configs(cell)
        paths = {}
        for arm, config in configs.items():
            path = GENERATED_ROOT / f"{cell}_{arm}.json"
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            paths[arm] = path
        for arm in arms:
            run([sys.executable, "scripts/check_environment.py", "--config", str(paths[arm])], environment)
            if not args.skip_train:
                run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(paths[arm])], environment)
        comparison = ROOT / f"outputs/cle_stress_{cell}_comparison.json"
        if all((ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").exists() for arm in ("control", "candidate")):
            run([
                sys.executable, "scripts/analyze_strict_pew_asymhfl_probe.py",
                "--control", str(ROOT / "outputs" / configs["control"]["experiment_name"]),
                "--candidate", str(ROOT / "outputs" / configs["candidate"]["experiment_name"]),
                "--output", str(comparison),
            ], environment)
            produced.append(comparison)
    archive = ROOT / "outputs/cle_stress_grid_12round_outputs.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in GENERATED_ROOT.glob("*.json"):
            handle.add(path, arcname=f"configs/generated/{path.name}")
        for path in produced:
            handle.add(path, arcname=f"outputs/{path.name}")
    log(f"Wrote {archive}")
    if not args.no_upload and context is not None:
        from c2net.context import upload_output

        destination = Path(context.output_path)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, destination / archive.name)
        upload_output()


if __name__ == "__main__":
    main()
