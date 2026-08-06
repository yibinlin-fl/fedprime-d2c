#!/usr/bin/python
# coding=utf-8
"""OpenI entry for the CIFAR-100-private/CIFAR-10-public CLE screening A/B."""
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


DATASET_NAME = "alpha05_gamma09_seed0_split0"
DATASET_ARCHIVE = f"cle_hfl_v2_prepared_cifar100_{DATASET_NAME}.tar.gz"
GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_cifar100"


def build_configs(rounds: int = 12) -> dict[str, dict]:
    templates = {
        "control": load_config(ROOT / "configs/openi_v100_rahfl_val_cle_v2_40round_probe.yaml"),
        "candidate": load_config(ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_40round_probe.yaml"),
    }
    configs = {}
    for arm, template in templates.items():
        config = copy.deepcopy(template)
        config["seed"] = 0
        config["experiment_name"] = f"cle_cifar100_12round_{arm}_seed0"
        config["data"].update({
            "private_dataset": "cifar100",
            "private_root": f"RAHFL-master/Dataset/cifar_100_cle_v2/{DATASET_NAME}",
            "public_dataset": "cifar10",
            "public_root": "RAHFL-master/Dataset/cifar_10",
            "num_classes": 100,
        })
        config["train"]["rounds"] = int(rounds)
        config["method"]["strict_fit_audit"].update({
            "split_path": "outputs/partitions/strict_cle_cifar100_seed0_split0.npz",
            "seed": 0,
        })
        config["checkpoints"]["save_final"] = False
        if arm == "candidate":
            config["method"]["fedease"]["pew"]["checkpoint"] = (
                "outputs/pew_checkpoints/cle_cifar100_public_cifar10_seed0.pt"
            )
        configs[arm] = config
    return configs


def find_archive(roots: list[Path]) -> Path:
    for root in roots:
        direct = root / DATASET_ARCHIVE
        if direct.is_file():
            return direct
        matches = list(root.rglob(DATASET_ARCHIVE))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {DATASET_ARCHIVE} under {roots}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CIFAR-100 CLE 12-round strict A/B.")
    parser.add_argument("--mode", choices=("control", "candidate", "both"), default="both")
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
        archive = find_archive(candidate_roots(args, context))
        run([sys.executable, "scripts/import_cle_data.py", "--source", str(archive), "--destination", "."], environment)

    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    configs = build_configs()
    paths = {}
    for arm, config in configs.items():
        path = GENERATED_ROOT / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        paths[arm] = path
    arms = ("control", "candidate") if args.mode == "both" else (args.mode,)
    for arm in arms:
        run([sys.executable, "scripts/check_environment.py", "--config", str(paths[arm])], environment)
        if not args.skip_train:
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(paths[arm])], environment)

    comparison = ROOT / "outputs/cle_cifar100_12round_comparison.json"
    if all((ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").is_file() for arm in ("control", "candidate")):
        run([
            sys.executable, "scripts/analyze_strict_pew_asymhfl.py",
            "--control", str(ROOT / "outputs" / configs["control"]["experiment_name"]),
            "--candidate", str(ROOT / "outputs" / configs["candidate"]["experiment_name"]),
            "--output", str(comparison),
        ], environment)

    archive = ROOT / "outputs/cle_cifar100_12round_outputs.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for arm in arms:
            output = ROOT / "outputs" / configs[arm]["experiment_name"]
            if output.exists():
                handle.add(output, arcname=f"outputs/{output.name}")
            handle.add(paths[arm], arcname=f"configs/generated/{arm}.json")
        if comparison.is_file():
            handle.add(comparison, arcname=f"outputs/{comparison.name}")
    log(f"Wrote {archive}")
    if not args.no_upload and context is not None:
        from c2net.context import upload_output

        destination = Path(context.output_path)
        destination.mkdir(parents=True, exist_ok=True)
        for source in (archive, comparison):
            if source.exists():
                shutil.copy2(source, destination / source.name)
        upload_output()


if __name__ == "__main__":
    main()
