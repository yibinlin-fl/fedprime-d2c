#!/usr/bin/python
# coding=utf-8
"""One-factor-at-a-time PEW/BER/CDep sensitivity screening on frozen CLE seed0."""
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
from scripts.openi_strict_pew_asymhfl_entry import candidate_roots, log, prepare_c2net, run


ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_sensitivity"
DATASET_ARCHIVE = "cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz"
ARMS = {
    "base": {},
    "pew_t045": {"pew.unknown_threshold": 0.45},
    "pew_t065": {"pew.unknown_threshold": 0.65},
    "ber_g000": {"ber.support_gamma": 0.0},
    "ber_g100": {"ber.support_gamma": 1.0},
    "cdep_l001": {"cdep.lambda": 0.01},
    "cdep_l010": {"cdep.lambda": 0.10},
}


def build_configs() -> dict[str, dict]:
    template = load_config(ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml")
    configs = {}
    for arm, overrides in ARMS.items():
        config = copy.deepcopy(template)
        config["experiment_name"] = f"cle_sensitivity_{arm}_seed0_12round"
        config["train"]["rounds"] = 12
        config["method"]["strict_fit_audit"]["split_path"] = (
            "outputs/partitions/strict_cle_v2_seed0_split0.npz"
        )
        config["method"]["fedease"]["pew"]["checkpoint"] = (
            "outputs/pew_checkpoints/cle_sensitivity_shared_seed0.pt"
        )
        for key, value in overrides.items():
            section, field = key.split(".")
            config["method"]["fedease"][section][field] = value
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
    parser = argparse.ArgumentParser(description="OpenI CLE PEW/BER/CDep sensitivity.")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def select_arms(raw: str) -> list[str]:
    selected = list(ARMS) if raw == "all" else [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(ARMS))
    if unknown:
        raise ValueError(f"Unknown sensitivity arms: {unknown}")
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
        run([sys.executable, "scripts/import_cle_data.py", "--source", str(find_archive(candidate_roots(args, context))), "--destination", "."], environment)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    configs = build_configs()
    selected = select_arms(args.arms)
    paths = {}
    for arm, config in configs.items():
        path = GENERATED_ROOT / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        paths[arm] = path
    for arm in selected:
        run([sys.executable, "scripts/check_environment.py", "--config", str(paths[arm])], environment)
        if not args.skip_train:
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(paths[arm])], environment)
    report = ROOT / "outputs/cle_sensitivity_report.json"
    available = [arm for arm in selected if (ROOT / "outputs" / configs[arm]["experiment_name"] / "metrics.csv").is_file()]
    if available:
        command = [sys.executable, "scripts/analyze_cle_diagnostics.py"]
        for arm in available:
            command.extend(["--run", f"{arm}={ROOT / 'outputs' / configs[arm]['experiment_name']}"])
        command.extend(["--output", str(report)])
        run(command, environment)
    archive = ROOT / "outputs/cle_sensitivity_12round_outputs.tar.gz"
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
        shutil.copy2(archive, destination / archive.name)
        upload_output()


if __name__ == "__main__":
    main()
