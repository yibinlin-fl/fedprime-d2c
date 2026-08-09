#!/usr/bin/python
# coding=utf-8
"""Run the matched hard PEW+BER versus Multi-label PEW+Soft-BER screen."""
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


TEMPLATE = ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml"
GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_multilabel_softber"
ARCHIVE_NAME = "cle_multilabel_softber_seed0_12round_outputs.tar.gz"
ARM_ORDER = ("hard_pew_ber", "multilabel_softber")


def build_configs() -> dict[str, dict]:
    base = load_config(TEMPLATE)
    base["seed"] = 0
    base["train"]["rounds"] = 12
    base["checkpoints"]["save_final"] = False

    hard = copy.deepcopy(base)
    hard["experiment_name"] = "cle_hard_pew_ber_seed0_12round_paired"
    hard_pew = hard["method"]["fedease"]["pew"]
    hard_pew["label_mode"] = "hard"
    hard_pew["checkpoint"] = "outputs/pew_checkpoints/cle_hard_pew_ber_seed0_paired.pt"
    hard_pew["reuse_checkpoint"] = True
    hard["method"]["fedease"]["ber"]["assignment"] = "hard"

    soft = copy.deepcopy(base)
    soft["experiment_name"] = "cle_multilabel_pew_softber_seed0_12round"
    soft_pew = soft["method"]["fedease"]["pew"]
    soft_pew["label_mode"] = "multi_label"
    soft_pew["checkpoint"] = "outputs/pew_checkpoints/cle_multilabel_pew_seed0.pt"
    soft_pew["reuse_checkpoint"] = True
    soft["method"]["fedease"]["ber"]["assignment"] = "soft"
    soft["method"]["fedease"]["ber"]["min_group_count"] = 2.0
    return {"hard_pew_ber": hard, "multilabel_softber": soft}


def write_configs(configs: dict[str, dict]) -> dict[str, Path]:
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for arm in ARM_ORDER:
        path = GENERATED_ROOT / f"{arm}.json"
        path.write_text(json.dumps(configs[arm], indent=2), encoding="utf-8")
        paths[arm] = path
    return paths


def compare(report: dict) -> dict:
    hard = report["runs"]["hard_pew_ber"]["last_five"]
    soft = report["runs"]["multilabel_softber"]["last_five"]
    delta = {key: float(soft[key]) - float(hard[key]) for key in ("avg_acc", "worst_acc", "wcca", "cfg")}
    gates = {
        "avg_gain": delta["avg_acc"] >= 0.5,
        "worst_noninferiority": delta["worst_acc"] >= 0.0,
        "wcca_noninferiority": delta["wcca"] >= 0.0,
        "cfg_noninferiority": delta["cfg"] <= 0.0,
    }
    return {
        "comparison_window": "last-five",
        "multilabel_softber_minus_hard": delta,
        "pre_registered_gates": gates,
        "pass": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI Multi-label PEW + Soft-BER paired screen.")
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
        run([sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."], environment)

    configs = build_configs()
    config_paths = write_configs(configs)
    outputs = {arm: ROOT / "outputs" / configs[arm]["experiment_name"] for arm in ARM_ORDER}
    for arm in ARM_ORDER:
        run([sys.executable, "scripts/check_environment.py", "--config", str(config_paths[arm])], environment)
        if not args.skip_train:
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_paths[arm])], environment)

    report_path = ROOT / "outputs/cle_multilabel_softber_report.json"
    decision_path = ROOT / "outputs/cle_multilabel_softber_decision.json"
    if all((outputs[arm] / "metrics.csv").is_file() for arm in ARM_ORDER):
        command = [sys.executable, "scripts/analyze_cle_diagnostics.py"]
        for arm in ARM_ORDER:
            command.extend(["--run", f"{arm}={outputs[arm]}"])
        command.extend(["--output", str(report_path)])
        run(command, environment)
        decision_path.write_text(
            json.dumps(compare(json.loads(report_path.read_text(encoding="utf-8"))), indent=2),
            encoding="utf-8",
        )

    archive = ROOT / "outputs" / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        for arm in ARM_ORDER:
            handle.add(config_paths[arm], arcname=f"configs/generated/{arm}.json")
            if outputs[arm].exists():
                handle.add(outputs[arm], arcname=f"outputs/{outputs[arm].name}")
        for artifact in (report_path, decision_path):
            if artifact.is_file():
                handle.add(artifact, arcname=f"outputs/{artifact.name}")
    log(f"Wrote {archive}")
    if not args.no_upload and context is not None:
        from c2net.context import upload_output

        destination = Path(context.output_path)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, destination / archive.name)
        upload_output()


if __name__ == "__main__":
    main()
