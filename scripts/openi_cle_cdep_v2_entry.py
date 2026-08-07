#!/usr/bin/python
# coding=utf-8
"""Run the single pre-registered CDep-v2 arm on frozen CLE seed0."""
from __future__ import annotations

import argparse
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
from scripts.openi_cle_sensitivity_entry import find_archive
from scripts.openi_strict_pew_asymhfl_entry import candidate_roots, log, prepare_c2net, run


GENERATED_CONFIG = ROOT / "local_runs/generated_configs/cle_cdep_v2/cdep_v2.json"
ARCHIVE_NAME = "cle_cdep_v2_12round_outputs.tar.gz"
REFERENCE_LAST_FIVE = {
    "avg_acc": 34.6320,
    "worst_acc": 29.4280,
    "wcca": 7.2500,
    "cfg": 24.6400,
}


def build_config() -> dict:
    config = load_config(ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml")
    config["experiment_name"] = "cle_cdep_v2_seed0_12round"
    config["train"]["rounds"] = 12
    config["method"]["strict_fit_audit"]["split_path"] = (
        "outputs/partitions/strict_cle_v2_seed0_split0.npz"
    )
    config["method"]["fedease"]["pew"]["checkpoint"] = (
        "outputs/pew_checkpoints/cle_cdep_v2_seed0.pt"
    )
    config["method"]["fedease"]["cdep"] = {
        "enabled": True,
        "version": "v2",
        "projection_dim": 64,
        # The normalized centroid-shift objective is bounded and roughly one
        # to two orders smaller than the standardized v1 covariance proxy.
        "lambda": 1.0,
        "buffer_size_per_group": 64,
        # Slightly above uniform six-way confidence (1/6), while the continuous
        # score still downweights uncertain samples inside each centroid.
        "min_confidence": 0.20,
        "min_group_count": 4,
        "min_environments": 2,
        "warmup_rounds": 2,
        "ramp_rounds": 3,
        "eps": 1.0e-6,
    }
    return config


def compare_with_reference(last_five: dict[str, float]) -> dict:
    delta = {
        metric: float(last_five[metric]) - reference
        for metric, reference in REFERENCE_LAST_FIVE.items()
    }
    gates = {
        "avg_noninferiority": delta["avg_acc"] >= 0.0,
        "worst_noninferiority": delta["worst_acc"] >= 0.0,
        "wcca_noninferiority": delta["wcca"] >= 0.0,
        "cfg_improvement": delta["cfg"] <= -0.5,
    }
    return {
        "reference": "matched calibrated PEW+BER A1, seed0 last-five",
        "reference_last_five": REFERENCE_LAST_FIVE,
        "candidate_last_five": {key: float(last_five[key]) for key in REFERENCE_LAST_FIVE},
        "candidate_minus_reference": delta,
        "pre_registered_gates": gates,
        "pass": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE CDep-v2 single-arm run.")
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
        source = find_archive(candidate_roots(args, context))
        run(
            [sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."],
            environment,
        )

    config = build_config()
    GENERATED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
    run([sys.executable, "scripts/check_environment.py", "--config", str(GENERATED_CONFIG)], environment)
    if not args.skip_train:
        run(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(GENERATED_CONFIG)],
            environment,
        )

    output = ROOT / "outputs" / config["experiment_name"]
    report = ROOT / "outputs/cle_cdep_v2_report.json"
    comparison = ROOT / "outputs/cle_cdep_v2_comparison.json"
    metrics = output / "metrics.csv"
    if metrics.is_file():
        run(
            [
                sys.executable,
                "scripts/analyze_cle_diagnostics.py",
                "--run",
                f"cdep_v2={output}",
                "--output",
                str(report),
            ],
            environment,
        )
        report_data = json.loads(report.read_text(encoding="utf-8"))
        decision = compare_with_reference(report_data["runs"]["cdep_v2"]["last_five"])
        comparison.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        log(f"CDep-v2 pre-registered decision: {'PASS' if decision['pass'] else 'FAIL'}")

    archive = ROOT / "outputs" / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(GENERATED_CONFIG, arcname="configs/generated/cdep_v2.json")
        if output.exists():
            handle.add(output, arcname=f"outputs/{output.name}")
        for artifact in (report, comparison):
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
