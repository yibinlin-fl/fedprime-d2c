#!/usr/bin/python
# coding=utf-8
"""Run matched PEW+BER control and shared-PEW CDep-v2 candidate on OpenI."""
from __future__ import annotations

import argparse
import copy
import hashlib
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


GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_cdep_v2_paired"
ARCHIVE_NAME = "cle_cdep_v2_paired_12round_outputs.tar.gz"
SHARED_CHECKPOINT = "outputs/pew_checkpoints/cle_cdep_v2_paired_seed0.pt"
ARM_NAMES = ("control", "candidate")


def _cdep_v2_config() -> dict:
    return {
        "enabled": True,
        "version": "v2",
        "projection_dim": 64,
        "lambda": 1.0,
        "buffer_size_per_group": 64,
        "min_confidence": 0.20,
        "min_group_count": 4,
        "min_environments": 2,
        "warmup_rounds": 2,
        "ramp_rounds": 3,
        "eps": 1.0e-6,
    }


def build_configs() -> dict[str, dict]:
    template = load_config(ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml")
    common = copy.deepcopy(template)
    common["train"]["rounds"] = 12
    common["method"]["strict_fit_audit"]["split_path"] = (
        "outputs/partitions/strict_cle_v2_seed0_split0.npz"
    )
    common["method"]["fedease"]["pew"]["checkpoint"] = SHARED_CHECKPOINT
    common["method"]["fedease"]["pew"]["reuse_checkpoint"] = True

    control = copy.deepcopy(common)
    control["experiment_name"] = "cle_cdep_v2_paired_control_seed0_12round"
    control["method"]["fedease"]["cdep"] = {
        "enabled": False,
        "version": "v2",
    }

    candidate = copy.deepcopy(common)
    candidate["experiment_name"] = "cle_cdep_v2_paired_candidate_seed0_12round"
    candidate["method"]["fedease"]["cdep"] = _cdep_v2_config()
    return {"control": control, "candidate": candidate}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_shared_pew(outputs: dict[str, Path], *, num_clients: int = 4) -> dict:
    hashes: dict[str, dict[str, str]] = {}
    for arm, output in outputs.items():
        hashes[arm] = {}
        for client_id in range(int(num_clients)):
            path = output / "pew_predictions" / f"client_{client_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing PEW annotation: {path}")
            hashes[arm][f"client_{client_id}"] = _sha256(path)
    identical = all(
        hashes["control"][client] == hashes["candidate"][client]
        for client in hashes["control"]
    )
    result = {"byte_identical": identical, "sha256": hashes}
    if not identical:
        raise RuntimeError("Paired CDep-v2 arms did not reuse byte-identical PEW annotations")
    return result


def compare_last_five(control: dict[str, float], candidate: dict[str, float]) -> dict:
    keys = ("avg_acc", "worst_acc", "wcca", "cfg")
    delta = {key: float(candidate[key]) - float(control[key]) for key in keys}
    gates = {
        "avg_noninferiority": delta["avg_acc"] >= 0.0,
        "worst_noninferiority": delta["worst_acc"] >= 0.0,
        "wcca_noninferiority": delta["wcca"] >= 0.0,
        "cfg_improvement": delta["cfg"] <= -0.5,
    }
    return {
        "comparison": "candidate minus matched control, last-five",
        "control_last_five": {key: float(control[key]) for key in keys},
        "candidate_last_five": {key: float(candidate[key]) for key in keys},
        "candidate_minus_control": delta,
        "pre_registered_gates": gates,
        "pass": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI matched CDep-v2 paired experiment.")
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

    configs = build_configs()
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    config_paths: dict[str, Path] = {}
    outputs: dict[str, Path] = {}
    for arm in ARM_NAMES:
        path = GENERATED_ROOT / f"{arm}.json"
        path.write_text(json.dumps(configs[arm], indent=2), encoding="utf-8")
        config_paths[arm] = path
        outputs[arm] = ROOT / "outputs" / configs[arm]["experiment_name"]
        run([sys.executable, "scripts/check_environment.py", "--config", str(path)], environment)

    for arm in ARM_NAMES:
        if not args.skip_train:
            run(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_paths[arm])],
                environment,
            )

    report_path = ROOT / "outputs/cle_cdep_v2_paired_report.json"
    comparison_path = ROOT / "outputs/cle_cdep_v2_paired_comparison.json"
    if all((outputs[arm] / "metrics.csv").is_file() for arm in ARM_NAMES):
        run(
            [
                sys.executable,
                "scripts/analyze_cle_diagnostics.py",
                "--run",
                f"control={outputs['control']}",
                "--run",
                f"candidate={outputs['candidate']}",
                "--output",
                str(report_path),
            ],
            environment,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        comparison = compare_last_five(
            report["runs"]["control"]["last_five"],
            report["runs"]["candidate"]["last_five"],
        )
        comparison["pew_annotations"] = verify_shared_pew(outputs)
        comparison["shared_checkpoint"] = SHARED_CHECKPOINT
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        log(f"Matched CDep-v2 decision: {'PASS' if comparison['pass'] else 'FAIL'}")

    archive = ROOT / "outputs" / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        for arm in ARM_NAMES:
            handle.add(config_paths[arm], arcname=f"configs/generated/{arm}.json")
            if outputs[arm].exists():
                handle.add(outputs[arm], arcname=f"outputs/{outputs[arm].name}")
        checkpoint = ROOT / SHARED_CHECKPOINT
        if checkpoint.is_file():
            handle.add(checkpoint, arcname=f"outputs/pew_checkpoints/{checkpoint.name}")
        for artifact in (report_path, comparison_path):
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
