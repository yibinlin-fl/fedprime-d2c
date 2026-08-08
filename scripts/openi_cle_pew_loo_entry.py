#!/usr/bin/python
# coding=utf-8
"""Run RAHFL, standard PEW+BER, and strict PEW-LOO+BER on OpenI."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.corruptions import DEFAULT_UNSEEN_CORRUPTIONS
from fedprime.utils.config import load_config
from scripts.openi_strict_pew_asymhfl_entry import (
    candidate_roots,
    find_dataset,
    log,
    prepare_c2net,
    run,
)


CONTROL_TEMPLATE = ROOT / "configs/openi_v100_rahfl_val_cle_v2_probe.yaml"
CANDIDATE_TEMPLATE = ROOT / "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml"
GENERATED_ROOT = ROOT / "local_runs/generated_configs/cle_pew_loo"
ARCHIVE_NAME = "cle_pew_loo_12round_seed0_outputs.tar.gz"
ARM_ORDER = ("rahfl", "standard_pew_ber", "strict_loo_pew_ber")
STANDARD_CHECKPOINT = "outputs/pew_checkpoints/cle_pew_loo_standard_seed0.pt"
STRICT_LOO_CHECKPOINT = "outputs/pew_checkpoints/cle_pew_loo_strict_seed0.pt"
STRICT_LOO_OPERATORS = tuple(DEFAULT_UNSEEN_CORRUPTIONS)


def build_configs() -> dict[str, dict]:
    control = load_config(CONTROL_TEMPLATE)
    candidate = load_config(CANDIDATE_TEMPLATE)

    for config in (control, candidate):
        config["seed"] = 0
        config["train"]["rounds"] = 12
        config["checkpoints"]["save_final"] = False
        config["method"]["strict_fit_audit"]["split_path"] = (
            "outputs/partitions/strict_cle_v2_seed0_split0.npz"
        )

    standard = copy.deepcopy(candidate)
    standard["experiment_name"] = "cle_pew_loo_standard_pew_ber_seed0_12round"
    standard["method"]["fedease"]["cdep"] = {"enabled": False}
    standard["method"]["fedease"]["pew"]["checkpoint"] = STANDARD_CHECKPOINT
    standard["method"]["fedease"]["pew"]["reuse_checkpoint"] = True
    standard["method"]["fedease"]["pew"]["exclude_operators"] = []

    strict_loo = copy.deepcopy(standard)
    strict_loo["experiment_name"] = "cle_pew_loo_strict_pew_ber_seed0_12round"
    strict_loo["method"]["fedease"]["pew"]["checkpoint"] = STRICT_LOO_CHECKPOINT
    strict_loo["method"]["fedease"]["pew"]["exclude_operators"] = list(
        STRICT_LOO_OPERATORS
    )

    control["experiment_name"] = "cle_pew_loo_rahfl_seed0_12round"
    return {
        "rahfl": control,
        "standard_pew_ber": standard,
        "strict_loo_pew_ber": strict_loo,
    }


def write_configs(configs: dict[str, dict]) -> dict[str, Path]:
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for arm in ARM_ORDER:
        path = GENERATED_ROOT / f"{arm}.json"
        path.write_text(json.dumps(configs[arm], indent=2), encoding="utf-8")
        paths[arm] = path
    return paths


def audit_private_fit_holdout(private_root: str | Path) -> dict:
    root = Path(private_root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    operator_to_id = {
        str(name): int(operator_id)
        for name, operator_id in metadata["operator_to_id"].items()
    }
    missing = sorted(set(STRICT_LOO_OPERATORS) - set(operator_to_id))
    if missing:
        raise ValueError(f"Strict PEW-LOO operators missing from metadata: {missing}")

    held_out_ids = {operator_to_id[name] for name in STRICT_LOO_OPERATORS}
    counts: dict[str, dict[str, int]] = {}
    for client_root in sorted(root.glob("client_*")):
        operator_ids = np.load(client_root / "train_corruption_ids.npy").astype(np.int64)
        client_counts = {
            name: int(np.count_nonzero(operator_ids == operator_to_id[name]))
            for name in STRICT_LOO_OPERATORS
        }
        if any(client_counts.values()):
            raise RuntimeError(
                f"Strict PEW-LOO private fit leakage in {client_root.name}: {client_counts}"
            )
        if any(int(operator_id) in held_out_ids for operator_id in np.unique(operator_ids)):
            raise RuntimeError(f"Strict PEW-LOO held-out ID leakage in {client_root.name}")
        counts[client_root.name] = client_counts
    if not counts:
        raise FileNotFoundError(f"No private client directories found under {root}")
    return {
        "held_out_operators": list(STRICT_LOO_OPERATORS),
        "private_fit_counts": counts,
        "passed": True,
    }


def _delta(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(reference[key])
        for key in ("avg_acc", "worst_acc", "wcca", "cfg")
    }


def compare_last_five(report: dict) -> dict:
    runs = report["runs"]
    rahfl = runs["rahfl"]["last_five"]
    standard = runs["standard_pew_ber"]["last_five"]
    strict_loo = runs["strict_loo_pew_ber"]["last_five"]
    strict_vs_rahfl = _delta(rahfl, strict_loo)
    gates = {
        "avg_gain": strict_vs_rahfl["avg_acc"] >= 1.5,
        "worst_gain": strict_vs_rahfl["worst_acc"] >= 1.0,
        "wcca_noninferiority": strict_vs_rahfl["wcca"] >= 0.0,
        "cfg_improvement": strict_vs_rahfl["cfg"] <= -1.0,
    }
    return {
        "comparison_window": "last-five",
        "last_five": {
            arm: {
                key: float(runs[arm]["last_five"][key])
                for key in ("avg_acc", "worst_acc", "wcca", "cfg")
            }
            for arm in ARM_ORDER
        },
        "standard_minus_rahfl": _delta(rahfl, standard),
        "strict_loo_minus_rahfl": strict_vs_rahfl,
        "strict_loo_minus_standard": _delta(standard, strict_loo),
        "pre_registered_strict_loo_gates": gates,
        "pass": all(gates.values()),
    }


def verify_pew_protocol(outputs: dict[str, Path]) -> dict:
    standard_report = json.loads(
        (outputs["standard_pew_ber"] / "pew_private_report.json").read_text(
            encoding="utf-8"
        )
    )
    strict_report = json.loads(
        (outputs["strict_loo_pew_ber"] / "pew_private_report.json").read_text(
            encoding="utf-8"
        )
    )
    standard_exclusions = tuple(standard_report.get("excluded_public_operators", ()))
    strict_exclusions = tuple(strict_report.get("excluded_public_operators", ()))
    if standard_exclusions:
        raise RuntimeError(f"Standard PEW unexpectedly excluded {standard_exclusions}")
    if set(strict_exclusions) != set(STRICT_LOO_OPERATORS):
        raise RuntimeError(
            "Strict PEW-LOO exclusion mismatch: "
            f"expected={list(STRICT_LOO_OPERATORS)} actual={list(strict_exclusions)}"
        )
    leaked = {
        group: sorted(set(operators) & set(STRICT_LOO_OPERATORS))
        for group, operators in strict_report["public_operator_pools"].items()
    }
    leaked = {group: operators for group, operators in leaked.items() if operators}
    if leaked:
        raise RuntimeError(f"Strict PEW-LOO public operator leakage: {leaked}")
    return {
        "standard_excluded_public_operators": list(standard_exclusions),
        "strict_loo_excluded_public_operators": list(strict_exclusions),
        "strict_loo_public_operator_pools": strict_report["public_operator_pools"],
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI strict PEW leave-one-operator-out audit.")
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
        run(
            [sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."],
            environment,
        )

    configs = build_configs()
    config_paths = write_configs(configs)
    holdout_audit = audit_private_fit_holdout(configs["strict_loo_pew_ber"]["data"]["private_root"])
    audit_path = ROOT / "outputs/cle_pew_loo_protocol_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(holdout_audit, indent=2), encoding="utf-8")

    outputs = {
        arm: ROOT / "outputs" / configs[arm]["experiment_name"]
        for arm in ARM_ORDER
    }
    for arm in ARM_ORDER:
        run([sys.executable, "scripts/check_environment.py", "--config", str(config_paths[arm])], environment)
        if not args.skip_train:
            run(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(config_paths[arm])],
                environment,
            )

    report_path = ROOT / "outputs/cle_pew_loo_report.json"
    decision_path = ROOT / "outputs/cle_pew_loo_decision.json"
    if all((outputs[arm] / "metrics.csv").is_file() for arm in ARM_ORDER):
        command = [sys.executable, "scripts/analyze_cle_diagnostics.py"]
        for arm in ARM_ORDER:
            command.extend(["--run", f"{arm}={outputs[arm]}"])
        command.extend(["--output", str(report_path)])
        run(command, environment)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        decision = compare_last_five(report)
        decision["protocol_audit"] = verify_pew_protocol(outputs)
        decision["private_fit_holdout_audit"] = holdout_audit
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        log(f"Strict PEW-LOO decision: {'PASS' if decision['pass'] else 'FAIL'}")

    archive = ROOT / "outputs" / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        for arm in ARM_ORDER:
            handle.add(config_paths[arm], arcname=f"configs/generated/{arm}.json")
            if outputs[arm].exists():
                handle.add(outputs[arm], arcname=f"outputs/{outputs[arm].name}")
        for checkpoint_name in (STANDARD_CHECKPOINT, STRICT_LOO_CHECKPOINT):
            checkpoint = ROOT / checkpoint_name
            if checkpoint.is_file():
                handle.add(checkpoint, arcname=f"outputs/pew_checkpoints/{checkpoint.name}")
        for artifact in (audit_path, report_path, decision_path):
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
