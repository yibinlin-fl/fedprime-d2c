#!/usr/bin/python
# coding=utf-8
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

try:
    from scripts.openi_strict_pew_asymhfl_entry import (
        ROOT,
        candidate_roots,
        find_dataset,
        log,
        prepare_c2net,
        run,
    )
except ModuleNotFoundError:  # Direct execution: Python puts scripts/ on sys.path.
    from openi_strict_pew_asymhfl_entry import (
        ROOT,
        candidate_roots,
        find_dataset,
        log,
        prepare_c2net,
        run,
    )


CONFIGS = {
    "control": "configs/openi_v100_rahfl_val_cle_v2_40round_probe.yaml",
    "candidate": "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_40round_probe.yaml",
}
EXPERIMENTS = {
    "control": "durability40_rahfl_val_cle_v2_alpha05_gamma09_seed0_split0",
    "candidate": "durability40_fedease_pew_asymhfl_val_cle_v2_alpha05_gamma09_seed0_split0",
}
COMPARISON = "outputs/strict_pew_asymhfl_val_40round_seed0_comparison.json"
ARCHIVE_NAME = "strict_pew_asymhfl_val_40round_seed0_outputs.tar.gz"
GUIDE = "docs/experiments/current/STRICT_PEW_ASYMHFL_VAL_40ROUND_OPENI_RUN_ZH.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenI strict PEW + AsymHFL-val 40-round seed-0 durability probe."
    )
    parser.add_argument("--mode", choices=["control", "candidate", "both"], default="both")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def package_outputs(methods: list[str]) -> Path:
    archive = ROOT / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        for method in methods:
            directory = ROOT / "outputs" / EXPERIMENTS[method]
            if directory.exists():
                handle.add(directory, arcname=f"outputs/{EXPERIMENTS[method]}")
            handle.add(ROOT / CONFIGS[method], arcname=CONFIGS[method])
        for relative in (
            COMPARISON,
            "outputs/partitions/strict_cle_v2_alpha05_gamma09_seed0_split0.npz",
            GUIDE,
            "docs/project/CURRENT_PROJECT_MEMORY.md",
        ):
            source = ROOT / relative
            if source.exists():
                handle.add(source, arcname=relative)
    log(f"Wrote {archive}")
    return archive


def upload_outputs(context, archive: Path) -> None:
    if context is None:
        log("[warning] c2net context unavailable; skip upload")
        return
    try:
        from c2net.context import upload_output
    except Exception as exc:  # pragma: no cover - OpenI integration.
        log(f"[warning] c2net upload_output unavailable: {exc}")
        return
    output_path = Path(context.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for source in (archive, ROOT / COMPARISON):
        if source.exists():
            destination = output_path / source.name
            shutil.copy2(source, destination)
            log(f"Copied {source} -> {destination}")
    upload_output()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    methods = ["control", "candidate"] if args.mode == "both" else [args.mode]
    log("===== Strict PEW + AsymHFL-val 40-round durability probe =====")
    log(f"Repository: {ROOT}")
    log(f"Methods: {methods}")
    log("Training seed: 0")
    log("Rounds: 40")
    log("CLE scenario/split: seed0/split0 (fixed)")
    context = prepare_c2net()

    if not args.skip_install:
        log("===== Installing dependencies =====")
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)

    if not args.skip_import:
        roots = candidate_roots(args, context)
        for root in roots:
            log(f"Dataset search root: {root}")
        source = find_dataset(roots)
        log(f"Prepared CLE-HFL v2 dataset: {source}")
        run(
            [
                sys.executable,
                "scripts/import_cle_data.py",
                "--source",
                str(source),
                "--destination",
                ".",
            ],
            environment,
        )

    for method in methods:
        config = CONFIGS[method]
        log(f"===== Environment check: {method} =====")
        run([sys.executable, "scripts/check_environment.py", "--config", config], environment)
        if not args.skip_train:
            log(f"===== Running {method} =====")
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", config], environment)

    if args.mode == "both" and not args.skip_train:
        run(
            [
                sys.executable,
                "scripts/analyze_strict_pew_asymhfl_40round.py",
                "--control",
                str(ROOT / "outputs" / EXPERIMENTS["control"]),
                "--candidate",
                str(ROOT / "outputs" / EXPERIMENTS["candidate"]),
                "--output",
                str(ROOT / COMPARISON),
            ],
            environment,
        )

    if not args.skip_summary:
        log("===== Summarizing outputs =====")
        run([sys.executable, "scripts/summarize_results.py", "--outputs", "outputs"], environment)

    archive = package_outputs(methods)
    if not args.no_upload:
        log("===== Uploading outputs through c2net =====")
        upload_outputs(context, archive)
    log("===== Strict PEW + AsymHFL-val 40-round durability probe complete =====")


if __name__ == "__main__":
    main()
