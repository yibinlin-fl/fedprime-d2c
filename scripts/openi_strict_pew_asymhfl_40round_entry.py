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
    0: {
        "control": "configs/openi_v100_rahfl_val_cle_v2_40round_probe.yaml",
        "candidate": "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_40round_probe.yaml",
    },
    1: {
        "control": "configs/openi_v100_rahfl_val_cle_v2_40round_trainseed1_probe.yaml",
        "candidate": "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_40round_trainseed1_probe.yaml",
    },
    2: {
        "control": "configs/openi_v100_rahfl_val_cle_v2_40round_trainseed2_probe.yaml",
        "candidate": "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_40round_trainseed2_probe.yaml",
    },
}
EXPERIMENTS = {
    0: {
        "control": "durability40_rahfl_val_cle_v2_alpha05_gamma09_seed0_split0",
        "candidate": "durability40_fedease_pew_asymhfl_val_cle_v2_alpha05_gamma09_seed0_split0",
    },
    1: {
        "control": "durability40_rahfl_val_cle_v2_alpha05_gamma09_seed0_split0_trainseed1",
        "candidate": "durability40_fedease_pew_asymhfl_val_cle_v2_alpha05_gamma09_seed0_split0_trainseed1",
    },
    2: {
        "control": "durability40_rahfl_val_cle_v2_alpha05_gamma09_seed0_split0_trainseed2",
        "candidate": "durability40_fedease_pew_asymhfl_val_cle_v2_alpha05_gamma09_seed0_split0_trainseed2",
    },
}
GUIDE = "docs/experiments/current/STRICT_PEW_ASYMHFL_VAL_40ROUND_OPENI_RUN_ZH.md"


def comparison_path(train_seed: int) -> str:
    if train_seed == 0:
        return "outputs/strict_pew_asymhfl_val_40round_seed0_comparison.json"
    return f"outputs/strict_pew_asymhfl_val_40round_trainseed{train_seed}_comparison.json"


def archive_name(train_seed: int) -> str:
    if train_seed == 0:
        return "strict_pew_asymhfl_val_40round_seed0_outputs.tar.gz"
    return f"strict_pew_asymhfl_val_40round_trainseed{train_seed}_outputs.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenI strict PEW + AsymHFL-val 40-round durability probe."
    )
    parser.add_argument("--mode", choices=["control", "candidate", "both"], default="both")
    parser.add_argument(
        "--train_seed",
        type=int,
        choices=sorted(CONFIGS),
        default=0,
        help="Training/init seed. CLE scenario and strict fit/audit split remain seed0/split0.",
    )
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def package_outputs(methods: list[str], train_seed: int) -> Path:
    configs = CONFIGS[train_seed]
    experiments = EXPERIMENTS[train_seed]
    comparison = comparison_path(train_seed)
    archive = ROOT / archive_name(train_seed)
    with tarfile.open(archive, "w:gz") as handle:
        for method in methods:
            directory = ROOT / "outputs" / experiments[method]
            if directory.exists():
                handle.add(directory, arcname=f"outputs/{experiments[method]}")
            handle.add(ROOT / configs[method], arcname=configs[method])
        for relative in (
            comparison,
            "outputs/partitions/strict_cle_v2_alpha05_gamma09_seed0_split0.npz",
            GUIDE,
            "docs/project/CURRENT_PROJECT_MEMORY.md",
        ):
            source = ROOT / relative
            if source.exists():
                handle.add(source, arcname=relative)
    log(f"Wrote {archive}")
    return archive


def upload_outputs(context, archive: Path, comparison: str) -> None:
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
    for source in (archive, ROOT / comparison):
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
    configs = CONFIGS[args.train_seed]
    experiments = EXPERIMENTS[args.train_seed]
    comparison = comparison_path(args.train_seed)
    log("===== Strict PEW + AsymHFL-val 40-round durability probe =====")
    log(f"Repository: {ROOT}")
    log(f"Methods: {methods}")
    log(f"Training seed: {args.train_seed}")
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
        config = configs[method]
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
                str(ROOT / "outputs" / experiments["control"]),
                "--candidate",
                str(ROOT / "outputs" / experiments["candidate"]),
                "--output",
                str(ROOT / comparison),
            ],
            environment,
        )

    if not args.skip_summary:
        log("===== Summarizing outputs =====")
        run([sys.executable, "scripts/summarize_results.py", "--outputs", "outputs"], environment)

    archive = package_outputs(methods, args.train_seed)
    if not args.no_upload:
        log("===== Uploading outputs through c2net =====")
        upload_outputs(context, archive, comparison)
    log("===== Strict PEW + AsymHFL-val 40-round durability probe complete =====")


if __name__ == "__main__":
    main()
