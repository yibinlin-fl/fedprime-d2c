#!/usr/bin/python
#coding=utf-8
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ARCHIVE = "cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz"
CONFIGS = {
    "control": "configs/openi_v100_rahfl_val_cle_v2_probe.yaml",
    "candidate": "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml",
}
EXPERIMENTS = {
    "control": "probe_rahfl_val_cle_v2_alpha05_gamma09_seed0_split0",
    "candidate": "probe_fedease_pew_asymhfl_val_cle_v2_alpha05_gamma09_seed0_split0",
}
COMPARISON = "outputs/strict_pew_asymhfl_val_comparison.json"


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], environment: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=environment)


def prepare_c2net():
    try:
        from c2net.context import prepare

        context = prepare()
        log(f"c2net dataset_path = {getattr(context, 'dataset_path', '')}")
        log(f"c2net output_path  = {getattr(context, 'output_path', '')}")
        return context
    except Exception as exc:  # pragma: no cover - OpenI integration.
        log(f"[warning] c2net prepare failed or unavailable: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI strict PEW + AsymHFL-val A/B probe.")
    parser.add_argument("--mode", choices=["control", "candidate", "both"], default="both")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def candidate_roots(args: argparse.Namespace, context) -> list[Path]:
    roots: list[Path] = []
    for raw in (
        args.data_source,
        os.environ.get("DATA_SOURCE", ""),
        getattr(context, "dataset_path", "") if context is not None else "",
        "/tmp/dataset",
        "/dataset",
        "/cache/dataset",
        "/tmp",
        "/cache",
    ):
        if raw:
            path = Path(raw)
            if path.exists() and path not in roots:
                roots.append(path)
    return roots


def find_dataset(roots: list[Path]) -> Path:
    for root in roots:
        direct = root / DATASET_ARCHIVE
        if direct.is_file():
            return direct
        matches = list(root.rglob(DATASET_ARCHIVE))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {DATASET_ARCHIVE} under: "
        + ", ".join(str(path) for path in roots)
    )


def package_outputs(methods: list[str]) -> Path:
    archive = ROOT / "strict_pew_asymhfl_val_probe_outputs.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for method in methods:
            directory = ROOT / "outputs" / EXPERIMENTS[method]
            if directory.exists():
                handle.add(directory, arcname=f"outputs/{EXPERIMENTS[method]}")
            handle.add(ROOT / CONFIGS[method], arcname=CONFIGS[method])
        for relative in (
            COMPARISON,
            "outputs/partitions/strict_cle_v2_alpha05_gamma09_seed0_split0.npz",
            "STRICT_PEW_ASYMHFL_VAL_OPENI_RUN_ZH.md",
            "CURRENT_PROJECT_MEMORY.md",
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
    log("===== Strict PEW + AsymHFL-val A/B probe =====")
    log(f"Repository: {ROOT}")
    log(f"Methods: {methods}")
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
            [sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."],
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
                "scripts/analyze_strict_pew_asymhfl_probe.py",
                "--control", str(ROOT / "outputs" / EXPERIMENTS["control"]),
                "--candidate", str(ROOT / "outputs" / EXPERIMENTS["candidate"]),
                "--output", str(ROOT / COMPARISON),
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
    log("===== Strict PEW + AsymHFL-val probe complete =====")


if __name__ == "__main__":
    main()
