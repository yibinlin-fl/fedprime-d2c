#!/usr/bin/python
# coding=utf-8
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ARCHIVE = "rahfl_coupling_phase_a_seed0_prepared.tar.gz"
CONFIGS = {
    "beta0": "configs/rahfl_coupling_phase_a_screen_beta0.yaml",
    "beta4": "configs/rahfl_coupling_phase_a_screen_beta4.yaml",
}
EXPERIMENTS = {
    "beta0": "rahfl_coupling_phase_a_screen_seed0_beta0",
    "beta4": "rahfl_coupling_phase_a_screen_seed0_beta4",
}
SUMMARY = "outputs/rahfl_coupling_phase_a_screen_seed0_summary.json"


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
    parser = argparse.ArgumentParser(description="OpenI RAHFL coupling Phase-A 10+10 screen.")
    parser.add_argument("--mode", choices=["beta0", "beta4", "both"], default="both")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
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


def safe_extract(source: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(source, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in prepared data: {member.name}")
        handle.extractall(destination, members=members)


def package_outputs(methods: list[str], suffix: str) -> Path:
    archive = ROOT / "outputs" / f"rahfl_coupling_phase_a_screen_seed0_{suffix}_outputs.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        for method in methods:
            directory = ROOT / "outputs" / EXPERIMENTS[method]
            if directory.exists():
                handle.add(directory, arcname=f"outputs/{EXPERIMENTS[method]}")
            handle.add(ROOT / CONFIGS[method], arcname=CONFIGS[method])
        for relative in (
            SUMMARY,
            "local_runs/rahfl_coupling_phase_a_seed0/experiment_manifest.json",
        ):
            source = ROOT / relative
            if source.exists():
                handle.add(source, arcname=relative)
    log(f"Wrote {archive}")
    return archive


def copy_to_openi_output(context, paths: list[Path]) -> None:
    if context is None:
        return
    output_path = Path(context.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for source in paths:
        if source.exists():
            destination = output_path / source.name
            shutil.copy2(source, destination)
            log(f"Copied {source} -> {destination}")


def upload_outputs(context) -> None:
    if context is None:
        log("[warning] c2net context unavailable; skip upload")
        return
    try:
        from c2net.context import upload_output
    except Exception as exc:  # pragma: no cover - OpenI integration.
        log(f"[warning] c2net upload_output unavailable: {exc}")
        return
    upload_output()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    methods = ["beta0", "beta4"] if args.mode == "both" else [args.mode]
    log("===== RAHFL corruption-label coupling Phase-A screen =====")
    log(f"Repository: {ROOT}")
    log(f"Conditions: {methods}")
    log("Frozen budget: 10 pretrain epochs + 10 communication rounds")
    context = prepare_c2net()

    if not args.skip_install:
        log("===== Installing dependencies =====")
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)

    if not args.skip_import:
        roots = candidate_roots(args, context)
        for root in roots:
            log(f"Dataset search root: {root}")
        source = find_dataset(roots)
        log(f"Prepared Phase-A data: {source}")
        safe_extract(source, ROOT)

    run([sys.executable, "scripts/verify_rahfl_coupling_phase_a.py"], environment)

    completed: list[str] = []
    archives: list[Path] = []
    for method in methods:
        config = CONFIGS[method]
        log(f"===== Environment check: {method} =====")
        run([sys.executable, "scripts/check_environment.py", "--config", config], environment)
        if not args.skip_train:
            log(f"===== Running {method} =====")
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", config], environment)
        completed.append(method)
        archive = package_outputs(completed, method)
        archives.append(archive)
        copy_to_openi_output(context, [archive])

    if args.mode == "both" and not args.skip_train:
        run(
            [
                sys.executable,
                "scripts/analyze_rahfl_coupling_phase_a.py",
                "--beta0", str(ROOT / "outputs" / EXPERIMENTS["beta0"]),
                "--beta4", str(ROOT / "outputs" / EXPERIMENTS["beta4"]),
                "--output", str(ROOT / SUMMARY),
            ],
            environment,
        )
        final_archive = package_outputs(methods, "both")
        archives.append(final_archive)
        copy_to_openi_output(context, [ROOT / SUMMARY, final_archive])

    if not args.no_upload:
        log("===== Uploading outputs through c2net =====")
        upload_outputs(context)
    log("===== Phase-A 10+10 screen complete =====")


if __name__ == "__main__":
    main()
