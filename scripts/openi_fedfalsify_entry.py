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
CONFIGS = [
    "configs/openi_v100_fedfalsify_fit_control_probe.yaml",
    "configs/openi_v100_fedfalsify_probe.yaml",
]
EXPERIMENTS = [
    "probe_fedfalsify_fit_control_alpha05_gamma09_seed0",
    "probe_fedfalsify_alpha05_gamma09_seed0",
]
DATASET_NAMES = (
    "cle_hfl_prepared_alpha05_gamma09_seed0",
    "fedease_cle_prepared_alpha05_gamma09_seed0",
)


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
    parser = argparse.ArgumentParser(description="OpenI FedFalsify strict A/B probe.")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def candidate_roots(args: argparse.Namespace, context) -> list[Path]:
    roots = []
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
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def find_dataset(roots: list[Path]) -> Path:
    for root in roots:
        for name in DATASET_NAMES:
            direct = root / f"{name}.tar.gz"
            if direct.is_file():
                return direct
            matches = list(root.rglob(f"{name}.tar.gz"))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Could not find the existing gamma=0.9 CLE-HFL archive under: "
        + ", ".join(str(path) for path in roots)
    )


def package_outputs() -> Path:
    archive_path = ROOT / "fedfalsify_probe_outputs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for experiment in EXPERIMENTS:
            directory = ROOT / "outputs" / experiment
            if directory.exists():
                archive.add(directory, arcname=f"outputs/{experiment}")
        for relative in (
            "outputs/fedfalsify_probe_comparison.json",
            "outputs/partitions/fedfalsify_v1_cle_alpha05_gamma09_seed0.npz",
            "FEDFALSIFY_AUDIT_GUIDE_ZH.md",
            "CURRENT_PROJECT_MEMORY.md",
        ):
            source = ROOT / relative
            if source.exists():
                archive.add(source, arcname=relative)
    log(f"Wrote {archive_path}")
    return archive_path


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
    for source in (
        archive,
        ROOT / "outputs" / "fedfalsify_probe_comparison.json",
    ):
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
    log("===== OpenI FedFalsify strict A/B probe =====")
    log(f"Repository: {ROOT}")
    for config in CONFIGS:
        log(f"Config: {config}")
    context = prepare_c2net()

    if not args.skip_install:
        log("===== Installing dependencies =====")
        run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            environment,
        )

    if not args.skip_import:
        roots = candidate_roots(args, context)
        for root in roots:
            log(f"Dataset search root: {root}")
        source = find_dataset(roots)
        log(f"Prepared dataset: {source}")
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

    log("===== Environment check =====")
    run([sys.executable, "scripts/check_environment.py", "--config", CONFIGS[0]], environment)

    if not args.skip_train:
        log("===== Running strict fit-only control then FedFalsify =====")
        run([sys.executable, "-u", "scripts/run_grid.py", *CONFIGS], environment)
        run(
            [
                sys.executable,
                "scripts/analyze_fedfalsify_probe.py",
                "--control",
                str(ROOT / "outputs" / EXPERIMENTS[0]),
                "--candidate",
                str(ROOT / "outputs" / EXPERIMENTS[1]),
                "--output",
                str(ROOT / "outputs" / "fedfalsify_probe_comparison.json"),
            ],
            environment,
        )

    if not args.skip_summary:
        log("===== Summarizing outputs =====")
        run(
            [sys.executable, "scripts/summarize_results.py", "--outputs", "outputs"],
            environment,
        )

    archive = package_outputs()
    if not args.no_upload:
        log("===== Uploading outputs through c2net =====")
        upload_outputs(context, archive)
    log("===== FedFalsify probe complete =====")


if __name__ == "__main__":
    main()
