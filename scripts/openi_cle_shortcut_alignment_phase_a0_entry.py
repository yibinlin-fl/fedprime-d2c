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
INPUT_ARCHIVE = "cle_shortcut_alignment_phase_a0_seed0_inputs.tar.gz"
INPUT_DIRECTORY = "cle_shortcut_alignment_phase_a0_seed0_inputs"
EXPERIMENT = "cle_shortcut_alignment_phase_a0_seed0"
OUTPUT_ARCHIVE = "cle_shortcut_alignment_phase_a0_seed0_outputs.tar.gz"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE Shortcut Alignment Phase-A0 inference.")
    parser.add_argument("--data-source", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


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


def find_input(roots: list[Path]) -> Path:
    for root in roots:
        direct = root / INPUT_ARCHIVE
        if direct.is_file():
            return direct
        matches = list(root.rglob(INPUT_ARCHIVE))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {INPUT_ARCHIVE} under: " + ", ".join(str(path) for path in roots)
    )


def safe_extract(source: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in Phase-A0 input: {member.name}")
        handle.extractall(destination, members=members)


def package_outputs(output_dir: Path) -> Path:
    archive = ROOT / "outputs" / OUTPUT_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_dir, arcname=f"outputs/{EXPERIMENT}")
        spec = ROOT / "docs/experiments/current/CLE_SHORTCUT_ALIGNMENT_PHASE_A0_OPENI_ZH.md"
        if spec.is_file():
            handle.add(spec, arcname="docs/experiments/current/CLE_SHORTCUT_ALIGNMENT_PHASE_A0_OPENI_ZH.md")
    log(f"Wrote {archive}")
    return archive


def upload_outputs(context, output_dir: Path, archive: Path) -> None:
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
    paths = [archive, *sorted(output_dir.iterdir())]
    for source in paths:
        if source.is_file():
            destination = output_path / source.name
            shutil.copy2(source, destination)
            log(f"Copied {source} -> {destination}")
    upload_output()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()

    if not args.skip_install:
        log("===== Installing dependencies =====")
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)

    roots = candidate_roots(args, context)
    for root in roots:
        log(f"Dataset search root: {root}")
    source = find_input(roots)
    log(f"Phase-A0 input archive: {source}")
    extraction_root = ROOT / "local_runs/cle_shortcut_alignment_phase_a0_openi_input"
    safe_extract(source, extraction_root)
    input_root = extraction_root / INPUT_DIRECTORY
    if not (input_root / "manifest.json").is_file():
        raise FileNotFoundError(input_root / "manifest.json")

    output_dir = ROOT / "outputs" / EXPERIMENT
    log("===== Running zero-training paired inference =====")
    run(
        [
            sys.executable,
            "scripts/audit_cle_shortcut_alignment.py",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--bootstrap-samples",
            "2000",
            "--permutations",
            "1000",
        ],
        environment,
    )
    archive = package_outputs(output_dir)
    if not args.no_upload:
        log("===== Uploading outputs through c2net =====")
        upload_outputs(context, output_dir, archive)
    log("===== CLE Shortcut Alignment Phase-A0 complete =====")


if __name__ == "__main__":
    main()
