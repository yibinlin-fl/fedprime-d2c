# coding=utf-8
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_ARCHIVE = "cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz"
INPUT_DIRECTORY = "cle_public_canonicalization_phase_b0_seed0_inputs"
INPUT_BYTES = 535256689
INPUT_SHA256 = "DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE K1-A head-only SDMN.")
    parser.add_argument("--mode", choices=("inspect", "smoke", "calibration", "formal"), default="smoke")
    parser.add_argument("--data-source", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def prepare_c2net():
    try:
        from c2net.context import prepare

        context = prepare()
        log(f"c2net dataset_path = {getattr(context, 'dataset_path', '')}")
        log(f"c2net output_path  = {getattr(context, 'output_path', '')}")
        return context
    except Exception as exc:  # pragma: no cover
        log(f"[warning] c2net prepare failed or unavailable: {exc}")
        return None


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
    raise FileNotFoundError(f"Could not find {INPUT_ARCHIVE}")


def safe_extract(source: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are forbidden: {member.name}")
        handle.extractall(destination, members=members)


def verify_manifest(input_root: Path) -> None:
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = input_root / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"input file mismatch: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"input hash mismatch: {path}")
    log(f"[integrity] verified {len(manifest['files'])} input files")


def run(command: list[str], environment: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=environment)


def package_outputs(output_dir: Path, mode: str) -> Path:
    archive = ROOT / "outputs" / f"cle_k1_sdmn_headonly_seed0_{mode}_outputs.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_dir, arcname=f"outputs/{output_dir.name}")
        spec = ROOT / "docs/experiments/current/CLE_K1_SDMN_HEADONLY_OPENI_ZH.md"
        if spec.is_file():
            handle.add(spec, arcname="docs/experiments/current/CLE_K1_SDMN_HEADONLY_OPENI_ZH.md")
    return archive


def upload_outputs(context, paths: list[Path]) -> None:
    if context is None:
        log("[warning] c2net context unavailable; skip upload")
        return
    from c2net.context import upload_output

    destination_root = Path(context.output_path)
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in paths:
        if source.is_file():
            destination = destination_root / source.name
            shutil.copy2(source, destination)
            log(f"Copied {source} -> {destination}")
    upload_output()


def main() -> None:
    args = parse_args()
    if args.mode == "formal":
        raise RuntimeError(
            "Formal K1-A remains locked. Run smoke, then calibration, audit its manifest, "
            "and freeze the maximum surgery-step budget first."
        )
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_input(candidate_roots(args, context))
    if source.stat().st_size != INPUT_BYTES or sha256_file(source) != INPUT_SHA256:
        raise ValueError("Phase-B0 input archive bytes/hash mismatch")
    extraction = ROOT / "local_runs/cle_k1_sdmn_openi_input"
    safe_extract(source, extraction)
    input_root = extraction / INPUT_DIRECTORY
    verify_manifest(input_root)
    output_dir = ROOT / "outputs" / f"cle_k1_sdmn_headonly_seed0_{args.mode}"
    command = [
        sys.executable,
        "scripts/run_cle_k1_sdmn_headonly.py",
        "--mode",
        args.mode,
        "--public-root",
        str(input_root / "public"),
        "--checkpoint-root",
        str(input_root / "checkpoints"),
        "--evaluation-root",
        str(input_root / "evaluation"),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    run(command, environment)
    run_manifest = {
        "protocol": "cle_k1a_sdmn_headonly_openi_v1",
        "mode": args.mode,
        "scientific_decision_allowed": False,
        "input_archive": source.name,
        "input_bytes": source.stat().st_size,
        "input_sha256": sha256_file(source),
        "full_training_performed": False,
        "communication_modified": False,
        "formal_locked": True,
    }
    manifest_path = output_dir / "openi_run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    archive = package_outputs(output_dir, args.mode)
    if not args.no_upload:
        upload_outputs(
            context,
            [
                archive,
                manifest_path,
                output_dir / "result.json",
                output_dir / "FINAL_REPORT_ZH.md",
                output_dir / "artifact_manifest.json",
            ],
        )
    log(f"===== CLE K1-A {args.mode} complete =====")


if __name__ == "__main__":
    main()
