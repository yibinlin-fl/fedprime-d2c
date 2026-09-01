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
ARMS = ("h0", "h9", "l0", "l9")


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE K0-A public-carrier oracle.")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--data-source", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


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
                raise ValueError(f"Links are not allowed in K0-A input: {member.name}")
        handle.extractall(destination, members=members)


def verify_manifest(input_root: Path) -> dict[str, object]:
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "cle_public_canonicalization_phase_b0_seed0_input":
        raise ValueError("Unexpected reused Phase-B0 input protocol")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise TypeError("input manifest files must be a list")
    expected = {
        "evaluation/test_images.npy",
        "evaluation/test_labels.npy",
        "public/cifar-100-python.tar.gz",
        *{
            f"checkpoints/{arm}/client_{client_id}.pt"
            for arm in ARMS
            for client_id in range(4)
        },
    }
    recorded = {str(row["path"]) for row in files}
    if recorded != expected:
        raise ValueError(f"input manifest file set mismatch: {sorted(recorded ^ expected)}")
    for row in files:
        path = input_root / str(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Byte-size mismatch: {path}")
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"SHA256 mismatch: {path}")
    log(f"[integrity] verified {len(files)} reused input files")
    return manifest


def package_outputs(output_dir: Path, mode: str) -> Path:
    archive = ROOT / "outputs" / f"cle_public_carrier_k0a_seed0_{mode}_outputs.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_dir, arcname=f"outputs/{output_dir.name}")
        spec = ROOT / "docs/experiments/current/CLE_PUBLIC_CARRIER_K0A_OPENI_ZH.md"
        if spec.is_file():
            handle.add(spec, arcname="docs/experiments/current/CLE_PUBLIC_CARRIER_K0A_OPENI_ZH.md")
    log(f"Wrote {archive}")
    return archive


def upload_outputs(context, paths: list[Path]) -> None:
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
    log(f"Reused Phase-B0 input archive: {source}")
    if source.stat().st_size != INPUT_BYTES or sha256_file(source) != INPUT_SHA256:
        raise ValueError(
            f"input archive mismatch: expected bytes={INPUT_BYTES}, sha256={INPUT_SHA256}"
        )
    extraction_root = ROOT / "local_runs/cle_public_carrier_k0a_openi_input"
    safe_extract(source, extraction_root)
    input_root = extraction_root / INPUT_DIRECTORY
    manifest = verify_manifest(input_root)
    log(f"[integrity] source arms={manifest['arms']} checkpoint_kind={manifest['checkpoint_kind']}")

    experiment = f"cle_public_carrier_k0a_seed0_{args.mode}"
    output_dir = ROOT / "outputs" / experiment
    command = [
        sys.executable,
        "scripts/analyze_cle_public_carrier_k0a.py",
        "--mode",
        args.mode,
        "--public-root",
        str(input_root / "public"),
        "--checkpoint-root",
        str(input_root / "checkpoints"),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    log(f"===== Running CLE K0-A {args.mode} =====")
    run(command, environment)

    run_manifest = {
        "protocol": "cle_public_carrier_k0a_openi_20260901",
        "mode": args.mode,
        "scientific_decision_allowed": args.mode == "formal",
        "input_archive": source.name,
        "input_archive_bytes": source.stat().st_size,
        "input_archive_sha256": sha256_file(source),
        "reused_phase_b0_input": True,
        "training_performed": False,
    }
    run_manifest_path = output_dir / "openi_run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    archive = package_outputs(output_dir, args.mode)

    if not args.no_upload:
        upload_paths = [archive, run_manifest_path]
        for path in (
            output_dir / "result.json",
            output_dir / "final_report.md",
            output_dir / "metrics/arm_metrics.csv",
            output_dir / "metrics/per_client_metrics.csv",
        ):
            if path.is_file():
                upload_paths.append(path)
        log("===== Uploading K0-A outputs through c2net =====")
        upload_outputs(context, upload_paths)
    log(f"===== CLE K0-A {args.mode} complete =====")


if __name__ == "__main__":
    main()
