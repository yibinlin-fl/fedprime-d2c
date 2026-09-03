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
    parser = argparse.ArgumentParser(description="OpenI K1-C CRSF checkpoint surgery.")
    parser.add_argument("--mode", choices=("inspect", "smoke", "calibration", "formal"), default="smoke")
    parser.add_argument("--data-source", default="")
    parser.add_argument("--calibration-manifest", default="configs/cle_k1_c_crsf_calibration_seed0.json")
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
    raise FileNotFoundError(INPUT_ARCHIVE)


def safe_extract(source: Path, destination: Path, *, include_evaluation: bool) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as handle:
        members = []
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are forbidden: {member.name}")
            if not include_evaluation and "evaluation" in Path(member.name).parts:
                continue
            members.append(member)
        handle.extractall(destination, members=members)


def verify_input(input_root: Path, *, include_evaluation: bool) -> dict[str, object]:
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    prefixes = ("checkpoints/", "public/", "evaluation/") if include_evaluation else ("checkpoints/", "public/")
    rows = [row for row in manifest["files"] if str(row["path"]).startswith(prefixes)]
    for row in rows:
        path = input_root / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"input file mismatch: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"input hash mismatch: {path}")
    log(f"[integrity] verified {len(rows)} input files")
    return manifest


def run(command: list[str], environment: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=environment)


def package(output_dir: Path, mode: str) -> Path:
    archive = ROOT / "outputs" / f"cle_k1_c_crsf_surgery_seed0_{mode}_outputs.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_dir, arcname=f"outputs/{output_dir.name}")
        spec = ROOT / "docs/experiments/current/CLE_K1_C_CRSF_SURGERY_OPENI_ZH.md"
        if spec.is_file():
            handle.add(spec, arcname="docs/experiments/current/CLE_K1_C_CRSF_SURGERY_OPENI_ZH.md")
    return archive


def upload(context, paths: list[Path]) -> None:
    if context is None:
        log("[warning] c2net context unavailable; skip upload")
        return
    from c2net.context import upload_output

    destination = Path(context.output_path)
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.is_file():
            target = destination / path.name
            shutil.copy2(path, target)
            log(f"Copied {path} -> {target}")
    upload_output()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_input(candidate_roots(args, context))
    if source.stat().st_size != INPUT_BYTES or sha256_file(source) != INPUT_SHA256:
        raise ValueError("Phase-B0 input archive bytes/hash mismatch")
    extraction = ROOT / "local_runs/cle_k1_c_crsf_openi_input"
    safe_extract(source, extraction, include_evaluation=args.mode == "formal")
    input_root = extraction / INPUT_DIRECTORY
    source_manifest = verify_input(input_root, include_evaluation=args.mode == "formal")
    output_dir = ROOT / "outputs" / f"cle_k1_c_crsf_surgery_seed0_{args.mode}"
    cache_dir = ROOT / "local_runs/cle_k1_c_crsf_openi_cache"
    command = [
        sys.executable,
        "scripts/run_cle_k1_c_crsf_surgery.py",
        "--mode",
        args.mode,
        "--public-root",
        str(input_root / "public"),
        "--checkpoint-root",
        str(input_root / "checkpoints"),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(cache_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.mode == "formal":
        calibration = (ROOT / args.calibration_manifest).resolve()
        if not calibration.is_file():
            raise FileNotFoundError(
                "Formal is locked until the audited calibration manifest is versioned at " + str(calibration)
            )
        command.extend(
            [
                "--evaluation-root",
                str(input_root / "evaluation"),
                "--calibration-manifest",
                str(calibration),
            ]
        )
    run(command, environment)
    run_manifest = {
        "protocol": "cle_k1_c_crsf_openi_v1",
        "mode": args.mode,
        "input_archive": source.name,
        "input_bytes": source.stat().st_size,
        "input_sha256": sha256_file(source),
        "source_checkpoint_kind": source_manifest.get("checkpoint_kind"),
        "evaluation_extracted": args.mode == "formal",
        "training_performed": False,
        "communication_modified": False,
        "full_checkpoints_written": False,
    }
    run_manifest_path = output_dir / "openi_run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    archive = package(output_dir, args.mode)
    if not args.no_upload:
        upload(
            context,
            [
                archive,
                run_manifest_path,
                output_dir / "result.json",
                output_dir / "calibration_manifest.json",
                output_dir / "independent_recomputation.json",
                output_dir / "FINAL_REPORT_ZH.md",
                output_dir / "artifact_manifest.json",
            ],
        )
    log(f"===== CLE K1-C {args.mode} complete =====")


if __name__ == "__main__":
    main()
