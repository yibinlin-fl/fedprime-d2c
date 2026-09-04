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
INPUT_ARCHIVE = "cle_cvrs_m0_seed0_inputs.tar.gz"
INPUT_DIRECTORY = "cle_cvrs_m0_seed0_inputs"
INPUT_BYTES = 109142359
INPUT_SHA256 = "E9427A55DBE2545AF9D5A1EBD8BEA5B18C41C84D7FE89D06674165F4109E3818"


def log(message: str) -> None:
    print(message, flush=True)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CVRS M0 cheap method gate")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="benchmark")
    parser.add_argument("--data-source", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--confirm-formal",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Formal cost approval; accepts a bare flag or explicit true in OpenI forms",
    )
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
    except Exception as exc:  # pragma: no cover - local execution
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
        if root.is_file() and root.name == INPUT_ARCHIVE:
            return root
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
    if manifest.get("protocol") != "cle_cvrs_m0_openi_input_v1":
        raise ValueError("unexpected CVRS M0 input protocol")
    verified = 0
    for row in manifest["files"]:
        relative = Path(str(row["path"]))
        if not include_evaluation and "evaluation" in relative.parts:
            continue
        path = input_root / relative
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"input file mismatch: {path}")
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"input hash mismatch: {path}")
        verified += 1
    log(f"[integrity] verified {verified} input files")
    return manifest


def run(command: list[str], environment: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=environment)


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


def package(output_dir: Path, mode: str) -> Path:
    archive = ROOT / "outputs" / f"cle_cvrs_m0_seed0_{mode}_outputs.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz", compresslevel=6) as handle:
        handle.add(output_dir, arcname=f"outputs/{output_dir.name}")
        for path in (
            ROOT / "configs/cle_cvrs_m0_seed0.json",
            ROOT / "docs/experiments/current/CLE_CVRS_M0_CHEAP_METHOD_GATE_ZH.md",
        ):
            handle.add(path, arcname=path.relative_to(ROOT).as_posix())
    return archive


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise ValueError("Formal is locked; run benchmark and obtain explicit cost approval first")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_input(candidate_roots(args, context))
    if source.stat().st_size != INPUT_BYTES or sha256_file(source) != INPUT_SHA256:
        raise ValueError("CVRS M0 input archive bytes/hash mismatch")

    extraction = ROOT / f"local_runs/cle_cvrs_m0_openi_extracted_{args.mode}"
    safe_extract(source, extraction, include_evaluation=args.mode == "formal")
    input_root = extraction / INPUT_DIRECTORY
    verify_input(input_root, include_evaluation=args.mode == "formal")

    config = json.loads((ROOT / "configs/cle_cvrs_m0_seed0.json").read_text(encoding="utf-8"))
    config["paths"] = {
        "private_root": str(input_root / "private"),
        "split_path": str(input_root / "splits/strict_cle_v1_alpha05_gamma_pair_seed0_split0.npz"),
        "evaluation_root": str(input_root / "evaluation"),
        "public_root": str(input_root / "public"),
        "checkpoint_root": str(input_root / "checkpoints"),
    }
    platform_config = extraction / "cle_cvrs_m0_openi_config.json"
    platform_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    output_dir = ROOT / "outputs" / f"cle_cvrs_m0_seed0_{args.mode}"
    command = [
        sys.executable,
        "scripts/run_cle_cvrs_m0.py",
        "--mode",
        args.mode,
        "--config",
        str(platform_config),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
    ]
    if args.mode == "formal":
        command.append("--confirm-formal")
    run(command, environment)

    run_manifest = {
        "protocol": "cle_cvrs_m0_openi_v1",
        "mode": args.mode,
        "input_archive": source.name,
        "input_bytes": source.stat().st_size,
        "input_sha256": sha256_file(source),
        "evaluation_extracted": args.mode == "formal",
        "formal_confirmed": bool(args.confirm_formal),
        "communication_modified": False,
    }
    run_manifest_path = output_dir / "openi_run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    archive = package(output_dir, args.mode)
    if not args.no_upload:
        upload(context, [archive, output_dir / "result.json", run_manifest_path])
    log(f"===== CVRS M0 {args.mode} complete =====")


if __name__ == "__main__":
    main()
