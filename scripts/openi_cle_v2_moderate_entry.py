#!/usr/bin/python
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openi_cle_v2_factorial_entry import candidate_roots, log, prepare_c2net, run  # noqa: E402
from scripts.openi_cle_v2_cross_scenario_entry import upload  # noqa: E402
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs  # noqa: E402
from scripts.run_cle_v2_moderate import PACKAGE_NAME, SHARDS  # noqa: E402


INPUT_ARCHIVE = f"{PACKAGE_NAME}_with_pew.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI moderate CLE dose/Axis-II experiment.")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="benchmark")
    parser.add_argument("--shard", choices=SHARDS, default="axis2_plain")
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


def find_archive(roots: list[Path]) -> Path:
    matches = []
    for root in roots:
        if root.is_file() and root.name == INPUT_ARCHIVE:
            matches.append(root)
        elif root.is_dir():
            direct = root / INPUT_ARCHIVE
            if direct.is_file():
                matches.append(direct)
            else:
                matches.extend(root.rglob(INPUT_ARCHIVE))
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        raise FileNotFoundError(f"Expected exactly one {INPUT_ARCHIVE}; found {len(unique)}")
    return unique[0]


def extract_archive(source: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    with tarfile.open(source, "r:gz") as archive:
        base = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if base != target and base not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)
    package_root = destination / PACKAGE_NAME
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    return package_root


def package_recovery_outputs(
    outputs_root: Path,
    configs_root: Path,
    archive_path: Path,
) -> Path:
    """Package train-time state, including checkpoints, before analysis.

    The ordinary light artifact intentionally excludes checkpoints.  This
    recovery artifact exists so that a post-training analysis failure never
    forces an otherwise completed Formal run to be trained again.
    """
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite recovery archive: {archive_path}")
    with tarfile.open(archive_path, "w:gz", compresslevel=4) as archive:
        if outputs_root.is_dir():
            archive.add(outputs_root, arcname="outputs")
        if configs_root.is_dir():
            archive.add(configs_root, arcname="configs")
    return archive_path


def write_timing(
    path: Path,
    *,
    mode: str,
    shard: str,
    training_seconds: float | None,
    analysis_seconds: float | None,
    status: str,
    error: str | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": mode,
                "shard": shard,
                "training_seconds": training_seconds,
                "analysis_seconds": analysis_seconds,
                "status": status,
                "scientific_evidence": mode == "formal" and status == "complete",
                "warning": (
                    None
                    if mode == "formal" and status == "complete"
                    else error or f"{mode} is not complete scientific evidence."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("Formal is locked; set confirm_formal=true after explicit approval")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_archive(candidate_roots(args, context))
    package_root = extract_archive(source, ROOT / "local_runs/openi_cle_v2_moderate_input")
    run_name = f"cle_v2_moderate_{args.shard}_{args.mode}"
    outputs_root = ROOT / "outputs" / run_name
    configs_root = ROOT / "local_runs" / f"{run_name}_configs"
    analysis_root = ROOT / "outputs" / f"{run_name}_analysis"
    outputs_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", "scripts/run_cle_v2_moderate.py",
        "--package-root", str(package_root), "--mode", args.mode, "--shard", args.shard,
        "--device", "cuda", "--output-root", str(outputs_root), "--config-root", str(configs_root),
    ]
    if args.mode == "formal":
        command.append("--confirm-formal")
    timing = outputs_root / "RUN_TIMING.json"
    recovery_archive = ROOT / f"cle_v2_moderate_{args.shard}_{args.mode}_recovery_outputs.tar.gz"
    final_archive = ROOT / f"cle_v2_moderate_{args.shard}_{args.mode}_outputs.tar.gz"
    training_seconds: float | None = None
    analysis_seconds: float | None = None
    artifacts: list[Path] = []
    training_complete = False
    recovery_uploaded = False
    started = time.perf_counter()
    try:
        run(command, environment)
        training_complete = True
        training_seconds = time.perf_counter() - started
        write_timing(
            timing,
            mode=args.mode,
            shard=args.shard,
            training_seconds=training_seconds,
            analysis_seconds=None,
            status="training_complete_analysis_pending",
        )
        # Freeze a checkpoint-bearing recovery artifact before launching the
        # separate analysis process.  Upload it immediately: a platform hard
        # stop cannot be handled by Python finally blocks.
        package_recovery_outputs(outputs_root, configs_root, recovery_archive)
        upload(context, [recovery_archive, timing])
        recovery_uploaded = True

        analysis_root.mkdir(parents=True, exist_ok=True)
        analysis_command = [
            sys.executable, "-u", "scripts/analyze_cle_v2_moderate.py",
            "--package-root", str(package_root), "--outputs-root", str(outputs_root),
            "--output-dir", str(analysis_root), "--mode", args.mode, "--shard", args.shard,
            # Formal training can consume the account's remaining GPU lease.
            # `auto` keeps CUDA when available and safely falls back to CPU.
            "--device", "auto",
        ]
        if args.mode == "formal":
            analysis_command.append("--confirm-formal")
        analysis_started = time.perf_counter()
        run(analysis_command, environment)
        analysis_seconds = time.perf_counter() - analysis_started
        write_timing(
            timing,
            mode=args.mode,
            shard=args.shard,
            training_seconds=training_seconds,
            analysis_seconds=analysis_seconds,
            status="complete",
        )
        archive = package_light_outputs(
            args.mode, outputs_root, configs_root, analysis_root, final_archive,
        )
        artifacts = [archive, timing]
        log(f"Complete: {archive}")
    except BaseException as exc:
        if training_seconds is None:
            training_seconds = time.perf_counter() - started
        write_timing(
            timing,
            mode=args.mode,
            shard=args.shard,
            training_seconds=training_seconds,
            analysis_seconds=analysis_seconds,
            status="failed_with_recovery_artifact",
            error=f"{type(exc).__name__}: {exc}",
        )
        if not recovery_archive.exists():
            package_recovery_outputs(outputs_root, configs_root, recovery_archive)
        artifacts = [timing] if recovery_uploaded else [recovery_archive, timing]
        if training_complete:
            log(
                "[recovery] Training completed but analysis failed; "
                f"checkpoint-bearing artifact is preserved: {recovery_archive}"
            )
        else:
            log(f"[recovery] Training failed; uploading partial state: {recovery_archive}")
            raise
    finally:
        if artifacts:
            upload(context, artifacts)


if __name__ == "__main__":
    main()
