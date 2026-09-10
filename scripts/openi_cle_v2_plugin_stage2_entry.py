#!/usr/bin/python
# coding=utf-8
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openi_cle_v2_factorial_entry import (  # noqa: E402
    candidate_roots,
    find_archive,
    log,
    prepare_c2net,
    run,
    safe_extract,
)


PACKAGE_DIRECTORY = "cle_hfl_v2_paired_factorial_seed0_split0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE-v2 PEW+BER Stage-2 A/B.")
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--confirm_formal", choices=["false", "true"], default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=["false", "true"], default="false")
    return parser.parse_args()


def package_light_outputs(
    mode: str,
    outputs_root: Path,
    configs_root: Path,
    analysis_root: Path,
    archive_path: Path | None = None,
) -> Path:
    archive_path = archive_path or ROOT / f"cle_v2_plugin_stage2_seed0_{mode}_outputs.tar.gz"
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite output archive: {archive_path}")

    def exclude_checkpoints(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "checkpoints" in PurePosixPath(member.name).parts:
            return None
        return member

    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        archive.add(outputs_root, arcname="outputs", filter=exclude_checkpoints)
        archive.add(configs_root, arcname="configs")
        archive.add(analysis_root, arcname="analysis")
    return archive_path


def upload(context, paths: list[Path]) -> None:
    if context is None:
        log("[warning] c2net context unavailable; outputs remain in repository root")
        return
    from c2net.context import upload_output

    output_path = Path(context.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for source in paths:
        destination = output_path / source.name
        shutil.copy2(source, destination)
        log(f"Copied {source} -> {destination}")
    upload_output()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("Stage-2 Formal is locked; explicit confirmation is required")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_archive(candidate_roots(args, context))
    import_root = ROOT / "local_runs/openi_cle_v2_plugin_stage2_input"
    package_root = safe_extract(source, import_root)
    if package_root.name != PACKAGE_DIRECTORY:
        raise ValueError(f"Unexpected extracted package: {package_root}")

    outputs_root = ROOT / f"outputs/cle_v2_plugin_stage2_seed0_{args.mode}"
    configs_root = ROOT / f"local_runs/cle_v2_plugin_stage2_seed0_{args.mode}_configs"
    analysis_root = ROOT / f"outputs/cle_v2_plugin_stage2_seed0_{args.mode}_analysis"
    outputs_root.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-u",
            "scripts/audit_cle_v2_factorial.py",
            "--package-root",
            str(package_root),
            "--output",
            str(outputs_root / "INPUT_AUDIT.json"),
        ],
        environment,
    )

    command = [
        sys.executable,
        "-u",
        "scripts/run_cle_v2_plugin_stage2.py",
        "--package-root",
        str(package_root),
        "--mode",
        args.mode,
        "--device",
        "cuda",
        "--output-root",
        str(outputs_root),
        "--config-root",
        str(configs_root),
    ]
    if args.mode == "formal":
        command.append("--confirm-formal")
    training_started = time.perf_counter()
    run(command, environment)
    training_seconds = time.perf_counter() - training_started

    analysis_command = [
        sys.executable,
        "-u",
        "scripts/analyze_cle_v2_plugin_stage2.py",
        "--package-root",
        str(package_root),
        "--outputs-root",
        str(outputs_root),
        "--output-dir",
        str(analysis_root),
        "--mode",
        args.mode,
        "--device",
        "cuda",
    ]
    if args.mode == "smoke":
        analysis_command.extend(["--max-sources", "20", "--bootstrap-samples", "100"])
    else:
        analysis_command.append("--confirm-formal")
    analysis_started = time.perf_counter()
    run(analysis_command, environment)
    analysis_seconds = time.perf_counter() - analysis_started

    timing_path = outputs_root / "RUN_TIMING.json"
    timing_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "training_seconds": training_seconds,
                "analysis_seconds": analysis_seconds,
                "total_measured_seconds": training_seconds + analysis_seconds,
                "rounds": 1 if args.mode == "smoke" else 12,
                "local_batches_per_client_round": 1 if args.mode == "smoke" else 16,
                "scientific_evidence": args.mode == "formal",
                "smoke_warning": (
                    "Smoke validates execution only; its metrics are not scientific evidence."
                    if args.mode == "smoke"
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = package_light_outputs(args.mode, outputs_root, configs_root, analysis_root)
    upload(context, [archive, timing_path])
    log(f"Complete: {archive}")


if __name__ == "__main__":
    main()
