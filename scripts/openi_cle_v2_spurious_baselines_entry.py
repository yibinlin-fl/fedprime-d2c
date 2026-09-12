#!/usr/bin/python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


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
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs, upload  # noqa: E402


PACKAGE_DIRECTORY = "cle_hfl_v2_paired_factorial_seed0_split0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE-v2 spurious baseline screen.")
    parser.add_argument("--mode", choices=["smoke", "benchmark", "screen"], default="smoke")
    parser.add_argument("--confirm_screen", choices=["false", "true"], default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=["false", "true"], default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "screen" and args.confirm_screen != "true":
        raise PermissionError("12-round spurious baseline screen is locked")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_archive(candidate_roots(args, context))
    package_root = safe_extract(source, ROOT / "local_runs/openi_cle_v2_spurious_input")
    if package_root.name != PACKAGE_DIRECTORY:
        raise ValueError(f"Unexpected extracted package: {package_root}")

    outputs_root = ROOT / f"outputs/cle_v2_spurious_seed0_{args.mode}"
    configs_root = ROOT / f"local_runs/cle_v2_spurious_seed0_{args.mode}_configs"
    analysis_root = ROOT / f"outputs/cle_v2_spurious_seed0_{args.mode}_analysis"
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
        "scripts/run_cle_v2_spurious_baselines.py",
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
    if args.mode == "screen":
        command.append("--confirm-screen")
    started = time.perf_counter()
    run(command, environment)
    training_seconds = time.perf_counter() - started
    analysis_command = [
        sys.executable,
        "-u",
        "scripts/analyze_cle_v2_spurious_baselines.py",
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
    if args.mode != "screen":
        analysis_command.extend(["--max-sources", "20"])
    analysis_started = time.perf_counter()
    run(analysis_command, environment)
    analysis_seconds = time.perf_counter() - analysis_started
    timing = outputs_root / "RUN_TIMING.json"
    timing.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "training_seconds": training_seconds,
                "analysis_seconds": analysis_seconds,
                "jtt_has_two_training_stages": True,
                "scientific_evidence": False,
                "warning": "Smoke/benchmark test execution; screen selects finalists only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = package_light_outputs(
        args.mode,
        outputs_root,
        configs_root,
        analysis_root,
        ROOT / f"cle_v2_spurious_seed0_{args.mode}_outputs.tar.gz",
    )
    upload(context, [archive, timing])
    log(f"Complete: {archive}")


if __name__ == "__main__":
    main()
