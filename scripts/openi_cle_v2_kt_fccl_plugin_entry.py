#!/usr/bin/python
# coding=utf-8
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
    parser = argparse.ArgumentParser(description="OpenI CLE-v2 KT-pFL/FCCL PEW+BER pairs.")
    parser.add_argument("--mode", choices=["smoke", "benchmark", "formal"], default="smoke")
    parser.add_argument("--confirm_formal", choices=["false", "true"], default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=["false", "true"], default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("KT/FCCL plugin Formal is locked")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_archive(candidate_roots(args, context))
    package_root = safe_extract(source, ROOT / "local_runs/openi_cle_v2_kt_fccl_plugin_input")
    if package_root.name != PACKAGE_DIRECTORY:
        raise ValueError(f"Unexpected extracted package: {package_root}")

    outputs_root = ROOT / f"outputs/cle_v2_kt_fccl_plugin_seed0_{args.mode}"
    configs_root = ROOT / f"local_runs/cle_v2_kt_fccl_plugin_seed0_{args.mode}_configs"
    analysis_root = ROOT / f"outputs/cle_v2_kt_fccl_plugin_seed0_{args.mode}_analysis"
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
        "scripts/run_cle_v2_kt_fccl_plugin.py",
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
    started = time.perf_counter()
    run(command, environment)
    training_seconds = time.perf_counter() - started

    analysis_command = [
        sys.executable,
        "-u",
        "scripts/analyze_cle_v2_kt_fccl_plugin.py",
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
    if args.mode == "formal":
        analysis_command.append("--confirm-formal")
    else:
        analysis_command.extend(
            ["--max-sources", "20", "--bootstrap-samples", "100", "--null-permutations", "100"]
        )
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
                "rounds": 12 if args.mode == "formal" else 1,
                "local_batches_per_client_round": {
                    "smoke": 1,
                    "benchmark": 8,
                    "formal": 16,
                }[args.mode],
                "scientific_evidence": args.mode == "formal",
                "warning": None if args.mode == "formal" else f"{args.mode} is not evidence.",
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
        ROOT / f"cle_v2_kt_fccl_plugin_seed0_{args.mode}_outputs.tar.gz",
    )
    upload(context, [archive, timing_path])
    log(f"Complete: {archive}")


if __name__ == "__main__":
    main()
