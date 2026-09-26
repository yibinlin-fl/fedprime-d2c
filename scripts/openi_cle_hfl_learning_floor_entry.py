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

from scripts.openi_cle_v2_cross_scenario_entry import find_input_archive, safe_extract_bundle, upload  # noqa: E402
from scripts.openi_cle_v2_factorial_entry import candidate_roots, log, prepare_c2net, run  # noqa: E402
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs  # noqa: E402
from scripts.run_cle_hfl_learning_floor import BUDGETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE-HFL Local learning-floor test.")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="benchmark")
    parser.add_argument("--local_batches", type=int, choices=BUDGETS, required=True)
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


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
    source = find_input_archive(candidate_roots(args, context))
    bundle_root = safe_extract_bundle(source, ROOT / "local_runs/openi_cle_hfl_learning_floor_input")
    package_root = bundle_root / "cle_hfl_v2_cross_map2_seed0_split0"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    run_name = f"cle_hfl_learning_floor_b{args.local_batches}_{args.mode}"
    outputs_root = ROOT / f"outputs/{run_name}"
    configs_root = ROOT / f"local_runs/{run_name}_configs"
    analysis_root = ROOT / f"outputs/{run_name}_analysis"
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
        "scripts/run_cle_hfl_learning_floor.py",
        "--package-root",
        str(package_root),
        "--mode",
        args.mode,
        "--local-batches",
        str(args.local_batches),
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
    analysis_root.mkdir(parents=True, exist_ok=True)
    analysis_command = [
        sys.executable,
        "-u",
        "scripts/analyze_cle_hfl_learning_floor.py",
        "--package-root",
        str(package_root),
        "--outputs-root",
        str(outputs_root),
        "--output-dir",
        str(analysis_root),
        "--mode",
        args.mode,
        "--local-batches",
        str(args.local_batches),
        "--device",
        "cuda",
    ]
    if args.mode == "formal":
        analysis_command.append("--confirm-formal")
    analysis_started = time.perf_counter()
    run(analysis_command, environment)
    analysis_seconds = time.perf_counter() - analysis_started
    timing_path = outputs_root / "RUN_TIMING.json"
    timing_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "local_batches": args.local_batches,
                "training_seconds": training_seconds,
                "analysis_seconds": analysis_seconds,
                "scientific_evidence": args.mode == "formal",
                "warning": None if args.mode == "formal" else f"{args.mode.title()} is not scientific evidence.",
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
        ROOT / f"cle_hfl_learning_floor_b{args.local_batches}_{args.mode}_outputs.tar.gz",
    )
    upload(context, [archive, timing_path])
    log(f"Complete: {archive}")


if __name__ == "__main__":
    main()
