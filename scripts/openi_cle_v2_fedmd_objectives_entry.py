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

from scripts.openi_cle_v2_cross_scenario_entry import (  # noqa: E402
    candidate_roots,
    find_input_archive,
    log,
    prepare_c2net,
    run,
    safe_extract_bundle,
    upload,
)
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI held-out map2 FedMD four-objective replication.")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="benchmark")
    parser.add_argument("--train_seed", choices=("0", "1", "2"), default="0")
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("FedMD four-objective Formal is locked; set confirm_formal=true")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_input_archive(candidate_roots(args, context))
    bundle_root = safe_extract_bundle(source, ROOT / "local_runs/openi_cle_v2_fedmd_objectives_input")
    package_root = bundle_root / "cle_hfl_v2_cross_map2_seed0_split0"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    run_name = f"cle_v2_fedmd_objectives_map2_trainseed{args.train_seed}_{args.mode}"
    outputs_root = ROOT / f"outputs/{run_name}"
    configs_root = ROOT / f"local_runs/{run_name}_configs"
    analysis_root = ROOT / f"outputs/{run_name}_analysis"
    outputs_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, "-u", "scripts/audit_cle_v2_factorial.py", "--package-root", str(package_root), "--output", str(outputs_root / "INPUT_AUDIT.json")],
        environment,
    )
    command = [
        sys.executable, "-u", "scripts/run_cle_v2_fedmd_objectives.py",
        "--package-root", str(package_root), "--mode", args.mode,
        "--train-seed", args.train_seed, "--device", "cuda",
        "--output-root", str(outputs_root), "--config-root", str(configs_root),
    ]
    if args.mode == "formal":
        command.append("--confirm-formal")
    started = time.perf_counter()
    run(command, environment)
    training_seconds = time.perf_counter() - started
    analysis_command = [
        sys.executable, "-u", "scripts/analyze_cle_v2_spurious_final.py",
        "--package-root", str(package_root), "--outputs-root", str(outputs_root),
        "--output-dir", str(analysis_root), "--mode", args.mode,
        "--train-seed", args.train_seed, "--device", "cuda",
        "--experiment-prefix", "cle_v2_fedmd_objectives_map2",
        "--protocol-label", "cle_v2_fedmd_four_objective_map2_analysis_v1",
        "--communication-label", "FedMD symmetric public-logit exchange",
    ]
    if args.mode == "formal":
        analysis_command.append("--confirm-formal")
    else:
        analysis_command.extend(["--max-sources", "20", "--bootstrap-samples", "100", "--null-permutations", "100"])
    analysis_started = time.perf_counter()
    run(analysis_command, environment)
    analysis_seconds = time.perf_counter() - analysis_started
    timing = outputs_root / "RUN_TIMING.json"
    timing.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "train_seed": int(args.train_seed),
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
        ROOT / f"{run_name}_outputs.tar.gz",
    )
    upload(context, [archive, timing])
    log(f"Complete: {archive}")


if __name__ == "__main__":
    main()
