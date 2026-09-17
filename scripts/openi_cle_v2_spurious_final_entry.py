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
    parser = argparse.ArgumentParser(description="OpenI held-out map2 four-arm Formal.")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument(
        "--train_seed",
        choices=("0", "1", "2", "all"),
        default="1",
        help="Training seed. 'all' runs the pending seeds 1 then 2; map2/partition stay fixed.",
    )
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


def selected_train_seeds(value: str) -> tuple[int, ...]:
    return (1, 2) if value == "all" else (int(value),)


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("40-round four-arm Formal is locked; set confirm_formal=true")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_input_archive(candidate_roots(args, context))
    bundle_root = safe_extract_bundle(source, ROOT / "local_runs/openi_cle_v2_spurious_final_input")
    package_root = bundle_root / "cle_hfl_v2_cross_map2_seed0_split0"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)

    for train_seed in selected_train_seeds(args.train_seed):
        label = f"trainseed{train_seed}_{args.mode}"
        outputs_root = ROOT / f"outputs/cle_v2_spurious_final_map2_{label}"
        configs_root = ROOT / f"local_runs/cle_v2_spurious_final_map2_{label}_configs"
        analysis_root = ROOT / f"outputs/cle_v2_spurious_final_map2_{label}_analysis"
        outputs_root.mkdir(parents=True, exist_ok=True)
        configs_root.mkdir(parents=True, exist_ok=True)
        analysis_root.mkdir(parents=True, exist_ok=True)
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
            "scripts/run_cle_v2_spurious_final.py",
            "--package-root",
            str(package_root),
            "--mode",
            args.mode,
            "--train-seed",
            str(train_seed),
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
            "scripts/analyze_cle_v2_spurious_final.py",
            "--package-root",
            str(package_root),
            "--outputs-root",
            str(outputs_root),
            "--output-dir",
            str(analysis_root),
            "--mode",
            args.mode,
            "--train-seed",
            str(train_seed),
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
        timing = outputs_root / "RUN_TIMING.json"
        timing.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "train_seed": train_seed,
                    "training_seconds": training_seconds,
                    "analysis_seconds": analysis_seconds,
                    "total_measured_seconds": training_seconds + analysis_seconds,
                    "scientific_evidence": args.mode == "formal",
                    "warning": None if args.mode == "formal" else "Smoke validates execution only.",
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
            ROOT / f"cle_v2_spurious_final_map2_trainseed{train_seed}_{args.mode}_outputs.tar.gz",
        )
        upload(context, [archive, timing])
        log(f"Complete seed {train_seed}: {archive}")


if __name__ == "__main__":
    main()
