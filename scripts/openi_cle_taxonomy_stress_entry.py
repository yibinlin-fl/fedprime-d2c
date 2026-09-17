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
    log,
    prepare_c2net,
    run,
    upload,
)
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs  # noqa: E402
from scripts.openi_cle_v2_factorial_entry import safe_extract  # noqa: E402


INPUT_ARCHIVE = "cle_hfl_v2_paired_factorial_seed0_split0_with_pew.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI bounded taxonomy held-out-operator stress.")
    parser.add_argument("--mode", choices=("benchmark", "formal"), default="benchmark")
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


def find_archive(roots: list[Path]) -> Path:
    matches = []
    for root in roots:
        if root.is_file() and root.name == INPUT_ARCHIVE:
            matches.append(root)
        elif (root / INPUT_ARCHIVE).is_file():
            matches.append(root / INPUT_ARCHIVE)
        else:
            matches.extend(root.rglob(INPUT_ARCHIVE))
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        raise FileNotFoundError(f"Expected exactly one {INPUT_ARCHIVE}; found {len(unique)}")
    return unique[0]


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
    package_root = safe_extract(source, ROOT / "local_runs/openi_cle_taxonomy_stress_input")
    pew_started = time.perf_counter()
    run(
        [
            sys.executable,
            "-u",
            "scripts/prepare_cle_v2_factorial_pew.py",
            "--package-root",
            str(package_root),
            "--device",
            "cuda",
            "--asset-name",
            "pew_loo_motion_blur",
            "--excluded-operators",
            "motion_blur",
            "--epochs",
            "5",
            "--public-size",
            "5000",
            "--batch-size",
            "128",
            "--inference-batch-size",
            "512",
            "--num-workers",
            "0",
        ],
        environment,
    )
    pew_seconds = time.perf_counter() - pew_started
    outputs_root = ROOT / f"outputs/cle_taxonomy_stress_motion_blur_{args.mode}"
    configs_root = ROOT / f"local_runs/cle_taxonomy_stress_motion_blur_{args.mode}_configs"
    analysis_root = ROOT / f"outputs/cle_taxonomy_stress_motion_blur_{args.mode}_analysis"
    outputs_root.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, "-u", "scripts/audit_cle_v2_factorial.py", "--package-root", str(package_root), "--output", str(outputs_root / "INPUT_AUDIT.json")],
        environment,
    )
    command = [
        sys.executable,
        "-u",
        "scripts/run_cle_taxonomy_stress.py",
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
    analysis_root.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "pew_training.csv", "pew_standard.pt"):
        source_asset = package_root / "pew_loo_motion_blur" / name
        if source_asset.is_file():
            import shutil

            shutil.copy2(source_asset, analysis_root / f"PEW_LOO_{name}")
    analysis_command = [
        sys.executable,
        "-u",
        "scripts/analyze_cle_taxonomy_stress.py",
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
        analysis_command.extend(["--bootstrap-samples", "100"])
    analysis_started = time.perf_counter()
    run(analysis_command, environment)
    analysis_seconds = time.perf_counter() - analysis_started
    timing = outputs_root / "RUN_TIMING.json"
    timing.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "pew_training_and_annotation_seconds": pew_seconds,
                "training_seconds": training_seconds,
                "analysis_seconds": analysis_seconds,
                "scientific_evidence": args.mode == "formal",
                "warning": None if args.mode == "formal" else "Benchmark is not scientific evidence.",
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
        ROOT / f"cle_taxonomy_stress_motion_blur_{args.mode}_outputs.tar.gz",
    )
    upload(context, [archive, timing])
    log(f"Complete: {archive}")


if __name__ == "__main__":
    main()
