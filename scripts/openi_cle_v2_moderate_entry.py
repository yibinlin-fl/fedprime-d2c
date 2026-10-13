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
    started = time.perf_counter()
    run(command, environment)
    training_seconds = time.perf_counter() - started
    analysis_root.mkdir(parents=True, exist_ok=True)
    analysis_command = [
        sys.executable, "-u", "scripts/analyze_cle_v2_moderate.py",
        "--package-root", str(package_root), "--outputs-root", str(outputs_root),
        "--output-dir", str(analysis_root), "--mode", args.mode, "--shard", args.shard,
        "--device", "cuda",
    ]
    if args.mode == "formal":
        analysis_command.append("--confirm-formal")
    analysis_started = time.perf_counter()
    run(analysis_command, environment)
    analysis_seconds = time.perf_counter() - analysis_started
    timing = outputs_root / "RUN_TIMING.json"
    timing.write_text(
        json.dumps(
            {
                "mode": args.mode, "shard": args.shard, "training_seconds": training_seconds,
                "analysis_seconds": analysis_seconds, "scientific_evidence": args.mode == "formal",
                "warning": None if args.mode == "formal" else f"{args.mode} is not scientific evidence.",
            }, indent=2
        ), encoding="utf-8"
    )
    archive = package_light_outputs(
        args.mode, outputs_root, configs_root, analysis_root,
        ROOT / f"cle_v2_moderate_{args.shard}_{args.mode}_outputs.tar.gz",
    )
    upload(context, [archive, timing])
    log(f"Complete: {archive}")


if __name__ == "__main__":
    main()
