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

from scripts.openi_cle_v2_factorial_entry import (  # noqa: E402
    candidate_roots,
    log,
    prepare_c2net,
    run,
    upload,
)
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs  # noqa: E402


INPUT_ARCHIVE = "cle_hfl_v2_cifar100_factorial_seed0_split0_input.tar.gz"
PACKAGE_DIRECTORY = "cle_hfl_v2_cifar100_factorial_seed0_split0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CIFAR-100-private CLE submission experiment.")
    parser.add_argument("--mode", choices=("benchmark", "formal"), default="benchmark")
    parser.add_argument("--train_seed", choices=("0", "1", "2", "all"), default="all")
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


def selected_train_seeds(value: str) -> tuple[int, ...]:
    return (0, 1, 2) if value == "all" else (int(value),)


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


def safe_extract(archive_path: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite extracted input: {destination}")
    destination.mkdir(parents=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        base = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if base != target and base not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)
    package_root = destination / PACKAGE_DIRECTORY
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    return package_root


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("Formal is locked; set confirm_formal=true only after explicit approval")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_archive(candidate_roots(args, context))
    package_root = safe_extract(source, ROOT / "local_runs/openi_cle_cifar100_submission_input")
    if package_root.name != PACKAGE_DIRECTORY:
        raise ValueError(f"Unexpected package directory: {package_root}")
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
    for train_seed in selected_train_seeds(args.train_seed):
        label = f"trainseed{train_seed}_{args.mode}"
        outputs_root = ROOT / f"outputs/cle_cifar100_submission_{label}"
        configs_root = ROOT / f"local_runs/cle_cifar100_submission_{label}_configs"
        analysis_root = ROOT / f"outputs/cle_cifar100_submission_{label}_analysis"
        outputs_root.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "-u",
                "scripts/audit_cle_cifar100_submission.py",
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
            "scripts/run_cle_cifar100_submission.py",
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
        run(
            [
                sys.executable,
                "-u",
                "scripts/audit_cle_cifar100_submission.py",
                "--package-root",
                str(package_root),
                "--outputs-root",
                str(outputs_root),
                "--train-seed",
                str(train_seed),
                "--output",
                str(outputs_root / "OUTPUT_AUDIT.json"),
            ],
            environment,
        )
        analysis_root.mkdir(parents=True, exist_ok=True)
        analysis_command = [
            sys.executable,
            "-u",
            "scripts/analyze_cle_cifar100_submission.py",
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
            analysis_command.extend(["--bootstrap-samples", "100", "--null-permutations", "100"])
        analysis_started = time.perf_counter()
        run(analysis_command, environment)
        analysis_seconds = time.perf_counter() - analysis_started
        timing = outputs_root / "RUN_TIMING.json"
        timing.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "train_seed": train_seed,
                    "pew_training_and_annotation_seconds_shared_across_seeds": pew_seconds,
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
            ROOT / f"cle_cifar100_submission_trainseed{train_seed}_{args.mode}_outputs.tar.gz",
        )
        upload(context, [archive, timing])
        log(f"Complete seed {train_seed}: {archive}")


if __name__ == "__main__":
    main()
