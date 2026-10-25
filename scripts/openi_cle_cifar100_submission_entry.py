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
)
from scripts.openi_cle_v2_cross_scenario_entry import upload  # noqa: E402
from scripts.openi_cle_v2_moderate_entry import (  # noqa: E402
    package_recovery_outputs,
    require_upload_context,
)
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs  # noqa: E402
from scripts.prepare_cle_v2_factorial_data import sha256_file  # noqa: E402


INPUT_ARCHIVE = "cle_hfl_v2_cifar100_factorial_seed0_split0_input.tar.gz"
PACKAGE_DIRECTORY = "cle_hfl_v2_cifar100_factorial_seed0_split0"
INPUT_ARCHIVE_BYTES = 681_493_067
INPUT_ARCHIVE_SHA256 = "650C3363B708554EB164BDEB94A3566B98CADBFF697FC093B9DE7F061C7DA708"


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


def validate_request(*, mode: str, confirm_formal: str) -> None:
    if mode == "formal" and confirm_formal != "true":
        raise PermissionError("Formal is locked; set confirm_formal=true only after explicit approval")


def verify_input_archive(path: Path) -> None:
    if path.stat().st_size != INPUT_ARCHIVE_BYTES:
        raise ValueError(
            f"Unexpected input archive size: {path.stat().st_size}; expected {INPUT_ARCHIVE_BYTES}"
        )
    digest = sha256_file(path)
    if digest != INPUT_ARCHIVE_SHA256:
        raise ValueError(f"Unexpected input archive SHA256: {digest}")


def write_timing(
    path: Path,
    *,
    mode: str,
    train_seed: int,
    pew_seconds: float,
    training_seconds: float | None,
    analysis_seconds: float | None,
    status: str,
    error: str | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": mode,
                "train_seed": int(train_seed),
                "pew_training_and_annotation_seconds_shared_across_seeds": pew_seconds,
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
    validate_request(mode=args.mode, confirm_formal=args.confirm_formal)
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    require_upload_context(context)
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_archive(candidate_roots(args, context))
    verify_input_archive(source)
    package_root = safe_extract(source, ROOT / "local_runs/openi_cle_cifar100_submission_input")
    if package_root.name != PACKAGE_DIRECTORY:
        raise ValueError(f"Unexpected package directory: {package_root}")
    # Audit immutable private/public data roles and paired inputs before paying
    # the PEW preparation or any four-arm training cost.
    preflight_root = ROOT / "outputs/cle_cifar100_submission_preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-u",
            "scripts/audit_cle_cifar100_submission.py",
            "--package-root",
            str(package_root),
            "--skip-pew",
            "--output",
            str(preflight_root / "INPUT_AUDIT_BEFORE_PEW.json"),
        ],
        environment,
    )
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
        timing = outputs_root / "RUN_TIMING.json"
        recovery_archive = ROOT / (
            f"cle_cifar100_submission_trainseed{train_seed}_{args.mode}_recovery_outputs.tar.gz"
        )
        final_archive = ROOT / f"cle_cifar100_submission_trainseed{train_seed}_{args.mode}_outputs.tar.gz"
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
                train_seed=train_seed,
                pew_seconds=pew_seconds,
                training_seconds=training_seconds,
                analysis_seconds=None,
                status="training_complete_analysis_pending",
            )
            package_recovery_outputs(outputs_root, configs_root, recovery_archive)
            upload(context, [recovery_archive, timing])
            recovery_uploaded = True

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
                # Formal analysis may outlive the GPU lease. Preserve the
                # completed checkpoints and fall back to CPU when necessary.
                "--device",
                "auto",
            ]
            if args.mode == "formal":
                analysis_command.append("--confirm-formal")
            else:
                analysis_command.extend(["--bootstrap-samples", "100", "--null-permutations", "100"])
            analysis_started = time.perf_counter()
            run(analysis_command, environment)
            analysis_seconds = time.perf_counter() - analysis_started
            write_timing(
                timing,
                mode=args.mode,
                train_seed=train_seed,
                pew_seconds=pew_seconds,
                training_seconds=training_seconds,
                analysis_seconds=analysis_seconds,
                status="complete",
            )
            archive = package_light_outputs(
                args.mode, outputs_root, configs_root, analysis_root, final_archive,
            )
            artifacts = [archive, timing]
            log(f"Complete seed {train_seed}: {archive}")
        except BaseException as exc:
            if training_seconds is None:
                training_seconds = time.perf_counter() - started
            write_timing(
                timing,
                mode=args.mode,
                train_seed=train_seed,
                pew_seconds=pew_seconds,
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
                    "[recovery] CIFAR-100 training completed but audit/analysis failed; "
                    f"checkpoints are preserved in {recovery_archive}"
                )
            else:
                log(f"[recovery] CIFAR-100 training failed; uploading partial state: {recovery_archive}")
            raise
        finally:
            if artifacts:
                upload(context, artifacts)


if __name__ == "__main__":
    main()
