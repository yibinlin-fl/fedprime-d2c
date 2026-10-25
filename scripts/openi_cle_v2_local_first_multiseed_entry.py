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

from scripts.openi_cle_v2_cross_scenario_entry import upload  # noqa: E402
from scripts.openi_cle_v2_factorial_entry import (  # noqa: E402
    candidate_roots,
    find_archive,
    log,
    prepare_c2net,
    run,
    safe_extract,
)
from scripts.openi_cle_v2_mechanism_stage1_entry import (  # noqa: E402
    PACKAGE_DIRECTORY,
    package_light_outputs,
)
from scripts.openi_cle_v2_moderate_entry import (  # noqa: E402
    package_recovery_outputs,
    require_upload_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenI CLE-v2 local-first training-seed replication."
    )
    parser.add_argument("--mode", choices=("benchmark", "formal"), default="benchmark")
    parser.add_argument(
        "--train_seed",
        choices=("1", "2", "all"),
        default="1",
        help="'all' runs pending training seeds 1 then 2; seed 0 is already complete.",
    )
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


def selected_train_seeds(value: str) -> tuple[int, ...]:
    return (1, 2) if value == "all" else (int(value),)


def validate_request(*, mode: str, confirm_formal: str) -> None:
    if mode == "formal" and confirm_formal != "true":
        raise PermissionError(
            "Formal is locked; set confirm_formal=true only after explicit approval"
        )


def write_timing(
    path: Path,
    *,
    mode: str,
    train_seed: int,
    training_seconds: float | None,
    analysis_seconds: float | None,
    status: str,
    error: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "protocol": "cle_v2_local_first_multiseed_12r16b_v1",
                "mode": mode,
                "train_seed": int(train_seed),
                "rounds": 1 if mode == "benchmark" else 12,
                "local_batches_per_client_round": 16,
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
    package_root = safe_extract(source, ROOT / "local_runs/openi_cle_v2_local_first_input")
    if package_root.name != PACKAGE_DIRECTORY:
        raise ValueError(f"Unexpected extracted package: {package_root}")
    run(
        [
            sys.executable,
            "-u",
            "scripts/audit_cle_v2_factorial.py",
            "--package-root",
            str(package_root),
            "--skip-pew",
        ],
        environment,
    )

    for train_seed in selected_train_seeds(args.train_seed):
        run_name = f"cle_v2_local_first_trainseed{train_seed}_{args.mode}"
        outputs_root = ROOT / "outputs" / run_name
        configs_root = ROOT / "local_runs" / f"{run_name}_configs"
        analysis_root = ROOT / "outputs" / f"{run_name}_analysis"
        outputs_root.mkdir(parents=True, exist_ok=True)
        timing_path = outputs_root / "RUN_TIMING.json"
        recovery_archive = ROOT / f"{run_name}_recovery_outputs.tar.gz"
        final_archive = ROOT / f"{run_name}_outputs.tar.gz"
        command = [
            sys.executable,
            "-u",
            "scripts/run_cle_v2_mechanism_stage1.py",
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

        training_seconds: float | None = None
        analysis_seconds: float | None = None
        training_complete = False
        recovery_uploaded = False
        artifacts: list[Path] = []
        started = time.perf_counter()
        try:
            run(command, environment)
            training_complete = True
            training_seconds = time.perf_counter() - started
            write_timing(
                timing_path,
                mode=args.mode,
                train_seed=train_seed,
                training_seconds=training_seconds,
                analysis_seconds=None,
                status="training_complete_analysis_pending",
            )
            package_recovery_outputs(outputs_root, configs_root, recovery_archive)
            upload(context, [recovery_archive, timing_path])
            recovery_uploaded = True

            analysis_root.mkdir(parents=True, exist_ok=True)
            analysis_command = [
                sys.executable,
                "-u",
                "scripts/analyze_cle_v2_mechanism_stage1.py",
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
                "auto",
            ]
            if args.mode == "benchmark":
                analysis_command.extend(["--bootstrap-samples", "100", "--permutations", "100"])
            else:
                analysis_command.append("--confirm-formal")
            analysis_started = time.perf_counter()
            run(analysis_command, environment)
            analysis_seconds = time.perf_counter() - analysis_started
            write_timing(
                timing_path,
                mode=args.mode,
                train_seed=train_seed,
                training_seconds=training_seconds,
                analysis_seconds=analysis_seconds,
                status="complete",
            )
            package_light_outputs(
                args.mode,
                outputs_root,
                configs_root,
                analysis_root,
                archive_path=final_archive,
            )
            artifacts = [final_archive, timing_path]
            log(f"Complete seed {train_seed}: {final_archive}")
        except BaseException as exc:
            if training_seconds is None:
                training_seconds = time.perf_counter() - started
            write_timing(
                timing_path,
                mode=args.mode,
                train_seed=train_seed,
                training_seconds=training_seconds,
                analysis_seconds=analysis_seconds,
                status="failed_with_recovery_artifact",
                error=f"{type(exc).__name__}: {exc}",
            )
            if not recovery_archive.exists():
                package_recovery_outputs(outputs_root, configs_root, recovery_archive)
            artifacts = [timing_path] if recovery_uploaded else [recovery_archive, timing_path]
            if training_complete:
                log(
                    "[recovery] Training completed but analysis failed; "
                    f"checkpoint-bearing artifact is preserved: {recovery_archive}"
                )
            else:
                log(f"[recovery] Training failed; uploading partial state: {recovery_archive}")
            raise
        finally:
            if artifacts:
                upload(context, artifacts)


if __name__ == "__main__":
    main()
