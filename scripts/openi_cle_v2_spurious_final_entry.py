#!/usr/bin/python
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from pathlib import PurePosixPath


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
from scripts.run_cle_v2_spurious_final import ARM_CHOICES, selected_arms  # noqa: E402


def _arm_output_name(arm: str, train_seed: int) -> str:
    return f"cle_v2_spurious_final_map2_{arm}_trainseed{train_seed}"


def package_arm_outputs(
    *,
    outputs_root: Path,
    configs_root: Path,
    analysis_root: Path | None,
    arm: str,
    train_seed: int,
    archive_path: Path,
    include_checkpoints: bool,
) -> Path:
    """Create one independently recoverable Axis-I arm artifact.

    The archive layout deliberately matches the shard merger's expected
    ``outputs/``, ``configs/`` and ``analysis/`` roots.  Recovery archives
    retain checkpoints; final analysis archives omit them.
    """
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite output archive: {archive_path}")
    arm_output = outputs_root / _arm_output_name(arm, train_seed)
    if not arm_output.is_dir():
        raise FileNotFoundError(arm_output)

    def exclude_checkpoints(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if not include_checkpoints and "checkpoints" in PurePosixPath(member.name).parts:
            return None
        return member

    config_names = (
        f"map2_{arm}_trainseed{train_seed}.json",
        f"SPURIOUS_FINAL_MAP2_CONTRACT_TRAINSEED{train_seed}_{arm}.json",
        f"SPURIOUS_FINAL_MAP2_COMPLETION_TRAINSEED{train_seed}_{arm}.json",
    )
    with tarfile.open(archive_path, "w:gz", compresslevel=4 if include_checkpoints else 6) as archive:
        audit = outputs_root / "INPUT_AUDIT.json"
        if audit.is_file():
            archive.add(audit, arcname="outputs/INPUT_AUDIT.json")
        archive.add(
            arm_output,
            arcname=f"outputs/{arm_output.name}",
            filter=exclude_checkpoints,
        )
        for name in config_names:
            path = configs_root / name
            if path.is_file():
                archive.add(path, arcname=f"configs/{name}")
        if analysis_root is not None:
            if not analysis_root.is_dir():
                raise FileNotFoundError(analysis_root)
            archive.add(analysis_root, arcname="analysis")
    return archive_path


def write_arm_timing(
    path: Path,
    *,
    mode: str,
    arm: str,
    train_seed: int,
    training_seconds: float | None,
    analysis_seconds: float | None,
    status: str,
    error: str | None = None,
) -> None:
    complete = status == "complete"
    path.write_text(
        json.dumps(
            {
                "mode": mode,
                "arm": arm,
                "train_seed": train_seed,
                "training_seconds": training_seconds,
                "analysis_seconds": analysis_seconds,
                "status": status,
                "scientific_evidence": mode == "formal" and complete,
                "warning": None if mode == "formal" and complete else (
                    error or f"{mode} is not complete scientific evidence."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI held-out map2 four-arm Formal.")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="smoke")
    parser.add_argument(
        "--train_seed",
        choices=("0", "1", "2", "all"),
        default="1",
        help="Training seed. 'all' runs the pending seeds 1 then 2; map2/partition stay fixed.",
    )
    parser.add_argument("--arm", choices=ARM_CHOICES, default="all")
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


def validate_execution_request(*, mode: str, arm: str, train_seed: str) -> None:
    if mode == "formal" and (arm == "all" or train_seed == "all"):
        raise PermissionError(
            "Failure-safe Axis-I Formal requires exactly one arm and one train_seed per OpenI task"
        )


def selected_train_seeds(value: str) -> tuple[int, ...]:
    return (1, 2) if value == "all" else (int(value),)


def main() -> None:
    args = parse_args()
    validate_execution_request(mode=args.mode, arm=args.arm, train_seed=args.train_seed)
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("40-round four-arm Formal is locked; set confirm_formal=true")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if context is None or not getattr(context, "output_path", None):
        raise RuntimeError("OpenI output context is required before training; recovery upload unavailable")
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_input_archive(candidate_roots(args, context))
    bundle_root = safe_extract_bundle(source, ROOT / "local_runs/openi_cle_v2_spurious_final_input")
    package_root = bundle_root / "cle_hfl_v2_cross_map2_seed0_split0"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)

    for train_seed in selected_train_seeds(args.train_seed):
        label = f"trainseed{train_seed}_{args.arm}_{args.mode}"
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
        # Non-Formal ``all`` requests are still executed and published one arm
        # at a time. Formal rejects multiplexed arms/seeds above.
        for arm in selected_arms(args.arm):
            arm_analysis_root = analysis_root / arm
            timing = outputs_root / f"RUN_TIMING_{arm}.json"
            recovery_archive = ROOT / (
                f"cle_v2_spurious_final_map2_trainseed{train_seed}_{arm}_{args.mode}"
                "_recovery_outputs.tar.gz"
            )
            final_archive = ROOT / (
                f"cle_v2_spurious_final_map2_trainseed{train_seed}_{arm}_{args.mode}_outputs.tar.gz"
            )
            command = [
                sys.executable, "-u", "scripts/run_cle_v2_spurious_final.py",
                "--package-root", str(package_root), "--mode", args.mode,
                "--train-seed", str(train_seed), "--arm", arm, "--device", "cuda",
                "--output-root", str(outputs_root), "--config-root", str(configs_root),
            ]
            if args.mode == "formal":
                command.append("--confirm-formal")

            training_seconds: float | None = None
            analysis_seconds: float | None = None
            training_complete = False
            recovery_uploaded = False
            started = time.perf_counter()
            try:
                run(command, environment)
                training_complete = True
                training_seconds = time.perf_counter() - started
                write_arm_timing(
                    timing,
                    mode=args.mode,
                    arm=arm,
                    train_seed=train_seed,
                    training_seconds=training_seconds,
                    analysis_seconds=None,
                    status="training_complete_analysis_pending",
                )
                package_arm_outputs(
                    outputs_root=outputs_root,
                    configs_root=configs_root,
                    analysis_root=None,
                    arm=arm,
                    train_seed=train_seed,
                    archive_path=recovery_archive,
                    include_checkpoints=True,
                )
                upload(context, [recovery_archive, timing])
                recovery_uploaded = True

                arm_analysis_root.mkdir(parents=True, exist_ok=True)
                analysis_command = [
                    sys.executable, "-u", "scripts/analyze_cle_v2_spurious_final.py",
                    "--package-root", str(package_root), "--outputs-root", str(outputs_root),
                    "--output-dir", str(arm_analysis_root), "--mode", args.mode,
                    "--train-seed", str(train_seed), "--arm", arm,
                    # The GPU lease can expire immediately after training.
                    "--device", "auto",
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
                write_arm_timing(
                    timing,
                    mode=args.mode,
                    arm=arm,
                    train_seed=train_seed,
                    training_seconds=training_seconds,
                    analysis_seconds=analysis_seconds,
                    status="complete",
                )
                package_arm_outputs(
                    outputs_root=outputs_root,
                    configs_root=configs_root,
                    analysis_root=arm_analysis_root,
                    arm=arm,
                    train_seed=train_seed,
                    archive_path=final_archive,
                    include_checkpoints=False,
                )
                upload(context, [final_archive, timing])
                log(f"Complete seed {train_seed} arm {arm}: {final_archive}")
            except BaseException as exc:
                if training_seconds is None:
                    training_seconds = time.perf_counter() - started
                write_arm_timing(
                    timing,
                    mode=args.mode,
                    arm=arm,
                    train_seed=train_seed,
                    training_seconds=training_seconds,
                    analysis_seconds=analysis_seconds,
                    status="failed_with_recovery_artifact",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if not recovery_archive.exists():
                    try:
                        package_arm_outputs(
                            outputs_root=outputs_root,
                            configs_root=configs_root,
                            analysis_root=None,
                            arm=arm,
                            train_seed=train_seed,
                            archive_path=recovery_archive,
                            include_checkpoints=True,
                        )
                    except FileNotFoundError:
                        # A very early failure may occur before the arm output
                        # directory exists; the timing file is still useful.
                        pass
                artifacts = [timing]
                if recovery_archive.exists() and not recovery_uploaded:
                    artifacts.insert(0, recovery_archive)
                upload(context, artifacts)
                if training_complete:
                    log(
                        f"[recovery] arm={arm} training completed; analysis failed, "
                        f"recovery artifact preserved: {recovery_archive}"
                    )
                else:
                    log(f"[recovery] arm={arm} training failed; partial state uploaded when available")
                raise


if __name__ == "__main__":
    main()
