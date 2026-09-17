#!/usr/bin/python
# coding=utf-8
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
INPUT_ARCHIVE = "cle_hfl_v2_paired_factorial_seed0_split0_with_pew.tar.gz"
PACKAGE_DIRECTORY = "cle_hfl_v2_paired_factorial_seed0_split0"


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], environment: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE-v2 matched factorial.")
    parser.add_argument("--mode", choices=["benchmark", "formal"], default="benchmark")
    parser.add_argument("--factorial_scope", choices=["mechanism", "all"], default="mechanism")
    parser.add_argument(
        "--train_seed",
        choices=["0", "1", "2", "all"],
        default="1",
        help="Training seed. 'all' runs pending seeds 1 then 2 on fixed seed0/split0 data.",
    )
    parser.add_argument("--confirm_formal", choices=["false", "true"], default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=["false", "true"], default="false")
    return parser.parse_args()


def selected_train_seeds(value: str) -> tuple[int, ...]:
    return (1, 2) if value == "all" else (int(value),)


def prepare_c2net():
    try:
        from c2net.context import prepare

        context = prepare()
        log(f"c2net dataset_path = {getattr(context, 'dataset_path', '')}")
        log(f"c2net output_path  = {getattr(context, 'output_path', '')}")
        return context
    except Exception as exc:  # pragma: no cover - OpenI only.
        log(f"[warning] c2net prepare failed or unavailable: {exc}")
        return None


def candidate_roots(args: argparse.Namespace, context) -> list[Path]:
    roots = []
    for raw in (
        args.data_source,
        os.environ.get("DATA_SOURCE", ""),
        getattr(context, "dataset_path", "") if context is not None else "",
        "/tmp/dataset",
        "/dataset",
        "/cache/dataset",
    ):
        if raw:
            path = Path(raw)
            if path.exists() and path not in roots:
                roots.append(path)
    return roots


def find_archive(roots: list[Path]) -> Path:
    matches = []
    for root in roots:
        if root.is_file() and root.name == INPUT_ARCHIVE:
            matches.append(root)
            continue
        direct = root / INPUT_ARCHIVE
        if direct.is_file():
            matches.append(direct)
            continue
        matches.extend(root.rglob(INPUT_ARCHIVE))
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {INPUT_ARCHIVE}; found {len(unique)} under "
            + ", ".join(str(path) for path in roots)
        )
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


def package_outputs(
    mode: str,
    scope: str,
    train_seed: int,
    outputs_root: Path,
    configs_root: Path,
    analysis_root: Path | None,
) -> Path:
    archive_path = ROOT / f"cle_v2_factorial_{scope}_trainseed{train_seed}_{mode}_outputs.tar.gz"
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite output archive: {archive_path}")

    def exclude_checkpoints(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "checkpoints" in PurePosixPath(member.name).parts:
            return None
        return member

    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        archive.add(outputs_root, arcname="outputs", filter=exclude_checkpoints)
        archive.add(configs_root, arcname="configs")
        if analysis_root is not None and analysis_root.exists():
            archive.add(analysis_root, arcname="analysis")
    return archive_path


def upload(context, paths: list[Path]) -> None:
    if context is None:
        log("[warning] c2net context unavailable; outputs remain in repository root")
        return
    from c2net.context import upload_output

    output_path = Path(context.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for source in paths:
        destination = output_path / source.name
        shutil.copy2(source, destination)
        log(f"Copied {source} -> {destination}")
    upload_output()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("Formal is locked; set --confirm_formal=true only after explicit approval")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_archive(candidate_roots(args, context))
    import_root = ROOT / "local_runs/openi_cle_v2_factorial_input"
    package_root = safe_extract(source, import_root)
    run(
        [
            sys.executable,
            "-u",
            "scripts/audit_cle_v2_factorial.py",
            "--package-root",
            str(package_root),
            *( ["--skip-pew"] if args.factorial_scope == "mechanism" else [] ),
        ],
        environment,
    )
    for train_seed in selected_train_seeds(args.train_seed):
        label = f"{args.factorial_scope}_trainseed{train_seed}_{args.mode}"
        outputs_root = ROOT / f"outputs/cle_v2_factorial_{label}"
        configs_root = ROOT / f"local_runs/cle_v2_factorial_{label}_configs"
        command = [
            sys.executable,
            "-u",
            "scripts/run_cle_v2_factorial.py",
            "--package-root",
            str(package_root),
            "--mode",
            args.factorial_scope,
            "--train-seed",
            str(train_seed),
            "--device",
            "cuda",
            "--output-root",
            str(outputs_root),
            "--config-root",
            str(configs_root),
        ]
        if args.mode == "benchmark":
            command.append("--benchmark")
        else:
            command.extend(["--rounds", "12", "--confirm-formal"])
        started = time.perf_counter()
        run(command, environment)
        elapsed = time.perf_counter() - started
        audit_command = [
            sys.executable,
            "-u",
            "scripts/audit_cle_v2_factorial.py",
            "--package-root",
            str(package_root),
            "--outputs-root",
            str(outputs_root),
            "--train-seed",
            str(train_seed),
            "--arm-set",
            args.factorial_scope,
            "--output",
            str(outputs_root / "INTEGRITY_AUDIT.json"),
        ]
        if args.factorial_scope == "mechanism":
            audit_command.append("--skip-pew")
        run(audit_command, environment)

        analysis_root = None
        if args.mode == "formal":
            analysis_root = ROOT / f"outputs/cle_v2_factorial_{label}_analysis"
            analyzer = (
                "scripts/analyze_cle_v2_local_first.py"
                if args.factorial_scope == "mechanism"
                else "scripts/analyze_cle_v2_factorial.py"
            )
            analysis_command = [
                sys.executable,
                "-u",
                analyzer,
                "--package-root",
                str(package_root),
                "--outputs-root",
                str(outputs_root),
                "--output-dir",
                str(analysis_root),
                "--train-seed",
                str(train_seed),
                "--device",
                "cuda",
                "--confirm-formal",
            ]
            if args.factorial_scope == "mechanism":
                analysis_command.extend(["--mode", "formal"])
            run(analysis_command, environment)
        timing_path = outputs_root / "RUN_TIMING.json"
        timing_path.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "factorial_scope": args.factorial_scope,
                    "train_seed": train_seed,
                    "elapsed_seconds": elapsed,
                    "scientific_evidence": args.mode == "formal",
                    "benchmark_warning": "Benchmark validates cost/execution only." if args.mode == "benchmark" else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        archive = package_outputs(
            args.mode, args.factorial_scope, train_seed, outputs_root, configs_root, analysis_root
        )
        upload(context, [archive, timing_path])
        log(f"Complete seed {train_seed}: {archive}")


if __name__ == "__main__":
    main()
