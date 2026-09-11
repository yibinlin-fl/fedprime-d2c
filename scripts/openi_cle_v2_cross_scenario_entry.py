#!/usr/bin/python
# coding=utf-8
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openi_cle_v2_factorial_entry import (  # noqa: E402
    candidate_roots,
    log,
    prepare_c2net,
    run,
)
from scripts.package_cle_v2_cross_scenario_inputs import BUNDLE_NAME  # noqa: E402


INPUT_ARCHIVE = f"{BUNDLE_NAME}.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI CLE-v2 cross-binding-map A/B.")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="benchmark")
    parser.add_argument("--map_seed", choices=("1", "2", "all"), default="1")
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    return parser.parse_args()


def find_input_archive(roots: list[Path]) -> Path:
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


def safe_extract_bundle(archive_path: Path, destination: Path) -> Path:
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
    bundle_root = destination / BUNDLE_NAME
    if not bundle_root.is_dir():
        raise FileNotFoundError(bundle_root)
    return bundle_root


def package_outputs(
    map_label: str,
    mode: str,
    outputs_root: Path,
    configs_root: Path,
    analysis_root: Path,
) -> Path:
    archive_path = ROOT / f"cle_v2_cross_{map_label}_{mode}_outputs.tar.gz"
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite output archive: {archive_path}")

    def exclude_checkpoints(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "checkpoints" in PurePosixPath(member.name).parts:
            return None
        return member

    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        archive.add(outputs_root, arcname="outputs", filter=exclude_checkpoints)
        archive.add(configs_root, arcname="configs")
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
        raise PermissionError("Cross-scenario Formal is locked; set --confirm_formal=true")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_input_archive(candidate_roots(args, context))
    import_root = ROOT / "local_runs/openi_cle_v2_cross_scenario_input"
    bundle_root = safe_extract_bundle(source, import_root)
    seeds = (1, 2) if args.map_seed == "all" else (int(args.map_seed),)
    label = "maps1_2" if len(seeds) == 2 else f"map{seeds[0]}"
    outputs_root = ROOT / f"outputs/cle_v2_cross_{label}_{args.mode}"
    configs_root = ROOT / f"local_runs/cle_v2_cross_{label}_{args.mode}_configs"
    analysis_root = ROOT / f"outputs/cle_v2_cross_{label}_{args.mode}_analysis"
    outputs_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    per_map = {}
    for seed in seeds:
        package_name = f"cle_hfl_v2_cross_map{seed}_seed0_split0"
        package_root = bundle_root / package_name
        audit_path = outputs_root / f"INPUT_AUDIT_MAP{seed}.json"
        run(
            [
                sys.executable,
                "-u",
                "scripts/audit_cle_v2_factorial.py",
                "--package-root",
                str(package_root),
                "--output",
                str(audit_path),
            ],
            environment,
        )
        command = [
            sys.executable,
            "-u",
            "scripts/run_cle_v2_cross_scenario.py",
            "--package-root",
            str(package_root),
            "--map-seed",
            str(seed),
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
        run(command, environment)
        analysis_command = [
            sys.executable,
            "-u",
            "scripts/analyze_cle_v2_cross_scenario.py",
            "--package-root",
            str(package_root),
            "--outputs-root",
            str(outputs_root),
            "--output-dir",
            str(analysis_root / f"map{seed}"),
            "--map-seed",
            str(seed),
            "--mode",
            args.mode,
            "--device",
            "cuda",
        ]
        if args.mode != "formal":
            analysis_command.extend(
                ["--max-sources", "20", "--bootstrap-samples", "100", "--null-permutations", "100"]
            )
        else:
            analysis_command.append("--confirm-formal")
        run(analysis_command, environment)
        per_map[str(seed)] = json.loads(
            (analysis_root / f"map{seed}" / "RESULT_SUMMARY.json").read_text(encoding="utf-8")
        )
    timing_path = outputs_root / "RUN_TIMING.json"
    timing_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "map_seeds": list(seeds),
                "total_measured_seconds": time.perf_counter() - started,
                "scientific_evidence": args.mode == "formal",
                "nonformal_warning": (
                    None
                    if args.mode == "formal"
                    else "Smoke/benchmark validates execution and cost only."
                ),
                "per_map_scientific_verdict": {
                    seed: summary["scientific_verdict"] for seed, summary in per_map.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = package_outputs(label, args.mode, outputs_root, configs_root, analysis_root)
    upload(context, [archive, timing_path])
    log(f"Complete: {archive}")


if __name__ == "__main__":
    main()
