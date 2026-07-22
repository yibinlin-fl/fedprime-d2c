#!/usr/bin/python
#coding=utf-8
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS_BY_MODE = {
    "oracle_probe": [
        "configs/openi_v100_fedease_oracle_control_probe.yaml",
        "configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml",
    ],
    "pew_probe": ["configs/openi_v100_fedease_pew_probe.yaml"],
    "ebst_probe": ["configs/openi_v100_fedease_ebst_probe.yaml"],
    "ebst_v2_probe": ["configs/openi_v100_fedease_ebst_v2_probe.yaml"],
    "pew_calibrated_local_probe": [
        "configs/openi_v100_fedease_pew_calibrated_local_probe.yaml"
    ],
    "pew_ebst_v2_probe": ["configs/openi_v100_fedease_pew_ebst_v2_probe.yaml"],
    "full": ["configs/openi_v100_fedease_full.yaml"],
}
EXPERIMENTS_BY_MODE = {
    "oracle_probe": [
        "probe_fedease_oracle_control_alpha05_gamma09_seed0",
        "probe_fedease_oracle_ber_cdep_alpha05_gamma09_seed0",
    ],
    "pew_probe": ["probe_fedease_pew_ber_cdep_alpha05_gamma09_seed0"],
    "ebst_probe": ["probe_fedease_oracle_ebst_alpha05_gamma09_seed0"],
    "ebst_v2_probe": ["probe_fedease_oracle_ebst_v2_alpha05_gamma09_seed0"],
    "pew_calibrated_local_probe": [
        "probe_fedease_pew_calibrated_local_alpha05_gamma09_seed0"
    ],
    "pew_ebst_v2_probe": ["probe_fedease_pew_ebst_v2_alpha05_gamma09_seed0"],
    "full": ["fedease_full_alpha05_gamma09_seed0"],
}
DATASET_NAMES = (
    "fedease_cle_prepared_alpha05_gamma09_seed0",
    "cle_hfl_prepared_alpha05_gamma09_seed0",
)


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], env: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=env)


def prepare_c2net():
    try:
        from c2net.context import prepare

        context = prepare()
        log(f"c2net dataset_path = {getattr(context, 'dataset_path', '')}")
        log(f"c2net output_path  = {getattr(context, 'output_path', '')}")
        return context
    except Exception as exc:  # pragma: no cover - OpenI integration.
        log(f"[warning] c2net prepare failed or unavailable: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI entry for FedEASE CLE-HFL experiments.")
    parser.add_argument("--mode", choices=sorted(CONFIGS_BY_MODE), default="oracle_probe")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def candidate_roots(args: argparse.Namespace, context) -> list[Path]:
    roots = []
    for raw in [
        args.data_source,
        os.environ.get("DATA_SOURCE", ""),
        getattr(context, "dataset_path", "") if context is not None else "",
        "/tmp/dataset",
        "/dataset",
        "/cache/dataset",
        "/tmp",
        "/cache",
    ]:
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def find_dataset(roots: list[Path]) -> Path:
    for root in roots:
        for dataset_name in DATASET_NAMES:
            for candidate in (root / f"{dataset_name}.tar.gz", root / dataset_name):
                if candidate.exists():
                    return candidate
            matches = list(root.rglob(f"{dataset_name}.tar.gz"))
            if matches:
                return matches[0]
            directories = [path for path in root.rglob(dataset_name) if path.is_dir()]
            if directories:
                return directories[0]
    raise FileNotFoundError(
        "Could not find a FedEASE/CLE-HFL gamma=0.9 dataset under: "
        + ", ".join(str(path) for path in roots)
    )


def package_outputs(mode: str) -> Path:
    path = ROOT / f"fedease_{mode}_outputs.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for experiment in EXPERIMENTS_BY_MODE[mode]:
            directory = ROOT / "outputs" / experiment
            if directory.exists():
                archive.add(directory, arcname=f"outputs/{experiment}")
        for name in ("summary.csv", "summary.md"):
            source = ROOT / "outputs" / name
            if source.exists():
                archive.add(source, arcname=f"outputs/{name}")
        comparison = ROOT / "outputs" / "fedease_oracle_probe_comparison.json"
        if mode == "oracle_probe" and comparison.exists():
            archive.add(comparison, arcname="outputs/fedease_oracle_probe_comparison.json")
        for document in ("FEDEASE_V2_1_FRAMEWORK_AND_IMPLEMENTATION_ZH.md", "CURRENT_PROJECT_MEMORY.md"):
            source = ROOT / document
            if source.exists():
                archive.add(source, arcname=document)
    log(f"Wrote {path}")
    return path


def upload_outputs(context, archive: Path) -> None:
    if context is None:
        log("[warning] c2net context unavailable; skip upload")
        return
    try:
        from c2net.context import upload_output
    except Exception as exc:  # pragma: no cover - OpenI integration.
        log(f"[warning] c2net upload_output unavailable: {exc}")
        return
    output_path = Path(context.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for source in (archive, ROOT / "outputs" / "summary.csv", ROOT / "outputs" / "summary.md"):
        if source.exists():
            destination = output_path / source.name
            shutil.copy2(source, destination)
            log(f"Copied {source} -> {destination}")
    upload_output()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    configs = CONFIGS_BY_MODE[args.mode]
    log("===== OpenI FedEASE entry =====")
    log(f"Mode: {args.mode}")
    log(f"Repository: {ROOT}")
    for config in configs:
        log(f"Config: {config}")
    context = prepare_c2net()

    if not args.skip_install:
        log("===== Installing dependencies =====")
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)

    if not args.skip_import:
        roots = candidate_roots(args, context)
        for root in roots:
            log(f"Dataset search root: {root}")
        source = find_dataset(roots)
        log(f"Prepared dataset: {source}")
        run(
            [sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."],
            environment,
        )

    log("===== Environment check =====")
    run([sys.executable, "scripts/check_environment.py", "--config", configs[0]], environment)

    if not args.skip_train:
        log("===== Running FedEASE experiments =====")
        run([sys.executable, "-u", "scripts/run_grid.py", *configs], environment)
        if args.mode == "oracle_probe":
            control, candidate = [ROOT / "outputs" / name for name in EXPERIMENTS_BY_MODE[args.mode]]
            run(
                [
                    sys.executable,
                    "scripts/analyze_fedease_probe.py",
                    "--control",
                    str(control),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(ROOT / "outputs" / "fedease_oracle_probe_comparison.json"),
                ],
                environment,
            )

    if not args.skip_summary:
        log("===== Summarizing outputs =====")
        run([sys.executable, "scripts/summarize_results.py", "--outputs", "outputs"], environment)

    archive = package_outputs(args.mode)
    if not args.no_upload:
        log("===== Uploading outputs through c2net =====")
        upload_outputs(context, archive)
    log("===== FedEASE run complete =====")


if __name__ == "__main__":
    main()
