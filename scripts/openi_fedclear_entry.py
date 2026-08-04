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
CONFIG_BY_MODE = {
    "debug": "configs/debug_fedclear_cle.yaml",
    "probe": "configs/openi_v100_fedclear_cle_gamma09_probe.yaml",
    "full": "configs/openi_v100_fedclear_cle_gamma09_full.yaml",
}
EXPERIMENT_BY_MODE = {
    "debug": "debug_fedclear_cle_alpha05_gamma09",
    "probe": "probe_fedclear_cle_alpha05_gamma09_seed0",
    "full": "fedclear_cle_alpha05_gamma09_seed0",
}
DATASET_NAME = "cle_hfl_prepared_alpha05_gamma09_seed0"


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], env: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=env)


def prepare_c2net():
    try:
        from c2net.context import prepare

        ctx = prepare()
        log(f"c2net dataset_path = {getattr(ctx, 'dataset_path', '')}")
        log(f"c2net output_path  = {getattr(ctx, 'output_path', '')}")
        return ctx
    except Exception as exc:  # pragma: no cover - OpenI-only integration.
        log(f"[warning] c2net prepare failed or unavailable: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI entry for FedCLEAR CLE-HFL experiments.")
    parser.add_argument(
        "--mode",
        choices=sorted(CONFIG_BY_MODE),
        default="probe",
        help="probe runs 12 rounds and is the default low-cost signal check; full runs 40 rounds.",
    )
    parser.add_argument(
        "--data_source",
        default="",
        help="Mounted dataset path. If empty, c2net and common OpenI mount paths are searched.",
    )
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def candidate_roots(args: argparse.Namespace, ctx) -> list[Path]:
    roots = []
    for raw in [
        args.data_source,
        os.environ.get("DATA_SOURCE", ""),
        getattr(ctx, "dataset_path", "") if ctx is not None else "",
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
        direct_dir = root / DATASET_NAME
        direct_tar = root / f"{DATASET_NAME}.tar.gz"
        if direct_dir.is_dir():
            return direct_dir
        if direct_tar.is_file():
            return direct_tar
        matches = list(root.rglob(f"{DATASET_NAME}.tar.gz"))
        if matches:
            return matches[0]
        dir_matches = [path for path in root.rglob(DATASET_NAME) if path.is_dir()]
        if dir_matches:
            return dir_matches[0]
    searched = ", ".join(str(path) for path in roots) or "<none>"
    raise FileNotFoundError(f"Could not find {DATASET_NAME}.tar.gz under: {searched}")


def package_outputs(mode: str) -> Path:
    tar_path = ROOT / f"fedclear_cle_gamma09_{mode}_outputs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        experiment_dir = ROOT / "outputs" / EXPERIMENT_BY_MODE[mode]
        if experiment_dir.exists():
            tar.add(experiment_dir, arcname=f"outputs/{experiment_dir.name}")
        for summary_name in ["summary.csv", "summary.md"]:
            summary = ROOT / "outputs" / summary_name
            if summary.exists():
                tar.add(summary, arcname=f"outputs/{summary.name}")
        method_doc = ROOT / "docs/archive/methods/FEDCLEAR_METHOD_DESIGN_REVIEW_ZH.md"
        if method_doc.exists():
            tar.add(method_doc, arcname=method_doc.name)
    log(f"Wrote {tar_path}")
    return tar_path


def upload_outputs(ctx, tar_path: Path) -> None:
    if ctx is None:
        log("[warning] c2net context unavailable; skip upload.")
        return
    try:
        from c2net.context import upload_output
    except Exception as exc:  # pragma: no cover - OpenI-only integration.
        log(f"[warning] c2net upload_output unavailable: {exc}")
        return

    output_path = Path(ctx.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for source in [tar_path, ROOT / "outputs" / "summary.csv", ROOT / "outputs" / "summary.md"]:
        if not source.exists():
            continue
        destination = output_path / source.name
        shutil.copy2(source, destination)
        log(f"Copied {source} -> {destination}")
    upload_output()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    config = CONFIG_BY_MODE[args.mode]

    log("===== OpenI FedCLEAR CLE-HFL entry =====")
    log(f"Repository root: {ROOT}")
    log(f"Mode: {args.mode}")
    log(f"Config: {config}")
    ctx = prepare_c2net()

    if not args.skip_install:
        log("===== Installing dependencies =====")
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], env)

    if not args.skip_import:
        log("===== Importing CLE-HFL gamma=0.9 data =====")
        roots = candidate_roots(args, ctx)
        for root in roots:
            log(f"Dataset search root: {root}")
        source = find_dataset(roots)
        log(f"Prepared dataset: {source}")
        run(
            [sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."],
            env,
        )

    log("===== Environment check =====")
    run([sys.executable, "scripts/check_environment.py", "--config", config], env)

    if not args.skip_train:
        log("===== Running FedCLEAR =====")
        run([sys.executable, "-u", "scripts/run_experiment.py", "--config", config], env)
        if args.mode == "probe":
            log("===== Comparing probe with same-round RAHFL =====")
            experiment_dir = ROOT / "outputs" / EXPERIMENT_BY_MODE[args.mode]
            run(
                [
                    sys.executable,
                    "scripts/analyze_fedclear_probe.py",
                    "--metrics",
                    str(experiment_dir / "metrics.csv"),
                    "--output-dir",
                    str(experiment_dir),
                ],
                env,
            )

    if not args.skip_summary:
        log("===== Summarizing outputs =====")
        run([sys.executable, "scripts/summarize_results.py", "--outputs", "outputs"], env)

    log("===== Packaging outputs =====")
    tar_path = package_outputs(args.mode)
    if not args.no_upload:
        log("===== Uploading outputs through c2net =====")
        upload_outputs(ctx, tar_path)
    log("===== FedCLEAR run complete =====")


if __name__ == "__main__":
    main()
