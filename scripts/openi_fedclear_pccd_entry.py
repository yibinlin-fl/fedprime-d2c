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
PRIVATE_DATASET_NAME = "cle_hfl_prepared_alpha05_gamma09_seed0"
PUBLIC_DATASET_NAME = "cle_hfl_indomain_public_alpha05_gamma09_seed0"
CONFIGS = {
    "rahfl": "configs/openi_v100_rahfl_cle_indomain_probe.yaml",
    "pccd": "configs/openi_v100_fedclear_pccd_probe.yaml",
}
EXPERIMENTS = {
    "rahfl": "probe_rahfl_cle_indomain_alpha05_gamma09_seed0",
    "pccd": "probe_fedclear_pccd_alpha05_gamma09_seed0",
}


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], env: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI entry for matching RAHFL/PCCD probes.")
    parser.add_argument("--method", choices=["rahfl", "pccd", "both"], default="pccd")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_import", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    return parser.parse_args()


def prepare_c2net():
    try:
        from c2net.context import prepare

        context = prepare()
        log(f"c2net dataset_path = {getattr(context, 'dataset_path', '')}")
        log(f"c2net output_path  = {getattr(context, 'output_path', '')}")
        return context
    except Exception as exc:  # pragma: no cover - OpenI-only integration.
        log(f"[warning] c2net prepare unavailable: {exc}")
        return None


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
    ]:
        if raw:
            path = Path(raw)
            if path.exists() and path not in roots:
                roots.append(path)
    return roots


def find_package(roots: list[Path], name: str) -> Path:
    for root in roots:
        for direct in [root / name, root / f"{name}.tar.gz"]:
            if direct.exists():
                return direct
        tar_matches = list(root.rglob(f"{name}.tar.gz"))
        if tar_matches:
            return tar_matches[0]
        directory_matches = [path for path in root.rglob(name) if path.is_dir()]
        if directory_matches:
            return directory_matches[0]
    searched = ", ".join(str(path) for path in roots) or "<none>"
    raise FileNotFoundError(f"Could not find {name} under: {searched}")


def selected_methods(method: str) -> list[str]:
    return ["rahfl", "pccd"] if method == "both" else [method]


def package_outputs(methods: list[str]) -> Path:
    tar_path = ROOT / "fedclear_pccd_probe_outputs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for method in methods:
            experiment_dir = ROOT / "outputs" / EXPERIMENTS[method]
            for name in [
                "metrics.csv",
                "corruption_group_acc.csv",
                "client_group_acc.csv",
                "class_corruption_acc.csv",
                "config.resolved.json",
            ]:
                path = experiment_dir / name
                if path.exists():
                    tar.add(path, arcname=f"outputs/{experiment_dir.name}/{name}")
        comparison = ROOT / "outputs" / "pccd_probe_comparison"
        if comparison.exists():
            for path in comparison.iterdir():
                if path.is_file():
                    tar.add(path, arcname=f"outputs/pccd_probe_comparison/{path.name}")
        document = ROOT / "docs/archive/methods/FEDCLEAR_LATEST_THEORY_FRAMEWORK_ZH.md"
        if document.exists():
            tar.add(document, arcname=document.name)
    log(f"Wrote metrics-only package: {tar_path}")
    return tar_path


def upload_outputs(context, tar_path: Path) -> None:
    if context is None:
        log("[warning] c2net context unavailable; skip upload.")
        return
    try:
        from c2net.context import upload_output
    except Exception as exc:  # pragma: no cover
        log(f"[warning] c2net upload_output unavailable: {exc}")
        return
    output_path = Path(context.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tar_path, output_path / tar_path.name)
    upload_output()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    methods = selected_methods(args.method)

    log("===== OpenI FedCLEAR-PCCD probe entry =====")
    log(f"Repository root: {ROOT}")
    log(f"Methods: {methods}")
    context = prepare_c2net()

    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], env)

    if not args.skip_import:
        roots = candidate_roots(args, context)
        for root in roots:
            log(f"Dataset search root: {root}")
        private_source = find_package(roots, PRIVATE_DATASET_NAME)
        public_source = find_package(roots, PUBLIC_DATASET_NAME)
        run(
            [sys.executable, "scripts/import_cle_data.py", "--source", str(private_source), "--destination", "."],
            env,
        )
        run(
            [sys.executable, "scripts/import_cle_public_data.py", "--source", str(public_source)],
            env,
        )

    for method in methods:
        run([sys.executable, "scripts/check_environment.py", "--config", CONFIGS[method]], env)
        if not args.skip_train:
            log(f"===== Running matching {method} probe =====")
            run([sys.executable, "-u", "scripts/run_experiment.py", "--config", CONFIGS[method]], env)

    rahfl_metrics = ROOT / "outputs" / EXPERIMENTS["rahfl"] / "metrics.csv"
    pccd_metrics = ROOT / "outputs" / EXPERIMENTS["pccd"] / "metrics.csv"
    if rahfl_metrics.exists() and pccd_metrics.exists():
        run(
            [
                sys.executable,
                "scripts/analyze_pccd_probe.py",
                "--rahfl",
                str(rahfl_metrics),
                "--pccd",
                str(pccd_metrics),
                "--output-dir",
                "outputs/pccd_probe_comparison",
            ],
            env,
        )

    tar_path = package_outputs(methods)
    if not args.no_upload:
        upload_outputs(context, tar_path)
    log("===== FedCLEAR-PCCD probe complete =====")


if __name__ == "__main__":
    main()
