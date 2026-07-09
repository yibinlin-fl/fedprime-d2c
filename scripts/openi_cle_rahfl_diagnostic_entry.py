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
CONFIG_BY_GAMMA = {
    "00": "configs/diagnostic_rahfl_cle_alpha05_gamma00.yaml",
    "06": "configs/diagnostic_rahfl_cle_alpha05_gamma06.yaml",
    "09": "configs/diagnostic_rahfl_cle_alpha05_gamma09.yaml",
}


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=env)


def prepare_c2net():
    try:
        from c2net.context import prepare

        ctx = prepare()
        log(f"c2net dataset_path = {getattr(ctx, 'dataset_path', '')}")
        log(f"c2net output_path  = {getattr(ctx, 'output_path', '')}")
        return ctx
    except Exception as exc:  # pragma: no cover - only used outside OpenI.
        log(f"[warning] c2net prepare failed or unavailable: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI entry for CLE-HFL RAHFL diagnostics.")
    parser.add_argument(
        "--gammas",
        default="00,06,09",
        help="Comma-separated gamma suffixes to run. Use 00,06,09 by default.",
    )
    parser.add_argument(
        "--data_source",
        default="",
        help="Mounted OpenI dataset path. If empty, c2net dataset_path and common paths are searched.",
    )
    parser.add_argument("--skip_install", action="store_true", help="Skip pip install -r requirements.txt.")
    parser.add_argument("--skip_import", action="store_true", help="Skip importing mounted prepared datasets.")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only package current outputs.")
    parser.add_argument("--skip_summary", action="store_true", help="Skip output summarization.")
    parser.add_argument("--no_upload", action="store_true", help="Do not upload outputs through c2net.")
    return parser.parse_args()


def candidate_roots(args: argparse.Namespace, ctx) -> list[Path]:
    roots: list[Path] = []
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


def find_dataset(gamma: str, roots: list[Path]) -> Path:
    pattern = f"cle_hfl_prepared_alpha05_gamma{gamma}_seed0"
    for root in roots:
        direct_dir = root / pattern
        direct_tar = root / f"{pattern}.tar.gz"
        if direct_dir.is_dir():
            return direct_dir
        if direct_tar.is_file():
            return direct_tar
        matches = list(root.rglob(f"{pattern}.tar.gz"))
        if matches:
            return matches[0]
        dir_matches = [p for p in root.rglob(pattern) if p.is_dir()]
        if dir_matches:
            return dir_matches[0]
    searched = ", ".join(str(p) for p in roots) or "<none>"
    raise FileNotFoundError(f"Could not find {pattern}.tar.gz under: {searched}")


def package_outputs() -> Path:
    out = ROOT / "outputs"
    tar_path = ROOT / "cle_rahfl_diagnostic_outputs.tar.gz"
    log("===== Packaging outputs =====")
    with tarfile.open(tar_path, "w:gz") as tar:
        if out.exists():
            tar.add(out, arcname="outputs")
    log(f"Wrote {tar_path}")
    return tar_path


def upload_outputs(ctx, tar_path: Path) -> None:
    if ctx is None:
        log("[warning] c2net context unavailable; skip upload.")
        return
    try:
        from c2net.context import upload_output
    except Exception as exc:  # pragma: no cover - only used outside OpenI.
        log(f"[warning] c2net upload_output unavailable: {exc}")
        return

    output_path = Path(ctx.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for name in ["outputs", tar_path.name]:
        src = ROOT / name
        if not src.exists():
            continue
        dst = output_path / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        log(f"Copied {src} -> {dst}")
    upload_output()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log("===== OpenI CLE-HFL RAHFL diagnostic entry =====")
    log(f"Repository root: {ROOT}")
    ctx = prepare_c2net()

    gammas = [g.strip() for g in args.gammas.split(",") if g.strip()]
    configs = [CONFIG_BY_GAMMA[g] for g in gammas]
    log("Configs:")
    for cfg in configs:
        log(f"  {cfg}")

    if not args.skip_install:
        log("===== Installing dependencies =====")
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], env=env)

    if not args.skip_import:
        log("===== Importing CLE-HFL datasets =====")
        roots = candidate_roots(args, ctx)
        log("Dataset search roots:")
        for root in roots:
            log(f"  {root}")
        for gamma in gammas:
            source = find_dataset(gamma, roots)
            log(f"Importing gamma={gamma}: {source}")
            run([sys.executable, "scripts/import_cle_data.py", "--source", str(source), "--destination", "."], env=env)

    log("===== Environment check =====")
    run([sys.executable, "scripts/check_environment.py", "--config", configs[0]], env=env)

    if not args.skip_train:
        log("===== Running RAHFL CLE-HFL diagnostics =====")
        run([sys.executable, "scripts/run_grid.py", *configs], env=env)

    if not args.skip_summary:
        log("===== Summarizing results =====")
        run([sys.executable, "scripts/summarize_results.py", "--outputs", "outputs"], env=env)

    tar_path = package_outputs()
    if not args.no_upload:
        log("===== Uploading outputs through c2net =====")
        upload_outputs(ctx, tar_path)

    log("===== Done =====")
    log("Metrics include avg_acc, worst_acc, WCCA, and CFG.")


if __name__ == "__main__":
    main()
