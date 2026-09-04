# coding=utf-8
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openi_cle_k1_c_crsf_surgery_entry import (  # noqa: E402
    INPUT_ARCHIVE,
    INPUT_BYTES,
    INPUT_DIRECTORY,
    INPUT_SHA256,
    candidate_roots,
    find_input,
    log,
    prepare_c2net,
    run,
    safe_extract,
    sha256_file,
    upload,
    verify_input,
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI K1-C-Minimal causal intervention gate.")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="smoke")
    parser.add_argument("--data-source", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--confirm-formal",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Formal cost approval; accepts a bare flag or an explicit true value for OpenI forms.",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def package(output_dir: Path, mode: str) -> Path:
    archive = ROOT / "outputs" / f"cle_k1_c_minimal_seed0_{mode}_outputs.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_dir, arcname=f"outputs/{output_dir.name}")
        for path in (
            ROOT / "docs/experiments/current/CLE_K1_C_MINIMAL_CAUSAL_GATE_OPENI_ZH.md",
            ROOT / "configs/cle_k1_c_minimal_seed0.json",
        ):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(ROOT).as_posix())
    return archive


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise ValueError("Formal is locked until benchmark cost approval; pass --confirm-formal explicitly")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    source = find_input(candidate_roots(args, context))
    if source.stat().st_size != INPUT_BYTES or sha256_file(source) != INPUT_SHA256:
        raise ValueError("Phase-B0 input archive bytes/hash mismatch")
    # Keep smoke/benchmark/formal extraction roots isolated.  In particular, a
    # later lightweight run must not inherit evaluation assets extracted by a
    # previous formal run.
    extraction = ROOT / f"local_runs/cle_k1_c_minimal_openi_input_{args.mode}"
    safe_extract(source, extraction, include_evaluation=args.mode == "formal")
    input_root = extraction / INPUT_DIRECTORY
    source_input_manifest = verify_input(input_root, include_evaluation=args.mode == "formal")
    output_dir = ROOT / "outputs" / f"cle_k1_c_minimal_seed0_{args.mode}"
    command = [
        sys.executable,
        "scripts/run_cle_k1_c_minimal.py",
        "--mode",
        args.mode,
        "--public-root",
        str(input_root / "public"),
        "--checkpoint-root",
        str(input_root / "checkpoints"),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.mode == "formal":
        command.extend(["--evaluation-root", str(input_root / "evaluation"), "--confirm-formal"])
    run(command, environment)
    run_manifest = {
        "protocol": "cle_k1_c_minimal_openi_v1",
        "mode": args.mode,
        "input_archive": source.name,
        "input_bytes": source.stat().st_size,
        "input_sha256": sha256_file(source),
        "source_checkpoint_kind": source_input_manifest.get("checkpoint_kind"),
        "evaluation_extracted": args.mode == "formal",
        "training_performed": False,
        "communication_modified": False,
        "full_checkpoints_written": False,
        "old_full_calibration_run": False,
    }
    run_manifest_path = output_dir / "openi_run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    archive = package(output_dir, args.mode)
    if not args.no_upload:
        upload(
            context,
            [
                archive,
                run_manifest_path,
                output_dir / "result.json",
                output_dir / "selection_manifest.json",
                output_dir / "artifact_manifest.json",
                output_dir / "FINAL_REPORT_ZH.md",
            ],
        )
    log(f"===== CLE K1-C-Minimal {args.mode} complete =====")


if __name__ == "__main__":
    main()
