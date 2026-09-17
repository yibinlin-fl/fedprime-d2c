#!/usr/bin/python
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

from scripts.openi_cle_v2_factorial_entry import candidate_roots, find_archive, prepare_c2net, run, safe_extract  # noqa: E402
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs, upload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenI matched ERM vs JTT Formal.")
    parser.add_argument("--mode", choices=("benchmark", "formal"), default="benchmark")
    parser.add_argument("--confirm_formal", choices=("false", "true"), default="false")
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("false", "true"), default="false")
    args = parser.parse_args()
    if args.mode == "formal" and args.confirm_formal != "true":
        raise PermissionError("Formal is locked; set confirm_formal=true after explicit approval")
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if args.skip_install != "true":
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    package_root = safe_extract(find_archive(candidate_roots(args, context)), ROOT / "local_runs/openi_cle_jtt_formal_input")
    outputs_root = ROOT / f"outputs/cle_jtt_formal_{args.mode}"
    configs_root = ROOT / f"local_runs/cle_jtt_formal_{args.mode}_configs"
    analysis_root = ROOT / f"outputs/cle_jtt_formal_{args.mode}_analysis"
    outputs_root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "scripts/run_cle_jtt_formal.py", "--package-root", str(package_root), "--mode", args.mode, "--device", "cuda", "--output-root", str(outputs_root), "--config-root", str(configs_root)]
    if args.mode == "formal":
        command.append("--confirm-formal")
    started = time.perf_counter()
    run(command, environment)
    training_seconds = time.perf_counter() - started
    analysis_root.mkdir(parents=True, exist_ok=True)
    analysis_command = [sys.executable, "-u", "scripts/analyze_cle_jtt_formal.py", "--package-root", str(package_root), "--outputs-root", str(outputs_root), "--output-dir", str(analysis_root), "--mode", args.mode, "--device", "cuda"]
    if args.mode == "formal":
        analysis_command.append("--confirm-formal")
    else:
        analysis_command.extend(["--bootstrap-samples", "100"])
    analysis_started = time.perf_counter()
    run(analysis_command, environment)
    analysis_seconds = time.perf_counter() - analysis_started
    timing = outputs_root / "RUN_TIMING.json"
    timing.write_text(json.dumps({"mode": args.mode, "training_seconds": training_seconds, "analysis_seconds": analysis_seconds, "jtt_two_training_stages": True, "scientific_evidence": args.mode == "formal", "warning": None if args.mode == "formal" else "Benchmark is not scientific evidence."}, indent=2), encoding="utf-8")
    archive = package_light_outputs(args.mode, outputs_root, configs_root, analysis_root, ROOT / f"cle_jtt_formal_{args.mode}_outputs.tar.gz")
    upload(context, [archive, timing])


if __name__ == "__main__":
    main()
