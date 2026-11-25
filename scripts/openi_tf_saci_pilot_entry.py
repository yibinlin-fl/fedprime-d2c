#!/usr/bin/env python
"""OpenI single-arm TF-SACI Pilot, with checkpoints uploaded before joint analysis."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import shutil
import sys
import tarfile
from pathlib import Path
import subprocess
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openi_cle_v2_factorial_entry import candidate_roots, prepare_c2net
from scripts.openi_cle_v2_cross_scenario_entry import upload
from scripts.openi_cle_v2_moderate_entry import package_recovery_outputs, require_upload_context


ARCHIVE = "tf_saci_gate1_pilot_training_input.tar.gz"
ARMS = ("ssa_poe", "ssa_poe_ce_anchor")


def find_input(roots):
    matches = {
        item.resolve()
        for root in roots
        for item in (root.rglob(ARCHIVE) if root.is_dir() else [root])
        if item.is_file() and item.name == ARCHIVE
    }
    if len(matches) != 1:
        raise ValueError(f"Expected one {ARCHIVE}, got {len(matches)}")
    return matches.pop()


def extract_input(source, target, expected_sha256):
    from experiments.tf_saci.gate0 import verify
    from experiments.tf_saci.pilot_data import FILES, MANIFEST, validate

    verify(source, expected_sha256)
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        if (
            len(members) != len(FILES) + 1 or {m.name for m in members} != FILES | {MANIFEST}
            or any(not m.isfile() for m in members)
        ):
            raise ValueError("OpenI training archive contains unexpected paths or links")
        target.mkdir(parents=True, exist_ok=False)
        for item in members:
            output = target / item.name
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as input_file, output.open("xb") as output_file:
                shutil.copyfileobj(input_file, output_file)
    validate(target)


class Tee:
    def __init__(self, main, log):
        self.main = main
        self.log = log

    def write(self, text):
        self.main.write(text)
        return self.log.write(text)

    def flush(self):
        self.main.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--confirm_pilot", choices=("true", "false"), default="false")
    parser.add_argument("--input_sha256", required=True)
    parser.add_argument("--data_source", default="")
    parser.add_argument("--skip_install", choices=("true", "false"), default="false")
    args = parser.parse_args()
    if args.confirm_pilot != "true":
        parser.error("Pilot requires confirm_pilot=true")

    context = prepare_c2net()
    require_upload_context(context)
    if args.skip_install != "true":
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    from experiments.tf_saci.gate0 import read_json, write_json
    from experiments.tf_saci.pilot_train import train

    source = find_input(candidate_roots(args, context))
    stage = ROOT / "local_runs" / f"openi_tf_saci_{args.arm}"
    stage.mkdir(parents=True, exist_ok=False)
    target = stage / "training_input"
    extract_input(source, target, args.input_sha256)
    result = stage / "results"
    log = stage / f"tf_saci_{args.arm}_TRAIN.log"
    status = stage / f"tf_saci_{args.arm}_STATUS.json"
    run = result / args.arm

    def milestone(output, name):
        archive = stage / f"tf_saci_{args.arm}_{name}_weights.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=2) as snapshot:
            snapshot.add(output / "checkpoints" / name, arcname="checkpoints")
            for entry in ("RUN_STARTED.json", "config.resolved.json", "metrics.csv",
                          "ssa_loss_trace.jsonl", "local_batch_trace.jsonl", "teacher_edges.csv"):
                if (output / entry).is_file():
                    snapshot.add(output / entry, arcname=entry)
            snapshot.add(log, arcname=log.name)
        upload(context, [archive])

    with log.open("x", encoding="utf-8", buffering=1) as handle:
        with redirect_stdout(Tee(sys.stdout, handle)), redirect_stderr(Tee(sys.stderr, handle)):
            try:
                train(target, run, args.arm, "pilot", "cuda", milestone)
            except Exception as exc:
                traceback.print_exc()
                write_json(status, {"arm": args.arm, "status": "training_incomplete",
                                    "scientific_evidence": False, "error_type": type(exc).__name__})
                artifacts = [log, status]
                if run.is_dir():
                    partial = stage / f"tf_saci_{args.arm}_partial_outputs.tar.gz"
                    package_recovery_outputs(result, stage / "no_configs", partial)
                    artifacts.append(partial)
                upload(context, artifacts)
                raise
    final = stage / f"tf_saci_{args.arm}_recovery_outputs.tar.gz"
    package_recovery_outputs(result, stage / "no_configs", final)
    write_json(status, read_json(run / "TRAINING_COMPLETE.json"))
    upload(context, [final, log, status])


if __name__ == "__main__":
    main()
