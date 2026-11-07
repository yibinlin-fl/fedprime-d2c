"""Publish one atomic rolling snapshot in a separate process."""
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

from scripts.openi_cle_v2_factorial_entry import prepare_c2net
from scripts.openi_cle_v2_cross_scenario_entry import upload


def publish(experiment_root, configs_root, archive, context):
    if context is None or not getattr(context, "output_path", None):
        raise RuntimeError("Progress upload context unavailable")
    recovery = experiment_root / "checkpoints" / "round_recovery"
    progress = json.loads((recovery / "PROGRESS.json").read_text(encoding="utf-8"))
    completed = int(progress["completed_rounds"])
    archive = archive.with_name(archive.name.replace(
        "_progress_outputs", f"_progress_round_{completed:03d}_outputs"
    ))
    if archive.exists():
        raise FileExistsError(f"Refusing to replace published round snapshot: {archive}")
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as handle:
        handle.add(recovery / "round_state.pt", arcname="round_state.pt")
        handle.add(recovery / "PROGRESS.json", arcname="PROGRESS.json")
        for path in sorted(experiment_root.iterdir()):
            if path.is_file() and path.suffix in {".csv", ".jsonl", ".json"}:
                handle.add(path, arcname="history/" + path.name)
        handle.add(configs_root, arcname="configs")
    os.replace(temporary, archive)
    upload(context, [archive, recovery / "PROGRESS.json"])
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--configs-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    publish(args.experiment_root, args.configs_root, args.archive, prepare_c2net())
