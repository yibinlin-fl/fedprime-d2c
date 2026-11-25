"""Compare the two Pilot smoke trajectories with the historical ERM fit trace."""

import argparse
import json
import math
from pathlib import Path
import tarfile

from .gate0 import read_json, verify, write_json
from .pilot_train import PILOT


def verify_smoke(smoke_root, erm_archive):
    arms = tuple(read_json(PILOT)["arms"])
    for arm in arms:
        completion = read_json(smoke_root / arm / "TRAINING_COMPLETE.json")
        if completion["mode"] != "smoke" or completion["completed_rounds"] != 2:
            raise ValueError(f"Not a complete two-round smoke: {arm}")
    names = [
        "local_batch_trace.jsonl", "fit_identity_trace.jsonl", "public_trace.jsonl",
        *(f"audit_trace_c{i}.jsonl" for i in range(4)),
    ]
    counts = {}
    for name in names:
        left, right = ((smoke_root / arm / name).read_bytes() for arm in arms)
        if left != right or not left.strip():
            raise AssertionError(f"Smoke pairing failed: {name}")
        counts[name] = len(left.splitlines())
    if [counts[n] for n in names[:3]] != [8, 16, 8]:
        raise AssertionError("Unexpected smoke trace lengths")
    gate0 = read_json(PILOT.parent / "gate0.json")
    verify(erm_archive, gate0["baseline_recovery_archives_sha256"][erm_archive.name])
    with tarfile.open(erm_archive, "r:gz") as archive:
        member, = [name for name in archive.getnames() if name.endswith("/local_batch_trace.jsonl")]
        original = [json.loads(line) for line in archive.extractfile(member) if line.strip()]
    smoke = [
        json.loads(line)
        for line in (smoke_root / arms[0] / "local_batch_trace.jsonl").read_text().splitlines()
    ]
    original = {(row["round"], row["client"]): row for row in original}
    matches = sum(
        all(row[k] == original[(row["round"], row["client"])][k] for k in ("label_sha256", "sha256"))
        for row in smoke
    )
    if matches != len(smoke):
        raise AssertionError(f"Only {matches}/{len(smoke)} held-out fit traces match the frozen ERM")
    for arm in arms:
        rows = [json.loads(line) for line in (smoke_root / arm / "ssa_loss_trace.jsonl").read_text().splitlines()]
        if len(rows) != 8 or any(not math.isclose(
            row["ce_weight"] * row["ce"] + row["poe_weight"] * row["poe"],
            row["total"], abs_tol=1e-6,
        ) for row in rows):
            raise AssertionError(f"Loss scale failed: {arm}")
    return {"status": "smoke_pass", "scientific_evidence": False, "paired_trace_counts": counts,
            "historical_erm_first_batch_view_and_label_matches": matches}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--erm-archive", required=True, type=Path)
    args = parser.parse_args()
    result = verify_smoke(args.smoke_root, args.erm_archive)
    write_json(args.smoke_root / "PAIRING_PASS.json", result)
    print(json.dumps(result, indent=2))
