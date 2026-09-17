from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cle_v2_factorial import load_trace, require_file, sha256_file  # noqa: E402
from scripts.run_cle_cifar100_submission import ARMS, PROTOCOL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CIFAR-100-private CLE submission assets/results.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--skip-pew", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    manifest_path = package_root / "manifest.json"
    require_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or int(manifest.get("num_classes", -1)) != 100:
        raise ValueError("Unexpected CIFAR-100 submission manifest")
    split_path = package_root / "splits/strict_cle_cifar100_seed0_split0.npz"
    require_file(split_path, manifest["strict_split"])
    split = np.load(split_path, allow_pickle=False)
    for client_id in range(4):
        fit, audit = split[f"client_{client_id}_fit"], split[f"client_{client_id}_audit"]
        if np.intersect1d(fit, audit).size:
            raise ValueError(f"fit/audit overlap for client {client_id}")
        for condition in ("gamma00", "gamma09"):
            root = package_root / "data" / condition / f"client_{client_id}"
            for name, record in manifest["conditions"][condition]["clients"][str(client_id)].items():
                if name.endswith(".npy"):
                    require_file(root / name, record)
    dsa_root = package_root / "evaluation/paired_dsa"
    dsa_record = manifest["evaluation"]["paired_dsa"]
    arrays = {}
    for name in ("test_images", "test_labels", "test_corruption_ids", "test_severity_ids", "test_source_ids"):
        path = dsa_root / f"{name}.npy"
        require_file(path, dsa_record[f"{name}.npy"])
        arrays[name] = np.load(path, allow_pickle=False)
    sources = np.unique(arrays["test_source_ids"])
    operators = list(manifest["operators"])
    if arrays["test_images"].shape[0] != sources.size * len(operators):
        raise ValueError("Incomplete paired DSA grid")
    if not np.all(arrays["test_severity_ids"] == 3):
        raise ValueError("DSA severity is not fixed at 3")
    labels = np.empty(sources.size, dtype=np.int64)
    for source in sources:
        values = np.unique(arrays["test_labels"][arrays["test_source_ids"] == source])
        if values.size != 1:
            raise ValueError("Inconsistent source label")
        labels[int(source)] = int(values[0])
    counts = np.bincount(labels, minlength=100)
    if not np.all(counts == counts[0]):
        raise ValueError("Evaluation sources are not class-balanced")
    pew = None
    if not args.skip_pew:
        pew_manifest = package_root / "pew_standard/manifest.json"
        require_file(pew_manifest)
        pew = json.loads(pew_manifest.read_text(encoding="utf-8"))
        checkpoint = package_root / "pew_standard/pew_standard.pt"
        require_file(checkpoint, {"bytes": pew["checkpoint_bytes"], "sha256": pew["checkpoint_sha256"]})

    trace_report = None
    if args.outputs_root is not None:
        root = args.outputs_root.resolve()
        traces = {
            arm: load_trace(root / f"cle_cifar100_submission_{arm}_trainseed{args.train_seed}" / "local_batch_trace.jsonl")
            for arm in ARMS
        }
        matched = all(traces[arm] == traces["erm"] for arm in ("cvar_dro", "pew_ber"))
        if not matched:
            raise ValueError("gamma09 local batch traces are not matched")
        trace_report = {"gamma09_matched": True, "rows": len(traces["erm"])}
    report = {
        "protocol": "cle_cifar100_submission_integrity_audit_v1",
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "private_dataset": "cifar100",
        "public_dataset": "cifar10",
        "dsa_sources": int(sources.size),
        "dsa_operators": len(operators),
        "pew_audited": not args.skip_pew,
        "pew_checkpoint_sha256": None if pew is None else pew["checkpoint_sha256"],
        "pew_reused_from": None if pew is None else pew.get("reused_from"),
        "paired_local_traces": trace_report,
        "scientific_evidence": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
