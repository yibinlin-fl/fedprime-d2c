from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_PROTOCOL = "cle_hfl_v2_paired_factorial_seed0_split0_v1"
PEW_PROTOCOL = "cle_v2_factorial_standard_pew_v1"
ANNOTATION_PROTOCOL = "cle_v2_factorial_frozen_pew_annotations_v1"
ARMS = ("h0_b", "h9_b", "l0_b", "l9_b", "h0_p", "h9_p", "l0_p", "l9_p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a paired CLE-v2 factorial package.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--test-sources", type=int, default=1000)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_file(path: Path, expected: dict | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected is not None:
        if int(expected.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Byte-size mismatch: {path}")
        if str(expected.get("sha256", "")).upper() != sha256_file(path):
            raise ValueError(f"SHA256 mismatch: {path}")


def load_trace(path: Path) -> list[tuple[int, int, str]]:
    require_file(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append((int(item["round"]), int(item["client"]), str(item["sha256"])))
    if not rows:
        raise ValueError(f"Empty local trace: {path}")
    return rows


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    manifest_path = package_root / "manifest.json"
    require_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("Unexpected paired factorial data protocol")

    operators = list(manifest["operators"])
    unseen_ids = {operators.index(name) for name in manifest["unseen_operators"]}
    all_source_ids: list[np.ndarray] = []
    for client_id in range(4):
        paired = {}
        for condition in ("gamma00", "gamma09"):
            client_root = package_root / "data" / condition / f"client_{client_id}"
            record = manifest["conditions"][condition]["clients"][str(client_id)]
            arrays = {}
            for name in (
                "train_images",
                "train_labels",
                "train_corruption_ids",
                "train_corruption_method_ids",
                "train_severity_ids",
                "train_source_indices",
            ):
                path = client_root / f"{name}.npy"
                require_file(path, record[f"{name}.npy"])
                arrays[name] = np.load(path, allow_pickle=False)
            if arrays["train_labels"].size != int(args.samples_per_client):
                raise ValueError(f"Unexpected client sample count: {condition}/client_{client_id}")
            if len(np.unique(arrays["train_source_indices"])) != arrays["train_labels"].size:
                raise ValueError(f"Duplicate source IDs within client {client_id}")
            if any(int(value) in unseen_ids for value in np.unique(arrays["train_corruption_ids"])):
                raise ValueError("Unseen operator leaked into private training")
            if not np.array_equal(
                arrays["train_corruption_ids"], arrays["train_corruption_method_ids"]
            ):
                raise ValueError("Reporting operator arrays disagree")
            paired[condition] = arrays
        for name in ("train_labels", "train_source_indices", "train_severity_ids"):
            if not np.array_equal(paired["gamma00"][name], paired["gamma09"][name]):
                raise ValueError(f"Gamma pairing mismatch: client={client_id} array={name}")
        all_source_ids.append(paired["gamma00"]["train_source_indices"])

    concatenated = np.concatenate(all_source_ids)
    if np.unique(concatenated).size != concatenated.size:
        raise ValueError("Private source IDs overlap across clients")

    split_path = package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz"
    require_file(split_path, manifest["strict_split"])
    split = np.load(split_path, allow_pickle=False)
    for client_id in range(4):
        fit = split[f"client_{client_id}_fit"]
        audit = split[f"client_{client_id}_audit"]
        if np.intersect1d(fit, audit).size or np.unique(np.concatenate([fit, audit])).size != int(
            args.samples_per_client
        ):
            raise ValueError(f"Invalid fit/audit partition for client {client_id}")

    dsa_root = package_root / "evaluation/paired_dsa"
    evaluation_record = manifest["evaluation"]["paired_dsa"]
    dsa = {}
    for name in (
        "test_images",
        "test_labels",
        "test_corruption_ids",
        "test_severity_ids",
        "test_source_ids",
    ):
        path = dsa_root / f"{name}.npy"
        require_file(path, evaluation_record[f"{name}.npy"])
        dsa[name] = np.load(path, allow_pickle=False)
    expected_rows = int(args.test_sources) * len(operators)
    if dsa["test_images"].shape[0] != expected_rows or not np.all(dsa["test_severity_ids"] == 3):
        raise ValueError("DSA grid size/severity invariant failed")
    pairs = np.stack([dsa["test_source_ids"], dsa["test_corruption_ids"]], axis=1)
    if np.unique(pairs, axis=0).shape[0] != expected_rows:
        raise ValueError("DSA grid does not contain one row per source/operator pair")
    source_labels = np.empty(int(args.test_sources), dtype=np.int64)
    for source_id in range(int(args.test_sources)):
        values = np.unique(dsa["test_labels"][dsa["test_source_ids"] == source_id])
        if values.size != 1:
            raise ValueError(f"Inconsistent DSA label for source {source_id}")
        source_labels[source_id] = int(values[0])
    if int(args.test_sources) % 10 != 0 or not np.array_equal(
        np.bincount(source_labels, minlength=10),
        np.full(10, int(args.test_sources) // 10),
    ):
        raise ValueError("DSA source labels are not class-balanced")

    public_path = package_root / "public" / manifest["public"]["file"]
    require_file(public_path, manifest["public"])
    for client_id in range(4):
        record = manifest["initialization"]["models"][str(client_id)]
        require_file(package_root / "initial_states" / f"client_{client_id}.pt", record)

    pew_manifest_path = package_root / "pew_standard/manifest.json"
    require_file(pew_manifest_path)
    pew = json.loads(pew_manifest_path.read_text(encoding="utf-8"))
    if pew.get("protocol") != PEW_PROTOCOL or pew.get("max_batches") is not None:
        raise ValueError("PEW is not the standard full preparation protocol")
    checkpoint = package_root / "pew_standard/pew_standard.pt"
    require_file(
        checkpoint,
        {"bytes": pew["checkpoint_bytes"], "sha256": pew["checkpoint_sha256"]},
    )
    for condition in ("gamma00", "gamma09"):
        annotation_root = package_root / "pew_standard/annotations" / condition
        annotation_manifest_path = annotation_root / "manifest.json"
        require_file(annotation_manifest_path)
        annotation_manifest = json.loads(annotation_manifest_path.read_text(encoding="utf-8"))
        if annotation_manifest.get("protocol") != ANNOTATION_PROTOCOL:
            raise ValueError("Unexpected frozen annotation protocol")
        if annotation_manifest.get("pew_checkpoint_sha256") != pew["checkpoint_sha256"]:
            raise ValueError("Annotation/checkpoint lineage mismatch")
        for client_id in range(4):
            record = annotation_manifest["clients"][str(client_id)]
            path = annotation_root / f"client_{client_id}.npz"
            require_file(path, record)
            payload = np.load(path, allow_pickle=False)
            if payload["environment_ids"].shape != (int(args.samples_per_client),):
                raise ValueError("Frozen annotation length mismatch")

    trace_audit = None
    if args.outputs_root is not None:
        outputs_root = args.outputs_root.resolve()
        traces = {
            arm: load_trace(
                outputs_root
                / f"cle_v2_factorial_{arm}_trainseed{args.train_seed}"
                / "local_batch_trace.jsonl"
            )
            for arm in ARMS
        }
        groups = {
            "gamma00": ("h0_b", "l0_b", "h0_p", "l0_p"),
            "gamma09": ("h9_b", "l9_b", "h9_p", "l9_p"),
        }
        for condition, arms in groups.items():
            reference = traces[arms[0]]
            if any(traces[arm] != reference for arm in arms[1:]):
                raise ValueError(f"Local batch/AugMix trace mismatch within {condition}")
        trace_audit = {condition: {"matched": True, "rows": len(traces[arms[0]])} for condition, arms in groups.items()}

    report = {
        "protocol": "cle_v2_factorial_integrity_audit_v1",
        "status": "PASS",
        "package_root": str(package_root),
        "manifest_sha256": sha256_file(manifest_path),
        "private_samples": int(concatenated.size),
        "dsa_sources": int(args.test_sources),
        "dsa_operators": len(operators),
        "pew_checkpoint_sha256": pew["checkpoint_sha256"],
        "paired_local_traces": trace_audit,
        "scientific_evidence": False,
    }
    output = args.output.resolve() if args.output else package_root / "integrity_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
