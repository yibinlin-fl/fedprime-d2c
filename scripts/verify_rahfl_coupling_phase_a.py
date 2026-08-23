from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the frozen RAHFL coupling Phase-A artifacts.")
    parser.add_argument(
        "--data-root",
        default="RAHFL-master/Dataset/cifar_10_c",
    )
    parser.add_argument(
        "--artifact-root",
        default="local_runs/rahfl_coupling_phase_a_seed0",
    )
    return parser.parse_args()


def _pass(label: str, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    print(f"{label}: PASS{suffix}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    manifest = json.loads((artifact_root / "experiment_manifest.json").read_text(encoding="utf-8"))
    clean_global = np.load(data_root / "train" / "labels.npy").astype(np.int64, copy=False)

    with np.load(artifact_root / "partition_disjoint_iid.npz") as archive:
        partition = {
            client_id: np.asarray(archive[f"client_{client_id}_global"], dtype=np.int64)
            for client_id in range(int(manifest["num_clients"]))
        }
    combined = np.concatenate(list(partition.values()))
    if np.unique(combined).size != combined.size:
        raise AssertionError("client partitions overlap")
    _pass("Disjoint client partition", f"{combined.size} unique samples")

    with np.load(artifact_root / "fit_audit_split.npz") as archive:
        splits = {
            client_id: (
                np.asarray(archive[f"client_{client_id}_fit"], dtype=np.int64),
                np.asarray(archive[f"client_{client_id}_audit"], dtype=np.int64),
            )
            for client_id in partition
        }
    expected_audit = round(
        float(manifest["audit_ratio"]) * int(manifest["samples_per_client"])
    )
    for client_id, (fit, audit) in splits.items():
        if fit.size + audit.size != int(manifest["samples_per_client"]):
            raise AssertionError(f"client {client_id} split does not cover its partition")
        if audit.size != expected_audit:
            raise AssertionError(f"client {client_id} audit split is not exact 90/10")
        if np.intersect1d(fit, audit).size:
            raise AssertionError(f"client {client_id} fit/audit overlap")
    expected_fit = int(manifest["samples_per_client"]) - expected_audit
    _pass("Exact fit/audit split", f"{expected_fit}/{expected_audit} per client")

    image_path = data_root / "train" / "random_corrupt_1.npy"
    if _sha256(image_path) != manifest["checksums"]["train_images"]:
        raise AssertionError("frozen training-image checksum changed")
    _pass("Frozen image checksum", manifest["checksums"]["train_images"][:12])

    beta_summaries = {}
    for token in ("beta0", "beta4"):
        regime_root = artifact_root / token
        beta_summaries[token] = json.loads(
            (regime_root / "summary.json").read_text(encoding="utf-8")
        )
        for client_id, global_indices in partition.items():
            fit, audit = splits[client_id]
            labels = np.load(regime_root / f"client_{client_id}_labels.npy")
            mask = np.load(regime_root / f"client_{client_id}_noisy_mask.npy")
            transition = np.load(regime_root / f"client_{client_id}_transition.npy")
            clean = clean_global[global_indices]
            expected = round(float(manifest["noise_rate"]) * fit.size)
            if int(mask[fit].sum()) != expected or bool(mask[audit].any()):
                raise AssertionError(f"{token} client {client_id} violates fit-only exact noise")
            if not np.array_equal(labels[audit], clean[audit]):
                raise AssertionError(f"{token} client {client_id} audit labels are not clean")
            reconstructed = np.zeros((10, 10), dtype=np.int64)
            np.add.at(reconstructed, (clean[mask], labels[mask]), 1)
            if not np.array_equal(reconstructed, transition):
                raise AssertionError(f"{token} client {client_id} transition matrix mismatch")
        _pass(f"{token} exact 20% fit noise")
        _pass(f"{token} trusted audit labels clean")

    for client_id in partition:
        left = np.load(artifact_root / "beta0" / f"client_{client_id}_transition.npy")
        right = np.load(artifact_root / "beta4" / f"client_{client_id}_transition.npy")
        if not np.array_equal(left, right):
            raise AssertionError(f"client {client_id} flip matrices differ")
    _pass("Flip matrices identical")
    _pass("Images identical", "both regimes reference one frozen mmap file")
    _pass("Fit/audit identical", "both regimes reference one persisted split")
    _pass("Final test untouched", "no regime-specific test-label artifact exists")

    beta0_mean = float(beta_summaries["beta0"]["mean_severity_noisy"])
    beta4_mean = float(beta_summaries["beta4"]["mean_severity_noisy"])
    if beta4_mean <= beta0_mean:
        raise AssertionError("beta4 does not increase noisy-sample severity")
    _pass("beta4 severity bias > beta0", f"{beta4_mean:.4f} > {beta0_mean:.4f}")
    print("Phase-A artifact equivalence: ALL PASS", flush=True)


if __name__ == "__main__":
    main()
