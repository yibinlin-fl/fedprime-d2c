"""Audit candidate CIFAR-10 train-source IDs using frozen metadata and index arrays."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MAIN_PATH = "cle_v2_moderate_alpha10_rho05/cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0"
A1A_PATH = "cle_shortcut_amplification_phase_a1a/cle_shortcut_amplification_phase_a1a_seed0"
AXIS1_PATH = (
    "cle_hfl_learning_floor_validation/cle_hfl_v2_cross_maps1_2_seed0_split0_with_pew/"
    "cle_hfl_v2_cross_map2_seed0_split0"
)
PEER_PATH = "peer_corruption_20261110/peer_corruption_input.npz"
MANIFEST_HASHES = {
    MAIN_PATH: "7255cd082055c9ca7135df1f26634765a2baf6d36d9559970e1b09ef5ee6db02",
    A1A_PATH: "a314bc4d1ad79d5bf784e20679e0f55220a4d51d825ca186d3e571d63cfc55f6",
    AXIS1_PATH: "a5d272149f6f31fc1ce117c6b936aa35fb12b423a1449a076dbd8a6094535196",
}
PEER_HASH = "29a9f8625ab432b2e552de07f2828fbdc97ba8d4477973cfd553a010a59def47"
SOURCE_NAME = "train_source_indices.npy"
TRAIN_SIZE = 50_000


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path}")


def check_ids(ids):
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids >= TRAIN_SIZE)):
        raise ValueError("Expected CIFAR-10 train IDs in [0, 50000)")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Repeated IDs in a disjoint source set")


def private_roles(root, relative, split_name, gammas):
    package = root / relative
    manifest_path = package / "manifest.json"
    verify(manifest_path, MANIFEST_HASHES[relative])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_path = package / "splits" / split_name
    hash_key = "file_sha256" if relative == A1A_PATH else "sha256"
    verify(split_path, manifest["strict_split"][hash_key])
    roles = {"fit": [], "audit": []}
    with np.load(split_path, allow_pickle=False) as split:
        for client in range(4):
            sources = {}
            for gamma in gammas:
                path = package / "data" / gamma / f"client_{client}" / SOURCE_NAME
                sha = manifest["conditions"][gamma]["clients"][str(client)][SOURCE_NAME][hash_key]
                verify(path, sha)
                sources[gamma] = np.load(path, allow_pickle=False)
            ids = sources["gamma09"]
            if not all(np.array_equal(ids, condition) for condition in sources.values()):
                raise ValueError(f"Gamma conditions are not source-matched for {package}, client {client}")
            fit_rows = split[f"client_{client}_fit"]
            audit_rows = split[f"client_{client}_audit"]
            if not np.array_equal(np.sort(np.r_[fit_rows, audit_rows]), np.arange(len(ids))):
                raise ValueError(f"Fit and audit are not a partition for {package}, client {client}")
            roles["fit"].append(ids[fit_rows])
            roles["audit"].append(ids[audit_rows])
    result = {role: np.concatenate(values) for role, values in roles.items()}
    check_ids(np.r_[result["fit"], result["audit"]])
    return result


def overlap(candidate, roles):
    return {
        role: {"size": int(ids.size), "overlap": int(np.intersect1d(ids, candidate).size)}
        for role, ids in roles.items()
    }


def audit(root):
    split = "strict_cle_v2_factorial_seed0_split0.npz"
    main = private_roles(root, MAIN_PATH, split, ("gamma00", "gamma05", "gamma09"))
    candidate = np.setdiff1d(np.arange(TRAIN_SIZE), np.r_[main["fit"], main["audit"]])
    a1a = private_roles(root, A1A_PATH, "strict_cle_v1_alpha05_gamma_pair_seed0_split0.npz", ("gamma00", "gamma09"))
    axis1 = private_roles(root, AXIS1_PATH, split, ("gamma00", "gamma09"))
    peer_path = root / PEER_PATH
    verify(peer_path, PEER_HASH)
    with np.load(peer_path, allow_pickle=False) as data:
        peer = {role: data[f"{role}_ids"].ravel() for role in ("fit", "route", "eval")}
    check_ids(np.concatenate(list(peer.values())))
    used = np.unique(np.concatenate([ids for roles in (a1a, axis1, peer) for ids in roles.values()]))
    known_overlap = np.intersect1d(candidate, used)
    return {
        "decision": "HOLD_DEV_PROVENANCE",
        "namespace": "original CIFAR-10 train split",
        "candidate_size": int(candidate.size),
        "candidate_sorted_le_int64_sha256": hashlib.sha256(candidate.astype("<i8").tobytes()).hexdigest(),
        "main_map2": overlap(candidate, main),
        "phase_a1a_40_round_formal": overlap(candidate, a1a),
        "axis1_map2_formal": overlap(candidate, axis1),
        "peer_corruption_kaggle_pilot": overlap(candidate, peer),
        "confirmed_historical_overlap_union": int(known_overlap.size),
        "unverified_remainder": int(candidate.size - known_overlap.size),
        "manifest_sha256": MANIFEST_HASHES,
        "peer_input_sha256": PEER_HASH,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-runs", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.local_runs), indent=2))


if __name__ == "__main__":
    main()
