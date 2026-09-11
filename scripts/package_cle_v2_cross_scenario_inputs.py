from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np


BUNDLE_NAME = "cle_hfl_v2_cross_maps1_2_seed0_split0_with_pew"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package two frozen cross-binding-map inputs.")
    parser.add_argument("--map1-root", type=Path, required=True)
    parser.add_argument("--map2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def manifest_for(root: Path, map_seed: int) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = f"cle_hfl_v2_cross_map{map_seed}_seed0_split0"
    if manifest.get("package") != expected or manifest.get("scenario_id") != expected:
        raise ValueError(f"Unexpected map{map_seed} package identity")
    if int(manifest.get("partition_seed", -1)) != 0:
        raise ValueError("partition_seed must remain zero")
    if int(manifest.get("binding_map_seed", -1)) != map_seed:
        raise ValueError("binding_map_seed mismatch")
    return manifest


def main() -> None:
    args = parse_args()
    roots = {1: args.map1_root.resolve(), 2: args.map2_root.resolve()}
    manifests = {seed: manifest_for(root, seed) for seed, root in roots.items()}
    if manifests[1]["class_operator_map"] == manifests[2]["class_operator_map"]:
        raise ValueError("Cross-scenario binding maps must differ")
    shared_hashes = {}
    for relative in (
        "splits/strict_cle_v2_factorial_seed0_split0.npz",
        "public/cifar-100-python.tar.gz",
        "initial_states/client_0.pt",
        "initial_states/client_1.pt",
        "initial_states/client_2.pt",
        "initial_states/client_3.pt",
        "evaluation/paired_dsa/test_images.npy",
        "evaluation/paired_dsa/test_labels.npy",
        "evaluation/paired_dsa/test_corruption_ids.npy",
        "evaluation/paired_dsa/test_severity_ids.npy",
        "evaluation/paired_dsa/test_source_ids.npy",
        "pew_standard/pew_standard.pt",
    ):
        values = {seed: sha256_file(root / relative) for seed, root in roots.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"Frozen shared asset differs across maps: {relative}")
        shared_hashes[relative] = values[1]
    pair_audit = {"clients": {}}
    for client_id in range(4):
        client_report = {}
        for condition in ("gamma00", "gamma09"):
            relative_root = Path("data") / condition / f"client_{client_id}"
            arrays = {}
            for name in (
                "train_labels",
                "train_source_indices",
                "train_severity_ids",
                "train_corruption_ids",
                "train_images",
            ):
                arrays[name] = {
                    seed: np.load(root / relative_root / f"{name}.npy", allow_pickle=False)
                    for seed, root in roots.items()
                }
            for name in ("train_labels", "train_source_indices", "train_severity_ids"):
                if not np.array_equal(arrays[name][1], arrays[name][2]):
                    raise ValueError(
                        f"Cross-map pairing mismatch: client={client_id} condition={condition} {name}"
                    )
            operator_change = float(
                np.mean(arrays["train_corruption_ids"][1] != arrays["train_corruption_ids"][2])
            )
            image_change = float(
                np.mean(
                    np.any(
                        arrays["train_images"][1] != arrays["train_images"][2],
                        axis=(1, 2, 3),
                    )
                )
            )
            if condition == "gamma00" and (operator_change != 0.0 or image_change != 0.0):
                raise ValueError("gamma00 control data must be identical across binding maps")
            if condition == "gamma09" and (operator_change <= 0.0 or image_change <= 0.0):
                raise ValueError("gamma09 data must respond to the changed binding map")
            client_report[condition] = {
                "labels_sources_severities_identical": True,
                "operator_change_fraction": operator_change,
                "image_change_fraction": image_change,
            }
        pair_audit["clients"][str(client_id)] = client_report
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite input archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        for seed, root in roots.items():
            archive.add(
                root,
                arcname=f"{BUNDLE_NAME}/{root.name}",
            )
    audit = {
        "protocol": "cle_v2_cross_binding_map_input_bundle_v1",
        "bundle": BUNDLE_NAME,
        "archive": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "map_packages": {
            str(seed): {
                "directory": root.name,
                "manifest_sha256": sha256_file(root / "manifest.json"),
                "binding_map_seed": seed,
            }
            for seed, root in roots.items()
        },
        "shared_asset_hashes": shared_hashes,
        "cross_map_pair_audit": pair_audit,
        "only_binding_map_changes": True,
    }
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
