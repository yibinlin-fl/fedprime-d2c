"""Prepare and verify the Gate 1-Pilot training-only allowlist."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .gate0 import FIT_FILES, SPLIT, read_json, sha256, verify, write_json


HERE = Path(__file__).resolve().parent
FILES = {
    SPLIT,
    "public/cifar-100-python.tar.gz",
    *(f"initial_states/client_{i}.pt" for i in range(4)),
    *(f"private/client_{i}/{name}" for i in range(4) for name in FIT_FILES),
    *(f"cache/client_{i}.npz" for i in range(4)),
}
MANIFEST = "TRAINING_INPUT.json"


def prepare(package, cache, dest):
    config = read_json(HERE / "gate0.json")
    manifest_path = package / "manifest.json"
    package_hash = verify(manifest_path, config["package_manifest_sha256"])
    manifest = read_json(manifest_path)
    fit = read_json(cache / "FIT_MANIFEST.json")
    if fit["config"] != config or fit["split_sha256"] != manifest["strict_split"]["sha256"].lower():
        raise ValueError("Gate 0 cache and main protocol do not match")
    if set(manifest["initialization"]["models"]) != {"0", "1", "2", "3"}:
        raise ValueError("Unexpected checkpoint set")
    public = manifest["public"]
    states = manifest["initialization"]["models"]
    assets = {
        SPLIT: (package / SPLIT, manifest["strict_split"]["sha256"]),
        f"public/{public['file']}": (package / "public" / public["file"], public["sha256"]),
        **{
            f"initial_states/client_{i}.pt": (package / f"initial_states/client_{i}.pt", states[str(i)]["sha256"])
            for i in range(4)
        },
    }
    for i in range(4):
        key = f"gamma09_client{i}"
        checked = fit["records"][key]
        item = checked["scores"]["phi_tex"]
        assets[f"cache/client_{i}.npz"] = (cache / item["file"], item["sha256"])
        for name in FIT_FILES:
            raw = package / f"data/gamma09/client_{i}" / name
            expected = manifest["conditions"]["gamma09"]["clients"][str(i)][name]["sha256"]
            if checked["inputs_sha256"][name].lower() != expected.lower():
                raise ValueError(f"Gate 0 cache used a different input file: {raw}")
            assets[f"private/client_{i}/{name}"] = (raw, expected)
    if set(assets) != FILES:
        raise ValueError("Unexpected training allowlist")
    frozen = {
        "package_manifest_sha256": package_hash,
        "protocol": manifest["protocol"],
        "fit_manifest_sha256": sha256(cache / "FIT_MANIFEST.json"),
        "fit_cache_cpu_seconds": {
            "input_verification_and_descriptors": sum(
                fit["records"][f"gamma09_client{i}"]["feature_seconds"] for i in range(4)
            ),
            "probe_fitting": sum(
                fit["records"][f"gamma09_client{i}"]["scores"]["phi_tex"]["fit_seconds"] for i in range(4)
            ),
        },
        "files": {name: verify(*assets[name]) for name in sorted(assets)},
    }
    dest.mkdir(parents=True, exist_ok=False)
    for name, (source, _) in assets.items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(dest / MANIFEST, frozen)
    validate(dest)


def validate(root):
    paths = list(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Unexpected link in Pilot training inputs")
    actual = {p.relative_to(root).as_posix() for p in paths if p.is_file()}
    if actual != FILES | {MANIFEST}:
        raise ValueError(f"Training allowlist mismatch: {sorted(actual ^ (FILES | {MANIFEST}))}")
    report = read_json(root / MANIFEST)
    if set(report["files"]) != FILES:
        raise ValueError("Training manifest file list differs from the allowlist")
    config = read_json(HERE / "gate0.json")
    if (
        report["package_manifest_sha256"] != config["package_manifest_sha256"]
        or report["protocol"] != config["protocol"]
    ):
        raise ValueError("Pilot input does not use the frozen main protocol")
    for name in sorted(FILES):
        verify(root / name, report["files"][name])
    return report


def load_aligned_cache(path, fit_rows, source_ids, labels):
    with np.load(path, allow_pickle=False) as data:
        ids, sources, targets, bias = (data[key] for key in ("sample_ids", "source_ids", "labels", "bias"))
    if (
        not np.array_equal(ids, np.sort(fit_rows))
        or not np.array_equal(sources, source_ids[ids])
        or not np.array_equal(targets, labels[ids])
        or bias.shape != (len(ids), 10)
        or not np.isfinite(bias).all()
        or np.any(np.abs(bias) > 3.0)
    ):
        raise ValueError(f"Gate 0 cache row/source/label alignment failed: {path}")
    return ids, bias.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.package_root, args.cache_root, args.output)
