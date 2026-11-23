"""Fit frozen OOF scores, then optionally audit them in a separate process."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits

from .ssa import ProbeResult, descriptors, fit_oof


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPLIT = "splits/strict_cle_v2_factorial_seed0_split0.npz"
FIT_FILES = ("train_images.npy", "train_labels.npy", "train_source_indices.npy")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify(path, expected):
    digest = sha256(path)
    if digest != expected.lower():
        raise ValueError(f"SHA256 mismatch: {path}")
    return digest


def load_fit(path, fit_rows, audit_rows, expected):
    for name in FIT_FILES:
        verify(path / name, expected[name]["sha256"])
    images, labels, sources = (
        np.load(path / name, mmap_mode="r", allow_pickle=False) for name in FIT_FILES
    )
    if not (len(images) == len(labels) == len(sources)):
        raise ValueError(f"Training arrays have different lengths: {path}")
    rows = np.concatenate((fit_rows, audit_rows))
    if not np.array_equal(np.sort(rows), np.arange(len(images))):
        raise ValueError(f"Expected a complete disjoint fit/audit partition: {path}")
    if np.intersect1d(sources[fit_rows], sources[audit_rows]).size:
        raise ValueError(f"Source IDs cross fit/audit: {path}")
    return images[fit_rows], labels[fit_rows], sources[fit_rows]


def fit(package, output, config):
    manifest_path = package / "manifest.json"
    verify(manifest_path, config["package_manifest_sha256"])
    manifest = read_json(manifest_path)
    if manifest["protocol"] != config["protocol"] or manifest["binding_map_seed"] != 2:
        raise ValueError("Expected the frozen main protocol with map2")
    split_path = package / SPLIT
    split_hash = verify(split_path, manifest["strict_split"]["sha256"])
    initial_states = {
        client: verify(package / "initial_states" / f"client_{client}.pt", record["sha256"])
        for client, record in manifest["initialization"]["models"].items()
    }
    artifacts = (
        "scripts/run_cle_v2_moderate.py",
        "scripts/analyze_cle_v2_moderate.py",
        "fedprime/engine/cle_v2_factorial.py",
        "docs/research/status/CLE_AXIS2_RAHFL_RECOVERY_2026_11_08_ZH.md",
    )
    evidence = {
        "config": config,
        "fit_code": {name: sha256(HERE / name) for name in ("ssa.py", "gate0.py", "gate0.json", "README.md")},
        "reference_files": {name: sha256(REPO / name) for name in artifacts},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "threadpool_limit": 1,
        },
        "split_sha256": split_hash,
        "initial_state_hashes": initial_states,
        "records": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    paired = {}
    with np.load(split_path, allow_pickle=False) as splits, threadpool_limits(limits=1):
        for condition in config["conditions"]:
            for client in config["clients"]:
                start = perf_counter()
                key = f"{condition}_client{client}"
                path = package / "data" / condition / f"client_{client}"
                expected = manifest["conditions"][condition]["clients"][str(client)]
                fit_rows = np.sort(splits[f"client_{client}_fit"])
                images, labels, sources = load_fit(path, fit_rows, splits[f"client_{client}_audit"], expected)
                if client in paired:
                    previous_labels, previous_sources = paired[client]
                    if not (np.array_equal(previous_labels, labels) and np.array_equal(previous_sources, sources)):
                        raise ValueError(f"Task labels or source IDs differ across conditions: {key}")
                else:
                    paired[client] = (labels, sources)
                features = descriptors(images)
                feature_seconds = perf_counter() - start
                files = {}
                for name in config["features"]:
                    fit_start = perf_counter()
                    result = fit_oof(features[name], labels, fit_rows, sources, **config["probe"])
                    fit_seconds = perf_counter() - fit_start
                    file = output / f"{key}_{name}.npz"
                    np.savez_compressed(file, **vars(result), labels=labels, source_ids=sources)
                    files[name] = {"file": file.name, "sha256": sha256(file), "fit_seconds": fit_seconds}
                evidence["records"][key] = {
                    "condition": condition, "client": client, "n_fit": len(labels),
                    "source_groups": int(np.unique(sources).size), "feature_seconds": feature_seconds,
                    "inputs_sha256": {name: expected[name]["sha256"].lower() for name in FIT_FILES},
                    "scores": files,
                }
                print(f"[fit] {key} n={len(labels)} {perf_counter() - start:.1f}s", flush=True)
    write_json(output / "FIT_MANIFEST.json", evidence)


def score(labels, predictions, prior, selection):
    per_class = {}
    for label in np.unique(labels[selection]):
        rows = selection & (labels == label)
        per_class[str(label)] = {
            "n": int(rows.sum()), "correct": int((predictions[rows] == label).sum()),
            "prior_correct": int((prior[rows] == label).sum()),
        }
    n = int(selection.sum())
    supported = len(per_class)
    return {
        "n": n, "class_count": supported,
        "balanced_accuracy": (
            float(np.mean([c["correct"] / c["n"] for c in per_class.values()])) if supported else None
        ),
        "prior_balanced_accuracy": (
            float(np.mean([c["prior_correct"] / c["n"] for c in per_class.values()])) if supported else None
        ),
        "per_class": per_class,
    }


def audit(package, output, config):
    frozen = read_json(output / "FIT_MANIFEST.json")
    if frozen["config"] != config:
        raise ValueError("Audit config differs from the fit freeze")
    for name, digest in frozen["fit_code"].items():
        verify(HERE / name, digest)
    manifest_path = package / "manifest.json"
    verify(manifest_path, config["package_manifest_sha256"])
    manifest = read_json(manifest_path)
    operator_names = manifest["operators"]
    records = []
    pairs = {}
    for record in frozen["records"].values():
        condition, client = record["condition"], record["client"]
        path = package / "data" / condition / f"client_{client}"
        metadata = manifest["conditions"][condition]["clients"][str(client)]
        for name in ("train_is_corrupted.npy", "train_corruption_ids.npy"):
            verify(path / name, metadata[name]["sha256"])
        corruption_flags = np.load(path / "train_is_corrupted.npy", allow_pickle=False)
        operator_ids = np.load(path / "train_corruption_ids.npy", allow_pickle=False)
        for feature, item in record["scores"].items():
            cache = output / item["file"]
            verify(cache, item["sha256"])
            with np.load(cache, allow_pickle=False) as data:
                result = ProbeResult(**{name: data[name] for name in ProbeResult.__dataclass_fields__})
                rows = result.sample_ids
                labels, sources = data["labels"], data["source_ids"]
            corrupted = corruption_flags[rows]
            operators = operator_ids[rows]
            if np.any(operators[corrupted] >= len(operator_names)):
                raise ValueError("Unexpected operator ID on corrupted fit rows")
            # The prepared CLE-v2 generator uses zero-based operator IDs and 255 for clean.
            if np.any(operators[~corrupted] != 255):
                raise ValueError("Clean fit rows have an unexpected operator ID")
            masks = {"all": np.ones(len(rows), bool), "clean": ~corrupted, "corrupted": corrupted}
            masks.update({
                f"operator/{name}": corrupted & (operators == index)
                for index, name in enumerate(operator_names)
            })
            for subset, mask in masks.items():
                records.append({
                    "condition": condition, "client": client, "feature": feature, "subset": subset,
                    **score(labels, result.prediction, result.prior_prediction, mask),
                })
            pairs[(client, feature, condition)] = (
                labels, sources, corrupted, result.prediction, result.prior_prediction, result.fold
            )
    contrasts = []
    for client in config["clients"]:
        for feature in config["features"]:
            left = pairs[(client, feature, "gamma00")]
            right = pairs[(client, feature, "gamma09")]
            if any(not np.array_equal(left[i], right[i]) for i in (0, 1, 5)):
                raise ValueError("Gamma contrast requires matched sources, labels, and OOF folds")
            for subset, rows in (
                ("clean_both", ~(left[2] | right[2])),
                ("corrupted_both", left[2] & right[2]),
            ):
                a = score(left[0], left[3], left[4], rows)
                b = score(right[0], right[3], right[4], rows)
                contrasts.append({
                    "client": client, "feature": feature, "subset": subset,
                    "gamma00": a, "gamma09": b,
                    "delta_pp": (
                        100 * (b["balanced_accuracy"] - a["balanced_accuracy"]) if a["n"] else None
                    ),
                })
    warnings = [
        {"condition": r["condition"], "client": r["client"],
         "excess_over_prior_pp": 100 * (r["balanced_accuracy"] - r["prior_balanced_accuracy"])}
        for r in records if r["feature"] == config["primary_feature"] and r["subset"] == "clean"
        and r["n"] and 100 * (r["balanced_accuracy"] - r["prior_balanced_accuracy"])
        >= config["clean_warning_pp"]
    ]
    report = {
        "stage": "gate0_fit_only",
        "fit_manifest_sha256": sha256(output / "FIT_MANIFEST.json"),
        "clean_warning_threshold_pp": config["clean_warning_pp"],
        "clean_semantic_warnings": warnings, "scores": records, "paired_gamma_contrasts": contrasts,
    }
    write_json(output / "GATE0_AUDIT.json", report)
    fields = [name for name in records[0] if name != "per_class"]
    with (output / "GATE0_SLICES.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"[audit] {len(records)} slices, {len(warnings)} clean semantic warnings", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "audit"))
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = read_json(HERE / "gate0.json")
    if any(config["tf_saci"].values()):
        raise ValueError("Gate 0 has no HFL integration. All TF-SACI training flags must remain off.")
    operation = fit if args.command == "fit" else audit
    operation(args.package_root.resolve(), args.output.resolve(), config)


if __name__ == "__main__":
    main()
