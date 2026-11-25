"""Evaluate both frozen Pilot arms and their historical baselines together."""

import argparse
import json
from pathlib import Path
import shutil
import tarfile
from time import perf_counter

import numpy as np
import torch

from fedprime.engine.cle_v2_factorial import compute_operator_dsa, paired_bootstrap_estimand
from scripts.analyze_cle_v2_factorial import infer_arm, load_grid
from scripts.analyze_cle_v2_mechanism_stage1 import grid_accuracy

from .gate0 import read_json, sha256, verify, write_json
from .pilot_train import PILOT, PROTOCOL


BASELINES = ("asymhfl_erm", "asymhfl_pew_ber")


def extract_baseline(source, target):
    config = read_json(PILOT.parent / "gate0.json")
    digest = verify(source, config["baseline_recovery_archives_sha256"][source.name])
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, "r:gz") as archive:
        for i in range(4):
            filename = f"client_{i}.pt"
            member, = [m for m in archive.getmembers()
                       if m.isfile() and m.name.endswith(f"/checkpoints/{filename}")]
            with archive.extractfile(member) as lead, (target / filename).open("xb") as result:
                shutil.copyfileobj(lead, result)
    return {"recovery_archive_sha256": digest,
            "checkpoint_sha256": {p.name: sha256(p) for p in sorted(target.glob("*.pt"))}}


def group_metrics(correct, labels, operators):
    outcomes = {}
    for label in np.unique(labels):
        outcomes[int(label)] = {}
        for operator in np.unique(operators[labels == label]):
            mask = (labels == label) & (operators == operator)
            count = int(mask.sum() * len(correct))
            successes = int(correct[:, mask].sum())
            outcomes[int(label)][int(operator)] = {"n": count, "correct": successes,
                                                    "accuracy": 100 * successes / count}
    per_class = [[x["accuracy"] for x in groups.values()] for groups in outcomes.values()]
    return {"wcca": min(min(a) for a in per_class), "cfg": float(np.mean([max(a) - min(a) for a in per_class])),
            "class_operator": outcomes}


def split_accuracy(checkpoints, split, device):
    images = np.load(split / "test_images.npy", allow_pickle=False)
    labels = np.load(split / "test_labels.npy", allow_pickle=False)
    probabilities = infer_arm(checkpoints, images[:, None], device, 256)
    correct = probabilities[:, :, 0].argmax(axis=2) == labels[None]
    values = correct.mean(axis=1) * 100
    return correct, labels, {"pooled": float(values.mean()), "clients": values.tolist(), "n_per_client": len(labels)}


def evaluate(package, runs, baseline_archives, output, device):
    pilot, protocol = read_json(PILOT), read_json(PROTOCOL)
    frozen = {}
    common = None
    expected_mode = "pilot"
    for arm, (root, eta) in zip(pilot["arms"], zip(runs, pilot["arms"].values()), strict=True):
        record = read_json(root / "TRAINING_COMPLETE.json")
        required = {"arm": arm, "mode": expected_mode, "completed_rounds": 40, "lambda": 0.5,
                    "eta": eta, "probe": "phi_tex", "pilot_sha256": sha256(PILOT),
                    "protocol_sha256": sha256(PROTOCOL), "reference_setup": protocol["backbone_protocol"]}
        if any(record.get(k) != v for k, v in required.items()):
            raise ValueError(f"Completion contract differs from frozen Pilot: {arm}")
        names = {f"client_{i}.pt" for i in range(4)}
        if set(record["checkpoint_sha256"]) != names:
            raise ValueError(f"Wrong checkpoint names: {arm}")
        verify(root / "config.resolved.json", record["config_sha256"])
        verify(root / "local_batch_trace.jsonl", record["paired_trace_sha256"])
        for name, digest in record["checkpoint_sha256"].items():
            verify(root / "checkpoints" / name, digest)
        identity = [record[key] for key in ("input_sha256", "device", "torch", "torchvision",
                                            "cuda", "paired_trace_sha256", "pilot_source_hashes")]
        if common is not None and identity != common:
            raise ValueError("The two runs do not share inputs, runtime and observed fit traces")
        common = identity
        frozen[arm] = record
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = {arm: root / "checkpoints" for arm, root in zip(pilot["arms"], runs, strict=True)}
    for arm, archive in zip(BASELINES, baseline_archives, strict=True):
        checkpoints[arm] = output / arm / "checkpoints"
        frozen[arm] = extract_baseline(archive, checkpoints[arm])
    # On disk before the first sealed file is loaded.
    write_json(output / "JOINT_FREEZE.json", frozen)
    cfg0 = read_json(PILOT.parent / "gate0.json")
    verify(package / "manifest.json", cfg0["package_manifest_sha256"])
    grid, labels, operator_names, binding = load_grid(package)
    split = package / "data/gamma09"
    operators = np.load(split / "test_balanced/test_corruption_ids.npy", allow_pickle=False)
    rows, dsa = {}, {}
    for arm, ckpt in checkpoints.items():
        start = perf_counter()
        prediction = infer_arm(ckpt, grid, device, 256)
        dsa[arm] = compute_operator_dsa(prediction, labels, binding)
        _, _, clean = split_accuracy(ckpt, split / "test_clean", device)
        correct, corrupt_labels, corrupt = split_accuracy(ckpt, split / "test_balanced", device)
        if len(operators) != len(corrupt_labels):
            raise ValueError("Corruption operator length mismatch")
        groups = group_metrics(correct, corrupt_labels, operators)
        mixed_clients = ((np.asarray(clean["clients"]) + corrupt["clients"]) / 2).tolist()
        rows[arm] = {
            "dsa": dsa[arm].pooled, "dsa_by_client": dsa[arm].client.tolist(),
            "operator_grid": grid_accuracy(prediction, labels),
            "source_count_operator_grid": len(labels), "operator_count_grid": len(operator_names),
            "clean": clean, "corrupted": corrupt,
            "mixed": {"pooled": float(np.mean(mixed_clients)), "clients": mixed_clients},
            "wcca": groups["wcca"], "cfg": groups["cfg"], "class_operator": groups["class_operator"],
            "wcca_cfg_by_client": [group_metrics(c[None], corrupt_labels, operators) for c in correct],
            "evaluation_seconds": perf_counter() - start, "training_seconds": frozen[arm].get("training_seconds"),
            "peak_training_cuda_memory_mb": frozen[arm].get("peak_cuda_memory_mb"),
        }
        np.save(output / f"{arm}_paired_probabilities.npy", prediction, allow_pickle=False)

    goals = protocol["formal_go_per_arm"]
    for arm in pilot["arms"]:
        comparisons = {}
        for base in BASELINES:
            draws = paired_bootstrap_estimand({arm: dsa[arm], base: dsa[base]}, {arm: 1, base: -1}, samples=2000)
            comparisons[base] = {
                "dsa_difference": rows[arm]["dsa"] - rows[base]["dsa"],
                "dsa_difference_source_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
                **{f"{m}_delta_pp": rows[arm][m]["pooled"] - rows[base][m]["pooled"]
                   for m in ("clean", "corrupted", "mixed")},
            }
        row = rows[arm]
        row["comparisons"] = comparisons
        row["predeclared_target"] = {
            "dsa": round(row["dsa"], 6) <= goals["max_dsa"],
            "clean": round(row["clean"]["pooled"], 3) >= goals["min_clean_accuracy_percent"],
            "mixed": round(row["mixed"]["pooled"], 3) >= goals["min_mixed_accuracy_percent"],
        }
        row["meets_all_targets"] = all(row["predeclared_target"].values())
    write_json(output / "RESULTS.json", {"evidence": pilot["claim"], "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("package-root", "poe-run", "anchor-run", "erm-archive", "ber-archive", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    evaluate(args.package_root, (args.poe_run, args.anchor_run), (args.erm_archive, args.ber_archive),
             args.output, torch.device(args.device))
