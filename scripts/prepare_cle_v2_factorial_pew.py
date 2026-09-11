from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.methods.environment_witness import (  # noqa: E402
    PEW_ENVIRONMENT_NAMES,
    PublicEnvironmentWitness,
    build_public_environment_loaders,
    calibrate_unknown_threshold,
    evaluate_environment_witness,
    infer_environment_annotations,
    load_environment_witness,
    save_environment_witness,
    train_environment_witness,
)
from fedprime.utils.env import seed_everything  # noqa: E402


DATA_PROTOCOL = "cle_hfl_v2_paired_factorial_seed0_split0_v1"
ANNOTATION_PROTOCOL = "cle_v2_factorial_frozen_pew_annotations_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one frozen PEW for the CLE-v2 factorial.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--public-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--reuse-pew-root",
        type=Path,
        help=(
            "Reuse a frozen standard PEW from another prepared package. The checkpoint and "
            "public-only calibration are copied, while private annotations are regenerated."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def oracle_family_ids(operator_ids: np.ndarray, metadata: dict) -> np.ndarray:
    family_to_pew = {"noise": 1, "blur": 2, "weather": 3, "digital": 4}
    id_to_family = {
        int(operator_id): family_to_pew[metadata["operator_families"][operator]]
        for operator, operator_id in metadata["operator_to_id"].items()
    }
    return np.asarray([id_to_family[int(value)] for value in operator_ids], dtype=np.int64)


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("Unexpected paired factorial data protocol")
    asset_root = package_root / "pew_standard"
    if asset_root.exists():
        raise FileExistsError(f"Refusing to overwrite frozen PEW assets: {asset_root}")
    asset_root.mkdir(parents=True)
    device = resolve_device(args.device)
    seed_everything(0)
    started = time.perf_counter()
    checkpoint = asset_root / "pew_standard.pt"
    reused_from = None
    if args.reuse_pew_root is not None:
        source_root = args.reuse_pew_root.resolve()
        source_asset = source_root / "pew_standard" if (source_root / "pew_standard").is_dir() else source_root
        source_manifest_path = source_asset / "manifest.json"
        source_checkpoint = source_asset / "pew_standard.pt"
        if not source_manifest_path.is_file() or not source_checkpoint.is_file():
            raise FileNotFoundError(f"Frozen PEW assets not found under {source_asset}")
        source_report = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_report.get("protocol") != "cle_v2_factorial_standard_pew_v1":
            raise ValueError("Reuse source is not a standard frozen CLE-v2 PEW")
        if source_report.get("max_batches") is not None:
            raise ValueError("Refusing to reuse a truncated PEW checkpoint")
        expected_sha = str(source_report.get("checkpoint_sha256", "")).upper()
        if sha256_file(source_checkpoint) != expected_sha:
            raise ValueError("Reuse PEW checkpoint hash does not match its manifest")
        shutil.copy2(source_checkpoint, checkpoint)
        source_history = source_asset / "pew_training.csv"
        if source_history.is_file():
            shutil.copy2(source_history, asset_root / "pew_training.csv")
        witness = load_environment_witness(checkpoint, device)
        threshold = float(source_report["unknown_threshold"])
        calibration = source_report["calibration"]
        validation = source_report["validation"]
        public_size = int(source_report["public_size"])
        epochs = int(source_report["epochs"])
        max_batches = None
        reused_from = {
            "source": str(source_asset),
            "checkpoint_sha256": expected_sha,
            "public_only_calibration_reused": True,
            "private_annotations_regenerated": True,
        }
    else:
        train_loader, validation_loader = build_public_environment_loaders(
            package_root / "public",
            public_size=int(args.public_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=0,
            validation_fraction=0.2,
            public_dataset="cifar100",
            excluded_operators=(),
            label_mode="hard",
        )
        witness = PublicEnvironmentWitness(
            embedding_dim=32,
            num_environments=len(PEW_ENVIRONMENT_NAMES),
            severity_levels=5,
        ).to(device)
        history = train_environment_witness(
            witness,
            train_loader,
            validation_loader,
            device,
            epochs=int(args.epochs),
            learning_rate=1.0e-3,
            severity_weight=0.25,
            max_batches=args.max_batches,
        )
        save_environment_witness(witness, checkpoint, excluded_operators=(), label_mode="hard")
        calibration = calibrate_unknown_threshold(witness, validation_loader, device)
        threshold = float(calibration["threshold"])
        validation = evaluate_environment_witness(witness, validation_loader, device).as_dict()
        with (asset_root / "pew_training.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        public_size = int(args.public_size)
        epochs = int(args.epochs)
        max_batches = args.max_batches

    condition_reports = {}
    for condition in ("gamma00", "gamma09"):
        data_root = package_root / "data" / condition
        metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
        annotation_root = asset_root / "annotations" / condition
        annotation_root.mkdir(parents=True)
        clients = {}
        correct = 0
        total = 0
        for client_id in range(4):
            client_root = data_root / f"client_{client_id}"
            images = np.load(client_root / "train_images.npy", allow_pickle=False)
            annotation = infer_environment_annotations(
                witness,
                images,
                device,
                batch_size=int(args.inference_batch_size),
                confidence_threshold=threshold,
                include_probabilities=False,
            )
            path = annotation_root / f"client_{client_id}.npz"
            np.savez_compressed(path, **annotation)
            oracle = oracle_family_ids(
                np.load(client_root / "train_corruption_ids.npy", allow_pickle=False), metadata
            )
            client_correct = int((annotation["environment_ids"] == oracle).sum())
            correct += client_correct
            total += int(oracle.size)
            clients[str(client_id)] = {
                "samples": int(oracle.size),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "diagnostic_group_accuracy": 100.0 * client_correct / max(int(oracle.size), 1),
                "unknown_rate": float(
                    (annotation["environment_ids"] == len(PEW_ENVIRONMENT_NAMES) - 1).mean()
                ),
            }
        annotation_manifest = {
            "protocol": ANNOTATION_PROTOCOL,
            "condition": condition,
            "pew_checkpoint": "../../pew_standard.pt",
            "pew_checkpoint_sha256": sha256_file(checkpoint),
            "unknown_threshold": threshold,
            "clients": clients,
        }
        (annotation_root / "manifest.json").write_text(
            json.dumps(annotation_manifest, indent=2), encoding="utf-8"
        )
        condition_reports[condition] = {
            "samples": total,
            "diagnostic_group_accuracy": 100.0 * correct / max(total, 1),
            "annotation_manifest_sha256": sha256_file(annotation_root / "manifest.json"),
        }

    report = {
        "protocol": "cle_v2_factorial_standard_pew_v1",
        "data_manifest_sha256": sha256_file(manifest_path),
        "device": str(device),
        "environment_names": PEW_ENVIRONMENT_NAMES,
        "public_size": public_size,
        "validation_fraction": 0.2,
        "epochs": epochs,
        "max_batches": max_batches,
        "learning_rate": 1.0e-3,
        "severity_weight": 0.25,
        "unknown_threshold": threshold,
        "calibration": calibration,
        "validation": validation,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "reused_from": reused_from,
        "conditions": condition_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (asset_root / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
