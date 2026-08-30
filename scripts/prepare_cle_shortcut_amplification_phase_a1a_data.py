from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.corruptions import CORRUPTION_GROUPS, GROUP_TO_ID, apply_corruption
from fedprime.data.loaders import partition_private_data
from fedprime.data.strict_fit_audit import stratified_fit_audit_indices
from scripts.prepare_corruption_skew_data import load_cifar10_arrays


PACKAGE_NAME = "cle_shortcut_amplification_phase_a1a_seed0"
PHASE_A0_INPUT = "cle_shortcut_alignment_phase_a0_seed0_inputs"
MODEL_NAMES = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]
GAMMAS = {"gamma00": 0.0, "gamma09": 0.9}
SEED = 0
EVAL_SEED = 20260830


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare matched CLE shortcut Phase-A1a data.")
    parser.add_argument(
        "--cifar10-root",
        type=Path,
        default=ROOT / "local_runs/rahfl_cifar10_clean_replay",
    )
    parser.add_argument(
        "--cifar100-tar",
        type=Path,
        default=ROOT / "RAHFL-master/Dataset/cifar_100/cifar-100-python.tar.gz",
    )
    parser.add_argument(
        "--phase-a0-input",
        type=Path,
        default=(
            ROOT
            / "local_runs/cle_shortcut_alignment_phase_a0"
            / "cle_shortcut_alignment_phase_a0_seed0_inputs.tar.gz"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "local_runs/cle_shortcut_amplification_phase_a1a",
    )
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--audit-ratio", type=float, default=0.15)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest().upper()


def load_phase_a0_clean(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with tarfile.open(path, "r:gz") as archive:
        prefix = f"{PHASE_A0_INPUT}/clean"
        image_stream = archive.extractfile(f"{prefix}/test_images.npy")
        label_stream = archive.extractfile(f"{prefix}/test_labels.npy")
        if image_stream is None or label_stream is None:
            raise FileNotFoundError("Phase-A0 clean evaluation arrays are missing")
        images = np.load(io.BytesIO(image_stream.read()), allow_pickle=False)
        labels = np.load(io.BytesIO(label_stream.read()), allow_pickle=False)
    return images.astype(np.uint8, copy=False), labels.astype(np.int64, copy=False)


def class_family_map(num_clients: int = 4, num_classes: int = 10) -> np.ndarray:
    clients = np.arange(num_clients, dtype=np.int64)[:, None]
    classes = np.arange(num_classes, dtype=np.int64)[None, :]
    return (clients + classes) % len(CORRUPTION_GROUPS)


def generate_client_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    client_id: int,
    gamma: float,
    binding: np.ndarray,
) -> dict[str, np.ndarray]:
    families = tuple(CORRUPTION_GROUPS)
    operators = tuple(operator for family in families for operator in CORRUPTION_GROUPS[family])
    operator_to_id = {name: index for index, name in enumerate(operators)}
    output_images = np.empty((indices.size, 32, 32, 3), dtype=np.uint8)
    output_labels = labels[indices].astype(np.uint8, copy=False)
    family_ids = np.empty(indices.size, dtype=np.uint8)
    operator_ids = np.empty(indices.size, dtype=np.uint8)
    severities = np.empty(indices.size, dtype=np.uint8)
    for local_id, source_id in enumerate(indices.tolist()):
        label = int(labels[source_id])
        probabilities = np.full(len(families), (1.0 - gamma) / len(families), dtype=np.float64)
        probabilities[int(binding[client_id, label])] += gamma
        group_rng = np.random.default_rng(np.random.SeedSequence([SEED, client_id, local_id, 0]))
        family_id = int(group_rng.choice(len(families), p=probabilities / probabilities.sum()))
        family = families[family_id]
        method_rng = np.random.default_rng(np.random.SeedSequence([SEED, client_id, local_id, 1]))
        family_operators = CORRUPTION_GROUPS[family]
        operator = family_operators[int(method_rng.integers(0, len(family_operators)))]
        severity_rng = np.random.default_rng(np.random.SeedSequence([SEED, client_id, local_id, 2]))
        severity = int(severity_rng.integers(1, 6))
        corruption_rng = np.random.default_rng(np.random.SeedSequence([SEED, client_id, local_id, 3]))
        output_images[local_id] = apply_corruption(
            images[source_id], operator, severity, corruption_rng
        )
        family_ids[local_id] = family_id
        operator_ids[local_id] = operator_to_id[operator]
        severities[local_id] = severity
    return {
        "train_images": output_images,
        "train_labels": output_labels,
        "train_corruption_ids": family_ids,
        "train_corruption_method_ids": operator_ids,
        "train_severity_ids": severities,
        "train_source_indices": indices.astype(np.int64, copy=False),
    }


def write_arrays(directory: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for name, array in arrays.items():
        path = directory / f"{name}.npy"
        np.save(path, array)
        records[path.name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "array_sha256": sha256_array(array),
            "file_sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return records


def build_reporting_test(clean_images: np.ndarray, clean_labels: np.ndarray) -> dict[str, np.ndarray]:
    representatives = [operators[0] for operators in CORRUPTION_GROUPS.values()]
    images, labels, family_ids = [], [], []
    for source_id, (image, label) in enumerate(zip(clean_images, clean_labels, strict=True)):
        for family_id, operator in enumerate(representatives):
            rng = np.random.default_rng(
                np.random.SeedSequence([EVAL_SEED, source_id, family_id, 91])
            )
            images.append(apply_corruption(image, operator, 3, rng))
            labels.append(int(label))
            family_ids.append(family_id)
    return {
        "test_images": np.asarray(images, dtype=np.uint8),
        "test_labels": np.asarray(labels, dtype=np.uint8),
        "test_corruption_ids": np.asarray(family_ids, dtype=np.uint8),
    }


def save_split(
    path: Path,
    labels_by_client: dict[int, np.ndarray],
    *,
    audit_ratio: float,
) -> dict[str, object]:
    payload: dict[str, np.ndarray] = {
        "audit_ratio": np.asarray([audit_ratio], dtype=np.float64),
        "min_audit_per_class": np.asarray([5], dtype=np.int64),
        "min_fit_per_class": np.asarray([2], dtype=np.int64),
        "seed": np.asarray([SEED], dtype=np.int64),
    }
    summary: dict[str, object] = {}
    for client_id, labels in labels_by_client.items():
        fit, audit = stratified_fit_audit_indices(
            labels,
            audit_ratio=audit_ratio,
            min_audit_per_class=5,
            min_fit_per_class=2,
            seed=client_id,
        )
        payload[f"client_{client_id}_fit"] = fit
        payload[f"client_{client_id}_audit"] = audit
        summary[str(client_id)] = {
            "fit": int(fit.size),
            "audit": int(audit.size),
            "fit_sha256": sha256_array(fit),
            "audit_sha256": sha256_array(audit),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {"file_sha256": sha256_file(path), "clients": summary}


def main() -> None:
    args = parse_args()
    package_root = args.output_root.resolve() / PACKAGE_NAME
    archive_path = args.output_root.resolve() / f"{PACKAGE_NAME}.tar.gz"
    if package_root.exists() or archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase-A1a data: {package_root}")
    args.cifar100_tar = args.cifar100_tar.resolve()
    args.phase_a0_input = args.phase_a0_input.resolve()
    if not args.cifar100_tar.is_file() or not args.phase_a0_input.is_file():
        raise FileNotFoundError("CIFAR-100 tar or Phase-A0 input package is missing")

    package_root.mkdir(parents=True)
    train_images, train_labels, _, _ = load_cifar10_arrays(args.cifar10_root.resolve(), False)
    clean_images, clean_labels = load_phase_a0_clean(args.phase_a0_input)
    if clean_images.shape != (1000, 32, 32, 3):
        raise ValueError(f"Unexpected Phase-A0 clean shape: {clean_images.shape}")
    partition = partition_private_data(
        labels=train_labels,
        num_clients=4,
        num_classes=10,
        partition="dirichlet",
        dirichlet_alpha=0.5,
        max_samples_per_client=int(args.samples_per_client),
        partition_seed=SEED,
    )
    binding = class_family_map()
    reporting_test = build_reporting_test(clean_images, clean_labels)
    manifest: dict[str, object] = {
        "protocol": PACKAGE_NAME,
        "seed": SEED,
        "alpha": 0.5,
        "gammas": GAMMAS,
        "model_names": MODEL_NAMES,
        "source": {
            "phase_a0_input": {
                "name": args.phase_a0_input.name,
                "sha256": sha256_file(args.phase_a0_input),
            },
            "cifar100_tar": {
                "name": args.cifar100_tar.name,
                "sha256": sha256_file(args.cifar100_tar),
            },
            "cifar10_train_images_sha256": sha256_array(train_images),
            "cifar10_train_labels_sha256": sha256_array(train_labels),
        },
        "binding": binding.tolist(),
        "conditions": {},
    }
    labels_by_client: dict[int, np.ndarray] = {}
    pairing_reference: dict[int, dict[str, np.ndarray]] = {}
    test_records = None
    for condition, gamma in GAMMAS.items():
        condition_root = package_root / "data" / condition
        condition_records: dict[str, object] = {"clients": {}}
        for client_id in range(4):
            arrays = generate_client_arrays(
                train_images,
                train_labels,
                np.asarray(partition[client_id], dtype=np.int64),
                client_id=client_id,
                gamma=gamma,
                binding=binding,
            )
            if condition == "gamma00":
                pairing_reference[client_id] = {
                    key: np.asarray(arrays[key]).copy()
                    for key in ("train_labels", "train_source_indices", "train_severity_ids")
                }
            else:
                for key, expected in pairing_reference[client_id].items():
                    if not np.array_equal(arrays[key], expected):
                        raise ValueError(
                            f"Paired condition mismatch for client={client_id}, array={key}"
                        )
            labels_by_client[client_id] = arrays["train_labels"].astype(np.int64, copy=False)
            condition_records["clients"][str(client_id)] = write_arrays(
                condition_root / f"client_{client_id}", arrays
            )
        test_records = write_arrays(condition_root / "test_balanced", reporting_test)
        metadata = {
            "dataset": "cifar10_cle_hfl_phase_a1a",
            "private_dataset": "cifar10",
            "condition": condition,
            "gamma": gamma,
            "alpha": 0.5,
            "seed": SEED,
            "num_clients": 4,
            "num_classes": 10,
            "corruption_groups": CORRUPTION_GROUPS,
            "group_to_id": GROUP_TO_ID,
            "class_corruption_map": binding.tolist(),
        }
        metadata_path = condition_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        condition_records["metadata_sha256"] = sha256_file(metadata_path)
        condition_records["test"] = test_records
        manifest["conditions"][condition] = condition_records

    manifest["paired_conditions"] = {
        "source_indices_identical": True,
        "labels_identical": True,
        "severity_draws_identical": True,
        "corruption_family_and_operator_may_differ_by_design": True,
    }

    split_path = package_root / "splits/strict_cle_v1_alpha05_gamma_pair_seed0_split0.npz"
    manifest["strict_split"] = save_split(
        split_path,
        labels_by_client,
        audit_ratio=float(args.audit_ratio),
    )
    evaluation_records = write_arrays(
        package_root / "evaluation",
        {"test_images": clean_images, "test_labels": clean_labels.astype(np.uint8)},
    )
    manifest["evaluation"] = evaluation_records
    public_dir = package_root / "public"
    public_dir.mkdir(parents=True)
    public_path = public_dir / args.cifar100_tar.name
    shutil.copy2(args.cifar100_tar, public_path)
    manifest["public_tar"] = {
        "bytes": public_path.stat().st_size,
        "sha256": sha256_file(public_path),
    }
    manifest_path = package_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        archive.add(package_root, arcname=PACKAGE_NAME)
    audit = {
        "archive": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "package_root": str(package_root),
    }
    (args.output_root.resolve() / f"{PACKAGE_NAME}_archive_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
