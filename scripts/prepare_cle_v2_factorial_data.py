from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.corruptions import (  # noqa: E402
    CIFAR_C_CORE_CORRUPTIONS,
    CORRUPTION_OPERATOR_FAMILIES,
    DEFAULT_UNSEEN_CORRUPTIONS,
    apply_corruption,
)
from fedprime.data.loaders import partition_private_data  # noqa: E402
from fedprime.data.strict_fit_audit import stratified_fit_audit_indices  # noqa: E402
from fedprime.models.factory import build_models  # noqa: E402
from fedprime.utils.env import seed_everything  # noqa: E402
from scripts.prepare_cle_v2_data import (  # noqa: E402
    balanced_test_indices,
    build_class_operator_map,
    parse_operator_split,
)
from scripts.prepare_corruption_skew_data import load_cifar10_arrays  # noqa: E402


PROTOCOL = "cle_hfl_v2_paired_factorial_seed0_split0_v1"
PACKAGE_NAME = "cle_hfl_v2_paired_factorial_seed0_split0"
MODEL_NAMES = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]
GAMMAS = {"gamma00": 0.0, "gamma09": 0.9}
SEED = 0
EVAL_SEED = 20260909


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paired CLE-HFL v2 factorial data.")
    parser.add_argument(
        "--cifar10-root",
        type=Path,
        default=ROOT / "RAHFL-master/Dataset/cifar_10",
    )
    parser.add_argument(
        "--cifar100-tar",
        type=Path,
        default=ROOT / "RAHFL-master/Dataset/cifar_100/cifar-100-python.tar.gz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "local_runs/cle_v2_factorial",
    )
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--test-samples-per-class", type=int, default=100)
    parser.add_argument("--audit-ratio", type=float, default=0.15)
    parser.add_argument("--unseen-operators", default=",".join(DEFAULT_UNSEEN_CORRUPTIONS))
    parser.add_argument("--no-archive", action="store_true")
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


def write_array(path: Path, array: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "array_sha256": sha256_array(array),
    }


def generate_client_condition(
    images: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    *,
    client_id: int,
    gamma: float,
    binding: dict[int, dict[int, str]],
    seen: list[str],
    operator_to_id: dict[str, int],
) -> dict[str, np.ndarray]:
    count = int(source_indices.size)
    output_images = np.empty((count, 32, 32, 3), dtype=np.uint8)
    output_labels = labels[source_indices].astype(np.uint8, copy=False)
    operator_ids = np.empty(count, dtype=np.uint8)
    severities = np.empty(count, dtype=np.uint8)
    for local_id, source_id in enumerate(source_indices.tolist()):
        label = int(labels[source_id])
        probabilities = np.full(len(seen), (1.0 - gamma) / len(seen), dtype=np.float64)
        probabilities[seen.index(binding[client_id][label])] += gamma
        operator_rng = np.random.default_rng(
            np.random.SeedSequence([SEED, client_id, local_id, 11])
        )
        operator = str(operator_rng.choice(seen, p=probabilities / probabilities.sum()))
        severity = int(
            np.random.default_rng(
                np.random.SeedSequence([SEED, client_id, local_id, 12])
            ).integers(1, 6)
        )
        corruption_rng = np.random.default_rng(
            np.random.SeedSequence([SEED, client_id, local_id, 13])
        )
        output_images[local_id] = apply_corruption(
            images[source_id], operator, severity, corruption_rng
        )
        operator_ids[local_id] = operator_to_id[operator]
        severities[local_id] = severity
    return {
        "train_images": output_images,
        "train_labels": output_labels,
        "train_corruption_ids": operator_ids,
        "train_corruption_method_ids": operator_ids.copy(),
        "train_severity_ids": severities,
        "train_source_indices": source_indices.astype(np.int64, copy=False),
    }


def make_test_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    operators: list[str],
    operator_to_id: dict[str, int],
    *,
    fixed_severity: int | None,
) -> dict[str, np.ndarray]:
    out_images, out_labels, out_operators, out_severities, out_sources = [], [], [], [], []
    for operator_id, operator in enumerate(operators):
        for source_position, source_index in enumerate(source_indices.tolist()):
            severity = (
                int(fixed_severity)
                if fixed_severity is not None
                else int(
                    np.random.default_rng(
                        np.random.SeedSequence([EVAL_SEED, source_position, operator_id, 21])
                    ).integers(1, 6)
                )
            )
            rng = np.random.default_rng(
                np.random.SeedSequence([EVAL_SEED, source_position, operator_id, 22])
            )
            out_images.append(apply_corruption(images[source_index], operator, severity, rng))
            out_labels.append(int(labels[source_index]))
            out_operators.append(operator_to_id[operator])
            out_severities.append(severity)
            out_sources.append(source_position)
    return {
        "test_images": np.asarray(out_images, dtype=np.uint8),
        "test_labels": np.asarray(out_labels, dtype=np.uint8),
        "test_corruption_ids": np.asarray(out_operators, dtype=np.uint8),
        "test_severity_ids": np.asarray(out_severities, dtype=np.uint8),
        "test_source_ids": np.asarray(out_sources, dtype=np.int64),
    }


def write_arrays(directory: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        f"{name}.npy": write_array(directory / f"{name}.npy", value)
        for name, value in arrays.items()
    }


def save_shared_split(path: Path, labels_by_client: dict[int, np.ndarray], audit_ratio: float) -> dict:
    payload: dict[str, np.ndarray] = {
        "audit_ratio": np.asarray([audit_ratio], dtype=np.float64),
        "min_audit_per_class": np.asarray([5], dtype=np.int64),
        "min_fit_per_class": np.asarray([2], dtype=np.int64),
        "seed": np.asarray([SEED], dtype=np.int64),
    }
    clients = {}
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
        clients[str(client_id)] = {"fit": int(fit.size), "audit": int(audit.size)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path), "clients": clients}


def prepare_initial_states(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    models = build_models(MODEL_NAMES, num_classes=10)
    records = {}
    for client_id, model_name in enumerate(MODEL_NAMES):
        path = directory / f"client_{client_id}.pt"
        torch.save(models[client_id].state_dict(), path)
        records[str(client_id)] = {
            "model": model_name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {"seed": SEED, "models": records}


def condition_metadata(
    condition: str,
    gamma: float,
    operators: list[str],
    seen: list[str],
    unseen: list[str],
    operator_to_id: dict[str, int],
    binding: dict[int, dict[int, str]],
) -> dict[str, object]:
    return {
        "dataset": "cifar10_cle_hfl_v2_factorial",
        "protocol_version": 2,
        "factorial_protocol": PROTOCOL,
        "condition": condition,
        "private_dataset": "cifar10",
        "public_dataset": "cifar100",
        "alpha": 0.5,
        "gamma": gamma,
        "seed": SEED,
        "num_clients": 4,
        "num_classes": 10,
        "operators": operators,
        "operator_to_id": operator_to_id,
        "operator_families": CORRUPTION_OPERATOR_FAMILIES,
        "operator_splits": {
            name: ("unseen" if name in unseen else "seen") for name in operators
        },
        "seen_operators": seen,
        "unseen_operators": unseen,
        "class_operator_map": {
            str(client): {str(label): operator for label, operator in mapping.items()}
            for client, mapping in binding.items()
        },
        "method_visibility": "operator/family/severity/binding metadata are reporting-only",
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    package_root = output_root / PACKAGE_NAME
    archive_path = output_root / f"{PACKAGE_NAME}.tar.gz"
    if package_root.exists() or archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing factorial artifact: {package_root}")
    if not args.cifar100_tar.resolve().is_file():
        raise FileNotFoundError(args.cifar100_tar)
    train_images, train_labels, test_images, test_labels = load_cifar10_arrays(
        args.cifar10_root.resolve(), False
    )
    seen, unseen = parse_operator_split(args.unseen_operators)
    operators = list(CIFAR_C_CORE_CORRUPTIONS)
    operator_to_id = {name: index for index, name in enumerate(operators)}
    binding = build_class_operator_map(4, 10, seen, SEED)
    partition = partition_private_data(
        labels=train_labels,
        num_clients=4,
        num_classes=10,
        partition="dirichlet",
        dirichlet_alpha=0.5,
        max_samples_per_client=int(args.samples_per_client),
        partition_seed=SEED,
    )
    test_rng = np.random.default_rng(EVAL_SEED)
    test_indices = balanced_test_indices(
        test_labels,
        samples_per_class=int(args.test_samples_per_class),
        num_classes=10,
        rng=test_rng,
    )
    test_seen = make_test_arrays(test_images, test_labels, test_indices, seen, operator_to_id, fixed_severity=None)
    test_unseen = make_test_arrays(test_images, test_labels, test_indices, unseen, operator_to_id, fixed_severity=None)
    test_all = {
        key: np.concatenate([test_seen[key], test_unseen[key]], axis=0)
        for key in test_seen
    }
    test_clean = {
        "test_images": test_images[test_indices].astype(np.uint8, copy=False),
        "test_labels": test_labels[test_indices].astype(np.uint8, copy=False),
        "test_corruption_ids": np.zeros(test_indices.size, dtype=np.uint8),
        "test_severity_ids": np.zeros(test_indices.size, dtype=np.uint8),
        "test_source_ids": np.arange(test_indices.size, dtype=np.int64),
    }
    dsa = make_test_arrays(
        test_images, test_labels, test_indices, operators, operator_to_id, fixed_severity=3
    )

    package_root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "package": PACKAGE_NAME,
        "gammas": GAMMAS,
        "model_names": MODEL_NAMES,
        "operators": operators,
        "seen_operators": seen,
        "unseen_operators": unseen,
        "class_operator_map": condition_metadata(
            "gamma00", 0.0, operators, seen, unseen, operator_to_id, binding
        )["class_operator_map"],
        "conditions": {},
    }
    labels_by_client: dict[int, np.ndarray] = {}
    pairing: dict[int, dict[str, np.ndarray]] = {}
    for condition, gamma in GAMMAS.items():
        condition_root = package_root / "data" / condition
        condition_record: dict[str, object] = {"clients": {}}
        for client_id in range(4):
            sources = np.asarray(partition[client_id], dtype=np.int64)
            arrays = generate_client_condition(
                train_images,
                train_labels,
                sources,
                client_id=client_id,
                gamma=gamma,
                binding=binding,
                seen=seen,
                operator_to_id=operator_to_id,
            )
            if condition == "gamma00":
                pairing[client_id] = {
                    key: arrays[key].copy()
                    for key in ("train_labels", "train_source_indices", "train_severity_ids")
                }
            else:
                for key, expected in pairing[client_id].items():
                    if not np.array_equal(arrays[key], expected):
                        raise ValueError(f"gamma pairing mismatch: client={client_id} key={key}")
            labels_by_client[client_id] = arrays["train_labels"].astype(np.int64)
            condition_record["clients"][str(client_id)] = write_arrays(
                condition_root / f"client_{client_id}", arrays
            )
        for split_name, arrays in (
            ("test_clean", test_clean),
            ("test_seen", test_seen),
            ("test_unseen", test_unseen),
            ("test_balanced", test_all),
        ):
            write_arrays(condition_root / split_name, arrays)
        metadata = condition_metadata(
            condition, gamma, operators, seen, unseen, operator_to_id, binding
        )
        metadata_path = condition_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        condition_record["metadata_sha256"] = sha256_file(metadata_path)
        manifest["conditions"][condition] = condition_record

    split_path = package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz"
    manifest["strict_split"] = save_shared_split(
        split_path, labels_by_client, float(args.audit_ratio)
    )
    manifest["evaluation"] = {
        "paired_dsa": write_arrays(package_root / "evaluation/paired_dsa", dsa),
        "source_class_counts": np.bincount(
            test_labels[test_indices].astype(np.int64), minlength=10
        ).tolist(),
        "severity": 3,
        "seed": EVAL_SEED,
    }
    manifest["initialization"] = prepare_initial_states(package_root / "initial_states")
    public_dir = package_root / "public"
    public_dir.mkdir(parents=True)
    public_path = public_dir / args.cifar100_tar.name
    shutil.copy2(args.cifar100_tar.resolve(), public_path)
    manifest["public"] = {
        "file": public_path.name,
        "bytes": public_path.stat().st_size,
        "sha256": sha256_file(public_path),
    }
    manifest["paired_conditions"] = {
        "source_indices_identical": True,
        "labels_identical": True,
        "severity_draws_identical": True,
        "shared_fit_audit_split": True,
        "shared_initial_states": True,
        "operator_draw_changes_only_with_gamma": True,
    }
    manifest_path = package_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.no_archive:
        output_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
            archive.add(package_root, arcname=PACKAGE_NAME)
        audit = {
            "archive": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        }
        (output_root / f"{PACKAGE_NAME}_archive_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print(json.dumps(audit, indent=2), flush=True)
    else:
        print(f"Prepared {package_root}", flush=True)


if __name__ == "__main__":
    main()
