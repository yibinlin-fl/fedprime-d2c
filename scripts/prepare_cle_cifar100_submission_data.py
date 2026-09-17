from __future__ import annotations

import argparse
import json
import pickle
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
)
from fedprime.data.loaders import partition_private_data  # noqa: E402
from fedprime.models.factory import build_models  # noqa: E402
from fedprime.utils.env import seed_everything  # noqa: E402
from scripts.prepare_cle_v2_data import (  # noqa: E402
    balanced_test_indices,
    build_class_operator_map,
    parse_operator_split,
)
from scripts.prepare_cle_v2_factorial_data import (  # noqa: E402
    EVAL_SEED,
    MODEL_NAMES,
    generate_client_condition,
    make_test_arrays,
    save_shared_split,
    sha256_file,
    write_arrays,
)


PROTOCOL = "cle_hfl_v2_cifar100_factorial_seed0_split0_v1"
PACKAGE_NAME = "cle_hfl_v2_cifar100_factorial_seed0_split0"
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the submission CIFAR-100-private CLE package.")
    parser.add_argument(
        "--cifar100-tar",
        type=Path,
        default=ROOT / "RAHFL-master/Dataset/cifar_100/cifar-100-python.tar.gz",
    )
    parser.add_argument(
        "--cifar10-tar",
        type=Path,
        default=ROOT / "RAHFL-master/Dataset/cifar_10/cifar-10-python.tar.gz",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "local_runs/cle_cifar100_submission"
    )
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--test-samples-per-class", type=int, default=10)
    parser.add_argument("--audit-ratio", type=float, default=0.15)
    parser.add_argument("--binding-map-seed", type=int, default=0)
    parser.add_argument("--unseen-operators", default=",".join(DEFAULT_UNSEEN_CORRUPTIONS))
    return parser.parse_args()


def _batch_arrays(batch: dict[bytes, object] | dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    data = batch.get(b"data", batch.get("data"))
    labels = batch.get(b"fine_labels", batch.get("fine_labels"))
    if data is None or labels is None:
        raise KeyError("CIFAR-100 batch is missing data or fine_labels")
    images = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.asarray(labels, dtype=np.int64)


def load_cifar100_tar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with tarfile.open(path, "r:gz") as archive:
        train_file = archive.extractfile("cifar-100-python/train")
        test_file = archive.extractfile("cifar-100-python/test")
        if train_file is None or test_file is None:
            raise FileNotFoundError("CIFAR-100 train/test members")
        train_images, train_labels = _batch_arrays(pickle.load(train_file, encoding="latin1"))
        test_images, test_labels = _batch_arrays(pickle.load(test_file, encoding="latin1"))
    return train_images, train_labels, test_images, test_labels


def prepare_initial_states(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    models = build_models(MODEL_NAMES, num_classes=100)
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


def metadata(
    condition: str,
    gamma: float,
    operators: list[str],
    seen: list[str],
    unseen: list[str],
    operator_to_id: dict[str, int],
    binding: dict[int, dict[int, str]],
) -> dict[str, object]:
    return {
        "dataset": "cifar100_cle_hfl_v2_factorial",
        "private_dataset": "cifar100",
        "public_dataset": "cifar10",
        "factorial_protocol": PROTOCOL,
        "condition": condition,
        "alpha": 0.5,
        "gamma": gamma,
        "seed": SEED,
        "partition_seed": SEED,
        "num_clients": 4,
        "num_classes": 100,
        "operators": operators,
        "operator_to_id": operator_to_id,
        "operator_families": CORRUPTION_OPERATOR_FAMILIES,
        "operator_splits": {name: ("unseen" if name in unseen else "seen") for name in operators},
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
    if package_root.exists():
        raise FileExistsError(f"Refusing to overwrite {package_root}")
    if not args.cifar10_tar.resolve().is_file():
        raise FileNotFoundError(args.cifar10_tar)
    train_images, train_labels, test_images, test_labels = load_cifar100_tar(
        args.cifar100_tar.resolve()
    )
    seen, unseen = parse_operator_split(args.unseen_operators)
    operators = list(CIFAR_C_CORE_CORRUPTIONS)
    operator_to_id = {name: index for index, name in enumerate(operators)}
    binding = build_class_operator_map(4, 100, seen, int(args.binding_map_seed))
    partition = partition_private_data(
        labels=train_labels,
        num_clients=4,
        num_classes=100,
        partition="dirichlet",
        dirichlet_alpha=0.5,
        max_samples_per_client=int(args.samples_per_client),
        partition_seed=SEED,
    )
    test_indices = balanced_test_indices(
        test_labels,
        samples_per_class=int(args.test_samples_per_class),
        num_classes=100,
        rng=np.random.default_rng(EVAL_SEED),
    )
    test_seen = make_test_arrays(test_images, test_labels, test_indices, seen, operator_to_id, fixed_severity=None)
    test_unseen = make_test_arrays(test_images, test_labels, test_indices, unseen, operator_to_id, fixed_severity=None)
    test_all = {key: np.concatenate([test_seen[key], test_unseen[key]]) for key in test_seen}
    test_clean = {
        "test_images": test_images[test_indices].astype(np.uint8, copy=False),
        "test_labels": test_labels[test_indices].astype(np.uint8, copy=False),
        "test_corruption_ids": np.zeros(test_indices.size, dtype=np.uint8),
        "test_severity_ids": np.zeros(test_indices.size, dtype=np.uint8),
        "test_source_ids": np.arange(test_indices.size, dtype=np.int64),
    }
    dsa = make_test_arrays(test_images, test_labels, test_indices, operators, operator_to_id, fixed_severity=3)

    package_root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "package": PACKAGE_NAME,
        "scenario_id": PACKAGE_NAME,
        "private_dataset": "cifar100",
        "public_dataset": "cifar10",
        "partition_seed": SEED,
        "binding_map_seed": int(args.binding_map_seed),
        "evaluation_seed": EVAL_SEED,
        "gammas": {"gamma00": 0.0, "gamma09": 0.9},
        "model_names": MODEL_NAMES,
        "num_classes": 100,
        "operators": operators,
        "seen_operators": seen,
        "unseen_operators": unseen,
        "class_operator_map": metadata(
            "gamma00", 0.0, operators, seen, unseen, operator_to_id, binding
        )["class_operator_map"],
        "conditions": {},
    }
    labels_by_client: dict[int, np.ndarray] = {}
    paired: dict[int, dict[str, np.ndarray]] = {}
    for condition, gamma in (("gamma00", 0.0), ("gamma09", 0.9)):
        condition_root = package_root / "data" / condition
        record: dict[str, object] = {"clients": {}}
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
                paired[client_id] = {
                    key: arrays[key].copy()
                    for key in ("train_labels", "train_source_indices", "train_severity_ids")
                }
            elif any(not np.array_equal(arrays[key], expected) for key, expected in paired[client_id].items()):
                raise ValueError(f"gamma pairing mismatch for client {client_id}")
            labels_by_client[client_id] = arrays["train_labels"].astype(np.int64)
            record["clients"][str(client_id)] = write_arrays(
                condition_root / f"client_{client_id}", arrays
            )
        for split_name, arrays in (
            ("test_clean", test_clean),
            ("test_seen", test_seen),
            ("test_unseen", test_unseen),
            ("test_balanced", test_all),
        ):
            write_arrays(condition_root / split_name, arrays)
        meta_path = condition_root / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata(condition, gamma, operators, seen, unseen, operator_to_id, binding), indent=2),
            encoding="utf-8",
        )
        record["metadata_sha256"] = sha256_file(meta_path)
        manifest["conditions"][condition] = record

    split_path = package_root / "splits/strict_cle_cifar100_seed0_split0.npz"
    manifest["strict_split"] = save_shared_split(split_path, labels_by_client, float(args.audit_ratio))
    manifest["evaluation"] = {
        "paired_dsa": write_arrays(package_root / "evaluation/paired_dsa", dsa),
        "source_class_counts": np.bincount(test_labels[test_indices], minlength=100).tolist(),
        "severity": 3,
        "seed": EVAL_SEED,
    }
    manifest["initialization"] = prepare_initial_states(package_root / "initial_states")
    public_dir = package_root / "public"
    public_dir.mkdir(parents=True)
    public_path = public_dir / args.cifar10_tar.name
    shutil.copy2(args.cifar10_tar.resolve(), public_path)
    manifest["public"] = {
        "file": public_path.name,
        "bytes": public_path.stat().st_size,
        "sha256": sha256_file(public_path),
    }
    manifest["pew_protocol"] = {
        "action": "train_once_on_packaged_cifar10_public_then_freeze",
        "reason": "avoid CIFAR-100 public/private source overlap in the second private task",
        "private_labels_used": False,
    }
    manifest["paired_conditions"] = {
        "source_indices_identical": True,
        "labels_identical": True,
        "severity_draws_identical": True,
        "shared_fit_audit_split": True,
        "shared_initial_states": True,
    }
    (package_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {package_root}", flush=True)


if __name__ == "__main__":
    main()
