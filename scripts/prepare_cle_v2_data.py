from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

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
from scripts.prepare_corruption_skew_data import (  # noqa: E402
    copy_public_cifar100,
    load_cifar10_arrays,
    make_tarball,
    write_counts_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build operator-level CLE-HFL v2 with seen/unseen corruptions."
    )
    parser.add_argument(
        "--cifar10-root",
        type=Path,
        default=Path("RAHFL-master/Dataset/cifar_10"),
    )
    parser.add_argument(
        "--cifar100-root",
        type=Path,
        default=Path("RAHFL-master/Dataset/cifar_100"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("local_runs/cle_hfl_v2_prepared"),
    )
    parser.add_argument("--dataset-name", default="alpha05_gamma09_seed0_split0")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operator-split-seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--severity-min", type=int, default=1)
    parser.add_argument("--severity-max", type=int, default=5)
    parser.add_argument(
        "--test-samples-per-class",
        type=int,
        default=100,
        help="Number of clean CIFAR-10 test examples per class before operator expansion.",
    )
    parser.add_argument(
        "--unseen-operators",
        default=",".join(DEFAULT_UNSEEN_CORRUPTIONS),
        help="Comma-separated operators held out from all client training data.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-public", action="store_true")
    parser.add_argument("--make-tar", action="store_true")
    return parser.parse_args()


def parse_operator_split(raw_unseen: str) -> tuple[list[str], list[str]]:
    unseen = [name.strip() for name in raw_unseen.split(",") if name.strip()]
    if not unseen:
        raise ValueError("At least one unseen operator is required.")
    unknown = sorted(set(unseen).difference(CIFAR_C_CORE_CORRUPTIONS))
    if unknown:
        raise ValueError(f"Unknown unseen operators: {unknown}")
    if len(set(unseen)) != len(unseen):
        raise ValueError("unseen operators must be unique")
    seen = [name for name in CIFAR_C_CORE_CORRUPTIONS if name not in unseen]
    if len(seen) < 2:
        raise ValueError("At least two seen operators are required.")
    return seen, unseen


def build_class_operator_map(
    num_clients: int,
    num_classes: int,
    seen_operators: list[str],
    seed: int,
) -> dict[int, dict[int, str]]:
    """Assign each client/class a concrete dominant seen operator.

    Every client receives a seeded permutation. Within a client, classes use
    distinct dominant operators whenever enough seen operators are available.
    """

    rng = np.random.default_rng(int(seed))
    mapping: dict[int, dict[int, str]] = {}
    for client_id in range(int(num_clients)):
        repeats = int(np.ceil(num_classes / len(seen_operators)))
        candidates = np.concatenate(
            [rng.permutation(seen_operators) for _ in range(repeats)]
        ).tolist()
        mapping[client_id] = {
            class_id: str(candidates[class_id])
            for class_id in range(int(num_classes))
        }
    return mapping


def sample_operator_for_label(
    *,
    label: int,
    client_id: int,
    class_operator_map: dict[int, dict[int, str]],
    seen_operators: list[str],
    gamma: float,
    rng: np.random.Generator,
) -> str:
    probabilities = np.full(
        len(seen_operators),
        (1.0 - float(gamma)) / len(seen_operators),
        dtype=np.float64,
    )
    dominant = class_operator_map[int(client_id)][int(label)]
    probabilities[seen_operators.index(dominant)] += float(gamma)
    return str(rng.choice(seen_operators, p=probabilities / probabilities.sum()))


def generate_client_data(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    client_id: int,
    class_operator_map: dict[int, dict[int, str]],
    seen_operators: list[str],
    operator_to_id: dict[str, int],
    gamma: float,
    severity_min: int,
    severity_max: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_images = []
    output_labels = []
    output_operator_ids = []
    output_severities = []
    for index in indices:
        label = int(labels[index])
        operator = sample_operator_for_label(
            label=label,
            client_id=client_id,
            class_operator_map=class_operator_map,
            seen_operators=seen_operators,
            gamma=gamma,
            rng=rng,
        )
        severity = int(rng.integers(severity_min, severity_max + 1))
        output_images.append(apply_corruption(images[index], operator, severity, rng))
        output_labels.append(label)
        output_operator_ids.append(operator_to_id[operator])
        output_severities.append(severity)
    return (
        np.asarray(output_images, dtype=np.uint8),
        np.asarray(output_labels, dtype=np.uint8),
        np.asarray(output_operator_ids, dtype=np.uint8),
        np.asarray(output_severities, dtype=np.uint8),
    )


def balanced_test_indices(
    labels: np.ndarray,
    *,
    samples_per_class: int,
    num_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    parts = []
    for class_id in range(int(num_classes)):
        candidates = np.flatnonzero(labels.astype(np.int64) == class_id)
        if candidates.size < int(samples_per_class):
            raise ValueError(
                f"class {class_id} has {candidates.size} test samples, "
                f"need {samples_per_class}"
            )
        parts.append(
            np.sort(
                rng.choice(candidates, size=int(samples_per_class), replace=False)
            )
        )
    return np.concatenate(parts).astype(np.int64, copy=False)


def generate_operator_test(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    operators: list[str],
    operator_to_id: dict[str, int],
    severity_min: int,
    severity_max: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_images = []
    output_labels = []
    output_operator_ids = []
    output_severities = []
    for operator in operators:
        for index in indices:
            severity = int(rng.integers(severity_min, severity_max + 1))
            output_images.append(
                apply_corruption(images[int(index)], operator, severity, rng)
            )
            output_labels.append(int(labels[int(index)]))
            output_operator_ids.append(operator_to_id[operator])
            output_severities.append(severity)
    return (
        np.asarray(output_images, dtype=np.uint8),
        np.asarray(output_labels, dtype=np.uint8),
        np.asarray(output_operator_ids, dtype=np.uint8),
        np.asarray(output_severities, dtype=np.uint8),
    )


def write_test_split(
    path: Path,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    images, labels, operator_ids, severities = arrays
    np.save(path / "test_images.npy", images)
    np.save(path / "test_labels.npy", labels)
    np.save(path / "test_corruption_ids.npy", operator_ids)
    np.save(path / "test_severity_ids.npy", severities)


def write_mapping_csv(
    path: Path,
    class_operator_map: dict[int, dict[int, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["client", "class_id", "dominant_operator", "family"])
        for client_id, mapping in class_operator_map.items():
            for class_id, operator in mapping.items():
                writer.writerow(
                    [
                        client_id,
                        class_id,
                        operator,
                        CORRUPTION_OPERATOR_FAMILIES[operator],
                    ]
                )


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 1 <= args.severity_min <= args.severity_max <= 5:
        raise ValueError("severity range must satisfy 1 <= min <= max <= 5")

    seen_operators, unseen_operators = parse_operator_split(args.unseen_operators)
    operators = list(CIFAR_C_CORE_CORRUPTIONS)
    operator_to_id = {name: idx for idx, name in enumerate(operators)}
    operator_splits = {
        name: ("unseen" if name in unseen_operators else "seen")
        for name in operators
    }
    package_root = args.output_root / f"cle_hfl_v2_prepared_{args.dataset_name}"
    cle_root = package_root / "cifar_10_cle_v2" / args.dataset_name
    audit_root = cle_root / "audit"
    cle_root.mkdir(parents=True, exist_ok=True)

    train_images, train_labels, test_images, test_labels = load_cifar10_arrays(
        args.cifar10_root,
        args.download,
    )
    dataidx_map = partition_private_data(
        labels=train_labels,
        num_clients=args.num_clients,
        num_classes=args.num_classes,
        partition="dirichlet",
        dirichlet_alpha=args.alpha,
        max_samples_per_client=args.samples_per_client,
        partition_seed=args.seed,
    )
    class_operator_map = build_class_operator_map(
        args.num_clients,
        args.num_classes,
        seen_operators,
        args.operator_split_seed,
    )

    operator_rows = []
    class_operator_rows = []
    for client_id in range(args.num_clients):
        client_rng = np.random.default_rng(args.seed * 100003 + client_id * 1009 + 17)
        client_dir = cle_root / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        c_images, c_labels, c_operator_ids, c_severities = generate_client_data(
            images=train_images,
            labels=train_labels,
            indices=dataidx_map[client_id],
            client_id=client_id,
            class_operator_map=class_operator_map,
            seen_operators=seen_operators,
            operator_to_id=operator_to_id,
            gamma=args.gamma,
            severity_min=args.severity_min,
            severity_max=args.severity_max,
            rng=client_rng,
        )
        np.save(client_dir / "train_images.npy", c_images)
        np.save(client_dir / "train_labels.npy", c_labels)
        np.save(client_dir / "train_corruption_ids.npy", c_operator_ids)
        np.save(client_dir / "train_corruption_method_ids.npy", c_operator_ids)
        np.save(client_dir / "train_severity_ids.npy", c_severities)

        counts = np.bincount(c_operator_ids.astype(np.int64), minlength=len(operators))
        operator_rows.append([client_id, int(c_labels.size), *counts.tolist()])
        labels64 = c_labels.astype(np.int64)
        operator_ids64 = c_operator_ids.astype(np.int64)
        for class_id in range(args.num_classes):
            for operator_id, operator in enumerate(operators):
                count = int(
                    (
                        (labels64 == class_id)
                        & (operator_ids64 == operator_id)
                    ).sum()
                )
                class_operator_rows.append(
                    [
                        client_id,
                        class_id,
                        operator,
                        operator_splits[operator],
                        count,
                    ]
                )

    test_rng = np.random.default_rng(args.seed * 200003 + args.operator_split_seed + 31)
    test_indices = balanced_test_indices(
        test_labels,
        samples_per_class=args.test_samples_per_class,
        num_classes=args.num_classes,
        rng=test_rng,
    )
    seen_arrays = generate_operator_test(
        images=test_images,
        labels=test_labels,
        indices=test_indices,
        operators=seen_operators,
        operator_to_id=operator_to_id,
        severity_min=args.severity_min,
        severity_max=args.severity_max,
        rng=np.random.default_rng(args.seed * 300007 + 41),
    )
    unseen_arrays = generate_operator_test(
        images=test_images,
        labels=test_labels,
        indices=test_indices,
        operators=unseen_operators,
        operator_to_id=operator_to_id,
        severity_min=args.severity_min,
        severity_max=args.severity_max,
        rng=np.random.default_rng(args.seed * 300007 + 43),
    )
    all_arrays = tuple(
        np.concatenate([seen, unseen], axis=0)
        for seen, unseen in zip(seen_arrays, unseen_arrays)
    )
    clean_arrays = (
        test_images[test_indices].astype(np.uint8, copy=False),
        test_labels[test_indices].astype(np.uint8, copy=False),
        np.zeros(test_indices.size, dtype=np.uint8),
        np.zeros(test_indices.size, dtype=np.uint8),
    )
    write_test_split(cle_root / "test_seen", seen_arrays)
    write_test_split(cle_root / "test_unseen", unseen_arrays)
    write_test_split(cle_root / "test_balanced", all_arrays)
    write_test_split(cle_root / "test_clean", clean_arrays)

    write_counts_csv(
        audit_root / "client_operator_counts.csv",
        ["client", "total", *operators],
        operator_rows,
    )
    write_counts_csv(
        audit_root / "client_class_operator_counts.csv",
        ["client", "class_id", "operator", "split", "count"],
        class_operator_rows,
    )
    write_mapping_csv(
        audit_root / "class_operator_map.csv",
        class_operator_map,
    )
    write_counts_csv(
        audit_root / "operator_manifest.csv",
        ["operator_id", "operator", "family", "split"],
        [
            [
                operator_to_id[operator],
                operator,
                CORRUPTION_OPERATOR_FAMILIES[operator],
                operator_splits[operator],
            ]
            for operator in operators
        ],
    )

    metadata = {
        "dataset": "cifar10_cle_hfl_v2",
        "protocol_version": 2,
        "dataset_name": args.dataset_name,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "seed": args.seed,
        "operator_split_seed": args.operator_split_seed,
        "num_clients": args.num_clients,
        "num_classes": args.num_classes,
        "samples_per_client": args.samples_per_client,
        "test_samples_per_class": args.test_samples_per_class,
        "severity_min": args.severity_min,
        "severity_max": args.severity_max,
        "operators": operators,
        "operator_to_id": operator_to_id,
        "operator_families": CORRUPTION_OPERATOR_FAMILIES,
        "operator_splits": operator_splits,
        "seen_operators": seen_operators,
        "unseen_operators": unseen_operators,
        "class_operator_map": {
            str(client_id): {
                str(class_id): operator
                for class_id, operator in mapping.items()
            }
            for client_id, mapping in class_operator_map.items()
        },
        "train_protocol": (
            "Dirichlet label-skew followed by client/class-conditional sampling "
            "over concrete seen corruption operators."
        ),
        "test_protocol": (
            "Class-balanced counterfactual evaluation over every concrete "
            "operator, with disjoint seen and unseen operator splits."
        ),
        "method_visibility": (
            "Operator IDs, names, families, and split labels are evaluation "
            "metadata only and must not be passed to the training method."
        ),
    }
    (cle_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.include_public:
        copy_public_cifar100(args.cifar100_root, package_root / "cifar_100")
    if args.make_tar:
        print(f"Wrote tarball: {make_tarball(package_root)}")
    print(f"Wrote CLE-HFL v2 dataset: {package_root}")
    print(f"Seen operators ({len(seen_operators)}): {seen_operators}")
    print(f"Unseen operators ({len(unseen_operators)}): {unseen_operators}")


if __name__ == "__main__":
    main()
