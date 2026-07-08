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
    CORRUPTION_GROUPS,
    GROUP_TO_ID,
    apply_corruption,
    sample_corruption_from_group,
)
from fedprime.data.loaders import partition_private_data  # noqa: E402
from scripts.prepare_corruption_skew_data import (  # noqa: E402
    copy_public_cifar100,
    load_cifar10_arrays,
    make_tarball,
    write_bar_plot,
    write_counts_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CLE-HFL CIFAR-10 protocol dataset.")
    parser.add_argument("--cifar10-root", type=Path, default=Path("RAHFL-master/Dataset/cifar_10"))
    parser.add_argument("--cifar100-root", type=Path, default=Path("RAHFL-master/Dataset/cifar_100"))
    parser.add_argument("--output-root", type=Path, default=Path("local_runs/cle_hfl_prepared"))
    parser.add_argument("--dataset-name", default="alpha05_gamma09_seed0")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9, help="Corruption-label entanglement strength.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--severity-min", type=int, default=1)
    parser.add_argument("--severity-max", type=int, default=5)
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=0,
        help="Optional debug cap before applying every corruption group. 0 means full CIFAR-10 test set.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-public", action="store_true")
    parser.add_argument("--make-tar", action="store_true")
    return parser.parse_args()


def build_class_corruption_map(
    num_clients: int,
    num_classes: int,
    group_names: list[str],
) -> dict[int, dict[int, str]]:
    """Create a deterministic per-client class -> corruption-group map.

    The cyclic offset keeps every client balanced over corruption groups while
    making the shortcut pattern different across clients.
    """

    class_map: dict[int, dict[int, str]] = {}
    num_groups = len(group_names)
    for client_id in range(num_clients):
        class_map[client_id] = {
            class_id: group_names[(class_id + client_id) % num_groups]
            for class_id in range(num_classes)
        }
    return class_map


def sample_group_for_label(
    label: int,
    client_id: int,
    class_group_map: dict[int, dict[int, str]],
    group_names: list[str],
    gamma: float,
    rng: np.random.Generator,
) -> str:
    num_groups = len(group_names)
    probs = np.full(num_groups, (1.0 - gamma) / num_groups, dtype=np.float64)
    dominant = class_group_map[client_id][int(label)]
    probs[group_names.index(dominant)] += gamma
    probs = probs / probs.sum()
    return str(rng.choice(group_names, p=probs))


def generate_cle_client_data(
    images: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    client_id: int,
    class_group_map: dict[int, dict[int, str]],
    gamma: float,
    severity_min: int,
    severity_max: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_names = list(CORRUPTION_GROUPS)
    all_methods = [method for methods in CORRUPTION_GROUPS.values() for method in methods]
    method_to_id = {method: idx for idx, method in enumerate(all_methods)}

    out_images = []
    out_labels = []
    out_group_ids = []
    out_method_ids = []
    for index in indices:
        label = int(labels[index])
        group = sample_group_for_label(label, client_id, class_group_map, group_names, gamma, rng)
        method = sample_corruption_from_group(group, rng)
        severity = int(rng.integers(severity_min, severity_max + 1))
        out_images.append(apply_corruption(images[index], method, severity, rng))
        out_labels.append(label)
        out_group_ids.append(GROUP_TO_ID[group])
        out_method_ids.append(method_to_id[method])
    return (
        np.asarray(out_images, dtype=np.uint8),
        np.asarray(out_labels, dtype=np.uint8),
        np.asarray(out_group_ids, dtype=np.uint8),
        np.asarray(out_method_ids, dtype=np.uint8),
    )


def generate_counterfactual_test(
    images: np.ndarray,
    labels: np.ndarray,
    severity_min: int,
    severity_max: int,
    rng: np.random.Generator,
    max_test_images: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(images))
    if max_test_images > 0 and max_test_images < len(indices):
        indices = rng.choice(indices, size=max_test_images, replace=False)

    out_images = []
    out_labels = []
    out_group_ids = []
    for group in CORRUPTION_GROUPS:
        for index in indices:
            method = sample_corruption_from_group(group, rng)
            severity = int(rng.integers(severity_min, severity_max + 1))
            out_images.append(apply_corruption(images[index], method, severity, rng))
            out_labels.append(int(labels[index]))
            out_group_ids.append(GROUP_TO_ID[group])
    return (
        np.asarray(out_images, dtype=np.uint8),
        np.asarray(out_labels, dtype=np.uint8),
        np.asarray(out_group_ids, dtype=np.uint8),
    )


def write_class_group_map(path: Path, class_group_map: dict[int, dict[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["client", "class_id", "dominant_corruption_group"])
        for client_id, mapping in class_group_map.items():
            for class_id, group in mapping.items():
                writer.writerow([client_id, class_id, group])


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {args.gamma}")
    if args.severity_min < 1 or args.severity_max > 5 or args.severity_min > args.severity_max:
        raise ValueError("severity range must satisfy 1 <= min <= max <= 5.")

    rng = np.random.default_rng(args.seed)
    package_root = args.output_root / f"cle_hfl_prepared_{args.dataset_name}"
    cle_root = package_root / "cifar_10_cle" / args.dataset_name
    audit_root = cle_root / "audit"
    cle_root.mkdir(parents=True, exist_ok=True)

    train_images, train_labels, test_images, test_labels = load_cifar10_arrays(args.cifar10_root, args.download)
    dataidx_map = partition_private_data(
        labels=train_labels,
        num_clients=args.num_clients,
        num_classes=args.num_classes,
        partition="dirichlet",
        dirichlet_alpha=args.alpha,
        max_samples_per_client=args.samples_per_client,
        partition_seed=args.seed,
    )

    group_names = list(CORRUPTION_GROUPS)
    class_group_map = build_class_corruption_map(args.num_clients, args.num_classes, group_names)
    label_rows = []
    group_rows = []
    class_group_rows = []

    for client_id in range(args.num_clients):
        client_dir = cle_root / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        c_images, c_labels, c_group_ids, c_method_ids = generate_cle_client_data(
            images=train_images,
            labels=train_labels,
            indices=dataidx_map[client_id],
            client_id=client_id,
            class_group_map=class_group_map,
            gamma=args.gamma,
            severity_min=args.severity_min,
            severity_max=args.severity_max,
            rng=rng,
        )
        np.save(client_dir / "train_images.npy", c_images)
        np.save(client_dir / "train_labels.npy", c_labels)
        np.save(client_dir / "train_corruption_ids.npy", c_group_ids)
        np.save(client_dir / "train_corruption_method_ids.npy", c_method_ids)

        label_counts = np.bincount(c_labels.astype(np.int64), minlength=args.num_classes)
        group_counts = np.bincount(c_group_ids.astype(np.int64), minlength=len(group_names))
        label_rows.append([client_id, int(c_labels.size), *label_counts.astype(int).tolist()])
        group_rows.append([client_id, int(c_group_ids.size), *group_counts.astype(int).tolist()])
        for class_id in range(args.num_classes):
            class_mask = c_labels.astype(np.int64) == class_id
            for group_id, group in enumerate(group_names):
                count = int(((c_group_ids.astype(np.int64) == group_id) & class_mask).sum())
                class_group_rows.append([client_id, class_id, group, count])

        write_bar_plot(
            audit_root / f"client_{client_id}_label_counts.png",
            f"client {client_id} label counts",
            [f"class_{idx}" for idx in range(args.num_classes)],
            label_counts,
        )
        write_bar_plot(
            audit_root / f"client_{client_id}_corruption_counts.png",
            f"client {client_id} corruption counts",
            group_names,
            group_counts,
        )

    test_dir = cle_root / "test_balanced"
    test_dir.mkdir(parents=True, exist_ok=True)
    t_images, t_labels, t_group_ids = generate_counterfactual_test(
        test_images,
        test_labels,
        severity_min=args.severity_min,
        severity_max=args.severity_max,
        rng=rng,
        max_test_images=args.max_test_images,
    )
    np.save(test_dir / "test_images.npy", t_images)
    np.save(test_dir / "test_labels.npy", t_labels)
    np.save(test_dir / "test_corruption_ids.npy", t_group_ids)

    write_counts_csv(
        audit_root / "client_label_counts.csv",
        ["client", "total", *[f"class_{idx}" for idx in range(args.num_classes)]],
        label_rows,
    )
    write_counts_csv(
        audit_root / "client_corruption_counts.csv",
        ["client", "total", *group_names],
        group_rows,
    )
    write_counts_csv(
        audit_root / "client_class_corruption_counts.csv",
        ["client", "class_id", "group", "count"],
        class_group_rows,
    )
    write_class_group_map(audit_root / "class_corruption_map.csv", class_group_map)

    metadata = {
        "dataset": "cifar10_cle_hfl",
        "dataset_name": args.dataset_name,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "num_classes": args.num_classes,
        "samples_per_client": args.samples_per_client,
        "severity_min": args.severity_min,
        "severity_max": args.severity_max,
        "max_test_images": args.max_test_images,
        "corruption_groups": CORRUPTION_GROUPS,
        "group_to_id": GROUP_TO_ID,
        "class_corruption_map": {
            str(client_id): {str(class_id): group for class_id, group in mapping.items()}
            for client_id, mapping in class_group_map.items()
        },
        "train_protocol": "Dirichlet label-skew followed by class-dependent corruption sampling.",
        "test_protocol": "Counterfactual class-corruption test; every selected CIFAR-10 test image is corrupted once per group.",
    }
    (cle_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.include_public:
        copy_public_cifar100(args.cifar100_root, package_root / "cifar_100")

    if args.make_tar:
        tar_path = make_tarball(package_root)
        print(f"Wrote tarball: {tar_path}")
    print(f"Wrote CLE-HFL dataset: {package_root}")
    print(f"Metadata: {cle_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
