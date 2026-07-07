from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.corruptions import (
    CORRUPTION_GROUPS,
    GROUP_TO_ID,
    apply_corruption,
    build_client_group_profiles,
    sample_corruption_from_group,
)
from fedprime.data.loaders import partition_private_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a corruption-skew CIFAR-10 protocol dataset.")
    parser.add_argument("--cifar10-root", type=Path, default=Path("RAHFL-master/Dataset/cifar_10"))
    parser.add_argument("--cifar100-root", type=Path, default=Path("RAHFL-master/Dataset/cifar_100"))
    parser.add_argument("--output-root", type=Path, default=Path("local_runs/fedsara_cs_prepared"))
    parser.add_argument("--dataset-name", default="alpha05_rho07_seed0")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.7, help="Dominant corruption-group probability.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=0,
        help="Optional debug cap before applying each corruption group to CIFAR-10 test images. 0 means full test set.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-public", action="store_true", help="Copy CIFAR-100 into the output package.")
    parser.add_argument("--make-tar", action="store_true")
    return parser.parse_args()


def write_counts_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_bar_plot(path: Path, title: str, labels: list[str], values: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 4))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_client_data(
    images: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    profile: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = list(profile.keys())
    probs = np.asarray([profile[group] for group in groups], dtype=np.float64)
    probs = probs / probs.sum()

    out_images = []
    out_labels = []
    out_group_ids = []
    out_method_ids = []
    all_methods = [method for methods in CORRUPTION_GROUPS.values() for method in methods]
    method_to_id = {method: idx for idx, method in enumerate(all_methods)}

    for index in indices:
        group = str(rng.choice(groups, p=probs))
        method = sample_corruption_from_group(group, rng)
        severity = int(rng.integers(1, 5))
        out_images.append(apply_corruption(images[index], method, severity, rng))
        out_labels.append(int(labels[index]))
        out_group_ids.append(GROUP_TO_ID[group])
        out_method_ids.append(method_to_id[method])

    return (
        np.asarray(out_images, dtype=np.uint8),
        np.asarray(out_labels, dtype=np.uint8),
        np.asarray(out_group_ids, dtype=np.uint8),
        np.asarray(out_method_ids, dtype=np.uint8),
    )


def generate_balanced_test(
    images: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    max_test_images: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_images = []
    out_labels = []
    out_group_ids = []
    indices = np.arange(len(images))
    if max_test_images > 0 and max_test_images < len(indices):
        indices = rng.choice(indices, size=max_test_images, replace=False)
    for group in CORRUPTION_GROUPS:
        for index in indices:
            method = sample_corruption_from_group(group, rng)
            severity = int(rng.integers(1, 5))
            out_images.append(apply_corruption(images[index], method, severity, rng))
            out_labels.append(int(labels[index]))
            out_group_ids.append(GROUP_TO_ID[group])
    return (
        np.asarray(out_images, dtype=np.uint8),
        np.asarray(out_labels, dtype=np.uint8),
        np.asarray(out_group_ids, dtype=np.uint8),
    )


def copy_public_cifar100(source: Path, destination: Path) -> None:
    candidates = [
        source / "cifar-100-python",
        source,
    ]
    cifar100_root = None
    for candidate in candidates:
        try:
            has_files = (candidate / "train").exists() and (candidate / "test").exists()
        except PermissionError:
            has_files = False
        if has_files:
            cifar100_root = candidate
            break

    destination.mkdir(parents=True, exist_ok=True)
    if cifar100_root is not None:
        target = destination / "cifar-100-python"
        shutil.copytree(cifar100_root, target, dirs_exist_ok=True)
        return

    tar_candidates = [
        source / "cifar-100-python.tar.gz",
        source / "cifar-100-python" / "cifar-100-python.tar.gz",
    ]
    tar_source = next((candidate for candidate in tar_candidates if candidate.exists()), None)
    if tar_source is None:
        raise FileNotFoundError(f"Could not locate CIFAR-100 files or tarball under {source}")
    shutil.copy2(tar_source, destination / "cifar-100-python.tar.gz")


def make_tarball(package_root: Path) -> Path:
    tar_path = package_root.with_suffix(".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(package_root, arcname=package_root.name)
    return tar_path


def _cifar10_batch_to_arrays(batch: dict[bytes, object] | dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    data = batch.get(b"data", batch.get("data"))
    labels = batch.get(b"labels", batch.get("labels"))
    if data is None or labels is None:
        raise KeyError("CIFAR-10 batch is missing data or labels.")
    images = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    targets = np.asarray(labels, dtype=np.int64)
    return images, targets


def _load_cifar10_from_tar(cifar10_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tar_path = cifar10_root / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing CIFAR-10 tarball: {tar_path}")

    train_images = []
    train_labels = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for batch_idx in range(1, 6):
            member = tar.getmember(f"cifar-10-batches-py/data_batch_{batch_idx}")
            file_obj = tar.extractfile(member)
            if file_obj is None:
                raise FileNotFoundError(member.name)
            images, labels = _cifar10_batch_to_arrays(pickle.load(file_obj, encoding="latin1"))
            train_images.append(images)
            train_labels.append(labels)

        member = tar.getmember("cifar-10-batches-py/test_batch")
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise FileNotFoundError(member.name)
        test_images, test_labels = _cifar10_batch_to_arrays(pickle.load(file_obj, encoding="latin1"))

    return (
        np.concatenate(train_images, axis=0),
        np.concatenate(train_labels, axis=0),
        test_images,
        test_labels,
    )


def load_cifar10_arrays(cifar10_root: Path, download: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        train_ds = datasets.CIFAR10(str(cifar10_root), train=True, download=download)
        test_ds = datasets.CIFAR10(str(cifar10_root), train=False, download=download)
        return (
            np.asarray(train_ds.data, dtype=np.uint8),
            np.asarray(train_ds.targets, dtype=np.int64),
            np.asarray(test_ds.data, dtype=np.uint8),
            np.asarray(test_ds.targets, dtype=np.int64),
        )
    except Exception as exc:
        print(f"torchvision CIFAR-10 loader failed: {exc}")
        print("Falling back to direct cifar-10-python.tar.gz reader.")
        return _load_cifar10_from_tar(cifar10_root)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    package_root = args.output_root / f"fedsara_cs_prepared_{args.dataset_name}"
    cs_root = package_root / "cifar_10_cs" / args.dataset_name
    audit_root = cs_root / "audit"
    cs_root.mkdir(parents=True, exist_ok=True)

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
    profiles = build_client_group_profiles(args.num_clients, args.rho)

    label_rows = []
    group_rows = []
    for client_id in range(args.num_clients):
        client_dir = cs_root / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        c_images, c_labels, c_group_ids, c_method_ids = generate_client_data(
            images=train_images,
            labels=train_labels,
            indices=dataidx_map[client_id],
            profile=profiles[client_id],
            rng=rng,
        )
        np.save(client_dir / "train_images.npy", c_images)
        np.save(client_dir / "train_labels.npy", c_labels)
        np.save(client_dir / "train_corruption_ids.npy", c_group_ids)
        np.save(client_dir / "train_corruption_method_ids.npy", c_method_ids)

        label_counts = np.bincount(c_labels.astype(np.int64), minlength=args.num_classes)
        group_counts = np.bincount(c_group_ids.astype(np.int64), minlength=len(CORRUPTION_GROUPS))
        label_rows.append([client_id, int(c_labels.size), *label_counts.astype(int).tolist()])
        group_rows.append([client_id, int(c_group_ids.size), *group_counts.astype(int).tolist()])
        write_bar_plot(
            audit_root / f"client_{client_id}_label_counts.png",
            f"client {client_id} label counts",
            [f"class_{idx}" for idx in range(args.num_classes)],
            label_counts,
        )
        write_bar_plot(
            audit_root / f"client_{client_id}_corruption_counts.png",
            f"client {client_id} corruption counts",
            list(CORRUPTION_GROUPS.keys()),
            group_counts,
        )

    test_dir = cs_root / "test_balanced"
    test_dir.mkdir(parents=True, exist_ok=True)
    t_images, t_labels, t_group_ids = generate_balanced_test(
        test_images,
        test_labels,
        rng,
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
        ["client", "total", *list(CORRUPTION_GROUPS.keys())],
        group_rows,
    )

    metadata = {
        "dataset": "cifar10_corruption_skew",
        "dataset_name": args.dataset_name,
        "alpha": args.alpha,
        "rho": args.rho,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "num_classes": args.num_classes,
        "samples_per_client": args.samples_per_client,
        "max_test_images": args.max_test_images,
        "corruption_groups": CORRUPTION_GROUPS,
        "group_to_id": GROUP_TO_ID,
        "client_profiles": profiles,
        "test_protocol": "balanced over corruption groups; every CIFAR-10 test image is corrupted once per group",
    }
    (cs_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.include_public:
        copy_public_cifar100(args.cifar100_root, package_root / "cifar_100")

    if args.make_tar:
        tar_path = make_tarball(package_root)
        print(f"Wrote tarball: {tar_path}")
    print(f"Wrote corruption-skew dataset: {package_root}")
    print(f"Metadata: {cs_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
