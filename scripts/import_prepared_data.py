from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PARTITION_NAME = "cifar10c_alpha05_seed0_clients4_samples10000.npz"

REQUIRED_FILES = (
    Path("RAHFL-master/Dataset/cifar_10_c/train/random_corrupt_1.npy"),
    Path("RAHFL-master/Dataset/cifar_10_c/train/labels.npy"),
    Path("RAHFL-master/Dataset/cifar_10_c/test/random_corrupt_1.npy"),
    Path("RAHFL-master/Dataset/cifar_10_c/test/labels.npy"),
    Path("RAHFL-master/Dataset/cifar_100/cifar-100-python/train"),
    Path(f"outputs/partitions/{PARTITION_NAME}"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a mounted FedPRIME prepared-data dataset into the repository."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/kaggle/input/fedprime-data"),
        help="Mounted dataset root containing cifar_10_c, cifar_100, and outputs.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("."),
        help="FedPRIME-D2C repository root.",
    )
    return parser.parse_args()


def copy_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Mounted dataset directory not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Copied {source} -> {destination}")


def find_unique_root(search_root: Path, marker: Path, root_parents: int) -> Path:
    search_roots = [search_root]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir() and kaggle_input != search_root:
        search_roots.append(kaggle_input)

    matches = []
    used_root = search_root
    for candidate in search_roots:
        matches = list(candidate.rglob(str(marker)))
        if matches:
            used_root = candidate
            break
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{marker}' below: "
            + ", ".join(str(candidate) for candidate in search_roots)
        )
    if len(matches) > 1:
        print(f"Found {len(matches)} matches for {marker}; using {matches[0]}")
    elif used_root != search_root:
        print(f"Located {marker} through fallback search under {used_root}")
    root = matches[0]
    for _ in range(root_parents):
        root = root.parent
    return root


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()

    print(f"Prepared-data source: {source}")
    print(f"Repository destination: {destination}")

    cifar10c_source = source / "cifar_10_c"
    if not cifar10c_source.is_dir():
        cifar10c_source = find_unique_root(
            source,
            Path("train/random_corrupt_1.npy"),
            root_parents=2,
        )

    cifar100_source = source / "cifar_100"
    if not cifar100_source.is_dir():
        cifar100_source = find_unique_root(
            source,
            Path("cifar-100-python/train"),
            root_parents=2,
        )

    partitions_source = source / "outputs/partitions"
    if not partitions_source.is_dir():
        partition_file = find_unique_root(
            source,
            Path(PARTITION_NAME),
            root_parents=0,
        )
        partitions_source = partition_file.parent

    copy_directory(
        cifar10c_source,
        destination / "RAHFL-master/Dataset/cifar_10_c",
    )
    copy_directory(
        cifar100_source,
        destination / "RAHFL-master/Dataset/cifar_100",
    )
    copy_directory(
        partitions_source,
        destination / "outputs/partitions",
    )

    missing = [path for path in REQUIRED_FILES if not (destination / path).exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Prepared-data import is incomplete:\n{missing_text}")

    print("Prepared-data import verified successfully.")


if __name__ == "__main__":
    main()
