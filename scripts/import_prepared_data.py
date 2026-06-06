from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED_FILES = (
    Path("RAHFL-master/Dataset/cifar_10_c/train/random_corrupt_1.npy"),
    Path("RAHFL-master/Dataset/cifar_10_c/train/labels.npy"),
    Path("RAHFL-master/Dataset/cifar_10_c/test/random_corrupt_1.npy"),
    Path("RAHFL-master/Dataset/cifar_10_c/test/labels.npy"),
    Path("RAHFL-master/Dataset/cifar_100/cifar-100-python/train"),
    Path("outputs/partitions/cifar10c_alpha05_seed0_clients4_samples10000.npz"),
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


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()

    print(f"Prepared-data source: {source}")
    print(f"Repository destination: {destination}")

    copy_directory(
        source / "cifar_10_c",
        destination / "RAHFL-master/Dataset/cifar_10_c",
    )
    copy_directory(
        source / "cifar_100",
        destination / "RAHFL-master/Dataset/cifar_100",
    )
    copy_directory(
        source / "outputs/partitions",
        destination / "outputs/partitions",
    )

    missing = [path for path in REQUIRED_FILES if not (destination / path).exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Prepared-data import is incomplete:\n{missing_text}")

    print("Prepared-data import verified successfully.")


if __name__ == "__main__":
    main()
