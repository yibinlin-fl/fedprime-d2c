from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


DATASET_NAME = "cle_hfl_indomain_public_alpha05_gamma09_seed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the CLE-HFL in-domain public pool.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("local_runs/cle_hfl_indomain_public"),
    )
    return parser.parse_args()


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if destination_resolved not in member_path.parents and member_path != destination_resolved:
            raise ValueError(f"Unsafe tar member path: {member.name}")
    tar.extractall(destination)


def locate_dataset(source: Path) -> Path:
    if (source / "public_images.npy").is_file():
        return source
    direct = source / DATASET_NAME
    if (direct / "public_images.npy").is_file():
        return direct
    matches = [path.parent for path in source.rglob("public_images.npy") if path.parent.name == DATASET_NAME]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate {DATASET_NAME}/public_images.npy under {source}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        if not source.name.endswith((".tar.gz", ".tgz")):
            raise ValueError(f"Unsupported public package: {source}")
        with tarfile.open(source, "r:gz") as tar:
            safe_extract(tar, destination)
        imported = locate_dataset(destination)
    elif source.is_dir():
        dataset = locate_dataset(source)
        imported = destination / DATASET_NAME
        shutil.copytree(dataset, imported, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(source)

    required = ["public_images.npy", "public_indices.npy", "reserved_indices.npy", "metadata.json"]
    missing = [name for name in required if not (imported / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Imported public pool is missing: {missing}")
    print(f"Imported CLE-HFL public pool: {imported}", flush=True)


if __name__ == "__main__":
    main()
