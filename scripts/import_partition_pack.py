from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import mounted extra partition files into outputs/partitions.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("."))
    return parser.parse_args()


def find_partition_files(source: Path) -> list[Path]:
    preferred = source / "outputs" / "partitions"
    if preferred.is_dir():
        return sorted(preferred.glob("*.npz"))
    return sorted(source.rglob("cifar10c_alpha*_seed*_clients*_samples*.npz"))


def import_from_archive(source: Path, target_dir: Path) -> int:
    copied = 0
    with tarfile.open(source, "r:*") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if not name.startswith("cifar10c_alpha") or not name.endswith(".npz"):
                continue
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            target = target_dir / name
            with target.open("wb") as out:
                shutil.copyfileobj(file_obj, out)
            print(f"Copied {member.name} -> {target}")
            copied += 1
    return copied


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    target_dir = destination / "outputs" / "partitions"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Partition pack source does not exist: {source}")

    if source.is_file() and source.suffixes[-2:] == [".tar", ".gz"]:
        copied = import_from_archive(source, target_dir)
    else:
        files = find_partition_files(source)
        if not files:
            raise FileNotFoundError(f"No partition .npz files found under: {source}")
        copied = 0
        for path in files:
            target = target_dir / path.name
            shutil.copy2(path, target)
            print(f"Copied {path} -> {target}")
            copied += 1

    if copied == 0:
        raise FileNotFoundError(f"No partition .npz files found under: {source}")

    print(f"Imported {copied} partition files into {target_dir}")


if __name__ == "__main__":
    main()
