from __future__ import annotations

import argparse
import shutil
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


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    target_dir = destination / "outputs" / "partitions"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Partition pack source does not exist: {source}")

    files = find_partition_files(source)
    if not files:
        raise FileNotFoundError(f"No partition .npz files found under: {source}")

    for path in files:
        target = target_dir / path.name
        shutil.copy2(path, target)
        print(f"Copied {path} -> {target}")

    print(f"Imported {len(files)} partition files into {target_dir}")


if __name__ == "__main__":
    main()
