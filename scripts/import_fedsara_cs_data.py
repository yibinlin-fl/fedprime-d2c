from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a mounted FedSARA-CS prepared dataset.")
    parser.add_argument("--source", type=Path, default=Path("/dataset"))
    parser.add_argument("--destination", type=Path, default=Path("."))
    parser.add_argument("--repo-root", dest="destination", type=Path)
    return parser.parse_args()


def maybe_extract_archive(source: Path, destination: Path) -> Path:
    if source.is_file() and source.suffixes[-2:] == [".tar", ".gz"]:
        extract_root = destination / "local_runs" / "openi_imported_fedsara_cs"
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {source} -> {extract_root}")
        with tarfile.open(source, "r:gz") as tar:
            tar.extractall(extract_root)
        return extract_root
    return source


def find_root(search_root: Path, marker: Path, root_parents: int) -> Path:
    candidates = [search_root]
    for extra in (Path("/dataset"), Path("/cache/dataset"), Path("/tmp/dataset"), Path("/mnt/data")):
        if extra.is_dir() and extra not in candidates:
            candidates.append(extra)
    for candidate in candidates:
        matches = list(candidate.rglob(str(marker))) if candidate.exists() else []
        if matches:
            root = matches[0]
            for _ in range(root_parents):
                root = root.parent
            print(f"Located {marker} under {candidate}: {root}")
            return root
    raise FileNotFoundError(f"Could not find marker {marker} under {[str(c) for c in candidates]}")


def copy_dir(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Copied {source} -> {destination}")


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except PermissionError:
        return False


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    source = maybe_extract_archive(args.source.resolve(), destination)
    print(f"FedSARA-CS data source: {source}")
    print(f"Repository destination: {destination}")

    cifar10cs = source / "cifar_10_cs"
    if not cifar10cs.is_dir():
        cifar10cs = find_root(source, Path("metadata.json"), root_parents=2)

    cifar100 = source / "cifar_100"
    if not cifar100.is_dir():
        try:
            cifar100 = find_root(source, Path("cifar-100-python/train"), root_parents=2)
        except FileNotFoundError:
            cifar100 = find_root(source, Path("cifar-100-python.tar.gz"), root_parents=1)

    copy_dir(cifar10cs, destination / "RAHFL-master/Dataset/cifar_10_cs")
    copy_dir(cifar100, destination / "RAHFL-master/Dataset/cifar_100")

    required = [
        destination / "RAHFL-master/Dataset/cifar_10_cs",
    ]
    missing = [path for path in required if not safe_exists(path)]
    if missing:
        raise FileNotFoundError("Import incomplete:\n" + "\n".join(str(path) for path in missing))
    public_train = destination / "RAHFL-master/Dataset/cifar_100/cifar-100-python/train"
    public_tar = destination / "RAHFL-master/Dataset/cifar_100/cifar-100-python.tar.gz"
    if not safe_exists(public_train) and not safe_exists(public_tar):
        raise FileNotFoundError(f"Missing CIFAR-100 public data: {public_train} or {public_tar}")
    print("FedSARA-CS prepared-data import verified successfully.")


if __name__ == "__main__":
    main()
