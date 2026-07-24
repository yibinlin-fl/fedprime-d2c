from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a mounted CLE-HFL prepared dataset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("."))
    parser.add_argument("--repo-root", dest="destination", type=Path)
    return parser.parse_args()


def maybe_extract_archive(source: Path, destination: Path) -> Path:
    if source.is_file() and source.suffixes[-2:] == [".tar", ".gz"]:
        archive_name = source.name[:-7]
        extract_root = destination / "local_runs" / "imported_cle_hfl" / archive_name
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {source} -> {extract_root}")
        with tarfile.open(source, "r:gz") as tar:
            tar.extractall(extract_root)
        nested_root = extract_root / archive_name
        if (
            (nested_root / "cifar_10_cle").is_dir()
            or (nested_root / "cifar_10_cle_v2").is_dir()
        ):
            return nested_root
        if (
            (extract_root / "cifar_10_cle").is_dir()
            or (extract_root / "cifar_10_cle_v2").is_dir()
        ):
            return extract_root
        return extract_root
    return source


def find_root(search_root: Path, marker: Path, root_parents: int) -> Path:
    candidates = [search_root]
    for candidate in candidates:
        matches = list(candidate.rglob(str(marker))) if candidate.exists() else []
        if matches:
            root = matches[0]
            for _ in range(root_parents):
                root = root.parent
            print(f"Located {marker}: {root}")
            return root
    raise FileNotFoundError(f"Could not find marker {marker} under {search_root}")


def find_package_root(search_root: Path) -> Path:
    if (
        (search_root / "cifar_10_cle").is_dir()
        or (search_root / "cifar_10_cle_v2").is_dir()
    ):
        return search_root
    matches = sorted(search_root.rglob("metadata.json")) if search_root.exists() else []
    if not matches:
        raise FileNotFoundError(f"Could not find metadata.json under {search_root}")
    root = matches[0].parent
    while root != search_root.parent:
        if (
            (root / "cifar_10_cle").is_dir()
            or (root / "cifar_10_cle_v2").is_dir()
        ):
            print(f"Located CLE-HFL package root: {root}")
            return root
        root = root.parent
    return find_root(search_root, Path("metadata.json"), root_parents=2).parent


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
    print(f"CLE-HFL data source: {source}")
    print(f"Repository destination: {destination}")
    package_root = find_package_root(source)
    print(f"CLE-HFL package root: {package_root}")

    copied_private_roots = []
    for directory_name in ("cifar_10_cle", "cifar_10_cle_v2"):
        cle = package_root / directory_name
        if cle.is_dir():
            target = destination / "RAHFL-master/Dataset" / directory_name
            copy_dir(cle, target)
            copied_private_roots.append(target)
    if not copied_private_roots:
        raise FileNotFoundError(
            f"No cifar_10_cle or cifar_10_cle_v2 directory under {package_root}"
        )

    cifar100 = package_root / "cifar_100"
    if cifar100.is_dir():
        copy_dir(cifar100, destination / "RAHFL-master/Dataset/cifar_100")

    for required in copied_private_roots:
        if not safe_exists(required):
            raise FileNotFoundError(f"Import incomplete: {required}")
    print("CLE-HFL prepared-data import verified successfully.")


if __name__ == "__main__":
    main()
