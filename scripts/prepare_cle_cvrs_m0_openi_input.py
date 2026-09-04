from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INPUT_DIRECTORY = "cle_cvrs_m0_seed0_inputs"
SOURCE_ROOT = (
    ROOT
    / "local_runs/cle_shortcut_amplification_phase_a1a/cle_shortcut_amplification_phase_a1a_seed0"
)
CHECKPOINT_ROOT = (
    ROOT
    / "local_runs/cle_public_canonicalization_phase_b0/cle_public_canonicalization_phase_b0_seed0_inputs/checkpoints/h9"
)
OUTPUT_PARENT = ROOT / "local_runs/cle_cvrs_m0_openi_input"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the compact CVRS M0 OpenI input archive")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def copy_file(source: Path, destination: Path, rows: list[dict[str, object]], input_root: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    rows.append(
        {
            "path": destination.relative_to(input_root).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output_parent = args.output_parent.resolve()
    input_root = output_parent / INPUT_DIRECTORY
    archive = output_parent / f"{INPUT_DIRECTORY}.tar.gz"
    if input_root.exists() or archive.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing M0 input artifact: {input_root} or {archive}"
        )
    input_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []

    private_source = source_root / "data/gamma09"
    for client_id in (0, 3):
        for name in ("train_images.npy", "train_labels.npy"):
            copy_file(
                private_source / f"client_{client_id}" / name,
                input_root / "private" / f"client_{client_id}" / name,
                rows,
                input_root,
            )
    copy_file(
        source_root / "splits/strict_cle_v1_alpha05_gamma_pair_seed0_split0.npz",
        input_root / "splits/strict_cle_v1_alpha05_gamma_pair_seed0_split0.npz",
        rows,
        input_root,
    )
    for name in ("test_images.npy", "test_labels.npy"):
        copy_file(source_root / "evaluation" / name, input_root / "evaluation" / name, rows, input_root)
    for client_id in (0, 3):
        copy_file(
            checkpoint_root / f"client_{client_id}.pt",
            input_root / "checkpoints" / f"client_{client_id}.pt",
            rows,
            input_root,
        )

    from fedprime.data.loaders import cifar100_train_images_from_tar
    from scripts.run_cle_cvrs_m0 import K0B_PUBLIC_HASH, K0B_PUBLIC_SEED, sha256_array

    cifar100 = cifar100_train_images_from_tar(source_root / "public")
    indices = np.random.default_rng(K0B_PUBLIC_SEED).choice(
        cifar100.shape[0], size=1000, replace=False
    ).astype(np.int64)
    if sha256_array(indices) != K0B_PUBLIC_HASH:
        raise ValueError("K0-B public selection mismatch while packaging")
    public_images_path = input_root / "public/k0b_public_images.npy"
    public_indices_path = input_root / "public/k0b_public_indices.npy"
    public_images_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(public_images_path, np.asarray(cifar100[indices], dtype=np.uint8), allow_pickle=False)
    np.save(public_indices_path, indices, allow_pickle=False)
    for path in (public_images_path, public_indices_path):
        rows.append(
            {
                "path": path.relative_to(input_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "protocol": "cle_cvrs_m0_openi_input_v1",
        "input_directory": INPUT_DIRECTORY,
        "contains_only_h9_clients": [0, 3],
        "private_files": "images_and_labels_only",
        "private_corruption_metadata_included": False,
        "public_labels_included": False,
        "k0b_public_indices_sha256": K0B_PUBLIC_HASH,
        "files": sorted(rows, key=lambda row: str(row["path"])),
    }
    manifest_path = input_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    output_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz", compresslevel=6) as handle:
        handle.add(input_root, arcname=INPUT_DIRECTORY)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
                "files": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
