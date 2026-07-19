from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import partition_private_data  # noqa: E402
from scripts.prepare_corruption_skew_data import load_cifar10_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a disjoint unlabeled CIFAR-10 public pool for CLE-HFL."
    )
    parser.add_argument("--cifar10-root", type=Path, default=Path("RAHFL-master/Dataset/cifar_10"))
    parser.add_argument(
        "--cle-private-root",
        type=Path,
        default=Path("RAHFL-master/Dataset/cifar_10_cle/alpha05_gamma09_seed0"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "local_runs/cle_hfl_indomain_public/cle_hfl_indomain_public_alpha05_gamma09_seed0"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--public-size", type=int, default=5000)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--make-tar", action="store_true")
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(memoryview(contiguous))
    return digest.hexdigest()


def verify_private_mapping(
    train_labels: np.ndarray,
    mapping: dict[int, list[int]],
    cle_private_root: Path,
) -> None:
    seen: set[int] = set()
    for client_id, indices in sorted(mapping.items()):
        overlap = seen.intersection(indices)
        if overlap:
            raise ValueError(f"Private partition overlap detected for client {client_id}.")
        seen.update(indices)
        observed_path = cle_private_root / f"client_{client_id}" / "train_labels.npy"
        if not observed_path.is_file():
            raise FileNotFoundError(f"Missing CLE private labels: {observed_path}")
        observed = np.load(observed_path).astype(np.int64)
        expected = train_labels[np.asarray(indices, dtype=np.int64)].astype(np.int64)
        if not np.array_equal(observed, expected):
            raise ValueError(
                f"Reconstructed partition does not match CLE client {client_id}; "
                "refuse to create a potentially unfair public split."
            )


def make_tarball(output_root: Path) -> Path:
    tar_path = output_root.parent / f"{output_root.name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_root, arcname=output_root.name)
    return tar_path


def main() -> None:
    args = parse_args()
    if args.public_size <= 0:
        raise ValueError("public-size must be positive.")

    train_images, train_labels, _, _ = load_cifar10_arrays(args.cifar10_root, args.download)
    mapping = partition_private_data(
        labels=train_labels,
        num_clients=args.num_clients,
        num_classes=args.num_classes,
        partition="dirichlet",
        dirichlet_alpha=args.alpha,
        max_samples_per_client=args.samples_per_client,
        partition_seed=args.seed,
    )
    verify_private_mapping(train_labels, mapping, args.cle_private_root)

    private_indices = np.asarray(
        sorted(index for indices in mapping.values() for index in indices),
        dtype=np.int64,
    )
    all_indices = np.arange(len(train_labels), dtype=np.int64)
    unused_indices = np.setdiff1d(all_indices, private_indices, assume_unique=True)
    if len(unused_indices) < args.public_size:
        raise ValueError(
            f"Only {len(unused_indices)} CIFAR-10 train samples remain after private partition; "
            f"cannot allocate public-size={args.public_size}."
        )

    rng = np.random.default_rng(args.seed + 2_026_0711)
    shuffled_unused = rng.permutation(unused_indices)
    public_indices = np.sort(shuffled_unused[: args.public_size]).astype(np.int64)
    reserved_indices = np.sort(shuffled_unused[args.public_size :]).astype(np.int64)
    if np.intersect1d(private_indices, public_indices).size:
        raise AssertionError("Private/public split overlap detected.")

    public_images = train_images[public_indices].astype(np.uint8)
    public_histogram = np.bincount(train_labels[public_indices].astype(np.int64), minlength=args.num_classes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    np.save(args.output_root / "public_images.npy", public_images)
    np.save(args.output_root / "public_indices.npy", public_indices)
    np.save(args.output_root / "reserved_indices.npy", reserved_indices)

    metadata = {
        "dataset": "cifar10_unlabeled_indomain_public",
        "alpha": args.alpha,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "samples_per_client": args.samples_per_client,
        "private_size": int(private_indices.size),
        "public_size": int(public_indices.size),
        "reserved_size": int(reserved_indices.size),
        "public_label_histogram_audit_only": public_histogram.astype(int).tolist(),
        "training_loader_reads_labels": False,
        "private_indices_sha256": sha256_array(private_indices),
        "public_indices_sha256": sha256_array(public_indices),
        "public_images_sha256": sha256_array(public_images),
        "split_rule": "Reconstruct CLE private Dirichlet indices, then sample only from their complement.",
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Verified CLE private partition: {private_indices.size} unique samples", flush=True)
    print(f"Wrote unlabeled public pool: {args.output_root}", flush=True)
    print(f"Public/reserved sizes: {public_indices.size}/{reserved_indices.size}", flush=True)
    print(f"Public label histogram (audit only): {public_histogram.tolist()}", flush=True)
    if args.make_tar:
        print(f"Wrote tarball: {make_tarball(args.output_root)}", flush=True)


if __name__ == "__main__":
    main()
