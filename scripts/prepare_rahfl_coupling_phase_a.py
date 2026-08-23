from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedprime.data.corruption_label_coupling import prepare_coupling_artifacts
from fedprime.utils.env import add_vendor_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_exact_metadata(
    *,
    data_root: Path,
    clean_root: Path,
    metadata_root: Path,
    seed: int,
) -> None:
    metadata_files = [
        metadata_root / "train" / "corruption_type.npy",
        metadata_root / "train" / "corruption_severity.npy",
        metadata_root / "train" / "corruption_mask.npy",
        metadata_root / "train" / "corruption_manifest.json",
    ]
    if not all(path.is_file() for path in metadata_files):
        add_vendor_paths()
        from Dataset.make_cifar_c import replay_corruption_metadata

        dataset_dir = PROJECT_ROOT / "RAHFL-master" / "Dataset"
        previous = Path.cwd()
        try:
            os.chdir(dataset_dir)
            replay_corruption_metadata(
                dataset_name="cifar10",
                dataset_root=str(clean_root.resolve()),
                output_root=str(metadata_root.resolve()),
                train=True,
                corrupt_rate=1,
                seed=int(seed),
            )
        finally:
            os.chdir(previous)

    original = data_root / "train" / "random_corrupt_1.npy"
    original_digest = _sha256(original)
    print(f"[prepare] frozen corruption-image checksum {original_digest}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the frozen RAHFL coupling Phase-A artifacts.")
    parser.add_argument(
        "--data-root",
        default="RAHFL-master/Dataset/cifar_10_c",
    )
    parser.add_argument(
        "--clean-root",
        default="RAHFL-master/Dataset/cifar_10",
    )
    parser.add_argument(
        "--metadata-root",
        default="local_runs/rahfl_coupling_metadata",
    )
    parser.add_argument(
        "--artifact-root",
        default="local_runs/rahfl_coupling_phase_a_seed0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--audit-ratio", type=float, default=0.10)
    parser.add_argument("--noise-rate", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    clean_root = (PROJECT_ROOT / args.clean_root).resolve()
    metadata_root = (PROJECT_ROOT / args.metadata_root).resolve()
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    _ensure_exact_metadata(
        data_root=data_root,
        clean_root=clean_root,
        metadata_root=metadata_root,
        seed=args.seed,
    )
    manifest = prepare_coupling_artifacts(
        data_root=data_root,
        metadata_root=metadata_root,
        artifact_root=artifact_root,
        num_clients=4,
        samples_per_client=args.samples_per_client,
        audit_ratio=args.audit_ratio,
        noise_rate=args.noise_rate,
        betas=(0.0, 4.0),
        partition_seed=args.seed,
        split_seed=args.seed,
        noise_seed=args.seed,
    )
    print(json.dumps(manifest["regimes"], ensure_ascii=False, indent=2), flush=True)
    print(f"[prepare] artifacts ready: {artifact_root}", flush=True)


if __name__ == "__main__":
    main()
