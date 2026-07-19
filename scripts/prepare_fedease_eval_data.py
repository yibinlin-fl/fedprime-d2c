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

from fedprime.data.corruptions import (  # noqa: E402
    CORRUPTION_GROUPS,
    GROUP_TO_ID,
    apply_corruption,
    sample_corruption_from_group,
)
from scripts.prepare_corruption_skew_data import load_cifar10_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add FedEASE clean/same/swapped/unseen evaluation splits.")
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("local_runs/cle_hfl_prepared/cle_hfl_prepared_alpha05_gamma09_seed0"),
    )
    parser.add_argument("--dataset-name", default="alpha05_gamma09_seed0")
    parser.add_argument("--cifar10-root", type=Path, default=Path("RAHFL-master/Dataset/cifar_10"))
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--severity-min", type=int, default=1)
    parser.add_argument("--severity-max", type=int, default=5)
    parser.add_argument("--max-test-images", type=int, default=0)
    parser.add_argument("--make-tar", action="store_true")
    return parser.parse_args()


def write_split(directory: Path, images: np.ndarray, labels: np.ndarray, environment_ids: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "test_images.npy", np.asarray(images, dtype=np.uint8))
    np.save(directory / "test_labels.npy", np.asarray(labels, dtype=np.uint8))
    np.save(directory / "test_corruption_ids.npy", np.asarray(environment_ids, dtype=np.int16))


def corrupt_by_mapping(
    images: np.ndarray,
    labels: np.ndarray,
    mapping: dict[int, str],
    rng: np.random.Generator,
    severity_min: int,
    severity_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = []
    environment_ids = []
    for index, (image, label) in enumerate(zip(images, labels)):
        group = mapping[int(label)]
        severity = int(rng.integers(severity_min, severity_max + 1))
        output.append(apply_corruption(image, sample_corruption_from_group(group, rng), severity, rng))
        environment_ids.append(GROUP_TO_ID[group])
        if (index + 1) % 2500 == 0:
            print(f"[heartbeat] generated {index + 1}/{len(images)} mapped evaluation images", flush=True)
    return np.asarray(output, dtype=np.uint8), np.asarray(environment_ids, dtype=np.int16)


def corrupt_unseen_compositions(
    images: np.ndarray,
    rng: np.random.Generator,
    severity_min: int,
    severity_max: int,
) -> np.ndarray:
    groups = list(CORRUPTION_GROUPS)
    output = []
    for index, image in enumerate(images):
        first = groups[index % len(groups)]
        second = groups[(index + 1) % len(groups)]
        severity = int(rng.integers(severity_min, severity_max + 1))
        value = apply_corruption(image, sample_corruption_from_group(first, rng), severity, rng)
        value = apply_corruption(value, sample_corruption_from_group(second, rng), severity, rng)
        output.append(value)
        if (index + 1) % 2500 == 0:
            print(f"[heartbeat] generated {index + 1}/{len(images)} unseen-composition images", flush=True)
    return np.asarray(output, dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(root: Path, output: Path) -> None:
    with tarfile.open(output, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    print(f"Wrote {output}", flush=True)


def main() -> None:
    args = parse_args()
    cle_root = args.prepared_root / "cifar_10_cle" / args.dataset_name
    metadata_path = cle_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_map = {
        int(client): {int(class_id): group for class_id, group in mapping.items()}
        for client, mapping in metadata["class_corruption_map"].items()
    }

    _, _, test_images, test_labels = load_cifar10_arrays(args.cifar10_root, download=False)
    if 0 < args.max_test_images < len(test_images):
        selected = np.random.default_rng(args.seed).choice(
            len(test_images), size=args.max_test_images, replace=False
        )
        test_images = test_images[selected]
        test_labels = test_labels[selected]

    print("[setup] writing clean evaluation split", flush=True)
    write_split(
        cle_root / "test_clean",
        test_images,
        test_labels,
        np.full(len(test_labels), -1, dtype=np.int16),
    )
    groups = list(CORRUPTION_GROUPS)
    for client_id in range(args.num_clients):
        same_mapping = class_map[client_id]
        swapped_mapping = {
            class_id: groups[(groups.index(group) + 1) % len(groups)]
            for class_id, group in same_mapping.items()
        }
        same_images, same_ids = corrupt_by_mapping(
            test_images,
            test_labels,
            same_mapping,
            np.random.default_rng(args.seed + client_id * 10_007 + 1),
            args.severity_min,
            args.severity_max,
        )
        write_split(cle_root / "test_same" / f"client_{client_id}", same_images, test_labels, same_ids)
        swapped_images, swapped_ids = corrupt_by_mapping(
            test_images,
            test_labels,
            swapped_mapping,
            np.random.default_rng(args.seed + client_id * 10_007 + 2),
            args.severity_min,
            args.severity_max,
        )
        write_split(
            cle_root / "test_swapped" / f"client_{client_id}",
            swapped_images,
            test_labels,
            swapped_ids,
        )

    unseen_images = corrupt_unseen_compositions(
        test_images,
        np.random.default_rng(args.seed + 97_003),
        args.severity_min,
        args.severity_max,
    )
    write_split(
        cle_root / "test_unseen",
        unseen_images,
        test_labels,
        np.full(len(test_labels), -1, dtype=np.int16),
    )

    files = sorted(
        path for split in ("test_clean", "test_same", "test_balanced", "test_swapped", "test_unseen")
        for path in (cle_root / split).rglob("*.npy")
        if path.is_file()
    )
    audit = {
        "protocol": "FedEASE CLE-HFL evaluation v1",
        "seed": args.seed,
        "splits": ["clean", "same", "random", "swapped", "unseen"],
        "ers_definition": "same_avg_acc - swapped_avg_acc",
        "unseen_definition": "composition of two corruption groups; training uses one group per sample",
        "sha256": {str(path.relative_to(cle_root)): sha256(path) for path in files},
    }
    audit_path = cle_root / "fedease_evaluation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {audit_path}", flush=True)

    if args.make_tar:
        output = args.prepared_root.parent / f"fedease_cle_prepared_{args.dataset_name}.tar.gz"
        package(args.prepared_root, output)


if __name__ == "__main__":
    main()
