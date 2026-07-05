from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import load_private_labels, partition_private_data
from fedprime.utils.env import seed_everything


def alpha_tag(alpha: float) -> str:
    value = int(round(float(alpha) * 10))
    return f"alpha{value:02d}"


def partition_name(alpha: float, seed: int, clients: int, samples: int | None) -> str:
    sample_text = "all" if samples is None else str(samples)
    return f"cifar10c_{alpha_tag(alpha)}_seed{seed}_clients{clients}_samples{sample_text}.npz"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(
    labels: np.ndarray,
    mapping: dict[int, list[int]],
    out_dir: Path,
    alpha: float,
    seed: int,
    num_classes: int,
) -> None:
    counts = np.zeros((len(mapping), num_classes), dtype=np.int64)
    for client_id, indices in mapping.items():
        client_labels = labels[np.asarray(indices, dtype=np.int64)]
        counts[client_id] = np.bincount(client_labels.astype(np.int64), minlength=num_classes)

    totals = counts.sum(axis=1, keepdims=True)
    proportions = counts / np.maximum(totals, 1)

    fields = ["client", "total"] + [f"class_{i}" for i in range(num_classes)]
    count_rows = []
    prop_rows = []
    for client_id in range(len(mapping)):
        count_row = {"client": client_id, "total": int(counts[client_id].sum())}
        prop_row = {"client": client_id, "total": int(counts[client_id].sum())}
        for class_id in range(num_classes):
            count_row[f"class_{class_id}"] = int(counts[client_id, class_id])
            prop_row[f"class_{class_id}"] = f"{proportions[client_id, class_id]:.6f}"
        count_rows.append(count_row)
        prop_rows.append(prop_row)

    write_csv(out_dir / "client_class_counts.csv", count_rows, fields)
    write_csv(out_dir / "client_class_proportions.csv", prop_rows, fields)

    summary = {
        "partition": "dirichlet",
        "dirichlet_alpha": float(alpha),
        "seed": int(seed),
        "num_clients": len(mapping),
        "num_classes": int(num_classes),
        "client_totals": counts.sum(axis=1).astype(int).tolist(),
        "class_totals": counts.sum(axis=0).astype(int).tolist(),
        "max_client_class_proportion": proportions.max(axis=1).round(6).tolist(),
        "nonzero_classes_per_client": (counts > 0).sum(axis=1).astype(int).tolist(),
    }
    (out_dir / "partition_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_archive(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in source_dir.rglob("*"):
            tar.add(path, arcname=path.relative_to(source_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small Kaggle dataset with fixed Non-IID partition files.")
    parser.add_argument("--private-root", default="RAHFL-master/Dataset/cifar_10_c")
    parser.add_argument("--corrupt-rate", type=float, default=1.0)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.3, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--out-dir", default="local_runs/sara_partitions_alpha03_alpha10")
    parser.add_argument("--archive", default="local_runs/sara_partitions_alpha03_alpha10.tar.gz")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    partitions_dir = out_dir / "outputs" / "partitions"
    audit_root = out_dir / "outputs" / "partition_audit"
    partitions_dir.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    private_root = Path(args.private_root)
    labels_path = private_root / "train" / "labels.npy"
    if labels_path.exists():
        labels = np.load(labels_path)
    else:
        labels = load_private_labels(private_root, args.corrupt_rate)

    manifest = {
        "description": "Fixed CIFAR-10-C Dirichlet partition files for SARA experiments.",
        "private_root_used": str(private_root),
        "corrupt_rate": float(args.corrupt_rate),
        "alphas": [float(x) for x in args.alphas],
        "seeds": [int(x) for x in args.seeds],
        "num_clients": int(args.num_clients),
        "num_classes": int(args.num_classes),
        "samples_per_client": int(args.samples_per_client),
        "files": [],
    }

    for alpha in args.alphas:
        for seed in args.seeds:
            seed_everything(seed)
            file_name = partition_name(alpha, seed, args.num_clients, args.samples_per_client)
            target_path = partitions_dir / file_name
            mapping = partition_private_data(
                labels=labels,
                num_clients=args.num_clients,
                num_classes=args.num_classes,
                partition="dirichlet",
                dirichlet_alpha=float(alpha),
                max_samples_per_client=args.samples_per_client,
                partition_indices_path=target_path,
            )
            audit_dir = audit_root / f"cifar10c_{alpha_tag(alpha)}_seed{seed}_clients{args.num_clients}"
            write_audit(
                labels=labels,
                mapping=mapping,
                out_dir=audit_dir,
                alpha=float(alpha),
                seed=int(seed),
                num_classes=args.num_classes,
            )
            manifest["files"].append(str(Path("outputs/partitions") / file_name))
            print(f"Wrote {target_path}")

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# SARA Partition Pack",
                "",
                "Suggested Kaggle dataset name: `sara-partitions-alpha03-alpha10`.",
                "",
                "Mount this dataset together with `fedprime-data`, then import these partition files into",
                "`/kaggle/working/fedprime-d2c/outputs/partitions` before running alpha=0.3/1.0 experiments.",
                "",
                "Contained partition files:",
                *[f"- `{name}`" for name in manifest["files"]],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.no_archive:
        archive_path = Path(args.archive)
        if archive_path.exists():
            archive_path.unlink()
        make_archive(out_dir, archive_path)
        print(f"Archive: {archive_path}")

    print(f"Partition pack directory: {out_dir}")


if __name__ == "__main__":
    main()
