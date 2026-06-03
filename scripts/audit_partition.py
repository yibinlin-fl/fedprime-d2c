from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import load_private_labels, partition_private_data
from fedprime.utils.config import load_config
from fedprime.utils.env import seed_everything


def load_labels(config: dict) -> np.ndarray:
    private_root = Path(config["data"]["private_root"])
    labels_path = private_root / "train" / "labels.npy"
    if labels_path.exists():
        return np.load(labels_path)
    return load_private_labels(private_root, config["data"]["private_corrupt_rate"])


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_heatmap(counts: np.ndarray, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, counts.shape[1] * 0.7), max(3, counts.shape[0] * 0.5)))
    im = ax.imshow(counts, aspect="auto", cmap="viridis")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_xticks(np.arange(counts.shape[1]))
    ax.set_yticks(np.arange(counts.shape[0]))
    ax.set_title("Client Class Counts")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Dirichlet client label skew.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="outputs/partition_audit")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)))

    labels = load_labels(config)
    num_clients = len(config["models"]["names"])
    num_classes = int(config["data"].get("num_classes", 10))
    dataidx_map = partition_private_data(
        labels=labels,
        num_clients=num_clients,
        num_classes=num_classes,
        partition=config["data"].get("partition", "dirichlet"),
        dirichlet_alpha=float(config["data"].get("dirichlet_alpha", 0.5)),
        max_samples_per_client=config["data"].get("private_samples_per_client"),
    )

    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for client_id, indices in dataidx_map.items():
        client_labels = labels[np.asarray(indices, dtype=np.int64)]
        counts[client_id] = np.bincount(client_labels.astype(np.int64), minlength=num_classes)

    totals = counts.sum(axis=1, keepdims=True)
    proportions = counts / np.maximum(totals, 1)

    out_dir = Path(args.out_dir) / config["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    count_rows = []
    prop_rows = []
    for client_id in range(num_clients):
        count_row = {"client": client_id, "total": int(counts[client_id].sum())}
        prop_row = {"client": client_id, "total": int(counts[client_id].sum())}
        for class_id in range(num_classes):
            count_row[f"class_{class_id}"] = int(counts[client_id, class_id])
            prop_row[f"class_{class_id}"] = f"{proportions[client_id, class_id]:.6f}"
        count_rows.append(count_row)
        prop_rows.append(prop_row)

    fields = ["client", "total"] + [f"class_{i}" for i in range(num_classes)]
    write_csv(out_dir / "client_class_counts.csv", count_rows, fields)
    write_csv(out_dir / "client_class_proportions.csv", prop_rows, fields)
    maybe_write_heatmap(counts, out_dir / "client_class_counts.png")

    summary = {
        "experiment_name": config["experiment_name"],
        "partition": config["data"].get("partition"),
        "dirichlet_alpha": config["data"].get("dirichlet_alpha"),
        "num_clients": num_clients,
        "num_classes": num_classes,
        "client_totals": counts.sum(axis=1).astype(int).tolist(),
        "class_totals": counts.sum(axis=0).astype(int).tolist(),
        "max_client_class_proportion": proportions.max(axis=1).round(6).tolist(),
        "nonzero_classes_per_client": (counts > 0).sum(axis=1).astype(int).tolist(),
    }
    (out_dir / "partition_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote partition audit to {out_dir}")
    for row in count_rows:
        print(row)


if __name__ == "__main__":
    main()

