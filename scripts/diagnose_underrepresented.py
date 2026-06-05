from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import (  # noqa: E402
    build_private_loaders,
    load_private_labels,
    partition_private_data,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402
from fedprime.utils.config import load_config  # noqa: E402
from fedprime.utils.env import resolve_device, seed_everything  # noqa: E402


def load_checkpoint(model, ckpt_dir: Path, client_id: int, device: torch.device) -> None:
    path = ckpt_dir / f"client_{client_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)


def evaluate_class_counts(model, loader, num_classes: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            pred = forward_logits(model, images).argmax(dim=1)
            for class_id in range(num_classes):
                mask = labels == class_id
                count = int(mask.sum().item())
                if count == 0:
                    continue
                total[class_id] += count
                correct[class_id] += int((pred[mask] == labels[mask]).sum().item())
    return correct, total


def safe_acc(correct: np.ndarray, total: np.ndarray, mask: np.ndarray) -> float:
    denom = int(total[mask].sum())
    if denom == 0:
        return float("nan")
    return 100.0 * float(correct[mask].sum()) / denom


def class_group_masks(
    counts: np.ndarray,
    tail_quantile: float,
    missing_threshold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = counts <= missing_threshold
    observed = ~missing
    if observed.any():
        threshold = float(np.quantile(counts[observed], tail_quantile))
        tail = observed & (counts <= threshold)
    else:
        tail = np.zeros_like(counts, dtype=bool)
    head = ~(missing | tail)
    return head, tail, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose client accuracy on head/tail/missing private-label classes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--out_csv")
    parser.add_argument("--tail_quantile", type=float, default=0.3)
    parser.add_argument("--missing_threshold", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)))
    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config["models"]
    num_clients = len(model_cfg["names"])
    num_classes = int(data_cfg.get("num_classes", 10))
    device = resolve_device(config.get("device", "auto"))

    labels = load_private_labels(data_cfg["private_root"], data_cfg["private_corrupt_rate"])
    dataidx_map = partition_private_data(
        labels=labels,
        num_clients=num_clients,
        num_classes=num_classes,
        partition=data_cfg.get("partition", "dirichlet"),
        dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
        max_samples_per_client=data_cfg.get("private_samples_per_client"),
        partition_indices_path=data_cfg.get("partition_indices_path"),
    )
    _, test_loader = build_private_loaders(
        cifar10c_root=data_cfg["private_root"],
        dataidx_map=dataidx_map,
        train_batch_size=train_cfg["batch_size"],
        test_batch_size=train_cfg.get("test_batch_size", 512),
        corrupt_rate=data_cfg["private_corrupt_rate"],
        test_corrupt_rate=data_cfg["test_corrupt_rate"],
        num_workers=int(config.get("num_workers", 2)),
        raw_for_prime=True,
    )

    models = build_models(model_cfg["names"], num_classes)
    ckpt_dir = Path(args.checkpoint_dir)
    rows = []
    for client_id, model in models.items():
        model = model.to(device)
        load_checkpoint(model, ckpt_dir, client_id, device)
        correct, total = evaluate_class_counts(model, test_loader, num_classes, device)

        client_indices = np.asarray(dataidx_map[client_id], dtype=np.int64)
        private_counts = np.bincount(labels[client_indices].astype(np.int64), minlength=num_classes)
        head, tail, missing = class_group_masks(
            private_counts,
            tail_quantile=float(args.tail_quantile),
            missing_threshold=int(args.missing_threshold),
        )
        rows.append({
            "client": client_id,
            "overall_acc": f"{safe_acc(correct, total, np.ones(num_classes, dtype=bool)):.4f}",
            "head_acc": f"{safe_acc(correct, total, head):.4f}",
            "tail_acc": f"{safe_acc(correct, total, tail):.4f}",
            "missing_acc": f"{safe_acc(correct, total, missing):.4f}",
            "head_classes": ";".join(str(i) for i in np.where(head)[0]),
            "tail_classes": ";".join(str(i) for i in np.where(tail)[0]),
            "missing_classes": ";".join(str(i) for i in np.where(missing)[0]),
            "private_class_counts": ";".join(str(int(v)) for v in private_counts),
            "test_class_correct": ";".join(str(int(v)) for v in correct),
            "test_class_total": ";".join(str(int(v)) for v in total),
        })

    out_csv = Path(args.out_csv) if args.out_csv else ckpt_dir.parent / "underrepresented_accuracy.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote underrepresented-class diagnosis to {out_csv}")


if __name__ == "__main__":
    main()
