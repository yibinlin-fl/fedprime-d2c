from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import CIFAR10_MEAN, CIFAR10_STD
from fedprime.models.factory import build_models, forward_logits
from fedprime.utils.config import load_config
from fedprime.utils.env import resolve_device


CORRUPTION_GROUPS = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


class NpyCifarDataset(Dataset):
    def __init__(self, images_path: Path, labels_path: Path):
        self.images = np.load(images_path).astype(np.uint8)
        self.labels = np.load(labels_path).astype(np.int64)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __len__(self) -> int:
        return min(len(self.images), len(self.labels))

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def find_corruption_file(root: Path, name: str) -> Path | None:
    candidates = [
        root / f"{name}.npy",
        root / "test" / f"{name}.npy",
        root / f"{name.replace('_', ' ').title()}.npy",
        root / "test" / f"{name.replace('_', ' ').title()}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_labels(root: Path) -> Path:
    candidates = [root / "labels.npy", root / "test" / "labels.npy"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find labels.npy under {root}")


def load_checkpoint(model, ckpt_dir: Path, client_id: int, device: torch.device) -> bool:
    path = ckpt_dir / f"client_{client_id}.pt"
    if not path.exists():
        return False
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return True


def evaluate_model(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            pred = forward_logits(model, images).argmax(dim=1)
            total += labels.numel()
            correct += (pred == labels).sum().item()
    return 100.0 * correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on CIFAR-C corruption groups.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--corruption_root", help="Root containing CIFAR-C npy files.")
    parser.add_argument("--out_csv", default="corruption_eval.csv")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    root = Path(args.corruption_root or cfg["data"]["private_root"])
    labels_path = find_labels(root)
    models = build_models(cfg["models"]["names"], int(cfg["data"].get("num_classes", 10)))
    ckpt_dir = Path(args.checkpoint_dir)
    models = {idx: model.to(device) for idx, model in models.items()}
    for client_id, model in models.items():
        if not load_checkpoint(model, ckpt_dir, client_id, device):
            raise FileNotFoundError(f"Missing checkpoint for client {client_id} in {ckpt_dir}")

    rows = []
    for group, corruptions in CORRUPTION_GROUPS.items():
        group_accs = []
        for corruption in corruptions:
            file_path = find_corruption_file(root, corruption)
            if file_path is None:
                continue
            loader = DataLoader(
                NpyCifarDataset(file_path, labels_path),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            client_accs = [evaluate_model(model, loader, device) for model in models.values()]
            avg_acc = sum(client_accs) / len(client_accs)
            group_accs.append(avg_acc)
            rows.append({
                "group": group,
                "corruption": corruption,
                "avg_acc": f"{avg_acc:.4f}",
                "client_accs": ";".join(f"{acc:.4f}" for acc in client_accs),
            })
        if group_accs:
            rows.append({
                "group": group,
                "corruption": "__group_avg__",
                "avg_acc": f"{sum(group_accs) / len(group_accs):.4f}",
                "client_accs": "",
            })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "corruption", "avg_acc", "client_accs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()

