from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import build_public_loader  # noqa: E402
from fedprime.models.public_canonicalizer import PublicNuisanceCanonicalizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase-B0 public nuisance canonicalizer.")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-size", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_x = F.avg_pool2d(prediction, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(prediction.square(), 3, 1, 1) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), 3, 1, 1) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, 3, 1, 1) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1.0e-8)
    return 1.0 - score.mean()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    if args.smoke and args.max_batches <= 0:
        args.max_batches = 2
    if args.epochs < 1 or args.batch_size < 1 or args.public_size < args.batch_size:
        raise ValueError("invalid public canonicalizer training schedule")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = resolve_device(args.device)

    loader = build_public_loader(
        args.public_root.resolve(),
        public_size=int(args.public_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        download=False,
        public_dataset="cifar100",
        public_views="augmix",
    )
    model = PublicNuisanceCanonicalizer(
        base_channels=int(args.base_channels),
        residual_scale=float(args.residual_scale),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    history: list[dict[str, float | int]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        total_l1 = total_ssim = total_loss = 0.0
        batches = 0
        for views, _labels in loader:
            clean, aug1, aug2 = views
            clean = clean.to(device=device, dtype=torch.float32)
            degraded = torch.cat(
                [
                    aug1.to(device=device, dtype=torch.float32),
                    aug2.to(device=device, dtype=torch.float32),
                ],
                dim=0,
            )
            target = clean.repeat(2, 1, 1, 1)
            restored = model(degraded)
            l1 = F.l1_loss(restored, target)
            ssim = _ssim_loss(restored, target)
            loss = l1 + float(args.ssim_weight) * ssim
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_l1 += float(l1.detach())
            total_ssim += float(ssim.detach())
            total_loss += float(loss.detach())
            batches += 1
            if args.max_batches > 0 and batches >= int(args.max_batches):
                break
        if batches == 0:
            raise RuntimeError("public canonicalizer loader yielded no batches")
        row = {
            "epoch": epoch,
            "batches": batches,
            "l1": total_l1 / batches,
            "ssim_loss": total_ssim / batches,
            "loss": total_loss / batches,
        }
        history.append(row)
        print(f"[epoch {epoch:03d}] {json.dumps(row)}", flush=True)

    checkpoint = output_dir / "public_nuisance_canonicalizer.pt"
    torch.save(
        {
            "protocol": "cle_public_canonicalizer_phase_b0",
            "state_dict": model.to("cpu").state_dict(),
            "model_config": model.config(),
            "train_config": {
                "public_size": int(args.public_size),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "ssim_weight": float(args.ssim_weight),
                "seed": int(args.seed),
                "max_batches": int(args.max_batches),
                "smoke": bool(args.smoke),
                "public_labels_used": False,
                "public_views": "rahfl_augmix",
            },
            "history": history,
        },
        checkpoint,
    )
    summary = {
        "protocol": "cle_public_canonicalizer_phase_b0",
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(device),
        "smoke": bool(args.smoke),
        "history": history,
    }
    summary_path = output_dir / "public_nuisance_canonicalizer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[complete] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
