from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.augmentations.prime_adapter import build_prime_module
from fedprime.data.loaders import (
    build_private_loaders,
    dataset_stats,
    load_private_labels,
    partition_private_data,
)
from fedprime.methods.local_prime import train_local_prime_epoch
from fedprime.models.factory import build_models
from fedprime.utils.config import load_config
from fedprime.utils.env import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only PRIME local training and stop at the first NaN/Inf."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--client", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--no-grad-clip", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)))
    device = resolve_device(config.get("device", "auto"))
    data_cfg = config["data"]
    train_cfg = config["train"]
    method_cfg = config["method"]
    model_names = config["models"]["names"]
    stats = dataset_stats(data_cfg.get("private_dataset", "cifar10"))
    max_grad_norm = (
        None
        if args.no_grad_clip
        else args.max_grad_norm
        if args.max_grad_norm is not None
        else train_cfg.get("max_grad_norm")
    )

    labels = load_private_labels(data_cfg["private_root"], data_cfg["private_corrupt_rate"])
    mapping = partition_private_data(
        labels=labels,
        num_clients=len(model_names),
        num_classes=int(data_cfg.get("num_classes", 10)),
        partition=data_cfg.get("partition", "dirichlet"),
        dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
        max_samples_per_client=data_cfg.get("private_samples_per_client"),
        partition_indices_path=data_cfg.get("partition_indices_path"),
    )
    loaders, _ = build_private_loaders(
        cifar10c_root=data_cfg["private_root"],
        dataidx_map=mapping,
        train_batch_size=int(train_cfg["batch_size"]),
        test_batch_size=int(train_cfg.get("test_batch_size", 256)),
        corrupt_rate=data_cfg["private_corrupt_rate"],
        test_corrupt_rate=data_cfg["test_corrupt_rate"],
        num_workers=args.num_workers,
        raw_for_prime=True,
    )
    models = build_models(model_names, int(data_cfg.get("num_classes", 10)))
    prime_aug = build_prime_module(stats, method_cfg.get("prime", {})).to(device)

    clients = range(len(model_names)) if args.client is None else [args.client]
    print(
        f"PRIME stability diagnostic: device={device}, batches/client={args.batches}, "
        f"max_grad_norm={max_grad_norm}"
    )
    for client_id in clients:
        model = models[client_id].to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg.get("optimizer", {}).get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("optimizer", {}).get("weight_decay", 0.0)),
        )
        loss = train_local_prime_epoch(
            model=model,
            loader=loaders[client_id],
            optimizer=optimizer,
            prime_aug=prime_aug,
            device=device,
            lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
            max_batches=args.batches,
            max_grad_norm=max_grad_norm,
            context=f"client={client_id}, model={model_names[client_id]}",
        )
        print(f"PASS client={client_id}, model={model_names[client_id]}, mean_loss={loss:.6f}")


if __name__ == "__main__":
    main()
