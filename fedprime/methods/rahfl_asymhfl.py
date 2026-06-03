from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from fedprime.augmentations.prime_adapter import build_prime_module
from fedprime.data.loaders import (
    build_augmix_private_loaders,
    build_prime_dcl_private_loaders,
    build_private_loaders,
    build_public_loader,
    dataset_stats,
    load_private_labels,
    normalize_batch,
    partition_private_data,
)
from fedprime.methods.local_prime import train_local_prime_epoch
from fedprime.methods.local_prime import train_local_prime_dcl_epoch
from fedprime.methods.local_rahfl import train_local_augmix_dcl_epoch
from fedprime.models.factory import build_models, forward_logits
from fedprime.utils.config import save_config
from fedprime.utils.env import resolve_device, seed_everything


class AsymHFLExperiment:
    """Unified runner for original RAHFL-style AsymHFL and RAHFL+PRIME."""

    def __init__(self, config: dict):
        self.config = config
        self.device = resolve_device(config.get("device", "auto"))
        self.output_dir = Path(config["output_root"]) / config["experiment_name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.output_dir / "config.resolved.json")
        seed_everything(int(config.get("seed", 0)))

    def run(self) -> None:
        data_cfg = self.config["data"]
        train_cfg = self.config["train"]
        method_cfg = self.config["method"]
        model_cfg = self.config["models"]

        num_clients = len(model_cfg["names"])
        num_classes = int(data_cfg.get("num_classes", 10))
        stats = dataset_stats(data_cfg.get("private_dataset", "cifar10"))

        labels = load_private_labels(data_cfg["private_root"], data_cfg["private_corrupt_rate"])
        dataidx_map = partition_private_data(
            labels=labels,
            num_clients=num_clients,
            num_classes=num_classes,
            partition=data_cfg.get("partition", "dirichlet"),
            dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
            max_samples_per_client=data_cfg.get("private_samples_per_client"),
        )

        use_prime = bool(method_cfg.get("use_prime", False))
        use_prime_dcl = use_prime and bool(method_cfg.get("use_dcl", True))
        if use_prime:
            if use_prime_dcl:
                private_loaders, test_loader = build_prime_dcl_private_loaders(
                    cifar10c_root=data_cfg["private_root"],
                    dataidx_map=dataidx_map,
                    train_batch_size=train_cfg["batch_size"],
                    test_batch_size=train_cfg.get("test_batch_size", 512),
                    corrupt_rate=data_cfg["private_corrupt_rate"],
                    test_corrupt_rate=data_cfg["test_corrupt_rate"],
                    num_workers=int(self.config.get("num_workers", 2)),
                )
            else:
                private_loaders, test_loader = build_private_loaders(
                    cifar10c_root=data_cfg["private_root"],
                    dataidx_map=dataidx_map,
                    train_batch_size=train_cfg["batch_size"],
                    test_batch_size=train_cfg.get("test_batch_size", 512),
                    corrupt_rate=data_cfg["private_corrupt_rate"],
                    test_corrupt_rate=data_cfg["test_corrupt_rate"],
                    num_workers=int(self.config.get("num_workers", 2)),
                    raw_for_prime=True,
                )
            prime_aug = build_prime_module(stats, method_cfg.get("prime", {})).to(self.device)
        else:
            private_loaders, test_loader, _, _ = build_augmix_private_loaders(
                cifar10c_root=data_cfg["private_root"],
                dataidx_map=dataidx_map,
                train_batch_size=train_cfg["batch_size"],
                test_batch_size=train_cfg.get("test_batch_size", 512),
                corrupt_rate=data_cfg["private_corrupt_rate"],
                test_corrupt_rate=data_cfg["test_corrupt_rate"],
                num_workers=int(self.config.get("num_workers", 2)),
                augmix_module=method_cfg.get("augmix_module", "jsd"),
            )
            prime_aug = None

        public_loader = build_public_loader(
            cifar100_root=data_cfg["public_root"],
            public_size=int(data_cfg.get("public_size", 5000)),
            batch_size=train_cfg["public_batch_size"],
            num_workers=int(self.config.get("num_workers", 2)),
            seed=int(self.config.get("seed", 0)),
            download=bool(data_cfg.get("download_public", False)),
        )
        public_iter = iter(public_loader)

        models = build_models(model_cfg["names"], num_classes)
        models = {idx: model.to(self.device) for idx, model in models.items()}
        self._load_models_if_configured(models)
        optimizers = {idx: self._build_optimizer(model) for idx, model in models.items()}

        metrics_path = self.output_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["round", "avg_acc", "worst_acc", "local_loss", "col_loss"],
            )
            writer.writeheader()

            for round_idx in range(int(train_cfg["rounds"])):
                accs_before = self._evaluate(models, test_loader)
                col_loss = self._collaborative_phase(
                    models=models,
                    optimizers=optimizers,
                    public_loader=public_loader,
                    public_iter=public_iter,
                    accs=accs_before,
                    stats=stats,
                )
                local_loss = self._local_phase(
                    models=models,
                    optimizers=optimizers,
                    private_loaders=private_loaders,
                    prime_aug=prime_aug,
                    use_prime=use_prime,
                    use_prime_dcl=use_prime_dcl,
                    train_cfg=train_cfg,
                    method_cfg=method_cfg,
                    stats=stats,
                )
                accs = self._evaluate(models, test_loader)
                row = {
                    "round": round_idx,
                    "avg_acc": sum(accs) / len(accs),
                    "worst_acc": min(accs),
                    "local_loss": local_loss,
                    "col_loss": col_loss,
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"[round {round_idx:03d}] "
                    f"avg_acc={row['avg_acc']:.2f} "
                    f"worst_acc={row['worst_acc']:.2f} "
                    f"local_loss={local_loss:.4f} "
                    f"col_loss={col_loss:.4f}"
                )

        self._save_models(models)

    def _build_optimizer(self, model):
        opt_cfg = self.config["train"].get("optimizer", {})
        name = opt_cfg.get("name", "adam").lower()
        lr = float(opt_cfg.get("lr", 1e-3))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        if name == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=float(opt_cfg.get("momentum", 0.9)),
                weight_decay=weight_decay,
            )
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _collaborative_phase(self, models, optimizers, public_loader, public_iter, accs, stats) -> float:
        losses = []
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        num_batches = int(self.config["train"].get("public_batches_per_round", 1))
        for _ in range(num_batches):
            try:
                images, _ = next(public_iter)
            except StopIteration:
                public_iter = iter(public_loader)
                images, _ = next(public_iter)
            images = images.to(self.device, non_blocking=True)
            images = normalize_batch(images, stats)

            target_probs = {}
            student_log_probs = {}
            for client_id in sorted(models):
                models[client_id].eval()
                with torch.no_grad():
                    logits = forward_logits(models[client_id], images)
                    target_probs[client_id] = F.softmax(logits, dim=1).detach()
                models[client_id].train()
                logits = forward_logits(models[client_id], images)
                student_log_probs[client_id] = F.log_softmax(logits, dim=1)

            for client_id in sorted(models):
                learn_losses = []
                for other_id in sorted(models):
                    if other_id == client_id:
                        continue
                    if accs[client_id] <= accs[other_id]:
                        learn_losses.append(criterion(student_log_probs[client_id], target_probs[other_id]))
                if not learn_losses:
                    continue
                loss = sum(learn_losses) / len(learn_losses)
                optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))
        return sum(losses) / max(len(losses), 1)

    def _local_phase(self, models, optimizers, private_loaders, prime_aug, use_prime, use_prime_dcl, train_cfg, method_cfg, stats) -> float:
        losses = []
        for client_id, loader in enumerate(private_loaders):
            for _ in range(int(train_cfg.get("local_epochs", 1))):
                if use_prime:
                    if use_prime_dcl:
                        loss = train_local_prime_dcl_epoch(
                            model=models[client_id],
                            loader=loader,
                            optimizer=optimizers[client_id],
                            prime_aug=prime_aug,
                            normalizer=lambda x: normalize_batch(x, stats),
                            device=self.device,
                            lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                            cl_module=method_cfg.get("cl_module", "dcl"),
                            max_batches=train_cfg.get("max_local_batches"),
                        )
                    else:
                        loss = train_local_prime_epoch(
                            model=models[client_id],
                            loader=loader,
                            optimizer=optimizers[client_id],
                            prime_aug=prime_aug,
                            device=self.device,
                            lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                            max_batches=train_cfg.get("max_local_batches"),
                        )
                else:
                    loss = train_local_augmix_dcl_epoch(
                        model=models[client_id],
                        loader=loader,
                        optimizer=optimizers[client_id],
                        device=self.device,
                        lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                        cl_module=method_cfg.get("cl_module", "dcl"),
                        max_batches=train_cfg.get("max_local_batches"),
                    )
                losses.append(loss)
        return sum(losses) / max(len(losses), 1)

    def _evaluate(self, models, test_loader) -> list[float]:
        accs = []
        for client_id in sorted(models):
            model = models[client_id]
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    logits = forward_logits(model, images)
                    pred = logits.argmax(dim=1)
                    total += labels.numel()
                    correct += (pred == labels).sum().item()
            accs.append(100.0 * correct / max(total, 1))
        return accs

    def _save_models(self, models) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for client_id, model in models.items():
            torch.save(model.state_dict(), ckpt_dir / f"client_{client_id}.pt")

    def _load_models_if_configured(self, models) -> None:
        ckpt_cfg = self.config.get("checkpoints", {})
        load_dir = ckpt_cfg.get("resume_dir") if ckpt_cfg.get("resume", False) else ckpt_cfg.get("load_dir")
        if not load_dir:
            return
        load_dir = Path(load_dir)
        for client_id, model in models.items():
            path = load_dir / f"client_{client_id}.pt"
            if not path.exists():
                continue
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            cleaned = {
                (key[7:] if key.startswith("module.") else key): value
                for key, value in state.items()
            }
            model.load_state_dict(cleaned, strict=bool(ckpt_cfg.get("strict", True)))
