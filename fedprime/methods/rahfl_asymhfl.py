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
from fedprime.methods.nir_dcl import NIRDCLFeatureQueue
from fedprime.models.factory import build_models, forward_logits
from fedprime.utils.config import save_config
from fedprime.utils.env import resolve_device, seed_everything


class AsymHFLExperiment:
    """Unified runner for RAHFL-style AsymHFL, RAHFL+PRIME, and FedCARA."""

    def __init__(self, config: dict):
        self.config = config
        self.device = resolve_device(config.get("device", "auto"))
        self.output_dir = Path(config["output_root"]) / config["experiment_name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.output_dir / "config.resolved.json")
        seed_everything(int(config.get("seed", 0)))
        self._nir_dcl_queues: dict[int, NIRDCLFeatureQueue] = {}

    def run(self) -> None:
        data_cfg = self.config["data"]
        train_cfg = self.config["train"]
        method_cfg = self.config["method"]
        model_cfg = self.config["models"]

        num_clients = len(model_cfg["names"])
        num_classes = int(data_cfg.get("num_classes", 10))
        stats = dataset_stats(data_cfg.get("private_dataset", "cifar10"))

        print("[setup] AsymHFL/FedCARA loading private labels", flush=True)
        labels = load_private_labels(data_cfg["private_root"], data_cfg["private_corrupt_rate"])
        print("[setup] AsymHFL/FedCARA loading/creating private partition", flush=True)
        dataidx_map = partition_private_data(
            labels=labels,
            num_clients=num_clients,
            num_classes=num_classes,
            partition=data_cfg.get("partition", "dirichlet"),
            dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
            max_samples_per_client=data_cfg.get("private_samples_per_client"),
            partition_indices_path=data_cfg.get("partition_indices_path"),
        )
        self._client_class_counts = self._build_client_class_counts(labels, dataidx_map, num_classes)

        use_prime = bool(method_cfg.get("use_prime", False))
        use_prime_dcl = use_prime and bool(method_cfg.get("use_dcl", True))
        if use_prime:
            print("[setup] building PRIME private loaders", flush=True)
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
            print("[setup] building AugMix private loaders", flush=True)
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

        print("[setup] building public loader", flush=True)
        public_loader = build_public_loader(
            cifar100_root=data_cfg["public_root"],
            public_size=int(data_cfg.get("public_size", 5000)),
            batch_size=train_cfg["public_batch_size"],
            num_workers=int(self.config.get("num_workers", 2)),
            seed=int(self.config.get("seed", 0)),
            download=bool(data_cfg.get("download_public", False)),
        )
        public_iter = iter(public_loader)

        print("[setup] building heterogeneous client models", flush=True)
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
                print(f"[heartbeat] round {round_idx:03d} start", flush=True)
                if self._use_cara_communication(method_cfg):
                    accs_before, class_accs_before = self._evaluate_detailed(models, test_loader, num_classes)
                else:
                    accs_before = self._evaluate(models, test_loader)
                    class_accs_before = None
                print(f"[heartbeat] round {round_idx:03d} collaborative phase", flush=True)
                col_loss = self._collaborative_phase(
                    models=models,
                    optimizers=optimizers,
                    public_loader=public_loader,
                    public_iter=public_iter,
                    accs=accs_before,
                    class_accs=class_accs_before,
                    stats=stats,
                )
                print(f"[heartbeat] round {round_idx:03d} local phase", flush=True)
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
                    num_classes=num_classes,
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
                    f"col_loss={col_loss:.4f}",
                    flush=True,
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

    def _use_cara_communication(self, method_cfg: dict) -> bool:
        return method_cfg.get("communication", "asymhfl").lower() in {"cara", "cara_c", "fedcara"}

    def _collaborative_phase(self, models, optimizers, public_loader, public_iter, accs, class_accs, stats) -> float:
        losses = []
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        method_cfg = self.config.get("method", {})
        use_cara = self._use_cara_communication(method_cfg)
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
                    if use_cara:
                        if class_accs is None:
                            raise RuntimeError("CARA-C communication requires class-wise accuracies.")
                        class_weights = self._cara_class_weights(
                            student_class_acc=class_accs[client_id],
                            teacher_class_acc=class_accs[other_id],
                            method_cfg=method_cfg,
                        )
                        if class_weights is None:
                            continue
                        learn_losses.append(self._weighted_kd_loss(
                            student_log_probs=student_log_probs[client_id],
                            teacher_probs=target_probs[other_id],
                            class_weights=class_weights,
                        ))
                    elif accs[client_id] <= accs[other_id]:
                        learn_losses.append(criterion(student_log_probs[client_id], target_probs[other_id]))
                if not learn_losses:
                    continue
                loss = sum(learn_losses) / len(learn_losses)
                optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))
        return sum(losses) / max(len(losses), 1)

    def _cara_class_weights(
        self,
        student_class_acc: torch.Tensor,
        teacher_class_acc: torch.Tensor,
        method_cfg: dict,
    ) -> torch.Tensor | None:
        cara_cfg = method_cfg.get("cara", {})
        student = student_class_acc.to(self.device).float().clamp(0.0, 1.0)
        teacher = teacher_class_acc.to(self.device).float().clamp(0.0, 1.0)
        margin = float(cara_cfg.get("better_margin", 0.0))
        teacher_power = float(cara_cfg.get("teacher_power", 1.0))
        need_power = float(cara_cfg.get("need_power", 1.0))
        min_weight = float(cara_cfg.get("min_weight", 1e-6))

        teacher_reliability = teacher.clamp_min(0.0).pow(teacher_power)
        receiver_need = (1.0 - student).clamp_min(0.0).pow(need_power)
        weights = teacher_reliability * receiver_need
        if bool(cara_cfg.get("better_only", True)):
            weights = weights * (teacher > student + margin).float()
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if float(weights.sum().detach().cpu()) <= min_weight:
            return None
        if bool(cara_cfg.get("normalize", True)):
            active = weights > min_weight
            weights = weights / weights[active].mean().clamp_min(min_weight)
        return weights.detach()

    def _weighted_kd_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        safe_teacher = teacher_probs.clamp_min(1e-8)
        per_class_kl = safe_teacher * (safe_teacher.log() - student_log_probs)
        return (per_class_kl * class_weights.view(1, -1)).sum(dim=1).mean()

    def _get_nir_dcl_queue(self, client_id: int, num_classes: int, method_cfg: dict) -> NIRDCLFeatureQueue:
        if client_id not in self._nir_dcl_queues:
            nir_cfg = method_cfg.get("cara_l", method_cfg.get("nir_dcl", {}))
            self._nir_dcl_queues[client_id] = NIRDCLFeatureQueue(
                num_classes=num_classes,
                max_size_per_class=int(nir_cfg.get("queue_size", 64)),
            )
        return self._nir_dcl_queues[client_id]

    def _build_client_class_counts(self, labels, dataidx_map, num_classes: int) -> dict[int, torch.Tensor]:
        label_tensor = torch.as_tensor(labels, dtype=torch.long)
        counts = {}
        for client_id, indices in dataidx_map.items():
            idx_tensor = torch.as_tensor(indices, dtype=torch.long)
            client_labels = label_tensor[idx_tensor]
            counts[int(client_id)] = torch.bincount(client_labels, minlength=num_classes).float()
        return counts

    def _local_phase(self, models, optimizers, private_loaders, prime_aug, use_prime, use_prime_dcl, train_cfg, method_cfg, stats, num_classes: int) -> float:
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
                    feature_queue = None
                    if method_cfg.get("cl_module", "dcl") in {"nir_dcl", "cara_l"}:
                        feature_queue = self._get_nir_dcl_queue(client_id, num_classes, method_cfg)
                    loss = train_local_augmix_dcl_epoch(
                        model=models[client_id],
                        loader=loader,
                        optimizer=optimizers[client_id],
                        device=self.device,
                        lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                        cl_module=method_cfg.get("cl_module", "dcl"),
                        num_classes=num_classes,
                        nir_dcl_cfg=method_cfg.get("cara_l", method_cfg.get("nir_dcl", {})),
                        sara_cfg=method_cfg.get("sara", {}),
                        feature_queue=feature_queue,
                        client_class_counts=getattr(self, "_client_class_counts", {}).get(client_id),
                        max_batches=train_cfg.get("max_local_batches"),
                        max_grad_norm=train_cfg.get("max_grad_norm"),
                        skip_nonfinite=bool(train_cfg.get("skip_nonfinite", False)),
                        log_interval=train_cfg.get("local_log_interval"),
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

    def _evaluate_detailed(self, models, test_loader, num_classes: int) -> tuple[list[float], dict[int, torch.Tensor]]:
        accs = []
        class_accs = {}
        for client_id in sorted(models):
            model = models[client_id]
            model.eval()
            correct = 0
            total = 0
            class_correct = torch.zeros(num_classes, dtype=torch.float64)
            class_total = torch.zeros(num_classes, dtype=torch.float64)
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    logits = forward_logits(model, images)
                    pred = logits.argmax(dim=1)
                    total += labels.numel()
                    correct_mask = pred == labels
                    correct += correct_mask.sum().item()
                    for class_id in range(num_classes):
                        mask = labels == class_id
                        if mask.any():
                            class_total[class_id] += mask.sum().item()
                            class_correct[class_id] += correct_mask[mask].sum().item()
            accs.append(100.0 * correct / max(total, 1))
            class_accs[client_id] = (class_correct / class_total.clamp_min(1.0)).float()
        return accs, class_accs

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
