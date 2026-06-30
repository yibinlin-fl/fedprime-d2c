from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import torch.optim as optim

from fedprime.data.loaders import (
    build_augmix_private_loaders,
    build_public_loader,
    dataset_stats,
    load_private_labels,
    normalize_batch,
    partition_private_data,
)
from fedprime.methods.local_rahfl import train_local_augmix_dcl_epoch
from fedprime.methods.nir_dcl import NIRDCLFeatureQueue
from fedprime.models.factory import build_models, forward_logits
from fedprime.utils.config import save_config
from fedprime.utils.env import resolve_device, seed_everything


def _linear_head(model):
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "linear"):
        raise AttributeError("PRAC-HFL head-only updates require models with a `.linear` classifier head.")
    return module.linear


def _clone_params(params: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def _restore_params(params: Iterable[torch.nn.Parameter], state: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, state):
            param.copy_(value)


class PRACHFLExperiment:
    """PRAC-HFL: RAHFL local training with receiver-adaptive safe logit routing."""

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

        print("[setup] PRAC-HFL loading private labels", flush=True)
        labels = load_private_labels(data_cfg["private_root"], data_cfg["private_corrupt_rate"])
        print("[setup] PRAC-HFL loading/creating private partition", flush=True)
        dataidx_map = partition_private_data(
            labels=labels,
            num_clients=num_clients,
            num_classes=num_classes,
            partition=data_cfg.get("partition", "dirichlet"),
            dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
            max_samples_per_client=data_cfg.get("private_samples_per_client"),
            partition_indices_path=data_cfg.get("partition_indices_path"),
        )

        print("[setup] PRAC-HFL building RAHFL AugMix/DCL private loaders", flush=True)
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
        warmup_rounds = int(method_cfg.get("prac", {}).get("warmup_rounds", 0))
        skip_prac_all_rounds = warmup_rounds >= int(train_cfg["rounds"])
        if skip_prac_all_rounds:
            print("[setup] PRAC-HFL local-only mode: skip public loader", flush=True)
            public_loader = None
        else:
            print("[setup] PRAC-HFL building public loader", flush=True)
            public_loader = build_public_loader(
                cifar100_root=data_cfg["public_root"],
                public_size=int(data_cfg.get("public_size", 5000)),
                batch_size=train_cfg["public_batch_size"],
                num_workers=int(self.config.get("num_workers", 2)),
                seed=int(self.config.get("seed", 0)),
                download=bool(data_cfg.get("download_public", False)),
            )

        print("[setup] PRAC-HFL building heterogeneous client models", flush=True)
        models = build_models(model_cfg["names"], num_classes)
        models = {idx: model.to(self.device) for idx, model in models.items()}
        self._load_models_if_configured(models)
        optimizers = {idx: self._build_optimizer(model) for idx, model in models.items()}

        metrics_path = self.output_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "avg_acc",
                    "worst_acc",
                    "local_loss",
                    "prac_loss",
                    "accept_rate",
                    "positive_teacher_ratio",
                    "avg_delta",
                    "avg_rho",
                ],
            )
            writer.writeheader()

            public_iter = iter(public_loader) if public_loader is not None else None
            private_iters = [iter(loader) for loader in private_loaders] if not skip_prac_all_rounds else None
            print("[setup] PRAC-HFL setup complete; entering training rounds", flush=True)
            for round_idx in range(int(train_cfg["rounds"])):
                round_start = time.perf_counter()
                print(f"[heartbeat] round {round_idx:03d} start", flush=True)

                local_loss = self._local_phase(
                    round_idx=round_idx,
                    models=models,
                    optimizers=optimizers,
                    private_loaders=private_loaders,
                    train_cfg=train_cfg,
                    method_cfg=method_cfg,
                    num_classes=num_classes,
                )

                print(f"[heartbeat] round {round_idx:03d} running PRAC communication", flush=True)
                warmup_rounds = int(method_cfg.get("prac", {}).get("warmup_rounds", 0))
                if round_idx < warmup_rounds:
                    print(
                        f"[heartbeat] round {round_idx:03d} PRAC warmup: skip communication",
                        flush=True,
                    )
                    prac_stats = self._average_prac_stats([])
                else:
                    if public_loader is None or public_iter is None:
                        raise RuntimeError("PRAC communication requires a public loader.")
                    prac_stats = []
                    for public_batch_idx in range(int(train_cfg.get("public_batches_per_round", 1))):
                        public_images, public_iter = self._next_public_batch(public_loader, public_iter, stats)
                        if private_iters is None:
                            raise RuntimeError("PRAC communication requires private route iterators.")
                        prac_stats.append(self._prac_phase(
                            round_idx=round_idx,
                            public_batch_idx=public_batch_idx,
                            models=models,
                            public_images=public_images,
                            private_loaders=private_loaders,
                            private_iters=private_iters,
                            num_classes=num_classes,
                            method_cfg=method_cfg,
                        ))
                    prac_stats = self._average_prac_stats(prac_stats)

                print(f"[heartbeat] round {round_idx:03d} evaluating clients", flush=True)
                accs = self._evaluate(models, test_loader)
                row = {
                    "round": round_idx,
                    "avg_acc": sum(accs) / len(accs),
                    "worst_acc": min(accs),
                    "local_loss": local_loss,
                    **prac_stats,
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"[round {round_idx:03d}] "
                    f"avg_acc={row['avg_acc']:.2f} "
                    f"worst_acc={row['worst_acc']:.2f} "
                    f"local_loss={local_loss:.4f} "
                    f"prac_loss={row['prac_loss']:.4f} "
                    f"accept_rate={row['accept_rate']:.2f} "
                    f"pos_teacher={row['positive_teacher_ratio']:.2f} "
                    f"avg_delta={row['avg_delta']:.4f} "
                    f"elapsed={time.perf_counter() - round_start:.1f}s",
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

    def _get_nir_dcl_queue(self, client_id: int, num_classes: int, method_cfg: dict) -> NIRDCLFeatureQueue:
        if client_id not in self._nir_dcl_queues:
            nir_cfg = method_cfg.get("nir_dcl", {})
            self._nir_dcl_queues[client_id] = NIRDCLFeatureQueue(
                num_classes=num_classes,
                max_size_per_class=int(nir_cfg.get("queue_size", 64)),
            )
        return self._nir_dcl_queues[client_id]

    def _local_phase(self, round_idx, models, optimizers, private_loaders, train_cfg, method_cfg, num_classes: int) -> float:
        losses = []
        for client_id, loader in enumerate(private_loaders):
            client_start = time.perf_counter()
            print(f"[heartbeat] round {round_idx:03d} local client {client_id} start", flush=True)
            feature_queue = None
            if method_cfg.get("cl_module", "dcl") == "nir_dcl":
                feature_queue = self._get_nir_dcl_queue(client_id, num_classes, method_cfg)
            for _ in range(int(train_cfg.get("local_epochs", 1))):
                loss = train_local_augmix_dcl_epoch(
                    model=models[client_id],
                    loader=loader,
                    optimizer=optimizers[client_id],
                    device=self.device,
                    lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                    cl_module=method_cfg.get("cl_module", "dcl"),
                    num_classes=num_classes,
                    nir_dcl_cfg=method_cfg.get("nir_dcl", {}),
                    feature_queue=feature_queue,
                    max_batches=train_cfg.get("max_local_batches"),
                    max_grad_norm=train_cfg.get("max_grad_norm"),
                    skip_nonfinite=bool(train_cfg.get("skip_nonfinite", False)),
                    log_interval=train_cfg.get("local_log_interval"),
                    context=f"PRAC-HFL local phase, round={round_idx}, client={client_id}",
                )
                losses.append(loss)
            print(
                f"[heartbeat] round {round_idx:03d} local client {client_id} done "
                f"loss={loss:.4f} elapsed={time.perf_counter() - client_start:.1f}s",
                flush=True,
            )
        local_loss = sum(losses) / max(len(losses), 1)
        if not torch.isfinite(torch.tensor(local_loss)):
            raise FloatingPointError(f"PRAC-HFL local phase produced non-finite loss: {local_loss}")
        return local_loss

    def _next_public_batch(self, public_loader, public_iter, stats):
        try:
            images, _ = next(public_iter)
        except StopIteration:
            public_iter = iter(public_loader)
            images, _ = next(public_iter)
        images = images.to(self.device, non_blocking=True)
        return normalize_batch(images, stats), public_iter

    def _next_private_batch(self, client_id, private_loaders, private_iters):
        try:
            batch = next(private_iters[client_id])
        except StopIteration:
            private_iters[client_id] = iter(private_loaders[client_id])
            batch = next(private_iters[client_id])
        return batch

    def _candidate_teachers(self, client_id: int, num_clients: int, max_teachers: int, round_idx: int) -> list[int]:
        candidates = [idx for idx in range(num_clients) if idx != client_id]
        if max_teachers >= len(candidates):
            return candidates
        seed = int(self.config.get("seed", 0)) + round_idx * 1009 + client_id * 9173
        return sorted(random.Random(seed).sample(candidates, max_teachers))

    def _prac_phase(
        self,
        round_idx: int,
        public_batch_idx: int,
        models,
        public_images: torch.Tensor,
        private_loaders,
        private_iters,
        num_classes: int,
        method_cfg: dict,
    ) -> dict[str, float]:
        prac_cfg = method_cfg.get("prac", {})
        num_clients = len(models)
        max_teachers = int(prac_cfg.get("num_candidate_teachers", num_clients - 1))
        temperature = float(prac_cfg.get("temperature", 3.0))
        virtual_lr = float(prac_cfg.get("virtual_lr", 0.05))
        head_max_grad_norm = prac_cfg.get("head_max_grad_norm")
        margin = float(prac_cfg.get("positive_margin", 0.0))
        gate_tau = float(prac_cfg.get("gate_tau", 0.05))
        accept_eps = float(prac_cfg.get("accept_epsilon", 0.0))
        use_classwise = bool(prac_cfg.get("classwise", True))

        with torch.no_grad():
            teacher_probs = {}
            own_probs = {}
            for client_id, model in models.items():
                model.eval()
                logits = forward_logits(model, public_images)
                probs = F.softmax(logits / temperature, dim=1).detach()
                teacher_probs[client_id] = probs
                own_probs[client_id] = probs

        accept_count = 0
        total_clients = 0
        positive_counts = []
        delta_values = []
        rho_values = []
        prac_losses = []

        for client_id, model in models.items():
            total_clients += 1
            route_batch = self._next_private_batch(client_id, private_loaders, private_iters)
            accept_batch = self._next_private_batch(client_id, private_loaders, private_iters)
            teachers = self._candidate_teachers(client_id, num_clients, max_teachers, round_idx)
            base_route = self._robust_risk_by_class(model, route_batch, num_classes, method_cfg)

            deltas = []
            class_deltas = []
            head = _linear_head(model)
            head_params = list(head.parameters())
            saved_head = _clone_params(head_params)

            for teacher_id in teachers:
                kd_loss = self._kd_loss(model, public_images, teacher_probs[teacher_id], temperature)
                step_ok = self._apply_head_step(
                    kd_loss,
                    head_params,
                    virtual_lr,
                    max_grad_norm=head_max_grad_norm,
                )
                if step_ok:
                    virt_route = self._robust_risk_by_class(model, route_batch, num_classes, method_cfg)
                else:
                    virt_route = {
                        "overall": base_route["overall"] + 1.0,
                        "per_class": [value + 1.0 for value in base_route["per_class"]],
                    }
                _restore_params(head_params, saved_head)

                delta = base_route["overall"] - virt_route["overall"]
                deltas.append(delta)
                delta_values.append(delta)
                if use_classwise:
                    class_delta = []
                    for class_id in range(num_classes):
                        if base_route["counts"][class_id] > 0:
                            class_delta.append(base_route["per_class"][class_id] - virt_route["per_class"][class_id])
                        else:
                            class_delta.append(delta)
                    class_deltas.append(class_delta)
                else:
                    class_deltas.append([delta] * num_classes)

            scores = torch.tensor(class_deltas, device=self.device, dtype=torch.float32)
            scores = torch.clamp(scores - margin, min=0.0)
            positive_counts.append(float((scores > 0).float().mean().detach().cpu()))
            score_sum = scores.sum(dim=0)
            rho = 1.0 - torch.exp(-score_sum / max(gate_tau, 1e-8))
            rho = torch.nan_to_num(rho, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            rho_values.append(float(rho.mean().detach().cpu()))

            if float(score_sum.max().detach().cpu()) <= 0.0:
                print(
                    f"[heartbeat] round {round_idx:03d} PRAC client {client_id} reject: no positive teacher",
                    flush=True,
                )
                continue

            beta = scores / (score_sum.unsqueeze(0) + 1e-12)
            mixed = (1.0 - rho).unsqueeze(0) * own_probs[client_id]
            for local_idx, teacher_id in enumerate(teachers):
                mixed = mixed + (rho * beta[local_idx]).unsqueeze(0) * teacher_probs[teacher_id]
            mixed = mixed.clamp_min(1e-8)
            mixed = mixed / mixed.sum(dim=1, keepdim=True)

            accept_before = self._robust_risk_by_class(model, accept_batch, num_classes, method_cfg)["overall"]
            saved_head = _clone_params(head_params)
            mix_loss = self._kd_loss(model, public_images, mixed.detach(), temperature)
            step_ok = self._apply_head_step(
                mix_loss,
                head_params,
                virtual_lr,
                max_grad_norm=head_max_grad_norm,
            )
            if step_ok:
                accept_after = self._robust_risk_by_class(model, accept_batch, num_classes, method_cfg)["overall"]
            else:
                accept_after = accept_before + 1.0

            if torch.isfinite(torch.tensor(accept_after)) and accept_after <= accept_before - accept_eps:
                accept_count += 1
                prac_losses.append(float(mix_loss.detach().cpu()))
                print(
                    f"[heartbeat] round {round_idx:03d} PRAC client {client_id} accept "
                    f"risk={accept_before:.4f}->{accept_after:.4f} "
                    f"rho={float(rho.mean().detach().cpu()):.3f}",
                    flush=True,
                )
            else:
                _restore_params(head_params, saved_head)
                print(
                    f"[heartbeat] round {round_idx:03d} PRAC client {client_id} reject "
                    f"risk={accept_before:.4f}->{accept_after:.4f} "
                    f"rho={float(rho.mean().detach().cpu()):.3f}",
                    flush=True,
                )

        return {
            "prac_loss": sum(prac_losses) / max(len(prac_losses), 1),
            "accept_rate": accept_count / max(total_clients, 1),
            "positive_teacher_ratio": sum(positive_counts) / max(len(positive_counts), 1),
            "avg_delta": sum(delta_values) / max(len(delta_values), 1),
            "avg_rho": sum(rho_values) / max(len(rho_values), 1),
        }

    def _average_prac_stats(self, stats: list[dict[str, float]]) -> dict[str, float]:
        if not stats:
            return {
                "prac_loss": 0.0,
                "accept_rate": 0.0,
                "positive_teacher_ratio": 0.0,
                "avg_delta": 0.0,
                "avg_rho": 0.0,
            }
        return {
            key: sum(item[key] for item in stats) / len(stats)
            for key in stats[0]
        }

    def _kd_loss(self, model, public_images: torch.Tensor, target_probs: torch.Tensor, temperature: float) -> torch.Tensor:
        model.train()
        logits = forward_logits(model, public_images)
        return (temperature**2) * F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            target_probs,
            reduction="batchmean",
        )

    def _apply_head_step(
        self,
        loss: torch.Tensor,
        params: list[torch.nn.Parameter],
        lr: float,
        max_grad_norm: float | None = None,
    ) -> bool:
        if not torch.isfinite(loss):
            return False
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
        finite_grads = [grad for grad in grads if grad is not None]
        if any(not torch.isfinite(grad).all() for grad in finite_grads):
            return False
        if max_grad_norm is not None and finite_grads:
            total_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(grad.detach(), 2) for grad in finite_grads]),
                2,
            )
            clip_coef = float(max_grad_norm) / (float(total_norm.detach().cpu()) + 1e-12)
            if clip_coef < 1.0:
                finite_grads = [grad * clip_coef if grad is not None else None for grad in grads]
            else:
                finite_grads = list(grads)
        else:
            finite_grads = list(grads)
        with torch.no_grad():
            for param, grad in zip(params, finite_grads):
                if grad is not None:
                    param.add_(grad, alpha=-lr)
                    if not torch.isfinite(param).all():
                        return False
        return True

    def _robust_risk_by_class(self, model, batch, num_classes: int, method_cfg: dict) -> dict[str, object]:
        model.eval()
        lambda_aug = float(method_cfg.get("prac", {}).get("risk_lambda_aug", 1.0))
        lambda_js = float(method_cfg.get("prac", {}).get("risk_lambda_js", method_cfg.get("lambda_jsd", 12.0)))
        with torch.no_grad():
            images, labels = batch
            labels = labels.to(self.device, non_blocking=True).long()
            if isinstance(images, (tuple, list)):
                views = [img.to(self.device, non_blocking=True) for img in images]
                clean = views[0]
                aug_views = views[1:3] if len(views) >= 3 else views[1:]
                logits_clean = forward_logits(model, clean)
                loss_vec = F.cross_entropy(logits_clean, labels, reduction="none")
                probs = [F.softmax(logits_clean, dim=1)]
                if aug_views:
                    aug_loss = torch.zeros_like(loss_vec)
                    for aug in aug_views:
                        logits_aug = forward_logits(model, aug)
                        aug_loss = aug_loss + F.cross_entropy(logits_aug, labels, reduction="none")
                        probs.append(F.softmax(logits_aug, dim=1))
                    loss_vec = loss_vec + lambda_aug * aug_loss / len(aug_views)
                if len(probs) > 1 and lambda_js > 0:
                    mix = torch.stack(probs, dim=0).mean(dim=0).clamp(1e-7, 1.0)
                    log_mix = mix.log()
                    js = torch.zeros_like(loss_vec)
                    for prob in probs:
                        js = js + F.kl_div(log_mix, prob, reduction="none").sum(dim=1)
                    loss_vec = loss_vec + lambda_js * js / len(probs)
            else:
                image_tensor = images.to(self.device, non_blocking=True)
                logits = forward_logits(model, image_tensor)
                loss_vec = F.cross_entropy(logits, labels, reduction="none")

            per_class = torch.zeros(num_classes, device=self.device)
            counts = torch.zeros(num_classes, device=self.device)
            for class_id in range(num_classes):
                mask = labels == class_id
                if mask.any():
                    per_class[class_id] = loss_vec[mask].mean()
                    counts[class_id] = mask.sum()
            return {
                "overall": float(loss_vec.mean().detach().cpu()),
                "per_class": per_class.detach().cpu().tolist(),
                "counts": counts.detach().cpu().tolist(),
            }

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
