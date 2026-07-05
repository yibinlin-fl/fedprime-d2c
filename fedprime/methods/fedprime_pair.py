from __future__ import annotations

import csv
from pathlib import Path
import time

import torch
import torch.optim as optim

from fedprime.augmentations.prime_adapter import build_prime_module
from fedprime.data.loaders import (
    build_private_loaders,
    build_public_loader,
    dataset_stats,
    load_private_labels,
    normalize_batch,
    partition_private_data,
)
from fedprime.methods.cpad import (
    cpad_pair_bce_loss,
    estimate_pair_expertise,
    save_pair_expertise_snapshot,
)
from fedprime.methods.local_prime import (
    optimizer_step_checked,
    require_finite,
    train_local_prime_cbcl_epoch,
    train_local_prime_epoch,
    train_local_standard_epoch,
)
from fedprime.models.factory import build_models, forward_logits
from fedprime.utils.config import save_config
from fedprime.utils.env import resolve_device, seed_everything


class FedPrimePairExperiment:
    """FedPRIME-PAIR: PRIME + optional CBCL + CPAD-PairBCE communication."""

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
        model_cfg = self.config["models"]
        method_cfg = self.config["method"]

        num_clients = len(model_cfg["names"])
        num_classes = int(data_cfg.get("num_classes", 10))
        stats = dataset_stats(data_cfg.get("private_dataset", "cifar10"))

        print("[setup] FedPRIME-PAIR loading private labels", flush=True)
        labels = load_private_labels(data_cfg["private_root"], data_cfg["private_corrupt_rate"])
        print("[setup] FedPRIME-PAIR loading/creating private partition", flush=True)
        dataidx_map = partition_private_data(
            labels=labels,
            num_clients=num_clients,
            num_classes=num_classes,
            partition=data_cfg.get("partition", "dirichlet"),
            dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
            max_samples_per_client=data_cfg.get("private_samples_per_client"),
            partition_indices_path=data_cfg.get("partition_indices_path"),
            partition_seed=int(self.config.get("seed", 0)),
        )

        use_prime = bool(method_cfg.get("use_prime", True))
        use_cbcl = use_prime and bool(method_cfg.get("use_cbcl", True))
        use_cpad = bool(method_cfg.get("use_cpad", True))

        print("[setup] FedPRIME-PAIR building private/test loaders", flush=True)
        private_loaders, test_loader = build_private_loaders(
            cifar10c_root=data_cfg["private_root"],
            dataidx_map=dataidx_map,
            train_batch_size=train_cfg["batch_size"],
            test_batch_size=train_cfg.get("test_batch_size", 512),
            corrupt_rate=data_cfg["private_corrupt_rate"],
            test_corrupt_rate=data_cfg["test_corrupt_rate"],
            num_workers=int(self.config.get("num_workers", 2)),
            raw_for_prime=use_prime,
        )
        print("[setup] FedPRIME-PAIR building public loader", flush=True)
        public_loader = build_public_loader(
            cifar100_root=data_cfg["public_root"],
            public_size=int(data_cfg.get("public_size", 5000)),
            batch_size=train_cfg["public_batch_size"],
            num_workers=int(self.config.get("num_workers", 2)),
            seed=int(self.config.get("seed", 0)),
            download=bool(data_cfg.get("download_public", False)),
        )

        print("[setup] FedPRIME-PAIR building heterogeneous client models", flush=True)
        models = build_models(model_cfg["names"], num_classes)
        models = {idx: model.to(self.device) for idx, model in models.items()}
        self._load_models_if_configured(models)
        print("[setup] FedPRIME-PAIR building optimizers and PRIME module", flush=True)
        optimizers = {idx: self._build_optimizer(model) for idx, model in models.items()}
        prime_aug = build_prime_module(stats, method_cfg.get("prime", {})).to(self.device) if use_prime else None
        print("[setup] FedPRIME-PAIR setup complete; entering training rounds", flush=True)

        metrics_path = self.output_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "avg_acc",
                    "worst_acc",
                    "local_loss",
                    "cpad_loss",
                    "expertise_mean",
                    "expertise_max",
                ],
            )
            writer.writeheader()

            public_iter = iter(public_loader)
            for round_idx in range(int(train_cfg["rounds"])):
                round_start = time.perf_counter()
                print(f"[heartbeat] round {round_idx:03d} start", flush=True)
                local_loss = self._local_phase(
                    round_idx=round_idx,
                    models=models,
                    optimizers=optimizers,
                    private_loaders=private_loaders,
                    prime_aug=prime_aug,
                    stats=stats,
                    train_cfg=train_cfg,
                    method_cfg=method_cfg,
                    use_prime=use_prime,
                    use_cbcl=use_cbcl,
                )

                expertise_weighted = None
                expertise_raw = None
                counts = None
                cpad_warmup = int(method_cfg.get("cpad", {}).get("warmup_rounds", 0))
                if use_cpad and round_idx >= cpad_warmup:
                    print(f"[heartbeat] round {round_idx:03d} estimating pair expertise", flush=True)
                    expertise_raw, expertise_weighted, counts = self._estimate_all_pair_expertise(
                        round_idx=round_idx,
                        models=models,
                        private_loaders=private_loaders,
                        prime_aug=prime_aug,
                        stats=stats,
                        method_cfg=method_cfg,
                        num_classes=num_classes,
                    )
                    self._maybe_save_expertise(round_idx, expertise_weighted, expertise_raw, counts, method_cfg)
                    print(f"[heartbeat] round {round_idx:03d} running CPAD public distillation", flush=True)
                    cpad_loss = self._cpad_phase(
                        round_idx=round_idx,
                        models=models,
                        optimizers=optimizers,
                        public_loader=public_loader,
                        public_iter=public_iter,
                        expertise_weighted=expertise_weighted,
                        stats=stats,
                        method_cfg=method_cfg,
                    )
                    expertise_mean = float(expertise_weighted.mean().detach().cpu())
                    expertise_max = float(expertise_weighted.max().detach().cpu())
                else:
                    cpad_loss = 0.0
                    expertise_mean = 0.0
                    expertise_max = 0.0

                print(f"[heartbeat] round {round_idx:03d} evaluating clients", flush=True)
                accs = self._evaluate(models, test_loader)
                row = {
                    "round": round_idx,
                    "avg_acc": sum(accs) / len(accs),
                    "worst_acc": min(accs),
                    "local_loss": local_loss,
                    "cpad_loss": cpad_loss,
                    "expertise_mean": expertise_mean,
                    "expertise_max": expertise_max,
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"[round {round_idx:03d}] "
                    f"avg_acc={row['avg_acc']:.2f} "
                    f"worst_acc={row['worst_acc']:.2f} "
                    f"local_loss={local_loss:.4f} "
                    f"cpad_loss={cpad_loss:.4f} "
                    f"expertise_mean={expertise_mean:.4f} "
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

    def _local_phase(
        self,
        round_idx: int,
        models,
        optimizers,
        private_loaders,
        prime_aug,
        stats,
        train_cfg,
        method_cfg,
        use_prime: bool,
        use_cbcl: bool,
    ) -> float:
        losses = []
        cbcl_cfg = method_cfg.get("cbcl", {})
        max_grad_norm = train_cfg.get("max_grad_norm")
        lambda_jsd = float(method_cfg.get("lambda_jsd", 12.0)) if bool(method_cfg.get("use_jsd", True)) else 0.0
        for client_id, loader in enumerate(private_loaders):
            client_start = time.perf_counter()
            print(f"[heartbeat] round {round_idx:03d} local client {client_id} start", flush=True)
            for _ in range(int(train_cfg.get("local_epochs", 1))):
                context = f"FedPRIME-PAIR local phase, client={client_id}"
                if use_prime and use_cbcl:
                    loss = train_local_prime_cbcl_epoch(
                        model=models[client_id],
                        loader=loader,
                        optimizer=optimizers[client_id],
                        prime_aug=prime_aug,
                        device=self.device,
                        lambda_jsd=lambda_jsd,
                        lambda_cbcl=float(cbcl_cfg.get("lambda_cbcl", 0.2)),
                        cbcl_temperature=float(cbcl_cfg.get("temperature", 0.2)),
                        cbcl_reliability_tau=float(cbcl_cfg.get("view_reliability_tau", 1.0)),
                        max_batches=train_cfg.get("max_local_batches"),
                        max_grad_norm=max_grad_norm,
                        progress_every=train_cfg.get("progress_every_batches"),
                        context=context,
                    )
                elif use_prime:
                    loss = train_local_prime_epoch(
                        model=models[client_id],
                        loader=loader,
                        optimizer=optimizers[client_id],
                        prime_aug=prime_aug,
                        device=self.device,
                        lambda_jsd=lambda_jsd,
                        max_batches=train_cfg.get("max_local_batches"),
                        max_grad_norm=max_grad_norm,
                        context=context,
                    )
                else:
                    loss = train_local_standard_epoch(
                        model=models[client_id],
                        loader=loader,
                        optimizer=optimizers[client_id],
                        normalizer=lambda x: normalize_batch(x, stats),
                        device=self.device,
                        max_batches=train_cfg.get("max_local_batches"),
                        max_grad_norm=max_grad_norm,
                        context=context,
                    )
                losses.append(loss)
            print(
                f"[heartbeat] round {round_idx:03d} local client {client_id} done "
                f"loss={loss:.4f} elapsed={time.perf_counter() - client_start:.1f}s",
                flush=True,
            )
        return sum(losses) / max(len(losses), 1)

    def _estimate_all_pair_expertise(
        self,
        round_idx: int,
        models,
        private_loaders,
        prime_aug,
        stats,
        method_cfg,
        num_classes: int,
    ):
        cpad_cfg = method_cfg.get("cpad", {})
        raw_list = []
        weighted_list = []
        count_list = []
        for client_id, loader in enumerate(private_loaders):
            start = time.perf_counter()
            expertise = estimate_pair_expertise(
                model=models[client_id],
                loader=loader,
                device=self.device,
                num_classes=num_classes,
                prime_aug=prime_aug,
                normalizer=lambda x: normalize_batch(x, stats),
                max_batches=cpad_cfg.get("expertise_batches"),
                softmin_tau=float(cpad_cfg.get("softmin_tau", 0.5)),
                expertise_tau=float(cpad_cfg.get("expertise_tau", 0.5)),
                support=cpad_cfg.get("support", "log"),
                support_gamma=float(cpad_cfg.get("support_gamma", 20.0)),
                eps=float(cpad_cfg.get("eps", 1e-6)),
            )
            raw_list.append(expertise.raw)
            weighted_list.append(expertise.weighted)
            count_list.append(expertise.counts)
            print(
                f"[heartbeat] round {round_idx:03d} expertise client {client_id} done "
                f"mean={float(expertise.weighted.mean().detach().cpu()):.4f} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
        raw = torch.stack(raw_list, dim=0)
        weighted = torch.stack(weighted_list, dim=0)
        counts = torch.stack(count_list, dim=0)
        require_finite(raw, "CPAD raw expertise", "CPAD expertise estimation")
        require_finite(weighted, "CPAD weighted expertise", "CPAD expertise estimation")
        return raw, weighted, counts

    def _cpad_phase(
        self,
        round_idx: int,
        models,
        optimizers,
        public_loader,
        public_iter,
        expertise_weighted,
        stats,
        method_cfg,
    ) -> float:
        losses = []
        cpad_cfg = method_cfg.get("cpad", {})
        num_batches = int(self.config["train"].get("public_batches_per_round", 1))
        for public_batch_idx in range(num_batches):
            batch_start = time.perf_counter()
            try:
                images, _ = next(public_iter)
            except StopIteration:
                public_iter = iter(public_loader)
                images, _ = next(public_iter)
            images = images.to(self.device, non_blocking=True)
            images_norm = normalize_batch(images, stats)

            logits = []
            for client_id in sorted(models):
                models[client_id].eval()
                with torch.no_grad():
                    logits.append(forward_logits(models[client_id], images_norm))
            logits_all = torch.stack(logits, dim=0).detach()
            require_finite(logits_all, "CPAD public client logits", "CPAD teacher construction")

            for client_id in sorted(models):
                model = models[client_id]
                model.train()
                student_logits = forward_logits(model, images_norm)
                loss = cpad_pair_bce_loss(
                    student_logits=student_logits,
                    public_logits_all=logits_all,
                    expertise_weighted=expertise_weighted,
                    student_id=client_id,
                    temperature=float(cpad_cfg.get("temperature", 2.0)),
                    gate_tau=float(cpad_cfg.get("gate_tau", 0.5)),
                    eps=float(cpad_cfg.get("eps", 1e-6)),
                    leave_one_out=bool(cpad_cfg.get("leave_one_out", True)),
                    use_gate=bool(cpad_cfg.get("use_gate", True)),
                    use_confidence=bool(cpad_cfg.get("use_confidence", True)),
                    use_agreement=bool(cpad_cfg.get("use_agreement", True)),
                    agreement_tau=float(cpad_cfg.get("agreement_tau", 0.05)),
                )
                require_finite(loss, "CPAD loss", f"CPAD phase, client={client_id}")
                loss = float(cpad_cfg.get("lambda_cpad", 1.0)) * loss
                optimizer_step_checked(
                    loss,
                    model,
                    optimizers[client_id],
                    context=f"CPAD phase, client={client_id}, public_batch={public_batch_idx}",
                    max_grad_norm=self.config["train"].get("max_grad_norm"),
                )
                losses.append(float(loss.detach().cpu()))
            print(
                f"[heartbeat] round {round_idx:03d} CPAD public_batch {public_batch_idx} done "
                f"elapsed={time.perf_counter() - batch_start:.1f}s",
                flush=True,
            )
        return sum(losses) / max(len(losses), 1)

    def _maybe_save_expertise(self, round_idx, expertise_weighted, expertise_raw, counts, method_cfg) -> None:
        diagnostics_cfg = method_cfg.get("pair_diagnostics", {})
        if not bool(diagnostics_cfg.get("enabled", True)):
            return
        save_rounds = diagnostics_cfg.get("save_rounds")
        save_every = diagnostics_cfg.get("save_every")
        should_save = False
        if save_rounds is not None:
            should_save = int(round_idx) in {int(value) for value in save_rounds}
        if save_every is not None and int(save_every) > 0:
            should_save = should_save or (int(round_idx) % int(save_every) == 0)
        if should_save:
            save_pair_expertise_snapshot(
                self.output_dir,
                round_idx=round_idx,
                expertise=expertise_weighted,
                expertise_raw=expertise_raw,
                counts=counts,
            )

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
