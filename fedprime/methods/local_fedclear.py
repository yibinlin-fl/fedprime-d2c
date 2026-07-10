from __future__ import annotations

import torch
import torch.nn.functional as F

from fedprime.augmentations.counterfactual import build_counterfactual_views
from fedprime.methods.ccre import class_conditional_counterfactual_risk
from fedprime.models.factory import forward_logits


def train_local_fedclear_epoch(
    model,
    loader,
    optimizer,
    normalizer,
    device: torch.device,
    lambda_jsd: float,
    ccre_cfg: dict,
    view_cfg: dict,
    round_idx: int,
    client_id: int,
    seed: int,
    client_class_counts: torch.Tensor | None = None,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    skip_nonfinite: bool = False,
    log_interval: int | None = None,
    diagnostics: dict[str, float] | None = None,
) -> float:
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    lambda_ccre = float(ccre_cfg.get("lambda_ccre", 1.0))
    risk_temperature = float(ccre_cfg.get("temperature", 0.5))
    losses = []
    ccre_losses = []
    worst_risks = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        view_seed = int(seed) + round_idx * 1_000_003 + client_id * 10_007 + batch_idx
        raw_views, operator_names = build_counterfactual_views(images, view_cfg, view_seed)
        views = [normalizer(view) for view in raw_views]
        logits_views = [forward_logits(model, view) for view in views]

        class_weights = None
        if client_class_counts is not None:
            counts = client_class_counts.to(device=device, dtype=torch.float32)
            frequencies = counts / counts.sum().clamp_min(1.0)
            presence_probability = 1.0 - (1.0 - frequencies).clamp(0.0, 1.0).pow(labels.numel())
            class_weights = presence_probability.clamp_min(1e-6).reciprocal()

        clean_loss = criterion(logits_views[0], labels)
        probs = [F.softmax(logits, dim=1) for logits in logits_views]
        mixture_log = torch.stack(probs, dim=0).mean(dim=0).clamp_min(1e-7).log()
        jsd_loss = sum(
            F.kl_div(mixture_log, prob, reduction="batchmean")
            for prob in probs
        ) / len(probs)
        ccre_result = class_conditional_counterfactual_risk(
            logits_views,
            labels,
            temperature=risk_temperature,
            class_weights=class_weights,
        )
        loss = clean_loss + float(lambda_jsd) * jsd_loss + lambda_ccre * ccre_result.loss

        if not torch.isfinite(loss):
            message = (
                f"FedCLEAR local phase, client={client_id}: non-finite loss "
                f"at batch {batch_idx}: {float(loss.detach().cpu())}"
            )
            if skip_nonfinite:
                print(f"[warning] {message}; skipping batch", flush=True)
                continue
            raise FloatingPointError(message)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads_finite = all(
            param.grad is None or bool(torch.isfinite(param.grad).all())
            for param in model.parameters()
        )
        if not grads_finite:
            optimizer.zero_grad(set_to_none=True)
            message = f"FedCLEAR local phase, client={client_id}: non-finite gradient at batch {batch_idx}"
            if skip_nonfinite:
                print(f"[warning] {message}; skipping batch", flush=True)
                continue
            raise FloatingPointError(message)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        ccre_losses.append(float(ccre_result.loss.detach().cpu()))
        worst_risks.append(float(ccre_result.mean_worst_view_risk.detach().cpu()))
        if batch_idx == 0:
            print(
                f"[heartbeat] FedCLEAR local views round={round_idx:03d} "
                f"client={client_id} operators={','.join(operator_names)}",
                flush=True,
            )
        if log_interval and (batch_idx + 1) % int(log_interval) == 0:
            print(
                f"[heartbeat] FedCLEAR local phase, client={client_id} "
                f"batch={batch_idx + 1} loss={losses[-1]:.4f} "
                f"ccre={ccre_losses[-1]:.4f}",
                flush=True,
            )

    if diagnostics is not None:
        diagnostics["ccre_loss"] = sum(ccre_losses) / max(len(ccre_losses), 1)
        diagnostics["ccre_worst_view_risk"] = sum(worst_risks) / max(len(worst_risks), 1)
    return sum(losses) / max(len(losses), 1)
