from __future__ import annotations

import torch
import torch.nn.functional as F

from fedprime.methods.balanced_environment_risk import balanced_environment_risk
from fedprime.models.factory import forward_logits
from fedprime.utils.env import add_vendor_paths


def _model_backbone(model):
    return model.module.backbone if hasattr(model, "module") else model.backbone


def train_local_fedease_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    fedease_cfg: dict,
    *,
    class_environment_counts: torch.Tensor | None = None,
    lambda_jsd: float = 12.0,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    skip_nonfinite: bool = False,
    log_interval: int | None = None,
    context: str = "PEW+BER local phase",
    diagnostics: dict[str, float] | None = None,
) -> float:
    """Train one PEW+BER local epoch with AugMix/JSD and optional DCL."""

    add_vendor_paths()
    from loss import DCLLoss

    ber_cfg = fedease_cfg.get("ber", {})
    use_ber = bool(ber_cfg.get("enabled", True))
    use_dcl = bool(fedease_cfg.get("preserve_dcl", True))

    model.train()
    metric_values: dict[str, list[float]] = {
        "clean_ce": [],
        "classification_loss": [],
        "ber_loss": [],
        "jsd_loss": [],
        "dcl_loss": [],
        "ber_valid_groups": [],
    }
    total_losses: list[float] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if len(batch) == 3:
            images, labels, environment_ids = batch
        elif len(batch) == 5:
            images, labels, environment_ids, _environment_features, _environment_confidence = batch
        else:
            raise ValueError(
                "PEW+BER requires (views, labels, environment_ids) or "
                "(views, labels, environment_ids, environment_features, confidence)"
            )
        if not isinstance(images, (tuple, list)) or len(images) < 4:
            raise ValueError("PEW+BER requires clean/strong/second-strong/weak AugMix views")

        images = [image.to(device, non_blocking=True) for image in images]
        labels = labels.to(device, non_blocking=True).long()
        environment_ids = environment_ids.to(device, non_blocking=True).long()

        batch_size = images[0].shape[0]
        logits_all = forward_logits(model, torch.cat(images[:3], dim=0))
        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, batch_size)
        sample_ce = F.cross_entropy(logits_clean, labels, reduction="none")
        clean_ce = sample_ce.mean()

        if use_ber:
            classification_loss, ber_stats = balanced_environment_risk(
                sample_ce,
                labels,
                environment_ids,
                group_counts=class_environment_counts,
                support_gamma=float(ber_cfg.get("support_gamma", 0.0)),
                count_cap=int(ber_cfg.get("count_cap", 32)),
                min_group_count=int(ber_cfg.get("min_group_count", 1)),
            )
            ber_loss = classification_loss
        else:
            classification_loss = clean_ce
            ber_loss = clean_ce.new_zeros(())
            ber_stats = {"valid_groups": clean_ce.new_zeros(())}

        p_clean = F.softmax(logits_clean, dim=1)
        p_aug1 = F.softmax(logits_aug1, dim=1)
        p_aug2 = F.softmax(logits_aug2, dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1.0e-7, 1.0).log()
        jsd_loss = (
            F.kl_div(p_mixture, p_clean, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
        ) / 3.0

        feature_views = _model_backbone(model)(torch.cat([images[0], images[1], images[3]], dim=0))
        feature_views = feature_views.reshape(feature_views.shape[0], -1)
        if use_dcl:
            normalized = F.normalize(feature_views, dim=1)
            fclean, fstrong, fweak = torch.split(normalized, batch_size)
            dcl_loss = DCLLoss(
                temperature=0.2,
                device=device,
                beta=1.0,
                ddm_temperature=0.2,
            )(
                original_feature=fclean.unsqueeze(1),
                weak_feature=fweak.unsqueeze(1),
                strong_feature=fstrong.unsqueeze(1),
                labels=labels,
            )
        else:
            dcl_loss = clean_ce.new_zeros(())

        loss = classification_loss + float(lambda_jsd) * jsd_loss + dcl_loss
        if not torch.isfinite(loss):
            message = f"{context}: non-finite loss at batch {batch_idx}"
            if skip_nonfinite:
                print(f"[warning] {message}; skipping batch", flush=True)
                continue
            raise FloatingPointError(message)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            message = f"{context}: non-finite gradient at batch {batch_idx}"
            if skip_nonfinite:
                print(f"[warning] {message}; skipping batch", flush=True)
                continue
            raise FloatingPointError(message)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()

        scalar_metrics = {
            "clean_ce": clean_ce,
            "classification_loss": classification_loss,
            "ber_loss": ber_loss,
            "jsd_loss": jsd_loss,
            "dcl_loss": dcl_loss,
            "ber_valid_groups": ber_stats["valid_groups"],
        }
        for name, value in scalar_metrics.items():
            metric_values[name].append(float(value.detach().cpu()))
        total_losses.append(float(loss.detach().cpu()))

        if log_interval and (batch_idx + 1) % int(log_interval) == 0:
            print(
                f"[heartbeat] {context} batch={batch_idx + 1} "
                f"loss={total_losses[-1]:.4f} "
                f"cls={metric_values['classification_loss'][-1]:.4f} "
                f"ber_groups={metric_values['ber_valid_groups'][-1]:.0f}",
                flush=True,
            )

    if diagnostics is not None:
        for name, values in metric_values.items():
            diagnostics[name] = sum(values) / max(len(values), 1)
        diagnostics["total_loss"] = sum(total_losses) / max(len(total_losses), 1)
    return sum(total_losses) / max(len(total_losses), 1)
