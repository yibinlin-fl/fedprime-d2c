from __future__ import annotations

import torch
import torch.nn.functional as F

from fedprime.models.factory import forward_logits
from fedprime.utils.env import add_vendor_paths


def _finite_summary(tensor: torch.Tensor) -> str:
    finite = tensor.detach()[torch.isfinite(tensor.detach())]
    if finite.numel() == 0:
        return "no finite values"
    return (
        f"shape={tuple(tensor.shape)}, min={finite.min().item():.4g}, "
        f"max={finite.max().item():.4g}, mean={finite.float().mean().item():.4g}"
    )


def require_finite(tensor: torch.Tensor, name: str, context: str) -> None:
    if not torch.isfinite(tensor).all():
        count = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(
            f"Non-finite {name} detected at {context}: {count}/{tensor.numel()} "
            f"values are NaN/Inf; finite {_finite_summary(tensor)}."
        )


def optimizer_step_checked(
    loss: torch.Tensor,
    model,
    optimizer,
    context: str,
    max_grad_norm: float | None = None,
) -> float:
    require_finite(loss, "loss", context)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    parameters = [param for param in model.parameters() if param.grad is not None]
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float("inf") if max_grad_norm is None else float(max_grad_norm),
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        raise FloatingPointError(f"Non-finite gradients detected at {context}.") from exc
    optimizer.step()
    return float(grad_norm.detach().cpu())


def jsd_loss_from_logits(logits_clean, logits_aug1, logits_aug2) -> torch.Tensor:
    # KLDiv also differentiates through its target probabilities. Softmax can
    # underflow to an exact zero for heterogeneous lightweight models, making
    # that target-side gradient contain log(0) even while the loss is finite.
    # Clamp and renormalize every target distribution to keep JSD gradients
    # finite without materially changing the augmentation objective.
    def stable_probs(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp_min(1e-7)
        return probs / probs.sum(dim=1, keepdim=True)

    p_clean = stable_probs(logits_clean)
    p_aug1 = stable_probs(logits_aug1)
    p_aug2 = stable_probs(logits_aug2)
    mixture = ((p_clean + p_aug1 + p_aug2) / 3.0).clamp_min(1e-7)
    p_mixture = (mixture / mixture.sum(dim=1, keepdim=True)).log()
    return (
        F.kl_div(p_mixture, p_clean, reduction="batchmean")
        + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
        + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
    ) / 3.0


def train_local_prime_epoch(
    model,
    loader,
    optimizer,
    prime_aug,
    device: torch.device,
    lambda_jsd: float,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    context: str = "PRIME local training",
) -> float:
    model.train()
    prime_aug.eval()
    losses = []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        images_all = prime_aug(images)
        batch_context = f"{context}, batch={batch_idx}"
        require_finite(images, "input images", batch_context)
        require_finite(images_all, "PRIME views", batch_context)
        logits_all = forward_logits(model, images_all)
        require_finite(logits_all, "model logits", batch_context)
        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images.size(0))

        ce_loss = F.cross_entropy(logits_clean, labels)
        jsd_loss = jsd_loss_from_logits(logits_clean, logits_aug1, logits_aug2)
        require_finite(ce_loss, "cross-entropy loss", batch_context)
        require_finite(jsd_loss, "JSD loss", batch_context)
        loss = ce_loss + lambda_jsd * jsd_loss

        optimizer_step_checked(loss, model, optimizer, batch_context, max_grad_norm)
        losses.append(float(loss.detach().cpu()))

    return sum(losses) / max(len(losses), 1)


def train_local_standard_epoch(
    model,
    loader,
    optimizer,
    normalizer,
    device: torch.device,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    context: str = "standard local training",
) -> float:
    model.train()
    losses = []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = forward_logits(model, normalizer(images))
        batch_context = f"{context}, batch={batch_idx}"
        require_finite(logits, "model logits", batch_context)
        loss = F.cross_entropy(logits, labels)

        optimizer_step_checked(loss, model, optimizer, batch_context, max_grad_norm)
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


def _model_backbone(model):
    return model.module.backbone if hasattr(model, "module") else model.backbone


def train_local_prime_dcl_epoch(
    model,
    loader,
    optimizer,
    prime_aug,
    normalizer,
    device: torch.device,
    lambda_jsd: float,
    cl_module: str | None = "dcl",
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    context: str = "PRIME+DCL local training",
) -> float:
    add_vendor_paths()
    from loss import DCLLoss, SupConLoss

    model.train()
    prime_aug.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    losses = []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if not isinstance(images, (tuple, list)) or len(images) != 2:
            raise ValueError("PRIME+DCL training expects (base_view, weak_view) images.")

        base_images = images[0].to(device, non_blocking=True)
        weak_images = images[1].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        prime_views = prime_aug(base_images)
        batch_context = f"{context}, batch={batch_idx}"
        require_finite(base_images, "input images", batch_context)
        require_finite(prime_views, "PRIME views", batch_context)
        logits_all = forward_logits(model, prime_views)
        require_finite(logits_all, "model logits", batch_context)
        logits_clean, logits_prime1, logits_prime2 = torch.split(logits_all, base_images.size(0))

        ce_loss = criterion(logits_clean, labels)
        jsd_loss = jsd_loss_from_logits(logits_clean, logits_prime1, logits_prime2)
        require_finite(ce_loss, "cross-entropy loss", batch_context)
        require_finite(jsd_loss, "JSD loss", batch_context)
        loss = ce_loss + lambda_jsd * jsd_loss

        if cl_module == "supcon":
            images_cont = torch.cat([prime_views[:base_images.size(0)], prime_views[base_images.size(0):2 * base_images.size(0)]], dim=0)
            features = _model_backbone(model)(images_cont)
            features = F.normalize(features.view(features.size(0), -1), dim=1)
            require_finite(features, "SupCon features", batch_context)
            fclean, fstrong = torch.split(features, base_images.size(0))
            features = torch.cat([fclean.unsqueeze(1), fstrong.unsqueeze(1)], dim=1)
            contrastive_loss = SupConLoss(temperature=0.2, device=device)(features, labels)
            require_finite(contrastive_loss, "SupCon loss", batch_context)
            loss = loss + contrastive_loss
        elif cl_module == "dcl":
            clean_norm = prime_views[:base_images.size(0)]
            strong_norm = prime_views[base_images.size(0):2 * base_images.size(0)]
            weak_norm = normalizer(weak_images)
            images_cont = torch.cat([clean_norm, strong_norm, weak_norm], dim=0)
            features = _model_backbone(model)(images_cont)
            features = F.normalize(features.view(features.size(0), -1), dim=1)
            require_finite(features, "DCL features", batch_context)
            fclean, fstrong, fweak = torch.split(features, base_images.size(0))
            contrastive_loss = DCLLoss(
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
            require_finite(contrastive_loss, "DCL loss", batch_context)
            loss = loss + contrastive_loss

        optimizer_step_checked(loss, model, optimizer, batch_context, max_grad_norm)
        losses.append(float(loss.detach().cpu()))

    return sum(losses) / max(len(losses), 1)
