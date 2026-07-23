from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def normalize_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each sample over classes to remove shift and positive scale."""

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, class], got {logits.shape}")
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True, unbiased=False)
    return (logits - mean) / (std + float(eps))


def _non_target_mask(num_classes: int, labels: torch.Tensor) -> torch.Tensor:
    class_ids = torch.arange(num_classes, device=labels.device).view(1, -1)
    return class_ids != labels.view(-1, 1)


def conservative_margin_transfer_loss(
    receiver_logits: torch.Tensor,
    source_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin_clip: float = 2.0,
    source_correct_only: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Transfer only positive source-over-receiver target-class margins."""

    if receiver_logits.shape != source_logits.shape:
        raise ValueError("receiver_logits and source_logits must have the same shape")
    if receiver_logits.ndim != 2:
        raise ValueError("logits must have shape [batch, class]")
    if labels.ndim != 1 or labels.shape[0] != receiver_logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if margin_clip <= 0:
        raise ValueError("margin_clip must be positive")

    labels = labels.long()
    receiver = normalize_logits(receiver_logits)
    source = normalize_logits(source_logits.detach())
    source_true = source.gather(1, labels.view(-1, 1))
    receiver_true = receiver.gather(1, labels.view(-1, 1))
    source_margins = source_true - source
    receiver_margins = receiver_true - receiver
    targets = source_margins.clamp(min=0.0, max=float(margin_clip))
    hinge = F.relu(targets - receiver_margins)
    mask = _non_target_mask(receiver_logits.shape[1], labels)
    per_sample = (hinge * mask).sum(dim=1) / max(receiver_logits.shape[1] - 1, 1)

    if source_correct_only:
        source_correct = source_logits.argmax(dim=1).eq(labels)
        per_sample = per_sample * source_correct.to(per_sample.dtype)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def fixed_margin_loss(
    receiver_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Architecture-independent large-margin control without a peer model."""

    if target_margin <= 0:
        raise ValueError("target_margin must be positive")
    labels = labels.long()
    receiver = normalize_logits(receiver_logits)
    true_logits = receiver.gather(1, labels.view(-1, 1))
    margins = true_logits - receiver
    mask = _non_target_mask(receiver_logits.shape[1], labels)
    per_sample = (
        F.relu(float(target_margin) - margins) * mask
    ).sum(dim=1) / max(receiver_logits.shape[1] - 1, 1)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def direct_peer_kd_loss(
    receiver_logits: torch.Tensor,
    source_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Ordinary peer KD control on the receiver's private inputs."""

    if receiver_logits.shape != source_logits.shape:
        raise ValueError("receiver_logits and source_logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    temperature = float(temperature)
    return F.kl_div(
        F.log_softmax(receiver_logits / temperature, dim=1),
        F.softmax(source_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def gradient_cosine_from_losses(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """Return cosine and norms for gradients of two losses at one model state."""

    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError("No trainable parameters were provided")
    first_grads = torch.autograd.grad(
        first_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    second_grads = torch.autograd.grad(
        second_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    dot = torch.zeros((), device=first_loss.device)
    first_sq = torch.zeros((), device=first_loss.device)
    second_sq = torch.zeros((), device=first_loss.device)
    for first, second in zip(first_grads, second_grads):
        if first is not None:
            first_sq = first_sq + first.detach().pow(2).sum()
        if second is not None:
            second_sq = second_sq + second.detach().pow(2).sum()
        if first is not None and second is not None:
            dot = dot + (first.detach() * second.detach()).sum()

    first_norm = first_sq.sqrt()
    second_norm = second_sq.sqrt()
    denominator = first_norm * second_norm
    if float(denominator.item()) <= float(eps):
        return 0.0, float(first_norm.item()), float(second_norm.item())
    cosine = dot / (denominator + float(eps))
    return float(cosine.item()), float(first_norm.item()), float(second_norm.item())
