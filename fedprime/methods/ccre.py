from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CCREResult:
    loss: torch.Tensor
    mean_view_risk: torch.Tensor
    mean_worst_view_risk: torch.Tensor
    num_present_classes: int


def class_conditional_counterfactual_risk(
    logits_views: list[torch.Tensor] | tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    temperature: float = 0.5,
    class_weights: torch.Tensor | None = None,
) -> CCREResult:
    """Class-balanced smooth maximum over per-view classification risks."""

    if not logits_views:
        raise ValueError("CCRE requires at least one logits view.")
    if temperature <= 0:
        raise ValueError(f"CCRE temperature must be positive, got {temperature}")
    batch_size = labels.numel()
    if any(logits.shape[0] != batch_size for logits in logits_views):
        raise ValueError("Every logits view must have the same batch size as labels.")

    per_sample_view_risk = torch.stack(
        [F.cross_entropy(logits, labels, reduction="none") for logits in logits_views],
        dim=1,
    )
    class_view_risks = []
    present_classes = labels.unique(sorted=True)
    for class_id in present_classes:
        mask = labels == class_id
        if mask.any():
            class_view_risks.append(per_sample_view_risk[mask].mean(dim=0))
    if not class_view_risks:
        zero = per_sample_view_risk.sum() * 0.0
        return CCREResult(zero, zero, zero, 0)

    risks = torch.stack(class_view_risks, dim=0)
    smooth_worst = temperature * torch.logsumexp(risks / temperature, dim=1)
    if class_weights is None:
        loss = smooth_worst.mean()
    else:
        weights = class_weights.to(device=labels.device, dtype=smooth_worst.dtype)[present_classes]
        weights = weights.clamp_min(0.0)
        loss = (smooth_worst * weights).sum() / weights.sum().clamp_min(1e-8)
    return CCREResult(
        loss=loss,
        mean_view_risk=risks.mean(),
        mean_worst_view_risk=risks.max(dim=1).values.mean(),
        num_present_classes=int(risks.shape[0]),
    )
