from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IRDLossResult:
    loss: torch.Tensor
    mean_kl: torch.Tensor
    worst_view_kl: torch.Tensor


def standardize_logits(logits: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Remove per-sample class-logit location and scale across heterogeneous models."""

    mean = logits.mean(dim=1, keepdim=True)
    std = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(float(eps))
    return (logits - mean) / std


def invariant_anchor(logits_views: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not logits_views:
        raise ValueError("Invariant anchor requires at least one logits view.")
    return torch.stack([standardize_logits(logits) for logits in logits_views], dim=0).mean(dim=0)


def leave_one_out_median(anchors: dict[int, torch.Tensor], receiver_id: int) -> torch.Tensor:
    others = [anchor for client_id, anchor in sorted(anchors.items()) if client_id != receiver_id]
    if not others:
        raise ValueError("Leave-one-out teacher requires at least two clients.")
    return torch.stack(others, dim=0).median(dim=0).values


def anchor_disagreement(anchors: dict[int, torch.Tensor]) -> torch.Tensor:
    if not anchors:
        raise ValueError("Anchor disagreement requires at least one client.")
    stacked = torch.stack([anchor for _, anchor in sorted(anchors.items())], dim=0)
    center = stacked.median(dim=0).values
    return (stacked - center.unsqueeze(0)).square().mean().sqrt()


def smooth_worst_view_distillation(
    student_logits_views: list[torch.Tensor] | tuple[torch.Tensor, ...],
    teacher_anchor: torch.Tensor,
    distill_temperature: float = 2.0,
    smooth_temperature: float = 0.5,
) -> IRDLossResult:
    if not student_logits_views:
        raise ValueError("IRD requires at least one student logits view.")
    if distill_temperature <= 0 or smooth_temperature <= 0:
        raise ValueError("IRD temperatures must be positive.")

    teacher_probs = F.softmax(teacher_anchor.detach() / distill_temperature, dim=1)
    per_view_kl = []
    for logits in student_logits_views:
        student = standardize_logits(logits)
        student_log_probs = F.log_softmax(student / distill_temperature, dim=1)
        per_sample_kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=1)
        per_view_kl.append(per_sample_kl)
    kl = torch.stack(per_view_kl, dim=0)
    smooth_worst = smooth_temperature * torch.logsumexp(
        kl / smooth_temperature,
        dim=0,
    )
    return IRDLossResult(
        loss=smooth_worst.mean(),
        mean_kl=kl.mean(),
        worst_view_kl=kl.max(dim=0).values.mean(),
    )
