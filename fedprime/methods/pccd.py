from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PCCDLossResult:
    loss: torch.Tensor
    mean_kl: torch.Tensor
    worst_view_kl: torch.Tensor


def _validate_probabilities(probabilities: torch.Tensor) -> None:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected [batch, classes] probabilities, got {tuple(probabilities.shape)}")
    if probabilities.shape[1] < 2:
        raise ValueError("PCCD requires at least two output classes.")


def log_opinion_consensus(
    probability_views: list[torch.Tensor] | tuple[torch.Tensor, ...],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Geometric-mean consensus that retains only cross-view class evidence."""

    if not probability_views:
        raise ValueError("PCCD requires at least one probability view.")
    reference_shape = probability_views[0].shape
    for probabilities in probability_views:
        _validate_probabilities(probabilities)
        if probabilities.shape != reference_shape:
            raise ValueError("Every PCCD probability view must have the same shape.")
    mean_log_probability = torch.stack(
        [probabilities.clamp_min(float(eps)).log() for probabilities in probability_views],
        dim=0,
    ).mean(dim=0)
    return F.softmax(mean_log_probability, dim=1)


def normalized_entropy_confidence(probabilities: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return per-sample confidence in [0, 1] without a threshold hyperparameter."""

    _validate_probabilities(probabilities)
    safe = probabilities.clamp_min(float(eps))
    entropy = -(safe * safe.log()).sum(dim=1)
    maximum_entropy = math.log(probabilities.shape[1])
    return (1.0 - entropy / maximum_entropy).clamp(0.0, 1.0)


def leave_one_out_consensus_teacher(
    consensuses: dict[int, torch.Tensor],
    confidences: dict[int, torch.Tensor],
    receiver_id: int,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a sample-wise teacher from all clients except the receiver."""

    sender_ids = [client_id for client_id in sorted(consensuses) if client_id != receiver_id]
    if not sender_ids:
        raise ValueError("Leave-one-out PCCD requires at least two clients.")
    if set(consensuses) != set(confidences):
        raise ValueError("PCCD consensuses and confidences must contain the same clients.")

    reference = consensuses[sender_ids[0]]
    _validate_probabilities(reference)
    weighted_sum = torch.zeros_like(reference)
    total_weight = torch.zeros(reference.shape[0], 1, device=reference.device, dtype=reference.dtype)
    for sender_id in sender_ids:
        consensus = consensuses[sender_id]
        confidence = confidences[sender_id]
        if consensus.shape != reference.shape or confidence.shape != (reference.shape[0],):
            raise ValueError("PCCD sender tensors have incompatible shapes.")
        weight = confidence.to(dtype=reference.dtype).unsqueeze(1)
        weighted_sum = weighted_sum + weight * consensus
        total_weight = total_weight + weight

    uniform = torch.full_like(reference, 1.0 / reference.shape[1])
    teacher = torch.where(total_weight > float(eps), weighted_sum / total_weight.clamp_min(float(eps)), uniform)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(float(eps))
    teacher_confidence = normalized_entropy_confidence(teacher, eps=eps)
    teacher_confidence = torch.where(total_weight.squeeze(1) > float(eps), teacher_confidence, torch.zeros_like(teacher_confidence))
    return teacher, teacher_confidence


def probability_view_disagreement(
    probability_views: list[torch.Tensor] | tuple[torch.Tensor, ...],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Jensen-Shannon disagreement among paired views."""

    if not probability_views:
        raise ValueError("View disagreement requires at least one probability view.")
    mixture = torch.stack(probability_views, dim=0).mean(dim=0).clamp_min(float(eps))
    divergences = []
    for probabilities in probability_views:
        safe = probabilities.clamp_min(float(eps))
        divergences.append((safe * (safe.log() - mixture.log())).sum(dim=1))
    return torch.stack(divergences, dim=0).mean()


def teacher_margin(probabilities: torch.Tensor) -> torch.Tensor:
    _validate_probabilities(probabilities)
    top_two = probabilities.topk(k=2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def paired_counterfactual_distillation(
    student_logits_views: list[torch.Tensor] | tuple[torch.Tensor, ...],
    teacher_probabilities: torch.Tensor,
    sample_weights: torch.Tensor,
    eps: float = 1e-7,
) -> PCCDLossResult:
    """Distill one invariant teacher into every paired public view."""

    if not student_logits_views:
        raise ValueError("PCCD requires at least one student view.")
    _validate_probabilities(teacher_probabilities)
    if sample_weights.shape != (teacher_probabilities.shape[0],):
        raise ValueError("PCCD sample weights must have shape [batch].")
    if any(logits.shape != teacher_probabilities.shape for logits in student_logits_views):
        raise ValueError("Every PCCD student view must match the teacher shape.")

    safe_teacher = teacher_probabilities.detach().clamp_min(float(eps))
    weights = sample_weights.detach().to(dtype=safe_teacher.dtype).clamp_min(0.0)
    per_view_kl = []
    for logits in student_logits_views:
        student_log_probabilities = F.log_softmax(logits, dim=1)
        per_view_kl.append(
            (safe_teacher * (safe_teacher.log() - student_log_probabilities)).sum(dim=1)
        )
    kl = torch.stack(per_view_kl, dim=0)
    normalizer = weights.sum().clamp_min(float(eps))
    weighted_per_view = (kl * weights.unsqueeze(0)).sum(dim=1) / normalizer
    weighted_worst = (kl.max(dim=0).values * weights).sum() / normalizer
    return PCCDLossResult(
        loss=weighted_per_view.mean(),
        mean_kl=weighted_per_view.mean(),
        worst_view_kl=weighted_worst,
    )
