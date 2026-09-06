from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


EPS = 1.0e-8


def center_logits(values: torch.Tensor) -> torch.Tensor:
    """Apply P_C by removing the common translation across class logits."""

    if values.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    return values - values.mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class ClassResponseStats:
    active_classes: torch.Tensor
    class_counts: torch.Tensor
    class_means: torch.Tensor
    between_class: torch.Tensor
    balanced_energy: torch.Tensor
    normalized: torch.Tensor
    singleton_count: int
    skipped: bool


def compute_class_response_stats(
    response: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float = EPS,
) -> ClassResponseStats:
    """Compute the frozen class-balanced LCRE statistics for [Q,B,C] responses."""

    if response.ndim != 3:
        raise ValueError("response must have shape [Q,B,C]")
    if labels.ndim != 1 or labels.shape[0] != response.shape[1]:
        raise ValueError("labels must have shape [B]")
    labels = labels.to(device=response.device, dtype=torch.long)
    unique, counts = torch.unique(labels, sorted=True, return_counts=True)
    active = unique[counts >= 2]
    active_counts = counts[counts >= 2]
    singleton_count = int((counts == 1).sum().item())
    if active.numel() < 2:
        zero = response.sum(dim=(1, 2)) * 0.0
        empty = response.new_empty((response.shape[0], 0, response.shape[2]))
        return ClassResponseStats(
            active_classes=active,
            class_counts=active_counts,
            class_means=empty,
            between_class=zero,
            balanced_energy=zero,
            normalized=zero,
            singleton_count=singleton_count,
            skipped=True,
        )

    means = []
    energies = []
    for class_id in active:
        selected = response[:, labels == class_id, :]
        means.append(selected.mean(dim=1))
        energies.append(selected.square().sum(dim=-1).mean(dim=1))
    class_means = torch.stack(means, dim=1)
    class_energy = torch.stack(energies, dim=1)
    center = class_means.mean(dim=1, keepdim=True)
    between = (class_means - center).square().sum(dim=-1).mean(dim=1)
    balanced_energy = class_energy.mean(dim=1)
    normalized = between / (balanced_energy.detach() + float(eps))
    return ClassResponseStats(
        active_classes=active,
        class_counts=active_counts,
        class_means=class_means,
        between_class=between,
        balanced_energy=balanced_energy,
        normalized=normalized,
        singleton_count=singleton_count,
        skipped=False,
    )


def compute_lcre_loss(
    base_logits: torch.Tensor,
    probe_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float = EPS,
) -> tuple[torch.Tensor, ClassResponseStats]:
    """Return mean_q B_q/stopgrad(E_q_bal) and its auditable statistics."""

    if base_logits.ndim != 2 or probe_logits.ndim != 3:
        raise ValueError("expected base [B,C] and probe [Q,B,C] logits")
    if tuple(probe_logits.shape[1:]) != tuple(base_logits.shape):
        raise ValueError("probe logits must match base batch and class dimensions")
    response = center_logits(probe_logits - base_logits.unsqueeze(0))
    stats = compute_class_response_stats(response, labels, eps=eps)
    return stats.normalized.mean(), stats


@contextmanager
def freeze_bn_running_stats(model: torch.nn.Module) -> Iterator[None]:
    """Freeze BN buffers while retaining train-mode behavior and affine gradients."""

    modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    states = [(module, bool(module.track_running_stats)) for module in modules]
    try:
        for module, _state in states:
            module.track_running_stats = False
        yield
    finally:
        for module, state in states:
            module.track_running_stats = state


__all__ = [
    "EPS",
    "ClassResponseStats",
    "center_logits",
    "compute_class_response_stats",
    "compute_lcre_loss",
    "freeze_bn_running_stats",
]
