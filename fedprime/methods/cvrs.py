from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fedprime.models.factory import forward_logits
from fedprime.utils.env import add_vendor_paths


EPS = 1.0e-12


def _model_backbone(model: torch.nn.Module) -> torch.nn.Module:
    return model.module.backbone if hasattr(model, "module") else model.backbone


def compute_rahfl_augmix_dcl_loss(
    model: torch.nn.Module,
    images: Sequence[torch.Tensor],
    labels: torch.Tensor,
    *,
    device: torch.device,
    lambda_jsd: float = 12.0,
) -> torch.Tensor:
    """Compute the unchanged RAHFL AugMix/JSD/DCL private objective.

    This deliberately mirrors ``train_local_augmix_dcl_epoch`` without owning
    the optimizer step, so M0 can interleave a public regularizer while keeping
    the original private objective intact.
    """

    if not isinstance(images, (tuple, list)) or len(images) < 4:
        raise ValueError("RAHFL AugMix/DCL expects clean, strong1, strong2 and weak views")
    add_vendor_paths()
    from loss import DCLLoss

    labels = labels.to(device=device, non_blocking=True).long()
    views = [image.to(device=device, non_blocking=True) for image in images]
    batch_size = int(views[0].shape[0])

    logits_all = forward_logits(model, torch.cat(views[:3], dim=0))
    logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, batch_size)
    loss = F.cross_entropy(logits_clean, labels)

    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1.0e-7, 1.0).log()
    jsd = (
        F.kl_div(p_mixture, p_clean, reduction="batchmean")
        + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
        + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
    ) / 3.0

    features = _model_backbone(model)(torch.cat([views[0], views[1], views[3]], dim=0))
    features = F.normalize(features.view(features.size(0), -1), dim=1)
    fclean, fstrong, fweak = torch.split(features, batch_size)
    dcl = DCLLoss(
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
    return loss + float(lambda_jsd) * jsd + dcl


def centered_class_response(base_logits: torch.Tensor, probe_logits: torch.Tensor) -> torch.Tensor:
    """Return P_C[z(A_q(u))-z(u)] for tensors shaped [Q,B,C]."""

    if probe_logits.ndim != 3 or base_logits.ndim != 2:
        raise ValueError("expected base [B,C] and probe [Q,B,C] logits")
    if tuple(probe_logits.shape[1:]) != tuple(base_logits.shape):
        raise ValueError("base/probe logit shapes do not match")
    response = probe_logits - base_logits.unsqueeze(0)
    return response - response.mean(dim=-1, keepdim=True)


def cvrs_statistics(
    base_logits: torch.Tensor,
    probe_logits: torch.Tensor,
    *,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-probe mu, energy and normalized persistent routing."""

    response = centered_class_response(base_logits, probe_logits)
    mean_response = response.mean(dim=1)
    energy = response.square().sum(dim=-1).mean(dim=1)
    routing = mean_response.square().sum(dim=-1) / (energy + float(eps))
    return mean_response, energy, routing


def cvrs_loss(
    base_logits: torch.Tensor,
    probe_logits: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Class-visible routing suppression with stop-gradient energy scale."""

    mean_response, energy, _routing = cvrs_statistics(
        base_logits, probe_logits, eps=eps
    )
    return (
        mean_response.square().sum(dim=-1) / (energy.detach() + float(eps))
    ).mean()


def pairwise_public_jsd_loss(
    base_logits: torch.Tensor,
    probe_logits: torch.Tensor,
    *,
    eps: float = 1.0e-7,
) -> torch.Tensor:
    """Mean two-view JSD between each carrier and each of its probe views."""

    if probe_logits.ndim != 3 or tuple(probe_logits.shape[1:]) != tuple(base_logits.shape):
        raise ValueError("expected base [B,C] and probe [Q,B,C] logits")
    p = F.softmax(base_logits, dim=-1).unsqueeze(0).expand_as(probe_logits)
    q = F.softmax(probe_logits, dim=-1)
    mixture = 0.5 * (p + q)
    log_mixture = torch.clamp(mixture, float(eps), 1.0).log()
    jsd = 0.5 * (
        F.kl_div(log_mixture, p, reduction="none").sum(dim=-1)
        + F.kl_div(log_mixture, q, reduction="none").sum(dim=-1)
    )
    return jsd.mean()


def gradient_l2_norm(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
) -> torch.Tensor:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss,
        trainable,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    squared = [gradient.detach().double().square().sum() for gradient in gradients if gradient is not None]
    if not squared:
        return loss.detach().new_zeros((), dtype=torch.float64)
    return torch.stack(squared).sum().sqrt()


def calibrated_regularizer_weight(
    task_gradient_norm: float,
    regularizer_gradient_norm: float,
    *,
    ratio: float = 0.1,
    eps: float = EPS,
) -> float:
    if task_gradient_norm < 0.0 or regularizer_gradient_norm < 0.0:
        raise ValueError("gradient norms must be non-negative")
    return float(ratio) * float(task_gradient_norm) / (float(regularizer_gradient_norm) + float(eps))


@dataclass
class ProbeSchedule:
    bank_size: int = 64
    probes_per_update: int = 4
    seed: int = 20260905

    def __post_init__(self) -> None:
        if self.bank_size <= 0 or self.probes_per_update <= 0:
            raise ValueError("probe schedule sizes must be positive")
        if self.bank_size % self.probes_per_update:
            raise ValueError("bank_size must be divisible by probes_per_update")
        self._rng = np.random.default_rng(int(self.seed))
        self._cycle = -1
        self._offset = self.bank_size
        self._permutation = np.empty(0, dtype=np.int64)

    def next_ids(self) -> np.ndarray:
        if self._offset >= self.bank_size:
            self._permutation = self._rng.permutation(self.bank_size).astype(np.int64)
            self._offset = 0
            self._cycle += 1
        stop = self._offset + self.probes_per_update
        values = self._permutation[self._offset:stop].copy()
        self._offset = stop
        return values

    @property
    def cycle(self) -> int:
        return int(self._cycle)


__all__ = [
    "EPS",
    "ProbeSchedule",
    "calibrated_regularizer_weight",
    "centered_class_response",
    "compute_rahfl_augmix_dcl_loss",
    "cvrs_loss",
    "cvrs_statistics",
    "gradient_l2_norm",
    "pairwise_public_jsd_loss",
]
