from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fedprime.methods.local_prime import require_finite
from fedprime.models.factory import forward_logits


@dataclass
class PairExpertise:
    raw: torch.Tensor
    weighted: torch.Tensor
    counts: torch.Tensor


def normalize_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = logits - logits.mean(dim=-1, keepdim=True)
    norm = centered.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return centered / norm


def pair_margins(normalized_logits: torch.Tensor) -> torch.Tensor:
    return normalized_logits.unsqueeze(-1) - normalized_logits.unsqueeze(-2)


def softmin_views(margins: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    if margins.size(0) == 1:
        return margins.squeeze(0)
    tau = max(float(tau), 1e-6)
    return -tau * torch.logsumexp(-margins / tau, dim=0) + tau * np.log(margins.size(0))


def class_balanced_cbcl_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
    reliability_tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Class-balanced supervised contrastive loss over clean and PRIME views."""
    views, batch_size = features.shape[:2]
    flat_features = F.normalize(features.reshape(views * batch_size, -1), dim=1)
    flat_labels = labels.repeat(views)

    logits_view = logits.reshape(views, batch_size, -1)
    true_logits = logits_view.gather(2, labels.view(1, batch_size, 1).expand(views, -1, -1)).squeeze(-1)
    masked_logits = logits_view.masked_fill(
        F.one_hot(labels, logits_view.size(-1)).bool().view(1, batch_size, -1),
        float("-inf"),
    )
    margin = true_logits - masked_logits.max(dim=-1).values
    view_weights = torch.sigmoid(margin / max(float(reliability_tau), eps)).reshape(-1).detach()

    sim = torch.matmul(flat_features, flat_features.t()) / max(float(temperature), eps)
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, -1e9)
    positive_mask = flat_labels.unsqueeze(0).eq(flat_labels.unsqueeze(1)) & ~eye

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    per_anchor = torch.zeros_like(view_weights)
    per_anchor[valid_anchor] = -(
        log_prob[valid_anchor].masked_fill(~positive_mask[valid_anchor], 0.0).sum(dim=1)
        / positive_count[valid_anchor].clamp_min(1)
    )
    per_anchor = per_anchor * view_weights

    losses = []
    for class_id in labels.unique(sorted=True):
        class_mask = (flat_labels == class_id) & valid_anchor
        if class_mask.any():
            losses.append(per_anchor[class_mask].mean())
    if not losses:
        return flat_features.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def estimate_pair_expertise(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    prime_aug=None,
    normalizer=None,
    max_batches: int | None = None,
    softmin_tau: float = 0.5,
    expertise_tau: float = 0.5,
    support: str = "log",
    support_gamma: float = 20.0,
    eps: float = 1e-6,
) -> PairExpertise:
    model.eval()
    score_sum = torch.zeros(num_classes, num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if isinstance(images, (tuple, list)):
            images = images[0]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        if prime_aug is not None:
            views = prime_aug(images)
            logits_all = forward_logits(model, views)
            view_logits = torch.stack(torch.split(logits_all, images.size(0)), dim=0)
        else:
            if normalizer is None:
                raise ValueError("normalizer is required when prime_aug is not provided.")
            view_logits = forward_logits(model, normalizer(images)).unsqueeze(0)

        require_finite(view_logits, "pair-expertise logits", f"client expertise batch={batch_idx}")
        margins = pair_margins(normalize_logits(view_logits, eps=eps))
        robust_margins = softmin_views(margins, tau=softmin_tau)

        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            class_count = int(class_mask.sum().item())
            if class_count == 0:
                continue
            class_scores = torch.sigmoid(
                robust_margins[class_mask, int(class_id)] / max(float(expertise_tau), eps)
            )
            score_sum[int(class_id)] += class_scores.sum(dim=0)
            counts[int(class_id)] += class_count

    raw = torch.zeros_like(score_sum)
    valid = counts > 0
    raw[valid] = score_sum[valid] / counts[valid].view(-1, 1).clamp_min(1.0)
    raw.fill_diagonal_(0.0)

    if support == "ratio":
        support_weight = counts / (counts + float(support_gamma))
    elif support == "sqrt":
        support_weight = counts.sqrt()
    elif support == "none":
        support_weight = torch.ones_like(counts)
    else:
        support_weight = torch.log1p(counts)
    weighted = raw * support_weight.view(-1, 1)
    weighted.fill_diagonal_(0.0)
    return PairExpertise(raw=raw.detach(), weighted=weighted.detach(), counts=counts.detach())


def cpad_pair_bce_loss(
    student_logits: torch.Tensor,
    public_logits_all: torch.Tensor,
    expertise_weighted: torch.Tensor,
    student_id: int,
    temperature: float = 2.0,
    gate_tau: float = 0.5,
    eps: float = 1e-6,
    leave_one_out: bool = True,
    use_gate: bool = True,
    use_confidence: bool = True,
    use_agreement: bool = True,
    agreement_tau: float = 0.05,
) -> torch.Tensor:
    num_clients, batch_size, num_classes = public_logits_all.shape
    student_norm = normalize_logits(student_logits, eps=eps)
    all_norm = normalize_logits(public_logits_all, eps=eps)

    student_margin = pair_margins(student_norm)
    all_margins = pair_margins(all_norm)
    weights = expertise_weighted.to(student_logits.device).clone()
    if leave_one_out and num_clients > 1:
        weights[student_id].zero_()

    denom = weights.sum(dim=0).clamp_min(eps)
    teacher_margin = (all_margins * weights[:, None, :, :]).sum(dim=0) / denom[None, :, :]
    teacher_prob = torch.sigmoid(teacher_margin / max(float(temperature), eps)).detach()
    student_prob = torch.sigmoid(student_margin / max(float(temperature), eps))

    valid_edge = (~torch.eye(num_classes, dtype=torch.bool, device=student_logits.device)).float()
    valid_edge = valid_edge * (weights.sum(dim=0) > eps).float()

    beta = valid_edge[None, :, :].expand(batch_size, -1, -1)
    if use_gate:
        if leave_one_out and num_clients > 1:
            teacher_strength = weights.max(dim=0).values
        else:
            other = weights.clone()
            other[student_id].zero_()
            teacher_strength = other.max(dim=0).values
        student_strength = expertise_weighted[student_id].to(student_logits.device)
        gate = torch.sigmoid((teacher_strength - student_strength) / max(float(gate_tau), eps))
        beta = beta * gate[None, :, :]
    if use_confidence:
        beta = beta * (2.0 * torch.abs(teacher_prob - 0.5))
    if use_agreement and num_clients > 2:
        probs = torch.sigmoid(all_margins / max(float(temperature), eps))
        if leave_one_out:
            mask = torch.ones(num_clients, dtype=torch.bool, device=student_logits.device)
            mask[student_id] = False
            probs = probs[mask]
        variance = probs.var(dim=0, unbiased=False)
        beta = beta * torch.exp(-variance / max(float(agreement_tau), eps))

    loss_matrix = F.binary_cross_entropy(
        student_prob.clamp(min=eps, max=1.0 - eps),
        teacher_prob.clamp(min=eps, max=1.0 - eps),
        reduction="none",
    )
    weighted = loss_matrix * beta
    return weighted.sum() / beta.sum().clamp_min(eps)


def save_pair_expertise_snapshot(
    output_dir: str | Path,
    round_idx: int,
    expertise: torch.Tensor,
    expertise_raw: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    path = Path(output_dir) / "pair_expertise"
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / f"round_{round_idx:03d}.npz",
        expertise=expertise.detach().cpu().numpy(),
        expertise_raw=expertise_raw.detach().cpu().numpy(),
        counts=counts.detach().cpu().numpy(),
    )
