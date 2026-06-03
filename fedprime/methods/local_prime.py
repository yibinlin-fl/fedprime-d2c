from __future__ import annotations

import torch
import torch.nn.functional as F

from fedprime.models.factory import forward_logits


def jsd_loss_from_logits(logits_clean, logits_aug1, logits_aug2) -> torch.Tensor:
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0).log()
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
        logits_all = forward_logits(model, images_all)
        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images.size(0))

        loss = F.cross_entropy(logits_clean, labels)
        loss = loss + lambda_jsd * jsd_loss_from_logits(logits_clean, logits_aug1, logits_aug2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return sum(losses) / max(len(losses), 1)


def train_local_standard_epoch(
    model,
    loader,
    optimizer,
    normalizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    losses = []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = forward_logits(model, normalizer(images))
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)
