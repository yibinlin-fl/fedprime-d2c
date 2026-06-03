from __future__ import annotations

import torch
import torch.nn.functional as F

from fedprime.models.factory import forward_logits
from fedprime.utils.env import add_vendor_paths


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
        logits_all = forward_logits(model, prime_views)
        logits_clean, logits_prime1, logits_prime2 = torch.split(logits_all, base_images.size(0))

        loss = criterion(logits_clean, labels)
        loss = loss + lambda_jsd * jsd_loss_from_logits(logits_clean, logits_prime1, logits_prime2)

        if cl_module == "supcon":
            images_cont = torch.cat([prime_views[:base_images.size(0)], prime_views[base_images.size(0):2 * base_images.size(0)]], dim=0)
            features = _model_backbone(model)(images_cont)
            features = F.normalize(features.view(features.size(0), -1), dim=1)
            fclean, fstrong = torch.split(features, base_images.size(0))
            features = torch.cat([fclean.unsqueeze(1), fstrong.unsqueeze(1)], dim=1)
            loss = loss + SupConLoss(temperature=0.2, device=device)(features, labels)
        elif cl_module == "dcl":
            clean_norm = prime_views[:base_images.size(0)]
            strong_norm = prime_views[base_images.size(0):2 * base_images.size(0)]
            weak_norm = normalizer(weak_images)
            images_cont = torch.cat([clean_norm, strong_norm, weak_norm], dim=0)
            features = _model_backbone(model)(images_cont)
            features = F.normalize(features.view(features.size(0), -1), dim=1)
            fclean, fstrong, fweak = torch.split(features, base_images.size(0))
            loss = loss + DCLLoss(
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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return sum(losses) / max(len(losses), 1)
