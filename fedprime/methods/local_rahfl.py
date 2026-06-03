from __future__ import annotations

import torch
import torch.nn.functional as F

from fedprime.models.factory import forward_logits
from fedprime.utils.env import add_vendor_paths


def _model_backbone(model):
    return model.module.backbone if hasattr(model, "module") else model.backbone


def train_local_augmix_dcl_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    lambda_jsd: float = 12.0,
    cl_module: str | None = "dcl",
    max_batches: int | None = None,
) -> float:
    add_vendor_paths()
    from loss import DCLLoss, SupConLoss

    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    losses = []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        labels = labels.to(device, non_blocking=True).long()
        if not isinstance(images, (tuple, list)):
            images = images.to(device, non_blocking=True)
            logits = forward_logits(model, images)
            loss = criterion(logits, labels)
        else:
            images = [img.to(device, non_blocking=True) for img in images]
            images_all = torch.cat([images[0], images[1], images[2]], dim=0)
            logits_all = forward_logits(model, images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))

            loss = criterion(logits_clean, labels)
            p_clean = F.softmax(logits_clean, dim=1)
            p_aug1 = F.softmax(logits_aug1, dim=1)
            p_aug2 = F.softmax(logits_aug2, dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0).log()
            jsd_loss = (
                F.kl_div(p_mixture, p_clean, reduction="batchmean")
                + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
                + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
            ) / 3.0
            loss = loss + lambda_jsd * jsd_loss

            if cl_module == "supcon":
                images_cont = torch.cat([images[0], images[1]], dim=0)
                features = _model_backbone(model)(images_cont)
                features = F.normalize(features.view(features.size(0), -1), dim=1)
                fclean, fstrong = torch.split(features, images[0].size(0))
                features = torch.cat([fclean.unsqueeze(1), fstrong.unsqueeze(1)], dim=1)
                loss = loss + SupConLoss(temperature=0.2, device=device)(features, labels)
            elif cl_module == "dcl":
                images_cont = torch.cat([images[0], images[1], images[3]], dim=0)
                features = _model_backbone(model)(images_cont)
                features = F.normalize(features.view(features.size(0), -1), dim=1)
                fclean, fstrong, fweak = torch.split(features, images[0].size(0))
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

