from __future__ import annotations

import torch
import torch.nn.functional as F

from fedprime.methods.nir_dcl import NIRDCLFeatureQueue, NIRDCLLoss
from fedprime.methods.sara import SARALoss
from fedprime.models.factory import forward_logits
from fedprime.communication.baselines import symmetric_cross_entropy
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
    num_classes: int = 10,
    nir_dcl_cfg: dict | None = None,
    sara_cfg: dict | None = None,
    feature_queue: NIRDCLFeatureQueue | None = None,
    client_class_counts: torch.Tensor | None = None,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    skip_nonfinite: bool = False,
    log_interval: int | None = None,
    context: str = "RAHFL local phase",
    communication_loss_fn=None,
    batch_trace_fn=None,
) -> float:
    add_vendor_paths()
    from loss import DCLLoss, SupConLoss

    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    nir_dcl_cfg = nir_dcl_cfg or {}
    sara_cfg = sara_cfg or {}
    losses = []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if batch_trace_fn is not None:
            batch_trace_fn(batch_idx=batch_idx, images=images, labels=labels)

        pending_queue_update = None
        labels = labels.to(device, non_blocking=True).long()
        if not isinstance(images, (tuple, list)):
            images = images.to(device, non_blocking=True)
            logits = forward_logits(model, images)
            loss = (
                symmetric_cross_entropy(logits, labels, num_classes=num_classes)
                if cl_module == "rhfl_sce"
                else criterion(logits, labels)
            )
        else:
            images = [img.to(device, non_blocking=True) for img in images]
            images_all = torch.cat([images[0], images[1], images[2]], dim=0)
            logits_all = forward_logits(model, images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))

            loss = (
                symmetric_cross_entropy(logits_clean, labels, num_classes=num_classes)
                if cl_module == "rhfl_sce"
                else criterion(logits_clean, labels)
            )
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
            elif cl_module == "nir_dcl":
                images_cont = torch.cat([images[0], images[1], images[3]], dim=0)
                features = _model_backbone(model)(images_cont)
                features = F.normalize(features.view(features.size(0), -1), dim=1)
                fclean, fstrong, fweak = torch.split(features, images[0].size(0))
                nir_loss, _ = NIRDCLLoss(
                    num_classes=num_classes,
                    temperature=float(nir_dcl_cfg.get("temperature", 0.2)),
                    relation_temperature=float(nir_dcl_cfg.get("relation_temperature", 0.2)),
                    beta=float(nir_dcl_cfg.get("beta", 1.0)),
                    reliability_tau=float(nir_dcl_cfg.get("reliability_tau", 1.0)),
                    reliability_min=float(nir_dcl_cfg.get("reliability_min", 0.05)),
                    use_class_balance=bool(nir_dcl_cfg.get("use_class_balance", True)),
                    use_queue=bool(nir_dcl_cfg.get("use_queue", True)),
                )(
                    original_feature=fclean,
                    weak_feature=fweak,
                    strong_feature=fstrong,
                    labels=labels,
                    strong_logits=logits_aug1,
                    feature_queue=feature_queue,
                )
                loss = loss + float(nir_dcl_cfg.get("lambda_nir", 1.0)) * nir_loss
                pending_queue_update = (fweak.detach(), labels.detach())
            elif cl_module in {"sara", "sara_cl"}:
                images_cont = torch.cat([images[0], images[1], images[3]], dim=0)
                features = _model_backbone(model)(images_cont)
                features = F.normalize(features.view(features.size(0), -1), dim=1)
                fclean, fstrong, fweak = torch.split(features, images[0].size(0))
                sara_loss, _ = SARALoss(
                    num_classes=num_classes,
                    temperature=float(sara_cfg.get("temperature", 0.2)),
                    relation_temperature=float(sara_cfg.get("relation_temperature", 0.2)),
                    beta=float(sara_cfg.get("beta", 1.0)),
                    class_weight_min=float(sara_cfg.get("class_weight_min", 0.75)),
                    class_weight_max=float(sara_cfg.get("class_weight_max", 1.5)),
                    class_weight_power=float(sara_cfg.get("class_weight_power", 0.5)),
                    reliability_tau=float(sara_cfg.get("reliability_tau", 1.0)),
                    reliability_min=float(sara_cfg.get("reliability_min", 0.05)),
                    use_class_calibration=bool(sara_cfg.get("use_class_calibration", True)),
                    use_view_reliability=bool(sara_cfg.get("use_view_reliability", True)),
                    use_relation_alignment=bool(sara_cfg.get("use_relation_alignment", True)),
                )(
                    original_feature=fclean,
                    weak_feature=fweak,
                    strong_feature=fstrong,
                    labels=labels,
                    strong_logits=logits_aug1,
                    class_counts=client_class_counts,
                )
                loss = loss + float(sara_cfg.get("lambda_sara", 1.0)) * sara_loss
            else:
                if cl_module not in (None, "none", "rhfl_sce"):
                    raise ValueError(f"Unknown contrastive module: {cl_module}")

            if communication_loss_fn is not None:
                loss = loss + communication_loss_fn(
                    receiver_logits=logits_clean,
                    clean_images=images[0],
                    labels=labels,
                )

        if not torch.isfinite(loss):
            message = f"{context}: non-finite loss at batch {batch_idx}: {float(loss.detach().cpu())}"
            if skip_nonfinite:
                print(f"[warning] {message}; skipping batch", flush=True)
                continue
            raise FloatingPointError(message)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads_finite = True
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grads_finite = False
                break
        if not grads_finite:
            optimizer.zero_grad(set_to_none=True)
            message = f"{context}: non-finite gradient at batch {batch_idx}"
            if skip_nonfinite:
                print(f"[warning] {message}; skipping batch", flush=True)
                continue
            raise FloatingPointError(message)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        if (
            isinstance(images, (tuple, list))
            and cl_module == "nir_dcl"
            and feature_queue is not None
            and pending_queue_update is not None
        ):
            queue_features, queue_labels = pending_queue_update
            feature_queue.enqueue(queue_features, queue_labels)
        losses.append(float(loss.detach().cpu()))
        if log_interval and (batch_idx + 1) % int(log_interval) == 0:
            print(
                f"[heartbeat] {context} batch={batch_idx + 1} "
                f"loss={float(loss.detach().cpu()):.4f}",
                flush=True,
            )

    return sum(losses) / max(len(losses), 1)
