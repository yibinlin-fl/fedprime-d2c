from __future__ import annotations

import torch
import torch.nn.functional as F

from fedprime.methods.balanced_environment_risk import balanced_environment_risk
from fedprime.methods.conditional_dependence import (
    FrozenRandomProjector,
    normalized_conditional_cross_covariance,
)
from fedprime.methods.environment_structural_transfer import (
    ebst_alignment_loss,
    update_relation_accumulator,
)
from fedprime.methods.safe_communication_projection import (
    add_projected_gradients,
    project_classifier_gradients_by_class,
    project_communication_gradients,
)
from fedprime.models.factory import forward_logits
from fedprime.utils.env import add_vendor_paths


def _model_backbone(model):
    return model.module.backbone if hasattr(model, "module") else model.backbone


def _model_head(model):
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "linear"):
        raise AttributeError("FedEASE EBST requires a `.linear` classifier head")
    return module.linear


def train_local_fedease_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    projector: FrozenRandomProjector,
    fedease_cfg: dict,
    *,
    class_environment_counts: torch.Tensor | None = None,
    lambda_jsd: float = 12.0,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    skip_nonfinite: bool = False,
    log_interval: int | None = None,
    context: str = "FedEASE local phase",
    diagnostics: dict[str, float] | None = None,
    relation_accumulator: dict[str, torch.Tensor] | None = None,
    global_relation_state: dict[str, torch.Tensor | float] | None = None,
    client_supported_classes: torch.Tensor | None = None,
) -> float:
    """Run one FedEASE local epoch with optional EBST and head-only SCP."""

    add_vendor_paths()
    from loss import DCLLoss

    ber_cfg = fedease_cfg.get("ber", {})
    cdep_cfg = fedease_cfg.get("cdep", {})
    ebst_cfg = fedease_cfg.get("ebst", fedease_cfg.get("structural_transfer", {}))
    scp_cfg = fedease_cfg.get("scp", fedease_cfg.get("safe_projection", {}))
    use_ber = bool(ber_cfg.get("enabled", True))
    use_cdep = bool(cdep_cfg.get("enabled", True))
    use_dcl = bool(fedease_cfg.get("preserve_dcl", True))
    num_environments = int(fedease_cfg.get("num_environments", 4))
    lambda_cdep = float(cdep_cfg.get("lambda", 0.05))
    lambda_ebst = float(ebst_cfg.get("lambda", ebst_cfg.get("lambda_max", 0.5)))

    model.train()
    metric_values: dict[str, list[float]] = {
        "clean_ce": [],
        "classification_loss": [],
        "ber_loss": [],
        "jsd_loss": [],
        "dcl_loss": [],
        "cdep_loss": [],
        "cdep_valid_classes": [],
        "cdep_mean_abs_covariance": [],
        "ber_valid_groups": [],
        "ebst_loss": [],
        "ebst_active_samples": [],
        "ebst_active_weight": [],
        "scp_gradient_dot": [],
        "scp_gradient_cosine": [],
        "scp_conflict": [],
        "scp_projection_norm_ratio": [],
    }
    total_losses = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if len(batch) == 3:
            images, labels, environment_ids = batch
            environment_features = None
        elif len(batch) == 5:
            images, labels, environment_ids, environment_features, _ = batch
        else:
            raise ValueError(
                "FedEASE requires (views, labels, environment_ids) or "
                "(views, labels, environment_ids, environment_features, confidence)"
            )
        if not isinstance(images, (tuple, list)) or len(images) < 4:
            raise ValueError("FedEASE requires clean/strong/second-strong/weak AugMix views")

        images = [image.to(device, non_blocking=True) for image in images]
        labels = labels.to(device, non_blocking=True).long()
        environment_ids = environment_ids.to(device, non_blocking=True).long()
        if environment_features is not None:
            environment_features = environment_features.to(device, non_blocking=True).float()

        batch_size = images[0].shape[0]
        logits_all = forward_logits(model, torch.cat(images[:3], dim=0))
        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, batch_size)
        if relation_accumulator is not None:
            update_relation_accumulator(
                relation_accumulator,
                logits_clean,
                labels,
                environment_ids,
            )
        sample_ce = F.cross_entropy(logits_clean, labels, reduction="none")
        clean_ce = sample_ce.mean()

        if use_ber:
            classification_loss, ber_stats = balanced_environment_risk(
                sample_ce,
                labels,
                environment_ids,
                group_counts=class_environment_counts,
                support_gamma=float(ber_cfg.get("support_gamma", 0.0)),
                count_cap=int(ber_cfg.get("count_cap", 32)),
                min_group_count=int(ber_cfg.get("min_group_count", 1)),
            )
            ber_loss = classification_loss
        else:
            classification_loss = clean_ce
            ber_loss = clean_ce.new_zeros(())
            ber_stats = {"valid_groups": clean_ce.new_zeros(())}

        p_clean = F.softmax(logits_clean, dim=1)
        p_aug1 = F.softmax(logits_aug1, dim=1)
        p_aug2 = F.softmax(logits_aug2, dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0).log()
        jsd_loss = (
            F.kl_div(p_mixture, p_clean, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
        ) / 3.0

        feature_views = _model_backbone(model)(torch.cat([images[0], images[1], images[3]], dim=0))
        feature_views = feature_views.reshape(feature_views.shape[0], -1)
        clean_feature, strong_feature, weak_feature = torch.split(feature_views, batch_size)

        if use_dcl:
            normalized = F.normalize(feature_views, dim=1)
            fclean, fstrong, fweak = torch.split(normalized, batch_size)
            dcl_loss = DCLLoss(
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
        else:
            dcl_loss = clean_ce.new_zeros(())

        if use_cdep:
            projected = projector(clean_feature)
            cdep_loss, cdep_stats = normalized_conditional_cross_covariance(
                projected,
                labels,
                environment_ids,
                num_environments=num_environments,
                environment_features=environment_features,
                min_class_count=int(cdep_cfg.get("min_class_count", 3)),
                eps=float(cdep_cfg.get("eps", 1.0e-5)),
            )
        else:
            cdep_loss = clean_ce.new_zeros(())
            cdep_stats = {
                "valid_classes": clean_ce.new_zeros(()),
                "mean_abs_covariance": clean_ce.new_zeros(()),
            }

        primary_loss = (
            classification_loss
            + float(lambda_jsd) * jsd_loss
            + dcl_loss
            + lambda_cdep * cdep_loss
        )
        ebst_loss = clean_ce.new_zeros(())
        ebst_stats = {
            "active_samples": clean_ce.new_zeros(()),
            "active_weight": clean_ce.new_zeros(()),
        }
        use_ebst_update = bool(ebst_cfg.get("enabled", False)) and global_relation_state is not None
        head = _model_head(model)
        if use_ebst_update:
            valid_classes = global_relation_state["global_valid"].bool()
            if client_supported_classes is not None:
                supported = client_supported_classes.cpu().bool()
                if valid_classes.ndim == 1:
                    valid_classes = valid_classes & supported
                else:
                    valid_classes = valid_classes & supported.unsqueeze(1)
            relation_logits = head(clean_feature.detach())
            ebst_loss, ebst_stats = ebst_alignment_loss(
                relation_logits,
                labels,
                global_relation_state["global_relation"],
                global_relation_state["gate"],
                valid_classes,
                huber_delta=float(ebst_cfg.get("huber_delta", 1.0)),
                eps=float(ebst_cfg.get("eps", 1.0e-6)),
            )
        loss = primary_loss + lambda_ebst * ebst_loss
        if not torch.isfinite(loss):
            message = f"{context}: non-finite loss at batch {batch_idx}"
            if skip_nonfinite:
                print(f"[warning] {message}; skipping batch", flush=True)
                continue
            raise FloatingPointError(message)

        optimizer.zero_grad(set_to_none=True)
        scp_stats = {
            "gradient_dot": 0.0,
            "gradient_cosine": 0.0,
            "conflict": 0.0,
            "projection_norm_ratio": 1.0,
        }
        if use_ebst_update and bool(ebst_stats["active_samples"].detach().item() > 0):
            primary_loss.backward(retain_graph=True)
            head_parameters = list(head.parameters())
            primary_gradients = [
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in head_parameters
            ]
            communication_gradients = list(torch.autograd.grad(
                ebst_loss,
                head_parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            ))
            scp_scope = str(scp_cfg.get("scope", "classifier_head")).lower()
            if scp_scope in {"classifier_class", "class", "class_row", "per_class"}:
                projected_gradients, scp_stats = project_classifier_gradients_by_class(
                    primary_gradients,
                    communication_gradients,
                    enabled=bool(scp_cfg.get("enabled", False)),
                    max_communication_norm_ratio=float(
                        scp_cfg.get("max_communication_norm_ratio", 1.0)
                    ),
                    eps=float(scp_cfg.get("eps", 1.0e-12)),
                )
            else:
                projected_gradients, scp_stats = project_communication_gradients(
                    primary_gradients,
                    communication_gradients,
                    enabled=bool(scp_cfg.get("enabled", False)),
                    eps=float(scp_cfg.get("eps", 1.0e-12)),
                )
            add_projected_gradients(head_parameters, projected_gradients, scale=lambda_ebst)
        else:
            primary_loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            message = f"{context}: non-finite gradient at batch {batch_idx}"
            if skip_nonfinite:
                print(f"[warning] {message}; skipping batch", flush=True)
                continue
            raise FloatingPointError(message)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()

        scalar_metrics = {
            "clean_ce": clean_ce,
            "classification_loss": classification_loss,
            "ber_loss": ber_loss,
            "jsd_loss": jsd_loss,
            "dcl_loss": dcl_loss,
            "cdep_loss": cdep_loss,
            "cdep_valid_classes": cdep_stats["valid_classes"],
            "cdep_mean_abs_covariance": cdep_stats["mean_abs_covariance"],
            "ber_valid_groups": ber_stats["valid_groups"],
            "ebst_loss": ebst_loss,
            "ebst_active_samples": ebst_stats["active_samples"],
            "ebst_active_weight": ebst_stats["active_weight"],
            "scp_gradient_dot": clean_ce.new_tensor(scp_stats["gradient_dot"]),
            "scp_gradient_cosine": clean_ce.new_tensor(scp_stats["gradient_cosine"]),
            "scp_conflict": clean_ce.new_tensor(scp_stats["conflict"]),
            "scp_projection_norm_ratio": clean_ce.new_tensor(scp_stats["projection_norm_ratio"]),
        }
        for name, value in scalar_metrics.items():
            metric_values[name].append(float(value.detach().cpu()))
        total_losses.append(float(loss.detach().cpu()))

        if log_interval and (batch_idx + 1) % int(log_interval) == 0:
            print(
                f"[heartbeat] {context} batch={batch_idx + 1} "
                f"loss={total_losses[-1]:.4f} cls={metric_values['classification_loss'][-1]:.4f} "
                f"cdep={metric_values['cdep_loss'][-1]:.4f} "
                f"ebst={metric_values['ebst_loss'][-1]:.4f} "
                f"conflict={metric_values['scp_conflict'][-1]:.0f}",
                flush=True,
            )

    if diagnostics is not None:
        for name, values in metric_values.items():
            diagnostics[name] = sum(values) / max(len(values), 1)
        diagnostics["total_loss"] = sum(total_losses) / max(len(total_losses), 1)
    return sum(total_losses) / max(len(total_losses), 1)
