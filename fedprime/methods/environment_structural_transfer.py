from __future__ import annotations

import torch
import torch.nn.functional as F


def new_relation_accumulator(
    num_classes: int,
    num_environments: int,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        "margin_sum": torch.zeros(num_classes, num_environments, num_classes, dtype=torch.float64, device=device),
        "count": torch.zeros(num_classes, num_environments, dtype=torch.float64, device=device),
    }


@torch.no_grad()
def update_relation_accumulator(
    accumulator: dict[str, torch.Tensor],
    logits: torch.Tensor,
    labels: torch.Tensor,
    environment_ids: torch.Tensor,
) -> None:
    logits = logits.detach().double()
    labels = labels.detach().long()
    environment_ids = environment_ids.detach().long()
    num_classes, num_environments = accumulator["count"].shape
    for class_id in torch.unique(labels, sorted=True):
        class_value = int(class_id.item())
        if class_value < 0 or class_value >= num_classes:
            continue
        for environment_id in torch.unique(environment_ids[labels == class_id], sorted=True):
            environment_value = int(environment_id.item())
            if environment_value < 0 or environment_value >= num_environments:
                continue
            mask = (labels == class_id) & (environment_ids == environment_id)
            margins = logits[mask, class_value].unsqueeze(1) - logits[mask]
            accumulator["margin_sum"][class_value, environment_value] += margins.sum(dim=0).to(
                accumulator["margin_sum"].device
            )
            accumulator["count"][class_value, environment_value] += float(mask.sum().item())


def normalize_margin_rows(rows: torch.Tensor, valid: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if rows.ndim != 3:
        raise ValueError("rows must have shape [class, environment, competing_class]")
    num_classes = rows.shape[0]
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=rows.device).unsqueeze(1)
    mask = mask.expand_as(rows)
    values = rows.masked_fill(~mask, 0.0)
    denominator = mask.sum(dim=2, keepdim=True).clamp_min(1)
    mean = values.sum(dim=2, keepdim=True) / denominator
    variance = ((values - mean).square() * mask).sum(dim=2, keepdim=True) / denominator
    normalized = (values - mean) / variance.sqrt().clamp_min(eps)
    normalized = normalized.masked_fill(~mask, 0.0)
    return torch.where(valid.unsqueeze(2), normalized, torch.zeros_like(normalized))


def finalize_client_relations(
    accumulator: dict[str, torch.Tensor],
    *,
    min_group_support: int,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    counts = accumulator["count"].cpu()
    valid = counts >= int(min_group_support)
    means = accumulator["margin_sum"].cpu() / counts.clamp_min(1.0).unsqueeze(2)
    normalized = normalize_margin_rows(means.float(), valid, eps=eps)
    return {
        "relations": normalized,
        "support": valid,
        "count": counts,
    }


def aggregate_environment_balanced_relations(
    client_states: dict[int, dict[str, torch.Tensor]],
    *,
    use_stability_gate: bool,
    variance_temperature: float,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor | float]:
    if not client_states:
        raise ValueError("client_states cannot be empty")
    first = next(iter(client_states.values()))
    relation_shape = first["relations"].shape
    environment_sum = torch.zeros(relation_shape, dtype=torch.float32)
    environment_support = torch.zeros(relation_shape[:2], dtype=torch.float32)
    for state in client_states.values():
        relations = state["relations"].float()
        support = state["support"].bool()
        if relations.shape != relation_shape or support.shape != relation_shape[:2]:
            raise ValueError("all client relation tensors must have matching shapes")
        environment_sum += relations * support.unsqueeze(2)
        environment_support += support.float()
    environment_valid = environment_support > 0
    environment_consensus = environment_sum / environment_support.clamp_min(1.0).unsqueeze(2)

    global_relation = (
        environment_consensus * environment_valid.unsqueeze(2)
    ).sum(dim=1) / environment_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    global_valid = environment_valid.any(dim=1)

    centered = environment_consensus - global_relation.unsqueeze(1)
    variance = (
        centered.square() * environment_valid.unsqueeze(2)
    ).sum(dim=1) / environment_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    num_classes = global_relation.shape[0]
    off_diagonal = ~torch.eye(num_classes, dtype=torch.bool)
    row_mean_variance = (
        variance.masked_fill(~off_diagonal, 0.0).sum(dim=1, keepdim=True)
        / off_diagonal.sum(dim=1, keepdim=True).clamp_min(1)
    )
    normalized_variance = variance / row_mean_variance.clamp_min(eps)
    if use_stability_gate:
        gate = torch.exp(-normalized_variance / max(float(variance_temperature), eps))
    else:
        gate = torch.ones_like(global_relation)
    gate = gate * off_diagonal.float() * global_valid.unsqueeze(1).float()
    global_relation = global_relation * off_diagonal.float()
    return {
        "global_relation": global_relation,
        "gate": gate,
        "environment_consensus": environment_consensus,
        "environment_valid": environment_valid,
        "variance": variance,
        "global_valid": global_valid,
        "valid_environment_fraction": float(environment_valid.float().mean().item()),
        "mean_gate": float(gate[off_diagonal].mean().item()),
    }


def ebst_alignment_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    global_relation: torch.Tensor,
    gate: torch.Tensor,
    valid_classes: torch.Tensor,
    *,
    huber_delta: float = 1.0,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    num_classes = logits.shape[1]
    labels = labels.long()
    true_logits = logits.gather(1, labels.unsqueeze(1))
    margins = true_logits - logits
    off_diagonal = ~F.one_hot(labels, num_classes=num_classes).bool()
    count = off_diagonal.sum(dim=1, keepdim=True).clamp_min(1)
    mean = margins.masked_fill(~off_diagonal, 0.0).sum(dim=1, keepdim=True) / count
    variance = (((margins - mean).square()) * off_diagonal).sum(dim=1, keepdim=True) / count
    normalized = (margins - mean) / variance.sqrt().clamp_min(eps)
    normalized = normalized.masked_fill(~off_diagonal, 0.0)

    target = global_relation.to(logits.device, logits.dtype)[labels]
    weights = gate.to(logits.device, logits.dtype)[labels] * off_diagonal.float()
    sample_valid = valid_classes.to(logits.device)[labels]
    weights = weights * sample_valid.unsqueeze(1).float()
    element_loss = F.huber_loss(normalized, target, reduction="none", delta=float(huber_delta))
    denominator = weights.sum().clamp_min(eps)
    loss = (element_loss * weights).sum() / denominator
    return loss, {
        "active_samples": sample_valid.float().sum(),
        "active_weight": weights.mean(),
    }
