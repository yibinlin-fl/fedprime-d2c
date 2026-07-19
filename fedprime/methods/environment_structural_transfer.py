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


def finalize_pair_qualified_client_relations(
    accumulator: dict[str, torch.Tensor],
    *,
    min_group_support: int,
    min_competing_class_support: int,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Finalize EBST-v2 statistics without exposing exact class counts."""

    state = finalize_client_relations(
        accumulator,
        min_group_support=min_group_support,
        eps=eps,
    )
    counts = state.pop("count")
    state["competing_class_support"] = (
        counts.sum(dim=1) >= int(min_competing_class_support)
    )
    return state


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


def _row_normalized_variance(
    variance: torch.Tensor,
    valid: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    num_classes = variance.shape[0]
    off_diagonal = ~torch.eye(num_classes, dtype=torch.bool, device=variance.device)
    mask = valid.bool() & off_diagonal
    row_scale = (
        variance.masked_fill(~mask, 0.0).sum(dim=1, keepdim=True)
        / mask.sum(dim=1, keepdim=True).clamp_min(1)
    )
    normalized = variance / row_scale.clamp_min(eps)
    return normalized.masked_fill(~mask, 0.0)


def aggregate_leave_one_out_pair_relations(
    client_states: dict[int, dict[str, torch.Tensor]],
    *,
    min_source_clients: int,
    use_stability_gate: bool,
    variance_temperature: float,
    eps: float = 1.0e-6,
) -> dict[str, object]:
    """Build recipient-specific relation teachers from eligible source clients.

    A source contributes to ``c-vs-j`` only when it has enough observations for
    class ``c`` in the current environment and enough total observations for
    competing class ``j``. The recipient is excluded from its own teacher.
    """

    if len(client_states) < 2:
        raise ValueError("leave-one-out aggregation requires at least two clients")
    if int(min_source_clients) < 1:
        raise ValueError("min_source_clients must be positive")

    client_ids = sorted(client_states)
    first = client_states[client_ids[0]]
    relation_shape = first["relations"].shape
    num_classes, num_environments, competing_classes = relation_shape
    if competing_classes != num_classes:
        raise ValueError("relation tensors must have shape [class, environment, class]")

    relations = []
    pair_supports = []
    off_diagonal = ~torch.eye(num_classes, dtype=torch.bool)
    for client_id in client_ids:
        state = client_states[client_id]
        relation = state["relations"].float()
        support = state["support"].bool()
        competing_support = state.get("competing_class_support")
        if competing_support is None:
            raise ValueError("EBST-v2 requires a client-side competing-class support mask")
        competing_support = competing_support.bool()
        if relation.shape != relation_shape or support.shape != relation_shape[:2]:
            raise ValueError("all client relation tensors must have matching shapes")
        if competing_support.shape != (num_classes,):
            raise ValueError("competing-class support masks must have shape [class]")
        pair_support = support.unsqueeze(2) & competing_support.view(1, 1, num_classes)
        pair_support = pair_support & off_diagonal.unsqueeze(1)
        relations.append(relation)
        pair_supports.append(pair_support)

    relation_stack = torch.stack(relations, dim=0)
    support_stack = torch.stack(pair_supports, dim=0)
    recipient_states: dict[int, dict[str, torch.Tensor | float]] = {}
    summary_values = {
        "valid_environment_fraction": [],
        "valid_pair_fraction": [],
        "mean_source_count": [],
        "mean_gate": [],
    }

    for recipient_index, recipient_id in enumerate(client_ids):
        source_selector = torch.ones(len(client_ids), dtype=torch.bool)
        source_selector[recipient_index] = False
        source_relations = relation_stack[source_selector]
        source_support = support_stack[source_selector]
        source_count = source_support.float().sum(dim=0)
        environment_valid = source_count >= int(min_source_clients)
        environment_consensus = (
            source_relations * source_support.float()
        ).sum(dim=0) / source_count.clamp_min(1.0)

        source_centered = source_relations - environment_consensus.unsqueeze(0)
        source_variance = (
            source_centered.square() * source_support.float()
        ).sum(dim=0) / source_count.clamp_min(1.0)

        valid_environment_count = environment_valid.sum(dim=1)
        global_relation = (
            environment_consensus * environment_valid.float()
        ).sum(dim=1) / valid_environment_count.clamp_min(1.0)
        global_valid = environment_valid.any(dim=1) & off_diagonal

        environment_centered = environment_consensus - global_relation.unsqueeze(1)
        environment_variance = (
            environment_centered.square() * environment_valid.float()
        ).sum(dim=1) / valid_environment_count.clamp_min(1.0)
        client_variance = (
            source_variance * environment_valid.float()
        ).sum(dim=1) / valid_environment_count.clamp_min(1.0)

        if use_stability_gate:
            temperature = max(float(variance_temperature), eps)
            environment_gate = torch.exp(
                -_row_normalized_variance(environment_variance, global_valid, eps=eps) / temperature
            )
            source_gate = torch.exp(
                -_row_normalized_variance(client_variance, global_valid, eps=eps) / temperature
            )
            gate = environment_gate * source_gate
        else:
            gate = torch.ones_like(global_relation)
        gate = gate * global_valid.float()
        global_relation = global_relation * global_valid.float()

        valid_environment_mask = environment_valid & off_diagonal.unsqueeze(1)
        valid_pair_mask = global_valid & off_diagonal
        valid_environment_fraction = float(valid_environment_mask.float().mean().item())
        valid_pair_fraction = float(valid_pair_mask.float().mean().item())
        mean_source_count = float(
            source_count[valid_environment_mask].mean().item()
            if bool(valid_environment_mask.any())
            else 0.0
        )
        mean_gate = float(gate[valid_pair_mask].mean().item() if bool(valid_pair_mask.any()) else 0.0)
        recipient_states[recipient_id] = {
            "global_relation": global_relation,
            "gate": gate,
            "global_valid": global_valid,
            "environment_valid": environment_valid,
            "source_count": source_count,
            "environment_variance": environment_variance,
            "client_variance": client_variance,
            "valid_environment_fraction": valid_environment_fraction,
            "valid_pair_fraction": valid_pair_fraction,
            "mean_source_count": mean_source_count,
            "mean_gate": mean_gate,
        }
        for name, value in (
            ("valid_environment_fraction", valid_environment_fraction),
            ("valid_pair_fraction", valid_pair_fraction),
            ("mean_source_count", mean_source_count),
            ("mean_gate", mean_gate),
        ):
            summary_values[name].append(value)

    diagnostics = {
        name: sum(values) / max(len(values), 1)
        for name, values in summary_values.items()
    }
    return {"recipients": recipient_states, **diagnostics}


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
    valid = valid_classes.to(logits.device).bool()
    if valid.ndim == 1:
        sample_valid = valid[labels]
        weights = weights * sample_valid.unsqueeze(1).float()
    elif valid.ndim == 2:
        pair_valid = valid[labels]
        weights = weights * pair_valid.float()
        sample_valid = pair_valid.any(dim=1)
    else:
        raise ValueError("valid_classes must have shape [class] or [class, class]")
    element_loss = F.huber_loss(normalized, target, reduction="none", delta=float(huber_delta))
    denominator = weights.sum().clamp_min(eps)
    loss = (element_loss * weights).sum() / denominator
    return loss, {
        "active_samples": sample_valid.float().sum(),
        "active_weight": weights.mean(),
    }
