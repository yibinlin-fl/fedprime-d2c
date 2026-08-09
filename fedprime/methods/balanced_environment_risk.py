from __future__ import annotations

import torch


def soft_balanced_environment_risk(
    sample_losses: torch.Tensor,
    labels: torch.Tensor,
    environment_probabilities: torch.Tensor,
    *,
    group_counts: torch.Tensor | None = None,
    support_gamma: float = 0.0,
    count_cap: int = 32,
    min_group_count: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balance class/environment risks using fractional PEW responsibilities."""

    if sample_losses.ndim != 1 or labels.shape != sample_losses.shape:
        raise ValueError("sample_losses and labels must have shape [batch]")
    if environment_probabilities.ndim != 2 or environment_probabilities.shape[0] != len(labels):
        raise ValueError("environment_probabilities must have shape [batch, environments]")
    if not 0.0 <= support_gamma <= 1.0:
        raise ValueError("support_gamma must be in [0, 1]")
    if count_cap < 1 or min_group_count <= 0:
        raise ValueError("count_cap and min_group_count must be positive")

    labels = labels.long()
    responsibilities = environment_probabilities.to(
        device=sample_losses.device, dtype=sample_losses.dtype
    )
    if bool((responsibilities < 0).any()):
        raise ValueError("environment probabilities cannot be negative")
    responsibilities = responsibilities / responsibilities.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(responsibilities.dtype).eps
    )

    if group_counts is None:
        num_classes = int(labels.max().item()) + 1
        counts = sample_losses.new_zeros((num_classes, responsibilities.shape[1]))
        counts.index_add_(0, labels, responsibilities)
    else:
        counts = group_counts.to(device=sample_losses.device, dtype=sample_losses.dtype)
        if counts.ndim != 2 or counts.shape[1] != responsibilities.shape[1]:
            raise ValueError("group_counts must match [classes, environments]")
        if int(labels.max().item()) >= counts.shape[0] or bool((counts < 0).any()):
            raise ValueError("invalid group_counts for batch labels")

    valid = counts >= float(min_group_count)
    support = counts.clamp(max=float(count_cap)).pow(support_gamma)
    support = torch.where(valid, support, torch.zeros_like(support))
    environment_weights = support / support.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(support.dtype).eps
    )
    valid_classes = valid.any(dim=1)
    class_count = valid_classes.sum().clamp_min(1).to(sample_losses.dtype)
    objective_weights = environment_weights / counts.clamp_min(
        torch.finfo(counts.dtype).eps
    ) / class_count
    sample_weights = (objective_weights[labels] * responsibilities).sum(dim=1)
    dataset_size = counts.sum()
    loss = dataset_size * (sample_weights * sample_losses).mean()
    effective = environment_weights.square().sum(dim=1).clamp_min(
        torch.finfo(counts.dtype).eps
    ).reciprocal()
    return loss, {
        "valid_classes": valid_classes.sum().to(sample_losses.dtype),
        "valid_groups": valid.sum().to(sample_losses.dtype),
        "effective_groups_per_class": (
            effective[valid_classes].mean() if bool(valid_classes.any()) else sample_losses.new_zeros(())
        ),
    }


def balanced_environment_risk(
    sample_losses: torch.Tensor,
    labels: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    group_counts: torch.Tensor | None = None,
    support_gamma: float = 0.0,
    count_cap: int = 32,
    min_group_count: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average classification risk across valid environments inside each class.

    Classes are averaged uniformly. Within a class, environment weights are
    proportional to ``min(count, count_cap) ** support_gamma``. Therefore
    ``support_gamma=0`` gives exact environment balancing.
    """

    if sample_losses.ndim != 1:
        raise ValueError("sample_losses must have shape [batch]")
    if labels.shape != sample_losses.shape or environment_ids.shape != sample_losses.shape:
        raise ValueError("labels and environment_ids must match sample_losses")
    if not 0.0 <= support_gamma <= 1.0:
        raise ValueError("support_gamma must be in [0, 1]")
    if count_cap < 1 or min_group_count < 1:
        raise ValueError("count_cap and min_group_count must be positive")

    labels = labels.long()
    environment_ids = environment_ids.long()

    if group_counts is not None:
        if group_counts.ndim != 2:
            raise ValueError("group_counts must have shape [classes, environments]")
        counts = group_counts.to(device=sample_losses.device, dtype=sample_losses.dtype)
        if bool((counts < 0).any()):
            raise ValueError("group_counts cannot contain negative values")
        if int(labels.max().item()) >= counts.shape[0] or int(environment_ids.max().item()) >= counts.shape[1]:
            raise ValueError("batch labels/environment IDs exceed group_counts dimensions")

        valid = counts >= float(min_group_count)
        capped_support = counts.clamp(max=float(count_cap)).pow(support_gamma)
        capped_support = torch.where(valid, capped_support, torch.zeros_like(capped_support))
        environment_weights = capped_support / capped_support.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(counts.dtype).eps
        )
        valid_classes_mask = valid.any(dim=1)
        valid_class_count = valid_classes_mask.sum().clamp_min(1)
        objective_weights = environment_weights / counts.clamp_min(1.0)
        objective_weights = objective_weights / valid_class_count.to(counts.dtype)
        sample_weights = objective_weights[labels, environment_ids]

        dataset_size = counts.sum()
        loss = dataset_size * (sample_weights * sample_losses).mean()
        effective_groups_by_class = environment_weights.square().sum(dim=1).clamp_min(
            torch.finfo(counts.dtype).eps
        ).reciprocal()
        diagnostics = {
            "valid_classes": valid_classes_mask.sum().to(sample_losses.dtype),
            "valid_groups": valid.sum().to(sample_losses.dtype),
            "effective_groups_per_class": effective_groups_by_class[valid_classes_mask].mean()
            if bool(valid_classes_mask.any())
            else sample_losses.new_zeros(()),
        }
        return loss, diagnostics

    class_risks = []
    valid_group_count = 0
    effective_weight_squares = []

    for class_id in torch.unique(labels, sorted=True):
        class_mask = labels == class_id
        group_risks = []
        group_support = []
        for environment_id in torch.unique(environment_ids[class_mask], sorted=True):
            group_mask = class_mask & (environment_ids == environment_id)
            count = int(group_mask.sum().item())
            if count < min_group_count:
                continue
            group_risks.append(sample_losses[group_mask].mean())
            group_support.append(min(count, count_cap))

        if not group_risks:
            continue
        risk_tensor = torch.stack(group_risks)
        support = sample_losses.new_tensor(group_support)
        weights = support.pow(support_gamma)
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        class_risks.append((weights * risk_tensor).sum())
        valid_group_count += len(group_risks)
        effective_weight_squares.append(weights.square().sum())

    if not class_risks:
        loss = sample_losses.mean()
        valid_classes = sample_losses.new_zeros(())
        effective_groups = sample_losses.new_zeros(())
    else:
        loss = torch.stack(class_risks).mean()
        valid_classes = sample_losses.new_tensor(float(len(class_risks)))
        effective_groups = torch.stack(effective_weight_squares).reciprocal().mean()

    diagnostics = {
        "valid_classes": valid_classes,
        "valid_groups": sample_losses.new_tensor(float(valid_group_count)),
        "effective_groups_per_class": effective_groups,
    }
    return loss, diagnostics
