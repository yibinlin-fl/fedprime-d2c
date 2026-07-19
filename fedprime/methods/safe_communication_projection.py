from __future__ import annotations

import torch


def project_communication_gradients(
    primary_gradients: list[torch.Tensor | None],
    communication_gradients: list[torch.Tensor | None],
    *,
    enabled: bool,
    eps: float = 1.0e-12,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    if len(primary_gradients) != len(communication_gradients):
        raise ValueError("primary and communication gradient lists must have equal length")
    pairs = [
        (primary, communication)
        for primary, communication in zip(primary_gradients, communication_gradients)
        if primary is not None and communication is not None
    ]
    if not pairs:
        return list(communication_gradients), {
            "gradient_dot": 0.0,
            "gradient_cosine": 0.0,
            "conflict": 0.0,
            "projection_norm_ratio": 1.0,
        }

    dot = sum((primary * communication).sum() for primary, communication in pairs)
    primary_norm_sq = sum(primary.square().sum() for primary, _ in pairs)
    communication_norm_sq = sum(communication.square().sum() for _, communication in pairs)
    cosine = dot / (primary_norm_sq.sqrt() * communication_norm_sq.sqrt()).clamp_min(eps)
    conflict = bool(dot.detach().item() < 0.0)

    projected = []
    coefficient = dot / primary_norm_sq.clamp_min(eps) if enabled and conflict else dot.new_zeros(())
    for primary, communication in zip(primary_gradients, communication_gradients):
        if communication is None:
            projected.append(None)
        elif primary is None or not (enabled and conflict):
            projected.append(communication)
        else:
            projected.append(communication - coefficient * primary)

    projected_norm_sq = sum(
        gradient.square().sum() for gradient in projected if gradient is not None
    )
    ratio = projected_norm_sq.sqrt() / communication_norm_sq.sqrt().clamp_min(eps)
    return projected, {
        "gradient_dot": float(dot.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
        "conflict": float(conflict),
        "projection_norm_ratio": float(ratio.detach().cpu()),
    }


def project_classifier_gradients_by_class(
    primary_gradients: list[torch.Tensor | None],
    communication_gradients: list[torch.Tensor | None],
    *,
    enabled: bool,
    max_communication_norm_ratio: float = 1.0,
    eps: float = 1.0e-12,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Project and cap classifier gradients independently for every class row."""

    if len(primary_gradients) != len(communication_gradients):
        raise ValueError("primary and communication gradient lists must have equal length")
    usable = [
        (primary, communication)
        for primary, communication in zip(primary_gradients, communication_gradients)
        if primary is not None and communication is not None
    ]
    if not usable:
        return list(communication_gradients), {
            "gradient_dot": 0.0,
            "gradient_cosine": 0.0,
            "conflict": 0.0,
            "projection_norm_ratio": 1.0,
        }

    num_classes = int(usable[0][0].shape[0])
    if any(primary.shape[0] != num_classes or communication.shape[0] != num_classes for primary, communication in usable):
        raise ValueError("classifier gradients must share a class-aligned first dimension")
    projected = [None if gradient is None else gradient.detach().clone() for gradient in communication_gradients]
    original_norm_sq = sum(communication.square().sum() for _, communication in usable)
    total_dot = sum((primary * communication).sum() for primary, communication in usable)
    primary_norm_sq = sum(primary.square().sum() for primary, _ in usable)
    conflict_count = 0

    for class_id in range(num_classes):
        class_pairs = []
        for parameter_index, (primary, communication) in enumerate(
            zip(primary_gradients, communication_gradients)
        ):
            if primary is None or communication is None:
                continue
            class_pairs.append((parameter_index, primary[class_id], communication[class_id]))
        if not class_pairs:
            continue
        class_dot = sum((primary * communication).sum() for _, primary, communication in class_pairs)
        class_primary_norm_sq = sum(primary.square().sum() for _, primary, _ in class_pairs)
        conflict = bool(class_dot.detach().item() < 0.0)
        conflict_count += int(conflict)
        coefficient = (
            class_dot / class_primary_norm_sq.clamp_min(eps)
            if enabled and conflict
            else class_dot.new_zeros(())
        )
        class_projected = []
        for parameter_index, primary, communication in class_pairs:
            gradient = communication - coefficient * primary
            class_projected.append((parameter_index, gradient))

        projected_norm = sum(gradient.square().sum() for _, gradient in class_projected).sqrt()
        primary_norm = class_primary_norm_sq.sqrt()
        if enabled:
            max_norm = max(float(max_communication_norm_ratio), 0.0) * primary_norm
            scale = torch.clamp(max_norm / projected_norm.clamp_min(eps), max=1.0)
            if bool(primary_norm.detach().item() <= eps):
                scale = scale.new_zeros(())
        else:
            scale = projected_norm.new_ones(())
        for parameter_index, gradient in class_projected:
            projected[parameter_index][class_id] = gradient * scale

    projected_norm_sq = sum(
        gradient.square().sum() for gradient in projected if gradient is not None
    )
    cosine = total_dot / (primary_norm_sq.sqrt() * original_norm_sq.sqrt()).clamp_min(eps)
    return projected, {
        "gradient_dot": float(total_dot.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
        "conflict": float(conflict_count / max(num_classes, 1)),
        "projection_norm_ratio": float(
            (projected_norm_sq.sqrt() / original_norm_sq.sqrt().clamp_min(eps)).detach().cpu()
        ),
    }


def add_projected_gradients(
    parameters: list[torch.nn.Parameter],
    gradients: list[torch.Tensor | None],
    *,
    scale: float,
) -> None:
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            update = gradient.detach() * float(scale)
            if parameter.grad is None:
                parameter.grad = update.clone()
            else:
                parameter.grad.add_(update)
