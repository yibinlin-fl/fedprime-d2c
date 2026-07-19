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
