from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from fedprime.models.factory import forward_logits


def jtt_reweighted_loss(
    sample_losses: torch.Tensor,
    error_mask: torch.Tensor,
    *,
    upweight: float,
) -> torch.Tensor:
    """JTT stage-2 objective, equivalent to repeating stage-1 error examples."""

    if sample_losses.ndim != 1 or error_mask.shape != sample_losses.shape:
        raise ValueError("JTT losses and error mask must be aligned vectors")
    if float(upweight) < 1.0:
        raise ValueError("JTT upweight must be at least one")
    weights = torch.ones_like(sample_losses)
    weights = weights + (float(upweight) - 1.0) * error_mask.to(sample_losses.dtype)
    return (weights * sample_losses).sum() / weights.sum().clamp_min(1.0e-12)


def cvar_dro_loss(sample_losses: torch.Tensor, *, tail_fraction: float) -> torch.Tensor:
    """Empirical upper-tail CVaR, with no group or environment information."""

    if sample_losses.ndim != 1 or sample_losses.numel() == 0:
        raise ValueError("CVaR losses must be a non-empty vector")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("CVaR tail_fraction must be in (0, 1]")
    count = max(1, int(math.ceil(float(tail_fraction) * sample_losses.numel())))
    return torch.topk(sample_losses, k=count, largest=True).values.mean()


@dataclass
class GroupDROState:
    """Persistent exponentiated-gradient weights for class x environment groups."""

    weights: torch.Tensor

    @classmethod
    def uniform(cls, num_classes: int, num_environments: int, device: torch.device):
        size = int(num_classes) * int(num_environments)
        if size <= 0:
            raise ValueError("GroupDRO requires positive class and environment counts")
        return cls(torch.full((size,), 1.0 / size, device=device))


def group_dro_loss(
    sample_losses: torch.Tensor,
    labels: torch.Tensor,
    environments: torch.Tensor,
    state: GroupDROState,
    *,
    num_environments: int,
    step_size: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-environment GroupDRO with persistent exponentiated group weights."""

    if sample_losses.ndim != 1 or labels.shape != sample_losses.shape:
        raise ValueError("GroupDRO losses and labels must be aligned vectors")
    if environments.shape != sample_losses.shape:
        raise ValueError("GroupDRO environments must align with losses")
    if float(step_size) < 0.0:
        raise ValueError("GroupDRO step_size must be non-negative")
    group_ids = labels.long() * int(num_environments) + environments.long()
    if bool((group_ids < 0).any()) or bool((group_ids >= state.weights.numel()).any()):
        raise ValueError("GroupDRO group ID is out of range")

    observed = torch.unique(group_ids, sorted=True)
    group_losses = torch.stack([sample_losses[group_ids == group].mean() for group in observed])
    with torch.no_grad():
        state.weights[observed] *= torch.exp(float(step_size) * group_losses.detach())
        state.weights /= state.weights.sum().clamp_min(1.0e-12)
    observed_weights = state.weights[observed]
    objective = (observed_weights * group_losses).sum() / observed_weights.sum().clamp_min(1.0e-12)
    entropy = -(state.weights.clamp_min(1.0e-12) * state.weights.clamp_min(1.0e-12).log()).sum()
    return objective, {
        "observed_groups": torch.as_tensor(float(observed.numel()), device=sample_losses.device),
        "group_weight_max": state.weights.max(),
        "group_weight_entropy": entropy,
    }


def train_local_spurious_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    *,
    objective: str,
    objective_cfg: dict,
    group_dro_state: GroupDROState | None = None,
    num_classes: int = 10,
    num_environments: int = 6,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    skip_nonfinite: bool = False,
    log_interval: int | None = None,
    context: str = "spurious baseline local phase",
    diagnostics: dict[str, float] | None = None,
    batch_trace_fn=None,
) -> float:
    """Train one isolated JTT, CVaR-DRO, or GroupDRO local epoch."""

    normalized = str(objective).lower()
    if normalized not in {"jtt", "cvar_dro", "pew_groupdro"}:
        raise ValueError(f"Unknown spurious objective: {objective}")
    if normalized == "pew_groupdro" and group_dro_state is None:
        raise ValueError("PEW+GroupDRO requires persistent state")

    model.train()
    totals: list[float] = []
    clean_values: list[float] = []
    aux_values: dict[str, list[float]] = {}
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images, labels = batch[:2]
        if isinstance(images, (tuple, list)):
            raise ValueError("Spurious baselines require standard single-view inputs")
        if batch_trace_fn is not None:
            batch_trace_fn(batch_idx=batch_idx, images=images, labels=labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        sample_ce = F.cross_entropy(forward_logits(model, images), labels, reduction="none")
        clean_ce = sample_ce.mean()
        stats: dict[str, torch.Tensor] = {}
        if normalized == "jtt":
            if len(batch) < 3:
                raise ValueError("JTT stage 2 requires a frozen stage-1 error mask")
            error_mask = batch[2].to(device, non_blocking=True).bool()
            loss = jtt_reweighted_loss(
                sample_ce, error_mask, upweight=float(objective_cfg.get("upweight", 20.0))
            )
            stats["error_fraction"] = error_mask.float().mean()
        elif normalized == "cvar_dro":
            loss = cvar_dro_loss(
                sample_ce, tail_fraction=float(objective_cfg.get("tail_fraction", 0.2))
            )
            stats["tail_fraction"] = torch.as_tensor(
                float(objective_cfg.get("tail_fraction", 0.2)), device=device
            )
        else:
            if len(batch) < 3:
                raise ValueError("PEW+GroupDRO requires frozen PEW environment IDs")
            environments = batch[2].to(device, non_blocking=True).long()
            assert group_dro_state is not None
            loss, stats = group_dro_loss(
                sample_ce,
                labels,
                environments,
                group_dro_state,
                num_environments=num_environments,
                step_size=float(objective_cfg.get("step_size", 0.01)),
            )
        if not torch.isfinite(loss):
            if skip_nonfinite:
                continue
            raise FloatingPointError(f"{context}: non-finite loss at batch {batch_idx}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite:
            optimizer.zero_grad(set_to_none=True)
            if skip_nonfinite:
                continue
            raise FloatingPointError(f"{context}: non-finite gradient at batch {batch_idx}")
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        totals.append(float(loss.detach().cpu()))
        clean_values.append(float(clean_ce.detach().cpu()))
        for name, value in stats.items():
            aux_values.setdefault(name, []).append(float(value.detach().cpu()))
        if log_interval and (batch_idx + 1) % int(log_interval) == 0:
            print(f"[heartbeat] {context} batch={batch_idx + 1} loss={totals[-1]:.4f}", flush=True)
    if diagnostics is not None:
        diagnostics["classification_loss"] = sum(totals) / max(len(totals), 1)
        diagnostics["clean_ce"] = sum(clean_values) / max(len(clean_values), 1)
        diagnostics.update(
            {name: sum(values) / max(len(values), 1) for name, values in aux_values.items()}
        )
    return sum(totals) / max(len(totals), 1)
