from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from fedprime.engine.cle_generic_probe_gate import generic_probe_statistics


SurgeryArm = Literal["targeted", "direction_sham", "random_probe", "generic_invariance"]


@dataclass(frozen=True)
class ProbeSelection:
    selected_probe_ids: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    rho: np.ndarray
    energy: np.ndarray
    active: np.ndarray


@dataclass(frozen=True)
class SurgeryTrace:
    objective_before: list[float]
    objective: list[float]
    anchor_kl: list[float]
    accepted: list[bool]
    learning_rate: list[float]
    trust_region_events: list[dict[str, float | int | str]]


def class_vs_rest_evidence_torch(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable z_c-logsumexp(z_not_c) used by the frozen K0-B response."""

    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("logits must end in a class dimension of size at least two")
    values = []
    for class_id in range(logits.shape[-1]):
        other = torch.cat((logits[..., :class_id], logits[..., class_id + 1 :]), dim=-1)
        values.append(logits[..., class_id] - torch.logsumexp(other, dim=-1))
    return torch.stack(values, dim=-1)


def centered_response_from_logits(
    base_logits: torch.Tensor,
    probe_logits: torch.Tensor,
) -> torch.Tensor:
    if base_logits.ndim != 2 or probe_logits.ndim != 3:
        raise ValueError("base/probe logits must have [source,class] and [source,probe,class]")
    if probe_logits.shape[0] != base_logits.shape[0] or probe_logits.shape[-1] != base_logits.shape[-1]:
        raise ValueError("base/probe logits have incompatible shapes")
    delta = class_vs_rest_evidence_torch(probe_logits) - class_vs_rest_evidence_torch(
        base_logits
    )[:, None, :]
    return delta - delta.mean(dim=-1, keepdim=True)


def centered_response_from_features(
    head: torch.nn.Linear,
    base_features: torch.Tensor,
    probe_features: torch.Tensor,
) -> torch.Tensor:
    if base_features.ndim != 2 or probe_features.ndim != 3:
        raise ValueError("features must have [source,feature] and [source,probe,feature]")
    if base_features.shape[0] != probe_features.shape[0] or base_features.shape[-1] != probe_features.shape[-1]:
        raise ValueError("base/probe features have incompatible shapes")
    base_logits = F.linear(base_features, head.weight, head.bias)
    probe_logits = F.linear(probe_features, head.weight, head.bias)
    return centered_response_from_logits(base_logits, probe_logits)


def select_high_risk_probes(centered_response: np.ndarray) -> ProbeSelection:
    """Reuse the exact K0-B active and top-20%-rho rules for one client and one bank."""

    values = np.asarray(centered_response, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("centered_response must have shape [source,probe,class]")
    statistics = generic_probe_statistics(values[None])
    active = statistics.active[0]
    active_ids = np.flatnonzero(active)
    count = max(1, int(np.ceil(0.20 * active_ids.size)))
    order = np.argsort(statistics.rho[0, active_ids], kind="stable")
    selected = active_ids[order[-count:]]
    selected = selected[np.argsort(statistics.rho[0, selected], kind="stable")[::-1]]
    mean_direction = 0.5 * (statistics.mu_a[0] + statistics.mu_b[0])
    directions = mean_direction[selected]
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("selected high-risk probe has a zero discover direction")
    directions = directions / norms
    selected_rho = statistics.rho[0, selected]
    if float(selected_rho.sum()) <= 0.0:
        raise ValueError("selected high-risk probes have zero total rho")
    weights = selected_rho / selected_rho.sum()
    return ProbeSelection(
        selected_probe_ids=selected.astype(np.int64),
        directions=directions.astype(np.float64),
        weights=weights.astype(np.float64),
        rho=statistics.rho[0].astype(np.float64),
        energy=statistics.energy[0].astype(np.float64),
        active=active.astype(bool),
    )


def make_direction_sham(
    directions: np.ndarray,
    *,
    seed: int,
    maximum_absolute_cosine: float = 0.20,
    maximum_attempts: int = 10000,
) -> tuple[np.ndarray, list[list[int]]]:
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("directions must have shape [probe,class]")
    rng = np.random.default_rng(int(seed))
    result = np.empty_like(values)
    permutations: list[list[int]] = []
    for probe_id, direction in enumerate(values):
        accepted = False
        for _ in range(int(maximum_attempts)):
            permutation = rng.permutation(direction.size)
            candidate = direction[permutation]
            cosine = float(
                np.dot(direction, candidate)
                / (np.linalg.norm(direction) * np.linalg.norm(candidate) + 1.0e-12)
            )
            if abs(cosine) <= float(maximum_absolute_cosine):
                result[probe_id] = candidate
                permutations.append(permutation.astype(int).tolist())
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"could not construct direction sham for probe {probe_id}")
    return result, permutations


def match_random_probes(selection: ProbeSelection) -> np.ndarray:
    selected = np.asarray(selection.selected_probe_ids, dtype=np.int64)
    candidates = np.flatnonzero(selection.active)
    candidates = candidates[~np.isin(candidates, selected)]
    if candidates.size < selected.size:
        raise ValueError("not enough non-high-risk active probes for one-to-one matching")
    remaining = candidates.tolist()
    matches: list[int] = []
    for probe_id in selected:
        target_log_energy = float(np.log(max(selection.energy[probe_id], 1.0e-12)))
        best = min(
            remaining,
            key=lambda candidate: (
                abs(float(np.log(max(selection.energy[candidate], 1.0e-12))) - target_log_energy),
                int(candidate),
            ),
        )
        matches.append(int(best))
        remaining.remove(best)
    return np.asarray(matches, dtype=np.int64)


def directional_moment_loss(response: torch.Tensor, directions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if response.ndim != 3:
        raise ValueError("response must have shape [source,probe,class]")
    mean_response = response.mean(dim=0)
    alignment = torch.sum(mean_response * directions, dim=-1)
    return torch.sum(weights * alignment.square())


def generic_invariance_loss(response: torch.Tensor) -> torch.Tensor:
    if response.ndim != 3:
        raise ValueError("response must have shape [source,probe,class]")
    return response.square().sum(dim=-1).mean()


def anchor_kl(
    reference_logits: torch.Tensor,
    current_logits: torch.Tensor,
) -> torch.Tensor:
    reference_probability = torch.softmax(reference_logits.detach(), dim=-1)
    return torch.sum(
        reference_probability
        * (torch.log_softmax(reference_logits.detach(), dim=-1) - torch.log_softmax(current_logits, dim=-1)),
        dim=-1,
    ).mean()


def clone_linear_head(head: torch.nn.Linear) -> torch.nn.Linear:
    clone = torch.nn.Linear(
        head.in_features,
        head.out_features,
        bias=head.bias is not None,
        device=head.weight.device,
        dtype=head.weight.dtype,
    )
    clone.load_state_dict(head.state_dict())
    return clone


def run_head_surgery(
    initial_head: torch.nn.Linear,
    base_features: torch.Tensor,
    probe_features: torch.Tensor,
    *,
    arm: SurgeryArm,
    directions: np.ndarray | None,
    weights: np.ndarray | None,
    learning_rate: float,
    steps: int,
    anchor_limit: float = 0.02,
    optimizer_name: str = "adam",
    backtracking_factor: float = 0.5,
    maximum_backtracks: int = 12,
) -> tuple[torch.nn.Linear, SurgeryTrace]:
    """Optimize the exact full-carrier head objective under a deterministic KL trust region."""

    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if arm not in ("targeted", "direction_sham", "random_probe", "generic_invariance"):
        raise ValueError(f"unsupported surgery arm: {arm}")
    head = clone_linear_head(initial_head)
    reference_logits = F.linear(base_features, initial_head.weight, initial_head.bias).detach()
    direction_tensor = None if directions is None else torch.as_tensor(
        directions, device=base_features.device, dtype=base_features.dtype
    )
    weight_tensor = None if weights is None else torch.as_tensor(
        weights, device=base_features.device, dtype=base_features.dtype
    )
    if arm != "generic_invariance" and (direction_tensor is None or weight_tensor is None):
        raise ValueError("directional surgery requires directions and weights")

    def make_optimizer(lr: float) -> torch.optim.Optimizer:
        if optimizer_name == "adam":
            return torch.optim.Adam(head.parameters(), lr=float(lr))
        if optimizer_name == "sgd":
            return torch.optim.SGD(head.parameters(), lr=float(lr))
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    current_lr = float(learning_rate)
    optimizer = make_optimizer(current_lr)
    objective_before_trace: list[float] = []
    objective_trace: list[float] = []
    anchor_trace: list[float] = []
    accepted_trace: list[bool] = []
    lr_trace: list[float] = []
    events: list[dict[str, float | int | str]] = []

    for step_id in range(int(steps)):
        before = {name: value.detach().clone() for name, value in head.state_dict().items()}
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        response = centered_response_from_features(head, base_features, probe_features)
        objective = (
            generic_invariance_loss(response)
            if arm == "generic_invariance"
            else directional_moment_loss(response, direction_tensor, weight_tensor)
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("non-finite surgery objective")
        objective_before = float(objective.detach().cpu())
        objective.backward()
        optimizer.step()
        current_logits = F.linear(base_features, head.weight, head.bias)
        current_anchor = anchor_kl(reference_logits, current_logits)
        accepted = bool(torch.isfinite(current_anchor) and current_anchor <= float(anchor_limit))
        if not accepted:
            head.load_state_dict(before)
            current_lr *= float(backtracking_factor)
            if len([event for event in events if event["event"] == "rollback"]) >= int(maximum_backtracks):
                raise RuntimeError("trust-region backtracking budget exhausted")
            events.append(
                {
                    "event": "rollback",
                    "step": int(step_id),
                    "anchor_kl": float(current_anchor.detach().cpu()),
                    "new_learning_rate": float(current_lr),
                }
            )
            optimizer = make_optimizer(current_lr)
            try:
                optimizer.load_state_dict(optimizer_state)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            except (ValueError, KeyError):
                optimizer = make_optimizer(current_lr)
            with torch.no_grad():
                response = centered_response_from_features(head, base_features, probe_features)
                objective = (
                    generic_invariance_loss(response)
                    if arm == "generic_invariance"
                    else directional_moment_loss(response, direction_tensor, weight_tensor)
                )
                current_logits = F.linear(base_features, head.weight, head.bias)
                current_anchor = anchor_kl(reference_logits, current_logits)
        else:
            with torch.no_grad():
                response = centered_response_from_features(head, base_features, probe_features)
                objective = (
                    generic_invariance_loss(response)
                    if arm == "generic_invariance"
                    else directional_moment_loss(response, direction_tensor, weight_tensor)
                )
        objective_before_trace.append(objective_before)
        objective_trace.append(float(objective.detach().cpu()))
        anchor_trace.append(float(current_anchor.detach().cpu()))
        accepted_trace.append(accepted)
        lr_trace.append(float(current_lr))

    return head, SurgeryTrace(
        objective_before=objective_before_trace,
        objective=objective_trace,
        anchor_kl=anchor_trace,
        accepted=accepted_trace,
        learning_rate=lr_trace,
        trust_region_events=events,
    )


__all__ = [
    "ProbeSelection",
    "SurgeryTrace",
    "anchor_kl",
    "centered_response_from_features",
    "centered_response_from_logits",
    "class_vs_rest_evidence_torch",
    "clone_linear_head",
    "directional_moment_loss",
    "generic_invariance_loss",
    "make_direction_sham",
    "match_random_probes",
    "run_head_surgery",
    "select_high_risk_probes",
]
