from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fedprime.data.fedfalsify import FedFalsifyClientSplit
from fedprime.methods.fedfalsify.evidence import compute_paired_advantage
from fedprime.methods.fedfalsify.transfer import (
    conservative_margin_transfer_loss,
    gradient_cosine_from_losses,
)
from fedprime.models.factory import forward_logits


def model_head(model: torch.nn.Module) -> torch.nn.Module:
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "linear"):
        raise AttributeError(f"{type(module).__name__} has no '.linear' classifier head")
    return module.linear


@dataclass(frozen=True)
class ClassRoute:
    receiver_id: int
    class_id: int
    source_id: int
    tau: float
    fra_strength: float
    fra_advantage: float
    fit_count: int
    audit_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateAudit:
    receiver_id: int
    class_id: int
    source_id: int
    tau: float
    fra_strength: float
    fra_advantage: float
    score: float
    selected: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def freeze_peer_snapshots(
    models: dict[int, torch.nn.Module],
    *,
    device: torch.device,
) -> dict[int, torch.nn.Module]:
    snapshots = {}
    for client_id, model in models.items():
        snapshot = copy.deepcopy(model).to(device)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        snapshots[int(client_id)] = snapshot
    return snapshots


def _stack_dataset_items(
    dataset,
    indices: np.ndarray,
    *,
    max_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = np.asarray(indices, dtype=np.int64)[: int(max_samples)]
    if selected.size == 0:
        raise ValueError("Cannot stack an empty index selection")
    items = [dataset[int(index)] for index in selected]
    images = torch.stack([item[0] for item in items]).to(device)
    labels = torch.as_tensor([int(item[1]) for item in items], device=device)
    return images, labels


class FedFalsifyTransferPlan:
    """Frozen per-round class routes used during receiver-local fitting."""

    def __init__(
        self,
        *,
        snapshots: dict[int, torch.nn.Module],
        routes: dict[int, dict[int, ClassRoute]],
        lambda_cmt: float,
        margin_clip: float,
        source_correct_only: bool,
    ):
        self.snapshots = snapshots
        self.routes = routes
        self.lambda_cmt = float(lambda_cmt)
        self.margin_clip = float(margin_clip)
        self.source_correct_only = bool(source_correct_only)
        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self.loss_sum = 0.0
        self.calls = 0
        self.active_samples = 0

    def loss_for_batch(
        self,
        *,
        receiver_id: int,
        receiver_logits: torch.Tensor,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        receiver_routes = self.routes.get(int(receiver_id), {})
        source_to_mask: dict[int, torch.Tensor] = {}
        for class_id, route in receiver_routes.items():
            class_mask = labels.eq(int(class_id))
            if not bool(class_mask.any()):
                continue
            if route.source_id in source_to_mask:
                source_to_mask[route.source_id] = source_to_mask[route.source_id] | class_mask
            else:
                source_to_mask[route.source_id] = class_mask

        if not source_to_mask:
            return receiver_logits.sum() * 0.0

        weighted_loss = receiver_logits.sum() * 0.0
        active = 0
        for source_id, mask in source_to_mask.items():
            source = self.snapshots[int(source_id)]
            with torch.no_grad():
                source_logits = forward_logits(source, clean_images[mask])
            per_sample = conservative_margin_transfer_loss(
                receiver_logits[mask],
                source_logits,
                labels[mask],
                margin_clip=self.margin_clip,
                source_correct_only=self.source_correct_only,
                reduction="none",
            )
            weighted_loss = weighted_loss + per_sample.sum()
            active += int(mask.sum().item())

        cmt = weighted_loss / max(active, 1)
        scaled = self.lambda_cmt * cmt
        self.loss_sum += float(cmt.detach().cpu())
        self.calls += 1
        self.active_samples += active
        return scaled

    def diagnostics(self) -> dict[str, float]:
        return {
            "cmt_loss": self.loss_sum / max(self.calls, 1),
            "cmt_active_samples": float(self.active_samples),
            "active_route_count": float(sum(len(routes) for routes in self.routes.values())),
        }


class FedFalsifyRouter:
    """Receiver-side class-conditional teacher selection using private audit data."""

    def __init__(
        self,
        *,
        num_classes: int,
        fit_samples_per_class: int = 16,
        audit_samples_per_class: int = 16,
        min_fit_count: int = 2,
        min_audit_count: int = 5,
        min_tau: float = 0.0,
        fra_weight: float = 0.0,
        fra_kappa: float = 1.0,
        fra_shrinkage_nu: float = 10.0,
        margin_clip: float = 2.0,
        source_correct_only: bool = True,
    ):
        self.num_classes = int(num_classes)
        self.fit_samples_per_class = int(fit_samples_per_class)
        self.audit_samples_per_class = int(audit_samples_per_class)
        self.min_fit_count = int(min_fit_count)
        self.min_audit_count = int(min_audit_count)
        self.min_tau = float(min_tau)
        self.fra_weight = float(fra_weight)
        self.fra_kappa = float(fra_kappa)
        self.fra_shrinkage_nu = float(fra_shrinkage_nu)
        self.margin_clip = float(margin_clip)
        self.source_correct_only = bool(source_correct_only)

    def build_plan(
        self,
        *,
        models: dict[int, torch.nn.Module],
        client_splits: dict[int, FedFalsifyClientSplit],
        device: torch.device,
        lambda_cmt: float,
    ) -> tuple[FedFalsifyTransferPlan, list[CandidateAudit]]:
        snapshots = freeze_peer_snapshots(models, device=device)
        routes: dict[int, dict[int, ClassRoute]] = {
            int(receiver_id): {} for receiver_id in models
        }
        candidates: list[CandidateAudit] = []

        for receiver_id, receiver in models.items():
            receiver_was_training = receiver.training
            receiver.eval()
            split = client_splits[int(receiver_id)]
            for class_id in range(self.num_classes):
                fit_indices = split.class_indices(class_id, split="fit")
                audit_indices = split.class_indices(class_id, split="audit")
                if (
                    fit_indices.size < self.min_fit_count
                    or audit_indices.size < self.min_audit_count
                ):
                    continue
                fit_images, fit_labels = _stack_dataset_items(
                    split.probe_dataset,
                    fit_indices,
                    max_samples=self.fit_samples_per_class,
                    device=device,
                )
                audit_images, audit_labels = _stack_dataset_items(
                    split.probe_dataset,
                    audit_indices,
                    max_samples=self.audit_samples_per_class,
                    device=device,
                )

                receiver_audit_predictions = None
                scored: list[tuple[float, int, float, float, float]] = []
                for source_id, source in snapshots.items():
                    if int(source_id) == int(receiver_id):
                        continue
                    receiver_fit_logits = forward_logits(receiver, fit_images)
                    with torch.no_grad():
                        source_fit_logits = forward_logits(source, fit_images)
                    cmt_loss = conservative_margin_transfer_loss(
                        receiver_fit_logits,
                        source_fit_logits,
                        fit_labels,
                        margin_clip=self.margin_clip,
                        source_correct_only=self.source_correct_only,
                    )
                    receiver_audit_logits = forward_logits(receiver, audit_images)
                    audit_ce = F.cross_entropy(receiver_audit_logits, audit_labels)
                    tau, _, _ = gradient_cosine_from_losses(
                        cmt_loss,
                        audit_ce,
                        model_head(receiver).parameters(),
                    )

                    with torch.no_grad():
                        if receiver_audit_predictions is None:
                            receiver_audit_predictions = (
                                forward_logits(receiver, audit_images).argmax(dim=1).cpu().numpy()
                            )
                        source_predictions = (
                            forward_logits(source, audit_images).argmax(dim=1).cpu().numpy()
                        )
                    fra = compute_paired_advantage(
                        source_predictions,
                        receiver_audit_predictions,
                        audit_labels.cpu().numpy(),
                        class_id=class_id,
                        kappa=self.fra_kappa,
                        shrinkage_nu=self.fra_shrinkage_nu,
                        min_count=self.min_audit_count,
                    )
                    score = float(tau) + self.fra_weight * float(fra.advantage_strength)
                    scored.append(
                        (
                            score,
                            int(source_id),
                            float(tau),
                            float(fra.advantage_strength),
                            float(fra.paired_advantage),
                        )
                    )

                selected_source = None
                if scored:
                    best = max(scored, key=lambda item: (item[0], item[2], -item[1]))
                    if best[2] > self.min_tau:
                        selected_source = best[1]
                        routes[int(receiver_id)][class_id] = ClassRoute(
                            receiver_id=int(receiver_id),
                            class_id=class_id,
                            source_id=best[1],
                            tau=best[2],
                            fra_strength=best[3],
                            fra_advantage=best[4],
                            fit_count=int(fit_indices.size),
                            audit_count=int(audit_indices.size),
                        )
                for score, source_id, tau, fra_strength, fra_advantage in scored:
                    candidates.append(CandidateAudit(
                        receiver_id=int(receiver_id),
                        class_id=class_id,
                        source_id=source_id,
                        tau=tau,
                        fra_strength=fra_strength,
                        fra_advantage=fra_advantage,
                        score=score,
                        selected=source_id == selected_source,
                    ))
            receiver.train(receiver_was_training)

        plan = FedFalsifyTransferPlan(
            snapshots=snapshots,
            routes=routes,
            lambda_cmt=lambda_cmt,
            margin_clip=self.margin_clip,
            source_correct_only=self.source_correct_only,
        )
        return plan, candidates


def summarize_routes(
    routes: dict[int, dict[int, ClassRoute]],
    candidates: list[CandidateAudit],
) -> dict[str, Any]:
    selected = [route for client_routes in routes.values() for route in client_routes.values()]
    return {
        "active_route_count": len(selected),
        "route_coverage": len(selected) / max(len(routes) * 10, 1),
        "mean_selected_tau": (
            float(np.mean([route.tau for route in selected])) if selected else 0.0
        ),
        "mean_selected_fra": (
            float(np.mean([route.fra_strength for route in selected])) if selected else 0.0
        ),
        "candidate_count": len(candidates),
    }
