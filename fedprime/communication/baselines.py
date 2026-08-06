from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fedprime.communication.public_logits import CommunicationContext, PublicLogitKDStrategy
from fedprime.data.loaders import normalize_batch
from fedprime.models.factory import forward_logits


def symmetric_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    alpha: float = 0.1,
    beta: float = 1.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels)
    probabilities = F.softmax(logits, dim=1).clamp(1.0e-7, 1.0)
    one_hot = F.one_hot(labels, num_classes=num_classes).float().clamp(1.0e-4, 1.0)
    reverse_ce = -(probabilities * one_hot.log()).sum(dim=1).mean()
    return float(alpha) * ce + float(beta) * reverse_ce


def _weighted_public_kd(
    context: CommunicationContext,
    teacher_weights: dict[int, float],
) -> float:
    if context.public_loader is None or context.public_iter is None:
        raise RuntimeError("weighted public-logit KD requires public data")
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    client_ids = sorted(context.models)
    public_iter = context.public_iter
    losses: list[float] = []
    for _ in range(int(context.public_batches_per_round)):
        try:
            images, _ = next(public_iter)
        except StopIteration:
            public_iter = iter(context.public_loader)
            images, _ = next(public_iter)
        images = normalize_batch(images.to(context.device, non_blocking=True), context.stats)
        targets = {}
        students = {}
        for client_id in client_ids:
            model = context.models[client_id]
            model.eval()
            with torch.no_grad():
                targets[client_id] = F.softmax(forward_logits(model, images), dim=1).detach()
            model.train()
            students[client_id] = F.log_softmax(forward_logits(model, images), dim=1)
        for client_id in client_ids:
            loss = sum(
                float(teacher_weights[teacher_id]) * criterion(students[client_id], targets[teacher_id])
                for teacher_id in client_ids
                if teacher_id != client_id
            )
            context.optimizers[client_id].zero_grad(set_to_none=True)
            loss.backward()
            context.optimizers[client_id].step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


class RHFLCommunicationStrategy:
    """Official RHFL client-confidence reweighting with fit-only SCE losses."""

    name = "rhfl"
    requires_public_data = True
    uses_accuracy_routing = False

    def __init__(self, beta: float = 0.5, max_quality_batches: int | None = None) -> None:
        self.beta = float(beta)
        self.max_quality_batches = max_quality_batches
        self.previous_losses: dict[int, float] | None = None

    def _fit_losses(self, context: CommunicationContext) -> dict[int, float]:
        if context.private_loaders is None:
            raise RuntimeError("RHFL requires private fit loaders for confidence estimation")
        values = {}
        for client_id, model in sorted(context.models.items()):
            loader = context.private_loaders[client_id]
            batch_losses = []
            model.eval()
            with torch.no_grad():
                for batch_idx, (images, labels, *_) in enumerate(loader):
                    if self.max_quality_batches is not None and batch_idx >= self.max_quality_batches:
                        break
                    if isinstance(images, (tuple, list)):
                        images = images[0]
                    images = images.to(context.device, non_blocking=True)
                    labels = labels.to(context.device, non_blocking=True).long()
                    logits = forward_logits(model, images)
                    batch_losses.append(float(symmetric_cross_entropy(
                        logits, labels, num_classes=context.num_classes
                    ).cpu()))
            values[client_id] = sum(batch_losses) / max(len(batch_losses), 1)
        return values

    def step(self, context: CommunicationContext) -> float:
        current = self._fit_losses(context)
        client_ids = sorted(context.models)
        if self.previous_losses is None:
            weights = {client_id: 1.0 / max(len(client_ids) - 1, 1) for client_id in client_ids}
        else:
            quality = {
                client_id: (self.previous_losses[client_id] - current[client_id]) / max(current[client_id], 1.0e-8)
                for client_id in client_ids
            }
            quality_sum = sum(quality.values())
            if abs(quality_sum) < 1.0e-8:
                logits = {client_id: 1.0 / max(len(client_ids) - 1, 1) for client_id in client_ids}
            else:
                logits = {
                    client_id: 1.0 / max(len(client_ids) - 1, 1)
                    + self.beta * quality[client_id] / quality_sum
                    for client_id in client_ids
                }
            maximum = max(logits.values())
            exponentials = {client_id: math.exp(logits[client_id] - maximum) for client_id in client_ids}
            normalizer = sum(exponentials.values())
            weights = {client_id: exponentials[client_id] / normalizer for client_id in client_ids}
        self.previous_losses = current
        return _weighted_public_kd(context, weights)


class AugHFLCommunicationStrategy:
    """Official AugHFL public-view consistency teacher reweighting."""

    name = "aughfl"
    requires_public_data = True
    uses_accuracy_routing = False

    def step(self, context: CommunicationContext) -> float:
        if context.public_loader is None or context.public_iter is None:
            raise RuntimeError("AugHFL requires an AugMix public loader")
        client_ids = sorted(context.models)
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        public_iter = context.public_iter
        losses: list[float] = []
        for _ in range(int(context.public_batches_per_round)):
            try:
                views, _ = next(public_iter)
            except StopIteration:
                public_iter = iter(context.public_loader)
                views, _ = next(public_iter)
            if not isinstance(views, (tuple, list)) or len(views) != 3:
                raise RuntimeError("AugHFL public loader must return clean/aug1/aug2 views")
            clean, aug1, aug2 = [
                normalize_batch(view.to(context.device, non_blocking=True), context.stats)
                for view in views
            ]
            targets = {}
            students = {}
            inverse_consistency = {}
            for client_id in client_ids:
                model = context.models[client_id]
                model.eval()
                with torch.no_grad():
                    logits = forward_logits(model, torch.cat([clean, aug1, aug2], dim=0))
                    clean_logits, aug1_logits, aug2_logits = torch.split(logits, clean.shape[0])
                    p_clean = F.softmax(clean_logits, dim=1)
                    p_aug1 = F.softmax(aug1_logits, dim=1).clamp_min(1.0e-7)
                    p_aug2 = F.softmax(aug2_logits, dim=1).clamp_min(1.0e-7)
                    consistency = F.kl_div(p_aug1.log(), p_clean, reduction="batchmean") + F.kl_div(
                        p_aug2.log(), p_clean, reduction="batchmean"
                    )
                    inverse_consistency[client_id] = 1.0 / max(float(consistency.cpu()), 1.0e-7)
                    targets[client_id] = p_clean.detach()
                model.train()
                students[client_id] = F.log_softmax(forward_logits(model, clean), dim=1)
            normalizer = sum(inverse_consistency.values())
            weights = {client_id: inverse_consistency[client_id] / normalizer for client_id in client_ids}
            for client_id in client_ids:
                loss = sum(
                    weights[teacher_id] * criterion(students[client_id], targets[teacher_id])
                    for teacher_id in client_ids
                    if teacher_id != client_id
                )
                context.optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                context.optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))
        return sum(losses) / max(len(losses), 1)


class FedProtoFeatureStrategy:
    """FedProto using class-wise prototypes in the shared model embedding space."""

    name = "fedproto"
    requires_public_data = False
    uses_accuracy_routing = False

    def __init__(self, proto_weight: float = 1.0, max_batches: int | None = None) -> None:
        self.proto_weight = float(proto_weight)
        self.max_batches = max_batches
        self.global_prototypes: torch.Tensor | None = None
        self.valid_classes: torch.Tensor | None = None

    @staticmethod
    def _embedding(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        output = model(images)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError(
                "FedProto requires models to return (logits, embedding)."
            )
        embedding = output[1]
        return embedding.flatten(1)

    def _client_prototypes(
        self,
        context: CommunicationContext,
        client_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.private_loaders is None:
            raise RuntimeError("FedProto requires private fit loaders")
        model = context.models[client_id]
        sums = None
        counts = torch.zeros(context.num_classes, device=context.device)
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels, *_) in enumerate(context.private_loaders[client_id]):
                if self.max_batches is not None and batch_idx >= self.max_batches:
                    break
                if isinstance(images, (tuple, list)):
                    images = images[0]
                images = images.to(context.device, non_blocking=True)
                labels = labels.to(context.device, non_blocking=True).long()
                embeddings = self._embedding(model, images)
                if sums is None:
                    sums = torch.zeros(
                        context.num_classes,
                        embeddings.shape[1],
                        device=context.device,
                    )
                sums.index_add_(0, labels, embeddings)
                counts.index_add_(
                    0, labels, torch.ones_like(labels, dtype=torch.float32)
                )
        if sums is None:
            raise RuntimeError(f"FedProto client {client_id} fit loader produced no batches")
        return sums / counts.clamp_min(1.0).unsqueeze(1), counts.gt(0)

    def step(self, context: CommunicationContext) -> float:
        # FedProto starts without global prototypes. At round r>0, models here
        # already contain the local update from round r-1, matching the
        # original aggregate-then-regularize schedule.
        if int(context.round_idx) == 0:
            self.global_prototypes = None
            self.valid_classes = None
            return 0.0
        local_prototypes = []
        local_validity = []
        embedding_dim = None
        for client_id in sorted(context.models):
            prototypes, valid = self._client_prototypes(context, client_id)
            if embedding_dim is None:
                embedding_dim = int(prototypes.shape[1])
            elif int(prototypes.shape[1]) != embedding_dim:
                raise ValueError(
                    "FedProto clients must expose the same embedding dimension; "
                    f"expected {embedding_dim}, got {prototypes.shape[1]} for client {client_id}."
                )
            local_prototypes.append(prototypes)
            local_validity.append(valid)

        prototype_stack = torch.stack(local_prototypes)
        validity_stack = torch.stack(local_validity)
        client_counts = validity_stack.sum(dim=0)
        self.global_prototypes = (
            prototype_stack * validity_stack.unsqueeze(2)
        ).sum(dim=0) / client_counts.clamp_min(1).unsqueeze(1)
        self.global_prototypes = self.global_prototypes.detach()
        self.valid_classes = client_counts.gt(0)
        return 0.0

    def local_loss(
        self,
        *,
        model: torch.nn.Module,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.global_prototypes is None or self.valid_classes is None:
            return clean_images.new_zeros(())
        supported = self.valid_classes[labels]
        if not bool(supported.any()):
            return clean_images.new_zeros(())
        embeddings = self._embedding(model, clean_images)
        targets = self.global_prototypes[labels[supported]].detach()
        return self.proto_weight * F.mse_loss(embeddings[supported], targets)


def build_baseline_communication_strategy(name: str, method_cfg: dict):
    normalized = str(name).lower()
    baseline_cfg = method_cfg.get("baseline", {})
    if normalized == "fedmd":
        return PublicLogitKDStrategy("symmetric")
    if normalized == "rhfl":
        return RHFLCommunicationStrategy(
            beta=float(baseline_cfg.get("beta", 0.5)),
            max_quality_batches=baseline_cfg.get("max_quality_batches"),
        )
    if normalized == "aughfl":
        return AugHFLCommunicationStrategy()
    if normalized == "fedproto":
        return FedProtoFeatureStrategy(
            proto_weight=float(baseline_cfg.get("proto_weight", 1.0)),
            max_batches=baseline_cfg.get("max_proto_batches"),
        )
    return None
