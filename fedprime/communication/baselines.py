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


def _next_public_batch(
    context: CommunicationContext,
    public_iter,
) -> tuple[torch.Tensor, object]:
    if context.public_loader is None or public_iter is None:
        raise RuntimeError("public-data communication requires a public loader")
    try:
        images, _ = next(public_iter)
    except StopIteration:
        public_iter = iter(context.public_loader)
        images, _ = next(public_iter)
    images = normalize_batch(images.to(context.device, non_blocking=True), context.stats)
    return images, public_iter


class FedDFCommunicationStrategy:
    """FedDF heterogeneous logit-ensemble fusion on public data.

    Each persistent client architecture is its own student.  The common teacher
    is the softmax of the mean client logits, matching the heterogeneous FedDF
    adaptation used by the FCCL/RAHFL comparison code.
    """

    name = "feddf"
    requires_public_data = True
    uses_accuracy_routing = False

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("FedDF temperature must be positive")
        self.temperature = float(temperature)

    @staticmethod
    def ensemble_probabilities(
        logits: list[torch.Tensor], temperature: float = 1.0
    ) -> torch.Tensor:
        return F.softmax(torch.stack(logits).mean(dim=0) / float(temperature), dim=1)

    def step(self, context: CommunicationContext) -> float:
        client_ids = sorted(context.models)
        public_iter = context.public_iter
        losses: list[float] = []
        for _ in range(int(context.public_batches_per_round)):
            images, public_iter = _next_public_batch(context, public_iter)
            teacher_logits = []
            for client_id in client_ids:
                model = context.models[client_id]
                model.eval()
                with torch.no_grad():
                    teacher_logits.append(forward_logits(model, images).detach())
            teacher = self.ensemble_probabilities(
                teacher_logits, self.temperature
            ).detach()
            for client_id in client_ids:
                model = context.models[client_id]
                model.train()
                student = F.log_softmax(
                    forward_logits(model, images) / self.temperature, dim=1
                )
                loss = F.kl_div(student, teacher, reduction="batchmean") * (
                    self.temperature**2
                )
                context.optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                context.optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))
        return sum(losses) / max(len(losses), 1)


class KTPFLCommunicationStrategy:
    """KT-pFL personalized public-logit transfer with learned coefficients."""

    name = "kt_pfl"
    requires_public_data = True
    uses_accuracy_routing = False

    def __init__(
        self,
        temperature: float = 1.0,
        coefficient_lr: float = 0.01,
        uniform_regularization: float = 0.5,
    ) -> None:
        if temperature <= 0:
            raise ValueError("KT-pFL temperature must be positive")
        self.temperature = float(temperature)
        self.coefficient_lr = float(coefficient_lr)
        self.uniform_regularization = float(uniform_regularization)
        self._coefficient_logits: torch.Tensor | None = None
        self._coefficient_optimizer: torch.optim.Optimizer | None = None

    def _ensure_coefficients(self, count: int, device: torch.device) -> None:
        if self._coefficient_logits is not None:
            if self._coefficient_logits.shape != (count, count):
                raise ValueError("KT-pFL client count changed after initialization")
            return
        self._coefficient_logits = torch.zeros(
            count, count, device=device, requires_grad=True
        )
        self._coefficient_optimizer = torch.optim.SGD(
            [self._coefficient_logits], lr=self.coefficient_lr
        )

    @property
    def coefficient_weights(self) -> torch.Tensor | None:
        if self._coefficient_logits is None:
            return None
        return F.softmax(self._coefficient_logits.detach(), dim=1)

    def step(self, context: CommunicationContext) -> float:
        client_ids = sorted(context.models)
        self._ensure_coefficients(len(client_ids), context.device)
        assert self._coefficient_logits is not None
        assert self._coefficient_optimizer is not None
        public_iter = context.public_iter
        losses: list[float] = []
        for _ in range(int(context.public_batches_per_round)):
            images, public_iter = _next_public_batch(context, public_iter)
            teacher_probs = []
            for client_id in client_ids:
                model = context.models[client_id]
                model.eval()
                with torch.no_grad():
                    teacher_probs.append(
                        F.softmax(
                            forward_logits(model, images) / self.temperature, dim=1
                        ).detach()
                    )
            teacher_stack = torch.stack(teacher_probs)
            weights = F.softmax(self._coefficient_logits, dim=1)
            personalized = torch.einsum("nt,tbc->nbc", weights, teacher_stack)

            # Alternate the paper's model and coefficient updates.  Targets are
            # detached for model KD so coefficient gradients cannot leak into a
            # client's optimizer.
            for student_index, client_id in enumerate(client_ids):
                model = context.models[client_id]
                model.train()
                student = F.log_softmax(
                    forward_logits(model, images) / self.temperature, dim=1
                )
                loss = F.kl_div(
                    student,
                    personalized[student_index].detach(),
                    reduction="batchmean",
                ) * (self.temperature**2)
                context.optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                context.optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))

            coefficient_fit = sum(
                F.kl_div(
                    personalized[index].clamp_min(1.0e-8).log(),
                    teacher_stack[index],
                    reduction="batchmean",
                )
                for index in range(len(client_ids))
            ) / len(client_ids)
            uniform = torch.full_like(weights, 1.0 / len(client_ids))
            coefficient_loss = coefficient_fit + self.uniform_regularization * F.mse_loss(
                weights, uniform
            )
            self._coefficient_optimizer.zero_grad(set_to_none=True)
            coefficient_loss.backward()
            self._coefficient_optimizer.step()
        return sum(losses) / max(len(losses), 1)


class FCCLCommunicationStrategy:
    """FCCL public-logit cross-correlation communication core."""

    name = "fccl"
    requires_public_data = True
    uses_accuracy_routing = False

    def __init__(self, offdiag_weight: float = 0.0051, eps: float = 1.0e-6) -> None:
        self.offdiag_weight = float(offdiag_weight)
        self.eps = float(eps)

    @staticmethod
    def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
        size, width = matrix.shape
        if size != width:
            raise ValueError("FCCL cross-correlation matrix must be square")
        return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()

    def correlation_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        student = (student_logits - student_logits.mean(0)) / student_logits.std(0).clamp_min(
            self.eps
        )
        teacher = (teacher_logits - teacher_logits.mean(0)) / teacher_logits.std(0).clamp_min(
            self.eps
        )
        correlation = student.T @ teacher / student.shape[0]
        on_diag = (torch.diagonal(correlation) - 1.0).pow(2).sum()
        # FCCL's released implementation explicitly targets -1 off diagonal.
        off_diag = (self.off_diagonal(correlation) + 1.0).pow(2).sum()
        return on_diag + self.offdiag_weight * off_diag

    def step(self, context: CommunicationContext) -> float:
        client_ids = sorted(context.models)
        public_iter = context.public_iter
        losses: list[float] = []
        for _ in range(int(context.public_batches_per_round)):
            images, public_iter = _next_public_batch(context, public_iter)
            targets = []
            for client_id in client_ids:
                model = context.models[client_id]
                model.eval()
                with torch.no_grad():
                    targets.append(forward_logits(model, images).detach())
            target_average = torch.stack(targets).mean(dim=0)
            for client_id in client_ids:
                model = context.models[client_id]
                model.train()
                student_logits = forward_logits(model, images)
                loss = self.correlation_loss(student_logits, target_average)
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
    if normalized == "feddf":
        return FedDFCommunicationStrategy(
            temperature=float(baseline_cfg.get("temperature", 1.0)),
        )
    if normalized in {"kt_pfl", "kt-pfl", "ktpfl"}:
        return KTPFLCommunicationStrategy(
            temperature=float(baseline_cfg.get("temperature", 1.0)),
            coefficient_lr=float(baseline_cfg.get("coefficient_lr", 0.01)),
            uniform_regularization=float(
                baseline_cfg.get("uniform_regularization", 0.5)
            ),
        )
    if normalized == "fccl":
        return FCCLCommunicationStrategy(
            offdiag_weight=float(baseline_cfg.get("offdiag_weight", 0.0051)),
            eps=float(baseline_cfg.get("eps", 1.0e-6)),
        )
    return None
