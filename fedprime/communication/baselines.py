from __future__ import annotations

import copy
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


class AugHFLFidelityCommunicationStrategy:
    """AugHFL PubAug with participant-specific public AugMix triplets.

    This keeps the released method's per-participant views, public normalization,
    global reliability weights, and fresh collaborative Adam optimizers.  It
    intentionally retains a numerically valid log-probability KL input instead
    of reproducing the released code's gradient-killing clamp of negative log
    probabilities.
    """

    name = "aughfl_fidelity"
    phase = "pre_local"
    requires_public_data = True
    uses_accuracy_routing = False

    def __init__(self, learning_rate: float = 1.0e-3) -> None:
        self.learning_rate = float(learning_rate)
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        return (images - 0.5) / 0.5

    def step(self, context: CommunicationContext) -> float:
        if context.public_loader is None or context.public_iter is None:
            raise RuntimeError("AugHFL fidelity strategy requires participant-specific AugMix views")
        client_ids = sorted(context.models)
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        public_iter = context.public_iter
        losses: list[float] = []
        all_consistencies: list[float] = []
        all_weights: list[float] = []
        for _ in range(int(context.public_batches_per_round)):
            try:
                client_views, _ = next(public_iter)
            except StopIteration:
                public_iter = iter(context.public_loader)
                client_views, _ = next(public_iter)
            if not isinstance(client_views, (tuple, list)) or len(client_views) != len(client_ids):
                raise RuntimeError(
                    "AugHFL fidelity loader must return one clean/aug1/aug2 triplet per client"
                )

            targets: dict[int, torch.Tensor] = {}
            students: dict[int, torch.Tensor] = {}
            inverse_consistency: dict[int, float] = {}
            for position, client_id in enumerate(client_ids):
                views = client_views[position]
                if not isinstance(views, (tuple, list)) or len(views) != 3:
                    raise RuntimeError("each AugHFL participant view must contain clean/aug1/aug2")
                clean, aug1, aug2 = [
                    self._normalize(view.to(context.device, non_blocking=True)) for view in views
                ]
                model = context.models[client_id]
                model.train()
                logits = forward_logits(model, torch.cat([clean, aug1, aug2], dim=0))
                clean_logits, aug1_logits, aug2_logits = torch.split(logits, clean.shape[0])
                p_clean = F.softmax(clean_logits, dim=1).clamp_min(1.0e-7)
                p_aug1 = F.softmax(aug1_logits, dim=1).clamp_min(1.0e-7)
                p_aug2 = F.softmax(aug2_logits, dim=1).clamp_min(1.0e-7)
                consistency = F.kl_div(p_aug1.log(), p_clean, reduction="batchmean") + F.kl_div(
                    p_aug2.log(), p_clean, reduction="batchmean"
                )
                consistency_value = max(float(consistency.detach().cpu()), 1.0e-7)
                inverse_consistency[client_id] = 1.0 / consistency_value
                all_consistencies.append(consistency_value)
                targets[client_id] = p_clean.detach()
                students[client_id] = p_clean.log()

            normalizer = sum(inverse_consistency.values()) + 1.0e-7
            weights = {
                client_id: inverse_consistency[client_id] / normalizer for client_id in client_ids
            }
            all_weights.extend(weights.values())
            for client_id in client_ids:
                loss = sum(
                    weights[teacher_id] * criterion(students[client_id], targets[teacher_id])
                    for teacher_id in client_ids
                    if teacher_id != client_id
                )
                optimizer = torch.optim.Adam(
                    context.models[client_id].parameters(), lr=self.learning_rate
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

        weight_tensor = torch.tensor(all_weights, dtype=torch.float32)
        self.last_metrics = {
            "teacher_weight_entropy": float(
                -(weight_tensor.clamp_min(1.0e-12) * weight_tensor.clamp_min(1.0e-12).log()).sum()
                / max(int(context.public_batches_per_round), 1)
            ),
            "teacher_weight_min": float(weight_tensor.min()) if weight_tensor.numel() else 0.0,
            "teacher_weight_max": float(weight_tensor.max()) if weight_tensor.numel() else 0.0,
            "view_consistency": sum(all_consistencies) / max(len(all_consistencies), 1),
        }
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


class FedDFFidelityCommunicationStrategy(FedDFCommunicationStrategy):
    """FedDF server fusion with frozen round teachers and post-local timing.

    With one client per architecture, architecture-wise FedAvg is an identity
    operation.  Each post-local client model therefore initializes the server
    student for that architecture, while frozen copies of all post-local models
    form the average-logit teacher throughout the server distillation phase.
    """

    name = "feddf_fidelity"
    phase = "post_local"

    def __init__(
        self,
        temperature: float = 1.0,
        student_learning_rate: float = 1.0e-3,
        server_steps_per_batch: int = 1,
    ) -> None:
        super().__init__(temperature=temperature)
        self.student_learning_rate = float(student_learning_rate)
        self.server_steps_per_batch = int(server_steps_per_batch)
        if self.server_steps_per_batch < 1:
            raise ValueError("server_steps_per_batch must be positive")
        self.last_metrics: dict[str, float] = {}

    def step(self, context: CommunicationContext) -> float:
        if context.public_loader is None or context.public_iter is None:
            raise RuntimeError("FedDF fidelity strategy requires public data")
        client_ids = sorted(context.models)
        teachers = {
            client_id: copy.deepcopy(context.models[client_id]).to(context.device).eval()
            for client_id in client_ids
        }
        for teacher in teachers.values():
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        optimizers = {
            client_id: torch.optim.Adam(
                context.models[client_id].parameters(), lr=self.student_learning_rate
            )
            for client_id in client_ids
        }
        schedulers = {
            client_id: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[client_id],
                T_max=max(
                    int(context.public_batches_per_round) * self.server_steps_per_batch,
                    1,
                ),
            )
            for client_id in client_ids
        }
        public_iter = context.public_iter
        losses: list[float] = []
        entropies: list[float] = []
        disagreements: list[float] = []
        for _ in range(int(context.public_batches_per_round)):
            images, public_iter = _next_public_batch(context, public_iter)
            with torch.no_grad():
                teacher_logits = [forward_logits(teachers[cid], images) for cid in client_ids]
                mean_logits = torch.stack(teacher_logits).mean(dim=0)
                target = F.softmax(mean_logits / self.temperature, dim=1).detach()
                entropies.append(
                    float(-(target * target.clamp_min(1.0e-12).log()).sum(dim=1).mean().cpu())
                )
                teacher_probs = torch.stack(
                    [F.softmax(logits / self.temperature, dim=1) for logits in teacher_logits]
                )
                disagreements.append(float(teacher_probs.var(dim=0, unbiased=False).mean().cpu()))
            for _server_step in range(self.server_steps_per_batch):
                for client_id in client_ids:
                    model = context.models[client_id]
                    model.train()
                    student = F.log_softmax(
                        forward_logits(model, images) / self.temperature, dim=1
                    )
                    loss = F.kl_div(student, target, reduction="batchmean")
                    optimizers[client_id].zero_grad(set_to_none=True)
                    loss.backward()
                    optimizers[client_id].step()
                    schedulers[client_id].step()
                    losses.append(float(loss.detach().cpu()))
        self.last_metrics = {
            "teacher_entropy": sum(entropies) / max(len(entropies), 1),
            "teacher_disagreement": sum(disagreements) / max(len(disagreements), 1),
            "server_updates": float(
                int(context.public_batches_per_round) * self.server_steps_per_batch
            ),
        }
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


class KTPFLFidelityCommunicationStrategy(KTPFLCommunicationStrategy):
    """Equation-oriented KT-pFL with post-local alternating updates.

    The fidelity variant performs private training first (through the runner),
    distills the current personalized soft predictions, then freezes the updated
    models and optimizes the row-stochastic knowledge coefficient matrix using
    the client-data-weighted objective in Eq. (7).
    """

    name = "kt_pfl_fidelity"
    phase = "post_local"

    def __init__(
        self,
        temperature: float = 1.0,
        coefficient_lr: float = 0.01,
        uniform_regularization: float = 0.5,
        distillation_lr: float = 0.02,
        distillation_steps: int = 1,
        knowledge_weight: float = 1.0,
    ) -> None:
        super().__init__(
            temperature=temperature,
            coefficient_lr=coefficient_lr,
            uniform_regularization=uniform_regularization,
        )
        self.distillation_lr = float(distillation_lr)
        self.distillation_steps = int(distillation_steps)
        self.knowledge_weight = float(knowledge_weight)
        if self.distillation_steps < 1:
            raise ValueError("distillation_steps must be positive")
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _client_data_weights(context: CommunicationContext, client_ids: list[int]) -> torch.Tensor:
        if context.private_loaders is None:
            return torch.full(
                (len(client_ids),), 1.0 / len(client_ids), device=context.device
            )
        sizes = []
        for client_id in client_ids:
            loader = (
                context.private_loaders[client_id]
                if isinstance(context.private_loaders, dict)
                else context.private_loaders[client_id]
            )
            sizes.append(float(len(loader.dataset)))
        result = torch.tensor(sizes, dtype=torch.float32, device=context.device)
        return result / result.sum().clamp_min(1.0)

    def step(self, context: CommunicationContext) -> float:
        if context.public_loader is None or context.public_iter is None:
            raise RuntimeError("KT-pFL fidelity strategy requires public data")
        client_ids = sorted(context.models)
        self._ensure_coefficients(len(client_ids), context.device)
        assert self._coefficient_logits is not None
        assert self._coefficient_optimizer is not None
        model_optimizers = {
            client_id: torch.optim.SGD(
                context.models[client_id].parameters(), lr=self.distillation_lr
            )
            for client_id in client_ids
        }
        data_weights = self._client_data_weights(context, client_ids)
        public_iter = context.public_iter
        model_losses: list[float] = []
        coefficient_losses: list[float] = []
        initial_weights = F.softmax(self._coefficient_logits.detach(), dim=1).clone()

        for _ in range(int(context.public_batches_per_round)):
            images, public_iter = _next_public_batch(context, public_iter)
            with torch.no_grad():
                pre_update_probs = torch.stack(
                    [
                        F.softmax(
                            forward_logits(context.models[client_id].eval(), images) / self.temperature,
                            dim=1,
                        )
                        for client_id in client_ids
                    ]
                ).detach()
            coefficient_weights = F.softmax(self._coefficient_logits.detach(), dim=1)
            personalized_targets = torch.einsum(
                "nt,tbc->nbc", coefficient_weights, pre_update_probs
            ).detach()

            for _distillation_step in range(self.distillation_steps):
                for position, client_id in enumerate(client_ids):
                    model = context.models[client_id]
                    model.train()
                    student = F.log_softmax(
                        forward_logits(model, images) / self.temperature, dim=1
                    )
                    loss = self.knowledge_weight * F.kl_div(
                        student, personalized_targets[position], reduction="batchmean"
                    )
                    model_optimizers[client_id].zero_grad(set_to_none=True)
                    loss.backward()
                    model_optimizers[client_id].step()
                    model_losses.append(float(loss.detach().cpu()))

            with torch.no_grad():
                updated_probs = torch.stack(
                    [
                        F.softmax(
                            forward_logits(context.models[client_id].eval(), images) / self.temperature,
                            dim=1,
                        )
                        for client_id in client_ids
                    ]
                ).detach()
            weights = F.softmax(self._coefficient_logits, dim=1)
            personalized = torch.einsum("nt,tbc->nbc", weights, updated_probs)
            coefficient_fit = sum(
                data_weights[position]
                * F.kl_div(
                    updated_probs[position].clamp_min(1.0e-12).log(),
                    personalized[position],
                    reduction="batchmean",
                )
                for position in range(len(client_ids))
            )
            uniform = torch.full_like(weights, 1.0 / len(client_ids))
            coefficient_loss = (
                self.knowledge_weight * coefficient_fit
                + self.uniform_regularization * (weights - uniform).pow(2).sum()
            )
            self._coefficient_optimizer.zero_grad(set_to_none=True)
            coefficient_loss.backward()
            self._coefficient_optimizer.step()
            coefficient_losses.append(float(coefficient_loss.detach().cpu()))

        final_weights = F.softmax(self._coefficient_logits.detach(), dim=1)
        entropy = -(final_weights.clamp_min(1.0e-12) * final_weights.clamp_min(1.0e-12).log()).sum(1)
        self.last_metrics = {
            "coefficient_loss": sum(coefficient_losses) / max(len(coefficient_losses), 1),
            "coefficient_entropy": float(entropy.mean().cpu()),
            "coefficient_diagonal": float(final_weights.diagonal().mean().cpu()),
            "coefficient_offdiagonal": float(
                (final_weights.sum() - final_weights.diagonal().sum()).cpu()
                / max(final_weights.numel() - len(client_ids), 1)
            ),
            "coefficient_drift": float((final_weights - initial_weights).abs().mean().cpu()),
        }
        return sum(model_losses) / max(len(model_losses), 1)


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
    if normalized == "aughfl_fidelity":
        return AugHFLFidelityCommunicationStrategy(
            learning_rate=float(baseline_cfg.get("collaborative_lr", 1.0e-3)),
        )
    if normalized == "fedproto":
        return FedProtoFeatureStrategy(
            proto_weight=float(baseline_cfg.get("proto_weight", 1.0)),
            max_batches=baseline_cfg.get("max_proto_batches"),
        )
    if normalized == "feddf":
        return FedDFCommunicationStrategy(
            temperature=float(baseline_cfg.get("temperature", 1.0)),
        )
    if normalized == "feddf_fidelity":
        return FedDFFidelityCommunicationStrategy(
            temperature=float(baseline_cfg.get("temperature", 1.0)),
            student_learning_rate=float(baseline_cfg.get("student_learning_rate", 1.0e-3)),
            server_steps_per_batch=int(baseline_cfg.get("server_steps_per_batch", 1)),
        )
    if normalized in {"kt_pfl", "kt-pfl", "ktpfl"}:
        return KTPFLCommunicationStrategy(
            temperature=float(baseline_cfg.get("temperature", 1.0)),
            coefficient_lr=float(baseline_cfg.get("coefficient_lr", 0.01)),
            uniform_regularization=float(
                baseline_cfg.get("uniform_regularization", 0.5)
            ),
        )
    if normalized in {"kt_pfl_fidelity", "kt-pfl-fidelity"}:
        return KTPFLFidelityCommunicationStrategy(
            temperature=float(baseline_cfg.get("temperature", 1.0)),
            coefficient_lr=float(baseline_cfg.get("coefficient_lr", 0.01)),
            uniform_regularization=float(
                baseline_cfg.get("uniform_regularization", 0.5)
            ),
            distillation_lr=float(baseline_cfg.get("distillation_lr", 0.02)),
            distillation_steps=int(baseline_cfg.get("distillation_steps", 1)),
            knowledge_weight=float(baseline_cfg.get("knowledge_weight", 1.0)),
        )
    if normalized == "fccl":
        return FCCLCommunicationStrategy(
            offdiag_weight=float(baseline_cfg.get("offdiag_weight", 0.0051)),
            eps=float(baseline_cfg.get("eps", 1.0e-6)),
        )
    return None
