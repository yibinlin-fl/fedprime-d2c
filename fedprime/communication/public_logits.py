from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from fedprime.data.loaders import normalize_batch
from fedprime.models.factory import forward_logits


@dataclass
class CommunicationContext:
    """Inputs shared by communication strategies for one federated round."""

    models: Mapping[int, torch.nn.Module]
    optimizers: Mapping[int, torch.optim.Optimizer]
    public_loader: object | None
    public_iter: object | None
    accuracies: list[float] | Mapping[int, float]
    stats: object
    device: torch.device
    public_batches_per_round: int
    private_loaders: Mapping[int, object] | list[object] | None = None
    num_classes: int = 10
    round_idx: int = 0


class NoCommunicationStrategy:
    name = "none"
    requires_public_data = False
    uses_accuracy_routing = False

    def step(self, context: CommunicationContext) -> float:
        del context
        return 0.0


class PublicLogitKDStrategy:
    """Public-logit KD with either symmetric or accuracy-routed teachers."""

    requires_public_data = True

    def __init__(self, routing: str) -> None:
        routing = str(routing).lower()
        if routing not in {"symmetric", "asymmetric"}:
            raise ValueError(f"Unknown public-logit routing: {routing}")
        self.routing = routing
        self.name = "hfl" if routing == "symmetric" else "asymhfl_val"
        self.uses_accuracy_routing = routing == "asymmetric"

    def teacher_ids(
        self,
        client_id: int,
        client_ids: list[int],
        accuracies: list[float] | Mapping[int, float],
    ) -> list[int]:
        if self.routing == "symmetric":
            return [other_id for other_id in client_ids if other_id != client_id]
        return [
            other_id
            for other_id in client_ids
            if other_id != client_id and accuracies[client_id] <= accuracies[other_id]
        ]

    def step(self, context: CommunicationContext) -> float:
        if context.public_loader is None or context.public_iter is None:
            raise RuntimeError(f"{self.name} communication requires a public loader")

        losses: list[float] = []
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        client_ids = sorted(context.models)
        public_iter = context.public_iter

        for _ in range(int(context.public_batches_per_round)):
            try:
                images, _ = next(public_iter)
            except StopIteration:
                public_iter = iter(context.public_loader)
                images, _ = next(public_iter)
            images = normalize_batch(images.to(context.device, non_blocking=True), context.stats)

            target_probs: dict[int, torch.Tensor] = {}
            student_log_probs: dict[int, torch.Tensor] = {}
            for client_id in client_ids:
                model = context.models[client_id]
                model.eval()
                with torch.no_grad():
                    logits = forward_logits(model, images)
                    target_probs[client_id] = F.softmax(logits, dim=1).detach()
                model.train()
                logits = forward_logits(model, images)
                student_log_probs[client_id] = F.log_softmax(logits, dim=1)

            for client_id in client_ids:
                teachers = self.teacher_ids(client_id, client_ids, context.accuracies)
                if not teachers:
                    continue
                loss = sum(
                    criterion(student_log_probs[client_id], target_probs[teacher_id])
                    for teacher_id in teachers
                ) / len(teachers)
                context.optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                context.optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))

        return sum(losses) / max(len(losses), 1)


def build_core_communication_strategy(name: str):
    """Build a core strategy, or return None for legacy specialized methods."""

    normalized = str(name or "asymhfl").lower()
    if normalized in {"none", "local_only"}:
        return NoCommunicationStrategy()
    if normalized in {"hfl", "symmetric_hfl", "public_logit_mean"}:
        return PublicLogitKDStrategy("symmetric")
    if normalized in {"asymhfl", "asymhfl_val"}:
        return PublicLogitKDStrategy("asymmetric")
    return None
