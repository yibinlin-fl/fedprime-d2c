from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fedprime.data.loaders import DatasetStats
from fedprime.communication.public_logits import (
    CommunicationContext,
    NoCommunicationStrategy,
    PublicLogitKDStrategy,
    build_core_communication_strategy,
)


class _LinearModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(images.flatten(1))


def test_asymhfl_teacher_selection_is_frozen() -> None:
    strategy = PublicLogitKDStrategy("asymmetric")
    accuracies = [55.0, 70.0, 70.0, 40.0]

    assert strategy.teacher_ids(0, [0, 1, 2, 3], accuracies) == [1, 2]
    assert strategy.teacher_ids(1, [0, 1, 2, 3], accuracies) == [2]
    assert strategy.teacher_ids(2, [0, 1, 2, 3], accuracies) == [1]
    assert strategy.teacher_ids(3, [0, 1, 2, 3], accuracies) == [0, 1, 2]


def test_symmetric_hfl_selects_every_other_client() -> None:
    strategy = PublicLogitKDStrategy("symmetric")
    assert strategy.teacher_ids(1, [0, 1, 2, 3], [90.0, 10.0, 20.0, 30.0]) == [0, 2, 3]


def test_core_strategy_registry_preserves_config_names() -> None:
    assert isinstance(build_core_communication_strategy("none"), NoCommunicationStrategy)
    assert build_core_communication_strategy("hfl").routing == "symmetric"
    assert build_core_communication_strategy("asymhfl_val").routing == "asymmetric"
    assert build_core_communication_strategy("pccd") is None


def test_public_logit_step_updates_only_clients_with_teachers() -> None:
    models = {
        0: _LinearModel(torch.tensor([[2.0, -1.0], [-1.0, 2.0]])),
        1: _LinearModel(torch.tensor([[-1.0, 2.0], [2.0, -1.0]])),
    }
    optimizers = {
        client_id: torch.optim.SGD(model.parameters(), lr=0.1)
        for client_id, model in models.items()
    }
    images = torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
    loader = DataLoader(TensorDataset(images, torch.zeros(2, dtype=torch.long)), batch_size=2)
    before = {client_id: model.linear.weight.detach().clone() for client_id, model in models.items()}
    context = CommunicationContext(
        models=models,
        optimizers=optimizers,
        public_loader=loader,
        public_iter=iter(loader),
        accuracies=[10.0, 90.0],
        stats=DatasetStats(mean=[0.0, 0.0], std=[1.0, 1.0]),
        device=torch.device("cpu"),
        public_batches_per_round=1,
    )

    loss = PublicLogitKDStrategy("asymmetric").step(context)

    assert loss > 0
    assert not torch.equal(models[0].linear.weight, before[0])
    assert torch.equal(models[1].linear.weight, before[1])


def _legacy_asymhfl_step(context: CommunicationContext) -> float:
    """Frozen pre-refactor implementation from commit 01d1185."""

    losses = []
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    public_iter = context.public_iter
    for _ in range(context.public_batches_per_round):
        try:
            images, _ = next(public_iter)
        except StopIteration:
            public_iter = iter(context.public_loader)
            images, _ = next(public_iter)
        images = images.to(context.device)
        target_probs = {}
        student_log_probs = {}
        for client_id in sorted(context.models):
            model = context.models[client_id]
            model.eval()
            with torch.no_grad():
                target_probs[client_id] = F.softmax(model(images), dim=1).detach()
            model.train()
            student_log_probs[client_id] = F.log_softmax(model(images), dim=1)
        for client_id in sorted(context.models):
            learn_losses = []
            for other_id in sorted(context.models):
                if other_id != client_id and context.accuracies[client_id] <= context.accuracies[other_id]:
                    learn_losses.append(
                        criterion(student_log_probs[client_id], target_probs[other_id])
                    )
            if not learn_losses:
                continue
            loss = sum(learn_losses) / len(learn_losses)
            context.optimizers[client_id].zero_grad(set_to_none=True)
            loss.backward()
            context.optimizers[client_id].step()
            losses.append(float(loss.detach()))
    return sum(losses) / max(len(losses), 1)


def _golden_context(models, loader) -> CommunicationContext:
    return CommunicationContext(
        models=models,
        optimizers={key: torch.optim.SGD(model.parameters(), lr=0.07) for key, model in models.items()},
        public_loader=loader,
        public_iter=iter(loader),
        accuracies=[41.0, 63.0, 63.0],
        stats=DatasetStats(mean=[0.0, 0.0], std=[1.0, 1.0]),
        device=torch.device("cpu"),
        public_batches_per_round=2,
    )


def test_refactored_asymhfl_matches_legacy_loss_and_parameter_updates() -> None:
    torch.manual_seed(20260806)
    base_models = {
        client_id: _LinearModel(torch.randn(2, 2))
        for client_id in range(3)
    }
    legacy_models = copy.deepcopy(base_models)
    strategy_models = copy.deepcopy(base_models)
    images = torch.tensor([
        [[[1.0]], [[0.0]]],
        [[[0.0]], [[1.0]]],
        [[[0.5]], [[-0.5]]],
        [[[-0.25]], [[0.75]]],
    ])
    loader = DataLoader(TensorDataset(images, torch.zeros(4, dtype=torch.long)), batch_size=2)

    legacy_loss = _legacy_asymhfl_step(_golden_context(legacy_models, loader))
    strategy_loss = PublicLogitKDStrategy("asymmetric").step(
        _golden_context(strategy_models, loader)
    )

    assert strategy_loss == pytest.approx(legacy_loss, abs=1.0e-8)
    for client_id in legacy_models:
        for legacy_parameter, strategy_parameter in zip(
            legacy_models[client_id].parameters(), strategy_models[client_id].parameters()
        ):
            torch.testing.assert_close(strategy_parameter, legacy_parameter, rtol=0.0, atol=1.0e-8)
