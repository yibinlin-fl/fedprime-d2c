from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from fedprime.communication.baselines import (
    AugHFLFidelityCommunicationStrategy,
    FCCLCommunicationStrategy,
    FedDFCommunicationStrategy,
    FedDFFidelityCommunicationStrategy,
    KTPFLCommunicationStrategy,
    KTPFLFidelityCommunicationStrategy,
    build_baseline_communication_strategy,
)
from fedprime.communication.public_logits import CommunicationContext
from fedprime.data.loaders import DatasetStats
from scripts.openi_cle_remaining_baselines_entry import ARM_ORDER, build_configs


class _LogitModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)

    def forward(self, images):
        return self.linear(images.flatten(1))


def _context(strategy_models: dict[int, torch.nn.Module]) -> CommunicationContext:
    images = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
            [[[1.0]], [[1.0]]],
            [[[2.0]], [[-1.0]]],
        ]
    )
    loader = DataLoader(TensorDataset(images, torch.zeros(4).long()), batch_size=4)
    return CommunicationContext(
        models=strategy_models,
        optimizers={
            key: torch.optim.SGD(model.parameters(), lr=0.01)
            for key, model in strategy_models.items()
        },
        public_loader=loader,
        public_iter=iter(loader),
        accuracies=[0.0] * len(strategy_models),
        stats=DatasetStats([0.0, 0.0], [1.0, 1.0]),
        device=torch.device("cpu"),
        public_batches_per_round=1,
        num_classes=2,
    )


def test_remaining_registry_exposes_three_distinct_strategies() -> None:
    assert isinstance(build_baseline_communication_strategy("feddf", {}), FedDFCommunicationStrategy)
    assert isinstance(build_baseline_communication_strategy("kt-pfl", {}), KTPFLCommunicationStrategy)
    assert isinstance(build_baseline_communication_strategy("fccl", {}), FCCLCommunicationStrategy)


def test_fidelity_registry_preserves_historical_adapters() -> None:
    assert isinstance(build_baseline_communication_strategy("feddf", {}), FedDFCommunicationStrategy)
    assert isinstance(
        build_baseline_communication_strategy("feddf_fidelity", {}),
        FedDFFidelityCommunicationStrategy,
    )
    assert isinstance(
        build_baseline_communication_strategy("kt_pfl_fidelity", {}),
        KTPFLFidelityCommunicationStrategy,
    )
    assert isinstance(
        build_baseline_communication_strategy("aughfl_fidelity", {}),
        AugHFLFidelityCommunicationStrategy,
    )
    assert getattr(build_baseline_communication_strategy("feddf", {}), "phase", "pre_local") == "pre_local"
    assert build_baseline_communication_strategy("feddf_fidelity", {}).phase == "post_local"
    assert build_baseline_communication_strategy("kt_pfl_fidelity", {}).phase == "post_local"


def test_feddf_uses_softmax_of_mean_logits() -> None:
    first = torch.tensor([[4.0, 0.0]])
    second = torch.tensor([[0.0, 2.0]])
    actual = FedDFCommunicationStrategy.ensemble_probabilities([first, second])
    expected = torch.softmax(torch.tensor([[2.0, 1.0]]), dim=1)
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, (first.softmax(1) + second.softmax(1)) / 2)


def test_kt_pfl_coefficients_are_persistent_row_stochastic() -> None:
    models = {
        0: _LogitModel(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        1: _LogitModel(torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
    }
    strategy = KTPFLCommunicationStrategy(coefficient_lr=0.1)
    assert torch.isfinite(torch.tensor(strategy.step(_context(models))))
    weights = strategy.coefficient_weights
    assert weights is not None
    assert weights.shape == (2, 2)
    assert torch.allclose(weights.sum(1), torch.ones(2), atol=1.0e-6)
    identity = strategy._coefficient_logits
    strategy.step(_context(models))
    assert strategy._coefficient_logits is identity


def test_fccl_loss_matches_released_cross_correlation_definition() -> None:
    strategy = FCCLCommunicationStrategy(offdiag_weight=0.0051)
    logits = torch.tensor(
        [[1.0, -1.0], [-1.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
        requires_grad=True,
    )
    loss = strategy.correlation_loss(logits, logits.detach())
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_feddf_and_fccl_execute_through_the_shared_context() -> None:
    for strategy in (FedDFCommunicationStrategy(), FCCLCommunicationStrategy()):
        models = {
            0: _LogitModel(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
            1: _LogitModel(torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
        }
        value = strategy.step(_context(models))
        assert torch.isfinite(torch.tensor(value))


def test_feddf_fidelity_uses_frozen_teachers_and_emits_diagnostics() -> None:
    models = {
        0: _LogitModel(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        1: _LogitModel(torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
    }
    strategy = FedDFFidelityCommunicationStrategy(student_learning_rate=0.01)
    value = strategy.step(_context(models))
    assert torch.isfinite(torch.tensor(value))
    assert strategy.last_metrics["server_updates"] == 1.0
    assert strategy.last_metrics["teacher_entropy"] > 0.0
    assert strategy.last_metrics["teacher_disagreement"] >= 0.0


def test_kt_pfl_fidelity_alternates_models_then_coefficients() -> None:
    models = {
        0: _LogitModel(torch.tensor([[2.0, -1.0], [-1.0, 2.0]])),
        1: _LogitModel(torch.tensor([[-0.5, 1.5], [1.0, -0.5]])),
    }
    strategy = KTPFLFidelityCommunicationStrategy(
        coefficient_lr=0.1,
        distillation_lr=0.02,
    )
    value = strategy.step(_context(models))
    assert torch.isfinite(torch.tensor(value))
    weights = strategy.coefficient_weights
    assert weights is not None
    assert torch.allclose(weights.sum(1), torch.ones(2), atol=1.0e-6)
    for name in (
        "coefficient_loss",
        "coefficient_entropy",
        "coefficient_diagonal",
        "coefficient_offdiagonal",
        "coefficient_drift",
    ):
        assert torch.isfinite(torch.tensor(strategy.last_metrics[name]))


def test_aughfl_fidelity_consumes_independent_participant_views() -> None:
    models = {
        0: _LogitModel(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        1: _LogitModel(torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
    }
    base = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
            [[[1.0]], [[1.0]]],
            [[[0.5]], [[-0.5]]],
        ]
    )
    client_views = (
        (base, base + 0.1, base - 0.1),
        (base, base + 0.2, base - 0.2),
    )
    labels = torch.zeros(4).long()
    loader = [(client_views, labels)]
    context = CommunicationContext(
        models=models,
        optimizers={key: torch.optim.SGD(model.parameters(), lr=0.01) for key, model in models.items()},
        public_loader=loader,
        public_iter=iter(loader),
        accuracies=[0.0, 0.0],
        stats=DatasetStats([0.0, 0.0], [1.0, 1.0]),
        device=torch.device("cpu"),
        public_batches_per_round=1,
        num_classes=2,
    )
    strategy = AugHFLFidelityCommunicationStrategy(learning_rate=0.01)
    value = strategy.step(context)
    assert torch.isfinite(torch.tensor(value))
    assert strategy.last_metrics["teacher_weight_max"] >= strategy.last_metrics["teacher_weight_min"]
    assert strategy.last_metrics["view_consistency"] >= 0.0


def test_remaining_configs_keep_cle_protocol_matched() -> None:
    configs = build_configs()
    assert tuple(configs) == ARM_ORDER
    for config in configs.values():
        assert config["seed"] == 0
        assert config["data"]["scenario"] == "cle_hfl_v2"
        assert config["method"]["strict_fit_audit"]["enabled"] is True
        assert config["train"]["rounds"] == 12
        assert config["train"]["public_batches_per_round"] == 4
    assert "cdep" not in configs["pew_ber"]["method"]["fedease"]
