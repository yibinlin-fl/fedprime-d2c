from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from fedprime.communication.baselines import (
    FedProtoFeatureStrategy,
    RHFLCommunicationStrategy,
    build_baseline_communication_strategy,
    symmetric_cross_entropy,
)
from fedprime.communication.public_logits import CommunicationContext
from fedprime.data.loaders import DatasetStats
from scripts.openi_cle_external_baselines_entry import ARM_ORDER, build_arm_configs


def test_baseline_registry_uses_distinct_official_core_mechanisms() -> None:
    assert build_baseline_communication_strategy("fedmd", {}).routing == "symmetric"
    assert isinstance(build_baseline_communication_strategy("rhfl", {}), RHFLCommunicationStrategy)
    assert build_baseline_communication_strategy("aughfl", {}).name == "aughfl"
    assert isinstance(build_baseline_communication_strategy("fedproto", {}), FedProtoFeatureStrategy)


def test_symmetric_cross_entropy_matches_ce_plus_reverse_ce_definition() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    labels = torch.tensor([0, 1])
    value = symmetric_cross_entropy(logits, labels, num_classes=2, alpha=0.1, beta=1.0)
    assert torch.isfinite(value)
    assert value > 0


def test_external_baseline_matrix_is_protocol_matched() -> None:
    configs = build_arm_configs()
    assert tuple(configs) == ARM_ORDER
    for config in configs.values():
        assert config["data"]["scenario"] == "cle_hfl_v2"
        assert config["method"]["strict_fit_audit"]["enabled"] is True
        assert config["train"]["rounds"] == 12
        assert config["checkpoints"]["save_final"] is False
    assert configs["rhfl"]["method"]["cl_module"] == "rhfl_sce"
    assert configs["aughfl"]["method"]["cl_module"] == "none"
    assert configs["rahfl"]["method"]["communication"] == "asymhfl_val"


class _FeatureModel(torch.nn.Module):
    def __init__(self, embedding_dim: int = 4) -> None:
        super().__init__()
        self.embedding = torch.nn.Linear(2, embedding_dim, bias=False)
        self.classifier = torch.nn.Linear(embedding_dim, 2, bias=False)

    def forward(self, images):
        embedding = self.embedding(images.flatten(1))
        return self.classifier(embedding), embedding


def test_fedproto_aggregates_feature_prototypes_and_builds_local_mse() -> None:
    images0 = torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
    images1 = torch.tensor([[[[2.0]], [[0.0]]], [[[0.0]], [[2.0]]]])
    labels = torch.tensor([0, 1])
    loaders = [
        DataLoader(TensorDataset(images0, labels), batch_size=2),
        DataLoader(TensorDataset(images1, labels), batch_size=2),
    ]
    models = {0: _FeatureModel(), 1: _FeatureModel()}
    strategy = FedProtoFeatureStrategy(proto_weight=1.0)
    context = CommunicationContext(
        models=models,
        optimizers={key: torch.optim.SGD(model.parameters(), lr=0.1) for key, model in models.items()},
        public_loader=None,
        public_iter=None,
        accuracies=[0.0, 0.0],
        stats=DatasetStats([0.0, 0.0], [1.0, 1.0]),
        device=torch.device("cpu"),
        public_batches_per_round=0,
        private_loaders=loaders,
        num_classes=2,
        round_idx=1,
    )

    assert strategy.step(context) == 0.0
    assert strategy.global_prototypes is not None
    assert strategy.global_prototypes.shape == (2, 4)
    loss = strategy.local_loss(model=models[0], clean_images=images0, labels=labels)
    assert torch.isfinite(loss)
    assert loss.requires_grad
