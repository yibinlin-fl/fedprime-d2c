from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedprime.communication.baselines import (
    FedProtoFeatureStrategy,
    FedTGPCommunicationStrategy,
    RHFLCommunicationStrategy,
    build_baseline_communication_strategy,
    symmetric_cross_entropy,
)
from fedprime.communication.public_logits import CommunicationContext
from fedprime.data.loaders import DatasetStats
from scripts.openi_cle_external_baselines_entry import ARM_ORDER, build_arm_configs
from scripts.merge_cle_hfl_context_shards import merge_shards, parse_shard_roots
from scripts.run_cle_hfl_context import ARMS, SHARDS, context_arm_config, selected_arms
from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment


def test_baseline_registry_uses_distinct_official_core_mechanisms() -> None:
    assert build_baseline_communication_strategy("fedmd", {}).routing == "symmetric"
    assert isinstance(build_baseline_communication_strategy("rhfl", {}), RHFLCommunicationStrategy)
    assert build_baseline_communication_strategy("aughfl", {}).name == "aughfl"
    assert isinstance(build_baseline_communication_strategy("fedproto", {}), FedProtoFeatureStrategy)
    assert isinstance(build_baseline_communication_strategy("fedtgp", {}), FedTGPCommunicationStrategy)


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


def test_fedtgp_trains_finite_global_prototypes_and_local_loss() -> None:
    images0 = torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
    images1 = torch.tensor([[[[2.0]], [[0.0]]], [[[0.0]], [[2.0]]]])
    labels = torch.tensor([0, 1])
    loaders = [
        DataLoader(TensorDataset(images0[:1], labels[:1]), batch_size=1),
        DataLoader(TensorDataset(images1[1:], labels[1:]), batch_size=1),
    ]
    models = {0: _FeatureModel(), 1: _FeatureModel()}
    strategy = FedTGPCommunicationStrategy(
        proto_weight=10.0,
        server_epochs=3,
        server_batch_size=2,
    )
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
        round_idx=0,
    )

    server_loss = strategy.step(context)
    assert torch.isfinite(torch.tensor(server_loss))
    assert strategy.global_prototypes is not None
    assert strategy.global_prototypes.shape == (2, 4)
    assert strategy.last_metrics["uploaded_prototypes"] == 2.0
    local_loss = strategy.local_loss(model=models[0], clean_images=images0, labels=labels)
    assert torch.isfinite(local_loss)
    assert local_loss.requires_grad


def test_private_loader_generator_state_is_restored_across_two_rounds() -> None:
    def build_loader(seed: int) -> DataLoader:
        generator = torch.Generator().manual_seed(seed)
        values = torch.arange(24)
        return DataLoader(
            TensorDataset(values, values),
            batch_size=4,
            shuffle=True,
            generator=generator,
        )

    control = build_loader(20260923)
    inspected = build_loader(20260923)
    control_trace = []
    inspected_trace = []
    for _ in range(2):
        snapshots = AsymHFLExperiment._capture_private_loader_generator_states([inspected])
        list(inspected)
        AsymHFLExperiment._restore_private_loader_generator_states(snapshots)
        inspected_trace.append(next(iter(inspected))[0].clone())
        control_trace.append(next(iter(control))[0].clone())

    assert all(torch.equal(left, right) for left, right in zip(control_trace, inspected_trace))


def test_hfl_pairing_mode_is_two_rounds_and_cheap(tmp_path) -> None:
    configs = {
        arm: context_arm_config(
            arm,
            package_root=tmp_path,
            mode="pairing",
            output_root=tmp_path / "outputs",
            device="cpu",
        )
        for arm in ARMS
    }
    assert all(config["train"]["rounds"] == 2 for config in configs.values())
    assert all(config["train"]["max_local_batches"] == 2 for config in configs.values())
    assert configs["fedtgp_adapter"]["method"]["baseline"]["server_epochs"] == 1
    assert configs["rhfl_adapter"]["method"]["baseline"]["max_quality_batches"] == 1


def test_hfl_context_shards_are_disjoint_and_cover_all_arms() -> None:
    execution_shards = ("cheap_a", "cheap_b", "fedtgp", "rhfl")
    flattened = tuple(arm for shard in execution_shards for arm in SHARDS[shard])
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(ARMS)
    assert selected_arms("cheap_b", "formal") == SHARDS["cheap_b"]
    assert selected_arms("cheap_b", "pairing") == ("local_erm", *SHARDS["cheap_b"])


def test_hfl_shard_root_parser_requires_exact_merge_partition(tmp_path) -> None:
    values = [f"{shard}={tmp_path / shard}" for shard in ("cheap_a", "cheap_b", "fedtgp", "rhfl")]
    roots = parse_shard_roots(values)
    assert tuple(roots) == ("cheap_a", "cheap_b", "fedtgp", "rhfl")


def test_hfl_shard_merger_requires_and_combines_matched_formal_outputs(tmp_path) -> None:
    roots = {}
    trace_line = json.dumps({"round": 0, "client": 0, "batch": 0, "sha256": "matched"}) + "\n"
    for shard in ("cheap_a", "cheap_b", "fedtgp", "rhfl"):
        root = tmp_path / shard
        roots[shard] = root
        for directory in ("configs", "outputs", "analysis"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        records = {}
        rows = {}
        for arm in SHARDS[shard]:
            config_path = root / "configs" / f"{arm}.json"
            config_path.write_text("{}", encoding="utf-8")
            records[arm] = {
                "config": str(config_path),
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest().upper(),
            }
            rows[arm] = {"pooled_dsa": 0.1}
            arm_root = root / "outputs" / f"cle_hfl_context_{arm}_trainseed0"
            arm_root.mkdir(parents=True)
            (arm_root / "local_batch_trace.jsonl").write_text(trace_line, encoding="utf-8")
        contract = {
            "protocol": "cle_hfl_context_table_v2",
            "mode": "formal",
            "rounds": 40,
            "train_seed": 0,
            "execution_shard": shard,
            "selected_arms": list(SHARDS[shard]),
            "arms": records,
        }
        completion = {
            "execution_shard": shard,
            "selected_arms": list(SHARDS[shard]),
            "completed_arms": list(SHARDS[shard]),
            "complete": True,
        }
        summary = {
            "mode": "formal",
            "execution_shard": shard,
            "selected_arms": list(SHARDS[shard]),
            "rows": rows,
            "scientific_evidence": True,
        }
        (root / "configs" / f"CONTRACT_{shard}.json").write_text(json.dumps(contract), encoding="utf-8")
        (root / "configs" / f"COMPLETION_{shard}.json").write_text(json.dumps(completion), encoding="utf-8")
        (root / "analysis" / "RESULT_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
        (root / "outputs" / "INPUT_AUDIT.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        np.savez_compressed(
            root / "analysis" / "HFL_CONTEXT_PREDICTIONS.npz",
            probabilities=np.zeros((len(SHARDS[shard]), 1, 1, 1, 2)),
            arms=np.asarray(SHARDS[shard]),
            labels=np.asarray([0]),
            binding=np.asarray([[0]]),
            operator_names=np.asarray(["blur"]),
        )
    merged = merge_shards(roots, tmp_path / "merged")
    assert tuple(merged["rows"]) == ARMS
    assert merged["cross_shard_local_batch_pairing"]["all_arms_match"] is True
