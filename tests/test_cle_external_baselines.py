from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
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
from scripts.merge_cle_hfl_context_shards import merge_shards, parse_shard_roots, trace_digest
from scripts.merge_cle_hfl_domain_table import FINAL_ROWS, merge_domain_table
from scripts.run_cle_hfl_context import ARMS, PRACTICAL_ARMS, SHARDS, context_arm_config, selected_arms
from scripts.run_cle_hfl_learning_floor import BUDGETS as LEARNING_FLOOR_BUDGETS, learning_floor_config
from scripts.run_cle_v2_fedmd_objectives import ARMS as FEDMD_OBJECTIVE_ARMS, fedmd_arm_config
from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment
from fedprime.methods.fedease import FedEASEExperiment
from fedprime.methods.local_prime import jsd_loss_from_logits


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
        # Local-batch FedProto must not rescan the private DataLoader during
        # communication; prototypes come from the exact optimization batches.
        private_loaders=None,
        num_classes=2,
        round_idx=0,
    )

    assert strategy.step(context) == 0.0
    for client_id in sorted(models):
        warmup_loss = strategy.local_loss(
            model=models[client_id],
            clean_images=(images0 if client_id == 0 else images1),
            labels=labels,
            client_id=client_id,
        )
        assert torch.isfinite(warmup_loss)
    context.round_idx = 1
    assert strategy.step(context) == 0.0
    assert strategy.global_prototypes is not None
    assert strategy.global_prototypes.shape == (2, 4)
    assert strategy.last_metrics["prototype_source"] == "local_batches"
    assert strategy.last_metrics["prototype_batches"] == 2.0
    assert strategy.last_metrics["prototype_samples"] == 4.0
    loss = strategy.local_loss(
        model=models[0], clean_images=images0, labels=labels, client_id=0
    )
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_augmix_jsd_has_finite_gradients_for_extreme_logits() -> None:
    logits = [
        torch.tensor([[1000.0, -1000.0]], requires_grad=True),
        torch.tensor([[-1000.0, 1000.0]], requires_grad=True),
        torch.tensor([[1000.0, -1000.0]], requires_grad=True),
    ]
    loss = jsd_loss_from_logits(*logits)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(item.grad is not None and torch.isfinite(item.grad).all() for item in logits)


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
    assert configs["fedproto_adapter"]["method"]["baseline"]["prototype_source"] == "local_batches"
    assert all(config["train"]["skip_nonfinite"] is False for config in configs.values())
    assert all(
        config["data"]["scenario_id"] == "cle_hfl_v2_cross_map2_seed0_split0"
        for config in configs.values()
    )


def test_learning_floor_configs_only_change_local_batch_budget(tmp_path) -> None:
    assert LEARNING_FLOOR_BUDGETS == (32, 64)
    for budget in LEARNING_FLOOR_BUDGETS:
        config = learning_floor_config(
            package_root=tmp_path,
            mode="formal",
            local_batches=budget,
            output_root=tmp_path / "outputs",
            device="cpu",
        )
        assert config["method"]["communication"] == "none"
        assert config["method"]["lambda_jsd"] == 0.0
        assert config["method"]["cl_module"] == "none"
        assert config["train"]["rounds"] == 40
        assert config["train"]["max_local_batches"] == budget
        assert config["train"]["batch_size"] == 64
        assert config["train"]["skip_nonfinite"] is False
        assert config["train"]["pretrain_epochs"] == 0


def test_fedmd_four_objective_replication_changes_only_local_objective(tmp_path) -> None:
    configs = {
        arm: fedmd_arm_config(
            arm,
            package_root=tmp_path,
            mode="formal",
            output_root=tmp_path / "outputs",
            device="cpu",
            train_seed=0,
        )
        for arm in FEDMD_OBJECTIVE_ARMS
    }
    assert all(config["method"]["communication"] == "fedmd" for config in configs.values())
    assert all(config["train"]["rounds"] == 40 for config in configs.values())
    assert all(config["train"]["max_local_batches"] == 16 for config in configs.values())
    assert configs["erm"]["method"]["cl_module"] == "none"
    assert configs["cvar_dro"]["method"]["cl_module"] == "cvar_dro"
    assert configs["pew_groupdro"]["method"]["fedease"]["objective"] == "pew_groupdro"
    assert configs["pew_ber"]["method"]["fedease"]["objective"] == "ce_ber"
    assert all("cdep" not in json.dumps(config).lower() for config in configs.values())
    experiment = FedEASEExperiment(configs["pew_ber"])
    assert experiment._communication_strategy.routing == "symmetric"


def test_hfl_context_shards_are_disjoint_and_cover_all_arms() -> None:
    execution_shards = ("cheap_a", "cheap_b", "fedtgp", "rhfl")
    flattened = tuple(arm for shard in execution_shards for arm in SHARDS[shard])
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(ARMS)
    assert selected_arms("cheap_b", "formal") == SHARDS["cheap_b"]
    assert selected_arms("cheap_b", "pairing") == ("local_erm", *SHARDS["cheap_b"])

    practical_shards = ("local", "fedmd", "fedproto", "feddf", "kt_pfl", "fccl", "aughfl", "rahfl")
    practical = tuple(arm for shard in practical_shards for arm in SHARDS[shard])
    assert practical == PRACTICAL_ARMS
    assert selected_arms("feddf", "formal") == ("feddf_fidelity",)


def test_fedmd_four_objective_erm_is_not_silently_reused_as_context_fedmd(tmp_path) -> None:
    common = {
        "package_root": tmp_path / "package",
        "mode": "formal",
        "output_root": tmp_path / "outputs",
        "device": "cpu",
    }
    objective = fedmd_arm_config("erm", train_seed=0, **common)
    context = context_arm_config("fedmd_adapter", **common)
    assert objective["method"]["local_loader_mode"] == "standard"
    assert context["method"].get("local_loader_mode", "augmix") == "augmix"
    assert objective != context


def test_hfl_shard_root_parser_requires_exact_merge_partition(tmp_path) -> None:
    values = [f"{shard}={tmp_path / shard}" for shard in ("cheap_a", "cheap_b", "fedtgp", "rhfl")]
    roots = parse_shard_roots(values)
    assert tuple(roots) == ("cheap_a", "cheap_b", "fedtgp", "rhfl")

    practical = ("local", "fedmd", "fedproto", "feddf", "kt_pfl", "fccl", "aughfl", "rahfl")
    roots = parse_shard_roots([f"{shard}={tmp_path / shard}" for shard in practical])
    assert tuple(roots) == practical


@pytest.mark.parametrize(
    ("profile", "shard_names", "expected_arms"),
    (
        ("full", ("cheap_a", "cheap_b", "fedtgp", "rhfl"), ARMS),
        (
            "practical",
            ("local", "fedmd", "fedproto", "feddf", "kt_pfl", "fccl", "aughfl", "rahfl"),
            PRACTICAL_ARMS,
        ),
    ),
)
def test_hfl_shard_merger_requires_and_combines_matched_formal_outputs(
    tmp_path, profile, shard_names, expected_arms
) -> None:
    roots = {}
    trace_line = json.dumps({"round": 0, "client": 0, "batch": 0, "sha256": "matched"}) + "\n"
    for shard in shard_names:
        root = tmp_path / profile / shard
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
            "protocol": "cle_hfl_context_table_map2_v3",
            "mode": "formal",
            "scenario_id": "cle_hfl_v2_cross_map2_seed0_split0",
            "partition_seed": 0,
            "binding_map_seed": 2,
            "evaluation_seed": 20260909,
            "rounds": 40,
            "local_batches_per_client_round": 16,
            "batch_size": 64,
            "public_batch_size": 128,
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
    merged = merge_shards(roots, tmp_path / profile / "merged", profile)
    assert tuple(merged["rows"]) == expected_arms
    assert merged["merge_profile"] == profile
    assert merged["cross_shard_local_batch_pairing"]["all_arms_match"] is True


def test_domain_table_merger_adds_only_matched_asymhfl_rows(tmp_path) -> None:
    context_root = tmp_path / "context"
    asym_root = tmp_path / "asym"
    output_root = tmp_path / "domain"
    context_root.mkdir()
    (asym_root / "analysis").mkdir(parents=True)
    (asym_root / "configs").mkdir(parents=True)
    trace = {(0, 0, 0): "matched"}
    context = {
        "mode": "formal",
        "scenario_id": "cle_hfl_v2_cross_map2_seed0_split0",
        "partition_seed": 0,
        "binding_map_seed": 2,
        "evaluation_seed": 20260909,
        "train_seed": 0,
        "rounds": 40,
        "rows": {arm: {"pooled_dsa": 0.1} for arm in ARMS},
        "cross_shard_local_batch_pairing": {
            "reference_trace_sha256": trace_digest(trace),
        },
        "scientific_evidence": True,
    }
    (context_root / "RESULT_SUMMARY_MERGED.json").write_text(json.dumps(context), encoding="utf-8")
    common = {
        "labels": np.asarray([0]),
        "binding": np.asarray([[0]]),
        "operator_names": np.asarray(["blur"]),
    }
    np.savez_compressed(
        context_root / "HFL_CONTEXT_PREDICTIONS_MERGED.npz",
        probabilities=np.zeros((len(ARMS), 1, 1, 1, 2)),
        arms=np.asarray(ARMS),
        **common,
    )
    asym = {
        "mode": "formal",
        "scenario_id": "cle_hfl_v2_cross_map2_seed0_split0",
        "train_seed": 0,
        "rounds": 40,
        "scientific_evidence": True,
        "pooled_dsa": {arm: 0.1 for arm in ("erm", "cvar_dro", "pew_groupdro", "pew_ber")},
        "client_dsa": {arm: [0.1] for arm in ("erm", "cvar_dro", "pew_groupdro", "pew_ber")},
        "operator_grid_accuracy": {arm: {"pooled": 20.0} for arm in ("erm", "cvar_dro", "pew_groupdro", "pew_ber")},
        "reporting_metrics": {arm: {"last10": {"avg_acc": 20.0}} for arm in ("erm", "cvar_dro", "pew_groupdro", "pew_ber")},
    }
    (asym_root / "analysis" / "RESULT_SUMMARY.json").write_text(json.dumps(asym), encoding="utf-8")
    contract = {
        "scenario_id": "cle_hfl_v2_cross_map2_seed0_split0",
        "binding_map_seed": 2,
        "partition_seed": 0,
        "evaluation_seed": 20260909,
        "rounds": 40,
        "train_seed": 0,
    }
    (asym_root / "configs" / "SPURIOUS_FINAL_MAP2_CONTRACT_TRAINSEED0.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    arm_root = asym_root / "outputs" / "cle_v2_spurious_final_map2_erm_trainseed0"
    arm_root.mkdir(parents=True)
    (arm_root / "local_batch_trace.jsonl").write_text(
        json.dumps({"round": 0, "client": 0, "batch": 0, "sha256": "matched"}) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        asym_root / "analysis" / "SPURIOUS_FINAL_MAP2_PREDICTIONS.npz",
        probabilities=np.zeros((4, 1, 1, 1, 2)),
        arms=np.asarray(("erm", "cvar_dro", "pew_groupdro", "pew_ber")),
        **common,
    )
    merged = merge_domain_table(context_root, asym_root, output_root)
    assert tuple(merged["rows"]) == FINAL_ROWS
    assert merged["local_batch_pairing"]["matched"] is True
