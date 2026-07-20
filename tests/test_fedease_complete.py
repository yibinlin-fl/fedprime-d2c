from __future__ import annotations

from pathlib import Path

import torch

from fedprime.methods.environment_structural_transfer import (
    aggregate_environment_balanced_relations,
    aggregate_leave_one_out_pair_relations,
    ebst_alignment_loss,
    finalize_client_relations,
    finalize_pair_qualified_client_relations,
    new_relation_accumulator,
    normalize_margin_rows,
    update_relation_accumulator,
)
from fedprime.methods.environment_witness import PublicEnvironmentWitness, select_unknown_threshold
from fedprime.methods.safe_communication_projection import (
    project_classifier_gradients_by_class,
    project_communication_gradients,
)
from fedprime.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_public_environment_witness_outputs_environment_severity_and_embedding():
    model = PublicEnvironmentWitness(embedding_dim=16, num_environments=6, severity_levels=5)
    environment, severity, embedding = model(torch.rand(4, 3, 32, 32))
    assert environment.shape == (4, 6)
    assert severity.shape == (4, 5)
    assert embedding.shape == (4, 16)


def test_unknown_threshold_calibration_prefers_public_validation_accuracy():
    probabilities = torch.tensor(
        [
            [0.60, 0.10, 0.10, 0.10, 0.05, 0.05],
            [0.24, 0.20, 0.16, 0.15, 0.13, 0.12],
            [0.05, 0.05, 0.05, 0.05, 0.10, 0.70],
        ]
    )
    targets = torch.tensor([0, 5, 5])
    result = select_unknown_threshold(
        probabilities,
        targets,
        unknown_id=5,
        thresholds=torch.tensor([0.0, 0.3, 0.8]),
    )
    assert result["threshold"] == torch.tensor(0.3).item()
    assert result["accuracy"] == 1.0


def test_relation_rows_are_invariant_to_positive_logit_scale():
    rows = torch.tensor([[[0.0, 1.0, 3.0]], [[2.0, 0.0, 4.0]], [[1.0, 2.0, 0.0]]])
    valid = torch.ones(3, 1, dtype=torch.bool)
    first = normalize_margin_rows(rows, valid)
    second = normalize_margin_rows(rows * 17.0 + 5.0, valid)
    assert torch.allclose(first, second, atol=1.0e-5)


def test_environment_balanced_aggregation_does_not_weight_by_sample_count():
    client_states = {
        0: {
            "relations": torch.tensor([[[0.0, 1.0], [0.0, 3.0]], [[-1.0, 0.0], [-3.0, 0.0]]]),
            "support": torch.ones(2, 2, dtype=torch.bool),
        },
        1: {
            "relations": torch.tensor([[[0.0, 1.0], [0.0, 3.0]], [[-1.0, 0.0], [-3.0, 0.0]]]),
            "support": torch.ones(2, 2, dtype=torch.bool),
        },
    }
    result = aggregate_environment_balanced_relations(
        client_states,
        use_stability_gate=False,
        variance_temperature=0.5,
    )
    assert torch.allclose(result["global_relation"], torch.tensor([[0.0, 2.0], [-2.0, 0.0]]))


def test_stability_gate_suppresses_environment_conflict():
    state = {
        0: {
            "relations": torch.tensor([[[0.0, 1.0, 1.0], [0.0, 1.0, -3.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "support": torch.tensor([[True, True], [False, False], [False, False]]),
        }
    }
    result = aggregate_environment_balanced_relations(
        state,
        use_stability_gate=True,
        variance_temperature=0.5,
    )
    assert result["gate"][0, 2] < result["gate"][0, 1]


def test_ebst_v2_excludes_source_without_competing_class_support():
    def state(row_value: float, class_one_count: float):
        counts = torch.tensor([[8.0], [class_one_count]])
        return {
            "relations": torch.tensor([[[0.0, row_value]], [[-row_value, 0.0]]]),
            "support": counts >= 4,
            "competing_class_support": counts.sum(dim=1) >= 4,
        }

    result = aggregate_leave_one_out_pair_relations(
        {
            0: state(99.0, 8.0),
            1: state(100.0, 0.0),
            2: state(1.0, 8.0),
            3: state(3.0, 8.0),
        },
        min_source_clients=2,
        use_stability_gate=False,
        variance_temperature=0.5,
    )
    recipient = result["recipients"][0]
    assert recipient["global_valid"][0, 1]
    assert recipient["source_count"][0, 0, 1] == 2
    assert torch.allclose(recipient["global_relation"][0, 1], torch.tensor(2.0))


def test_ebst_v2_leave_one_out_and_source_gate_suppress_client_disagreement():
    def state(stable: float, conflicting: float):
        counts = torch.full((3, 1), 8.0)
        return {
            "relations": torch.tensor(
                [[[0.0, stable, conflicting]], [[-stable, 0.0, 1.0]], [[-conflicting, -1.0, 0.0]]]
            ),
            "support": torch.ones(3, 1, dtype=torch.bool),
            "competing_class_support": counts.sum(dim=1) >= 4,
        }

    result = aggregate_leave_one_out_pair_relations(
        {
            0: state(50.0, 50.0),
            1: state(1.0, -3.0),
            2: state(1.0, 3.0),
            3: state(1.0, 0.0),
        },
        min_source_clients=2,
        use_stability_gate=True,
        variance_temperature=0.5,
    )
    recipient = result["recipients"][0]
    assert torch.allclose(recipient["global_relation"][0, 1], torch.tensor(1.0))
    assert recipient["gate"][0, 2] < recipient["gate"][0, 1]


def test_ebst_v2_client_state_exposes_support_mask_not_exact_counts():
    accumulator = new_relation_accumulator(2, 1)
    logits = torch.tensor([[3.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0, 1])
    environments = torch.zeros(3, dtype=torch.long)
    update_relation_accumulator(accumulator, logits, labels, environments)
    state = finalize_pair_qualified_client_relations(
        accumulator,
        min_group_support=1,
        min_competing_class_support=2,
    )
    assert "count" not in state
    assert state["competing_class_support"].tolist() == [True, False]


def test_relation_accumulator_and_alignment_loss_backpropagate_to_logits():
    accumulator = new_relation_accumulator(3, 2)
    logits = torch.tensor([[3.0, 1.0, 0.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    environments = torch.tensor([0, 1, 0])
    update_relation_accumulator(accumulator, logits, labels, environments)
    state = finalize_client_relations(accumulator, min_group_support=1)
    global_state = aggregate_environment_balanced_relations(
        {0: state}, use_stability_gate=False, variance_temperature=0.5
    )
    train_logits = logits.clone().requires_grad_(True)
    loss, diagnostics = ebst_alignment_loss(
        train_logits,
        labels,
        global_state["global_relation"],
        global_state["gate"],
        global_state["global_valid"],
    )
    loss.backward()
    assert diagnostics["active_samples"] > 0
    assert train_logits.grad is not None
    assert torch.isfinite(train_logits.grad).all()


def test_safe_projection_removes_negative_first_order_dot_product():
    primary = [torch.tensor([1.0, 0.0])]
    communication = [torch.tensor([-1.0, 1.0])]
    projected, diagnostics = project_communication_gradients(
        primary, communication, enabled=True
    )
    assert diagnostics["conflict"] == 1.0
    assert torch.dot(primary[0], projected[0]) >= -1.0e-7


def test_classwise_safe_projection_prevents_hidden_class_conflicts_and_caps_norm():
    primary = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.zeros(2)]
    communication = [torch.tensor([[-1.0, 1.0], [0.0, 2.0]]), torch.zeros(2)]
    projected, diagnostics = project_classifier_gradients_by_class(
        primary,
        communication,
        enabled=True,
        max_communication_norm_ratio=1.0,
    )
    for class_id in range(2):
        dot = sum(
            first[class_id].mul(second[class_id]).sum()
            for first, second in zip(primary, projected)
        )
        primary_norm = sum(value[class_id].square().sum() for value in primary).sqrt()
        projected_norm = sum(value[class_id].square().sum() for value in projected).sqrt()
        assert dot >= -1.0e-7
        assert projected_norm <= primary_norm + 1.0e-7
    assert diagnostics["conflict"] == 0.5


def test_openi_oracle_probe_configs_are_fair_except_fedease_switches():
    control = load_config(ROOT / "configs/openi_v100_fedease_oracle_control_probe.yaml")
    candidate = load_config(ROOT / "configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml")
    for key in ("data", "models", "train", "seed", "device"):
        assert control[key] == candidate[key]
    assert control["method"]["fedease"]["ber"]["enabled"] is False
    assert candidate["method"]["fedease"]["ber"]["enabled"] is True
    assert control["method"]["fedease"]["cdep"]["enabled"] is False
    assert candidate["method"]["fedease"]["cdep"]["enabled"] is True


def test_ebst_v2_probe_matches_local_probe_training_budget():
    local = load_config(ROOT / "configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml")
    candidate = load_config(ROOT / "configs/openi_v100_fedease_ebst_v2_probe.yaml")
    for key in ("data", "models", "train", "seed", "device"):
        assert local[key] == candidate[key]
    assert candidate["method"]["communication"] == "ebst_v2"
    assert candidate["method"]["fedease"]["ebst"]["version"] == 2
    assert candidate["method"]["fedease"]["scp"]["scope"] == "classifier_class"


def test_pew_ebst_v2_probe_combines_only_validated_switches():
    local = load_config(ROOT / "configs/openi_v100_fedease_pew_probe.yaml")
    candidate = load_config(ROOT / "configs/openi_v100_fedease_pew_ebst_v2_probe.yaml")
    for key in ("data", "models", "train", "seed", "device"):
        assert local[key] == candidate[key]
    assert candidate["method"]["communication"] == "ebst_v2"
    assert candidate["method"]["fedease"]["environment_mode"] == "learned"
    assert candidate["method"]["fedease"]["pew"]["unknown_threshold"] == "auto"
    assert candidate["method"]["fedease"]["ebst"]["version"] == 2
    assert candidate["method"]["fedease"]["scp"]["scope"] == "classifier_class"
