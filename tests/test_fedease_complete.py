from __future__ import annotations

from pathlib import Path

import torch

from fedprime.methods.environment_structural_transfer import (
    aggregate_environment_balanced_relations,
    ebst_alignment_loss,
    finalize_client_relations,
    new_relation_accumulator,
    normalize_margin_rows,
    update_relation_accumulator,
)
from fedprime.methods.environment_witness import PublicEnvironmentWitness
from fedprime.methods.safe_communication_projection import project_communication_gradients
from fedprime.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_public_environment_witness_outputs_environment_severity_and_embedding():
    model = PublicEnvironmentWitness(embedding_dim=16, num_environments=6, severity_levels=5)
    environment, severity, embedding = model(torch.rand(4, 3, 32, 32))
    assert environment.shape == (4, 6)
    assert severity.shape == (4, 5)
    assert embedding.shape == (4, 16)


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


def test_openi_oracle_probe_configs_are_fair_except_fedease_switches():
    control = load_config(ROOT / "configs/openi_v100_fedease_oracle_control_probe.yaml")
    candidate = load_config(ROOT / "configs/openi_v100_fedease_oracle_ber_cdep_probe.yaml")
    for key in ("data", "models", "train", "seed", "device"):
        assert control[key] == candidate[key]
    assert control["method"]["fedease"]["ber"]["enabled"] is False
    assert candidate["method"]["fedease"]["ber"]["enabled"] is True
    assert control["method"]["fedease"]["cdep"]["enabled"] is False
    assert candidate["method"]["fedease"]["cdep"]["enabled"] is True
