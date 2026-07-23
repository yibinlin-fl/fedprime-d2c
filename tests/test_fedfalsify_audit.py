from __future__ import annotations

import numpy as np
import pytest
import torch

from fedprime.data.fedfalsify import stratified_fit_audit_indices
from fedprime.methods.fedfalsify.evidence import (
    classwise_accuracy_tensor,
    compute_paired_advantage,
    planned_stratified_audit_counts,
)
from fedprime.methods.fedfalsify.transfer import (
    conservative_margin_transfer_loss,
    direct_peer_kd_loss,
    fixed_margin_loss,
    gradient_cosine_from_losses,
    normalize_logits,
)
from fedprime.methods.fedfalsify.router import (
    ClassRoute,
    FedFalsifyTransferPlan,
)
from scripts.audit_fedfalsify_source_ranking import summarize_policy


def test_paired_advantage_rewards_corrections_and_penalizes_regressions() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    receiver = np.asarray([0, 1, 1, 0, 1, 0])
    source = np.asarray([0, 0, 0, 1, 1, 1])

    evidence = compute_paired_advantage(
        source,
        receiver,
        labels,
        class_id=0,
        kappa=0.0,
        shrinkage_nu=0.0,
        min_count=2,
    )

    assert evidence.count == 4
    assert evidence.source_accuracy == pytest.approx(0.75)
    assert evidence.receiver_accuracy == pytest.approx(0.50)
    assert evidence.paired_advantage == pytest.approx(0.25)
    assert evidence.advantage_strength == pytest.approx(0.25)
    assert evidence.is_active


def test_paired_advantage_disables_unsupported_class() -> None:
    evidence = compute_paired_advantage(
        [0, 0],
        [0, 0],
        [0, 0],
        class_id=1,
        min_count=1,
    )

    assert evidence.count == 0
    assert not evidence.is_auditable
    assert not evidence.is_active
    assert np.isnan(evidence.paired_advantage)


def test_classwise_accuracy_tensor_uses_receiver_specific_labels() -> None:
    predictions = np.asarray([
        [[0, 1, 0, 1], [1, 1, 0, 0]],
        [[0, 0, 0, 1], [1, 0, 0, 0]],
    ])
    labels = np.asarray([
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ])

    accuracy, counts = classwise_accuracy_tensor(
        predictions,
        labels,
        num_classes=2,
    )

    assert accuracy.shape == (2, 2, 2)
    assert counts.tolist() == [[2, 2], [2, 2]]
    assert accuracy[0, 0].tolist() == pytest.approx([0.5, 0.5])
    assert accuracy[1, 1].tolist() == pytest.approx([1.0, 0.5])


def test_planned_audit_counts_preserve_at_least_one_fit_sample() -> None:
    labels = np.asarray([0, 1, 1] + [2] * 10)
    counts = planned_stratified_audit_counts(
        labels,
        num_classes=4,
        audit_ratio=0.15,
    )

    assert counts.tolist() == [0, 1, 2, 0]


def test_logit_import_does_not_change_torch_default_dtype() -> None:
    assert torch.get_default_dtype() == torch.float32


def test_normalize_logits_removes_per_sample_shift_and_positive_scale() -> None:
    logits = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.0, 3.0]])
    transformed = logits * torch.tensor([[3.0], [0.5]]) + torch.tensor([[7.0], [-4.0]])

    assert torch.allclose(normalize_logits(logits), normalize_logits(transformed), atol=1e-5)


def test_cmt_is_zero_when_receiver_has_stronger_margins() -> None:
    source = torch.tensor([[2.0, 0.0, -1.0]])
    receiver = torch.tensor([[4.0, -2.0, -3.0]], requires_grad=True)
    labels = torch.tensor([0])

    loss = conservative_margin_transfer_loss(receiver, source, labels)

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert receiver.grad is not None
    assert torch.isfinite(receiver.grad).all()


def test_cmt_ignores_incorrect_source_prediction() -> None:
    source = torch.tensor([[0.0, 3.0, -1.0]])
    receiver = torch.tensor([[0.5, 0.0, -0.5]], requires_grad=True)
    labels = torch.tensor([0])

    loss = conservative_margin_transfer_loss(
        receiver,
        source,
        labels,
        source_correct_only=True,
    )

    assert loss.item() == pytest.approx(0.0)


def test_fixed_margin_and_direct_kd_are_finite() -> None:
    receiver = torch.tensor([[0.2, 0.1, -0.2], [0.0, 0.3, -0.1]], requires_grad=True)
    source = torch.tensor([[1.2, 0.0, -0.7], [-0.1, 1.1, 0.0]])
    labels = torch.tensor([0, 1])

    margin_loss = fixed_margin_loss(receiver, labels)
    kd_loss = direct_peer_kd_loss(receiver, source)

    assert torch.isfinite(margin_loss)
    assert torch.isfinite(kd_loss)
    assert margin_loss.item() >= 0.0
    assert kd_loss.item() >= 0.0


def test_gradient_cosine_reports_aligned_and_conflicting_directions() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    aligned_first = (parameter ** 2).sum()
    aligned_second = (2.0 * parameter ** 2).sum()
    cosine, first_norm, second_norm = gradient_cosine_from_losses(
        aligned_first,
        aligned_second,
        [parameter],
    )
    assert cosine == pytest.approx(1.0)
    assert first_norm > 0.0
    assert second_norm > 0.0

    conflicting_first = (parameter ** 2).sum()
    conflicting_second = -(parameter ** 2).sum()
    cosine, _, _ = gradient_cosine_from_losses(
        conflicting_first,
        conflicting_second,
        [parameter],
    )
    assert cosine == pytest.approx(-1.0)


def test_tau_top1_selects_highest_positive_utility() -> None:
    grouped = {
        (0, 2): [
            {
                "source_client": 1.0,
                "advantage_gate_active": False,
                "advantage_strength": 0.0,
                "action_utility": 0.2,
                "cmt_increment": 0.01,
            },
            {
                "source_client": 3.0,
                "advantage_gate_active": True,
                "advantage_strength": 0.1,
                "action_utility": 0.8,
                "cmt_increment": 0.03,
            },
        ]
    }

    summary, selections = summarize_policy("tau_top1", grouped)

    assert summary["coverage"] == 1.0
    assert summary["positive_precision"] == 1.0
    assert selections[0]["source_client"] == 3
    assert selections[0]["cmt_increment_over_ce"] == pytest.approx(0.03)


def test_fra_top1_can_abstain() -> None:
    grouped = {
        (1, 4): [
            {
                "source_client": 0.0,
                "advantage_gate_active": False,
                "advantage_strength": 0.0,
                "action_utility": 0.5,
                "cmt_increment": 0.01,
            }
        ]
    }

    summary, selections = summarize_policy("fra_top1", grouped)

    assert summary["coverage"] == 0.0
    assert selections[0]["selected"] == 0


def test_stratified_fit_audit_split_is_disjoint_complete_and_repeatable() -> None:
    labels = np.asarray([0] * 3 + [1] * 20 + [2] * 8)
    first_fit, first_audit = stratified_fit_audit_indices(
        labels,
        audit_ratio=0.2,
        min_audit_per_class=5,
        min_fit_per_class=2,
        seed=17,
    )
    second_fit, second_audit = stratified_fit_audit_indices(
        labels,
        audit_ratio=0.2,
        min_audit_per_class=5,
        min_fit_per_class=2,
        seed=17,
    )

    assert np.array_equal(first_fit, second_fit)
    assert np.array_equal(first_audit, second_audit)
    assert np.intersect1d(first_fit, first_audit).size == 0
    assert np.array_equal(
        np.sort(np.concatenate([first_fit, first_audit])),
        np.arange(len(labels)),
    )
    assert np.sum(labels[first_audit] == 0) == 0
    assert np.sum(labels[first_audit] == 1) == 5
    assert np.sum(labels[first_audit] == 2) == 5


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, images):
        return self.linear(images)


def test_transfer_plan_applies_only_to_routed_classes() -> None:
    source = _TinyModel()
    with torch.no_grad():
        source.linear.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    for parameter in source.parameters():
        parameter.requires_grad_(False)

    route = ClassRoute(
        receiver_id=0,
        class_id=0,
        source_id=1,
        tau=0.5,
        fra_strength=0.0,
        fra_advantage=0.0,
        fit_count=8,
        audit_count=5,
    )
    plan = FedFalsifyTransferPlan(
        snapshots={1: source},
        routes={0: {0: route}},
        lambda_cmt=0.5,
        margin_clip=2.0,
        source_correct_only=True,
    )
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    receiver_logits = torch.tensor(
        [[0.1, 0.0], [0.0, 0.1]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1])

    loss = plan.loss_for_batch(
        receiver_id=0,
        receiver_logits=receiver_logits,
        clean_images=images,
        labels=labels,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert plan.diagnostics()["cmt_active_samples"] == 1.0
