from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from fedprime.methods.cvrs import ProbeSchedule
from fedprime.methods.lcre import (
    center_logits,
    compute_class_response_stats,
    compute_lcre_loss,
    freeze_bn_running_stats,
)


def _logits_from_response(response: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.zeros(response.shape[1:], dtype=response.dtype, device=response.device)
    return base, response


def test_zero_conditional_dependence_with_nonzero_response() -> None:
    response = torch.tensor(
        [[[2.0, -2.0], [2.0, -2.0], [2.0, -2.0], [2.0, -2.0]]]
    )
    base, probes = _logits_from_response(response)
    loss, stats = compute_lcre_loss(base, probes, torch.tensor([0, 0, 1, 1]))
    assert response.square().sum() > 0
    assert loss.item() == pytest.approx(0.0, abs=1.0e-8)
    assert not stats.skipped


def test_positive_class_dependence_matches_hand_calculation() -> None:
    response = torch.tensor(
        [[[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]]]
    )
    base, probes = _logits_from_response(response)
    loss, stats = compute_lcre_loss(base, probes, torch.tensor([0, 0, 1, 1]))
    assert stats.between_class.item() == pytest.approx(2.0)
    assert stats.balanced_energy.item() == pytest.approx(2.0)
    assert loss.item() == pytest.approx(1.0)


def test_common_logit_shift_invariance() -> None:
    generator = torch.Generator().manual_seed(4)
    base = torch.randn(6, 4, generator=generator)
    probes = torch.randn(3, 6, 4, generator=generator)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    expected, _ = compute_lcre_loss(base, probes, labels)
    observed, _ = compute_lcre_loss(base + 9.0, probes - 13.0, labels)
    assert observed.item() == pytest.approx(expected.item(), rel=1.0e-6, abs=1.0e-7)
    assert torch.allclose(center_logits(torch.ones(2, 4) * 7.0), torch.zeros(2, 4))


def test_label_name_permutation_invariance() -> None:
    generator = torch.Generator().manual_seed(5)
    base = torch.randn(6, 4, generator=generator)
    probes = torch.randn(2, 6, 4, generator=generator)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    renamed = torch.tensor([7, 7, 3, 3, 9, 9])
    left, _ = compute_lcre_loss(base, probes, labels)
    right, _ = compute_lcre_loss(base, probes, renamed)
    assert right.item() == pytest.approx(left.item(), rel=1.0e-7, abs=1.0e-8)


def test_response_scale_value_invariance() -> None:
    generator = torch.Generator().manual_seed(6)
    response = torch.randn(3, 6, 5, generator=generator)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    base, probes = _logits_from_response(response)
    left, _ = compute_lcre_loss(base, probes, labels, eps=0.0)
    right, _ = compute_lcre_loss(base, probes * 3.7, labels, eps=0.0)
    assert right.item() == pytest.approx(left.item(), rel=1.0e-6, abs=1.0e-7)


def test_denominator_is_stop_gradient() -> None:
    generator = torch.Generator().manual_seed(7)
    probes = torch.randn(2, 6, 4, generator=generator, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    base = torch.zeros(6, 4)
    loss, stats = compute_lcre_loss(base, probes, labels)
    expected = (stats.between_class / (stats.balanced_energy.detach() + 1.0e-8)).mean()
    observed_grad = torch.autograd.grad(loss, probes, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, probes)[0]
    assert torch.allclose(observed_grad, expected_grad, atol=1.0e-8, rtol=1.0e-6)


@pytest.mark.parametrize(
    ("labels", "skipped", "active", "singletons"),
    [
        ([0, 0, 1, 1], False, 2, 0),
        ([0, 0, 1], True, 1, 1),
        ([0, 1, 2], True, 0, 3),
        ([0, 0, 1, 1, 2], False, 2, 1),
    ],
)
def test_active_class_handling(labels, skipped, active, singletons) -> None:
    response = torch.randn(2, len(labels), 3, generator=torch.Generator().manual_seed(8))
    stats = compute_class_response_stats(response, torch.tensor(labels))
    assert stats.skipped is skipped
    assert stats.active_classes.numel() == active
    assert stats.singleton_count == singletons
    if skipped:
        assert torch.equal(stats.normalized, torch.zeros_like(stats.normalized))


def test_gradient_matches_direct_reference() -> None:
    generator = torch.Generator().manual_seed(9)
    base = torch.randn(6, 4, generator=generator, requires_grad=True)
    probes = torch.randn(2, 6, 4, generator=generator, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss, _ = compute_lcre_loss(base, probes, labels)
    response = center_logits(probes - base.unsqueeze(0))
    means = torch.stack([response[:, labels == value].mean(1) for value in (0, 1, 2)], 1)
    numerator = (means - means.mean(1, keepdim=True)).square().sum(-1).mean(1)
    energy = torch.stack(
        [response[:, labels == value].square().sum(-1).mean(1) for value in (0, 1, 2)], 1
    ).mean(1)
    reference = (numerator / (energy.detach() + 1.0e-8)).mean()
    actual_grad = torch.autograd.grad(loss, (base, probes), retain_graph=True)
    reference_grad = torch.autograd.grad(reference, (base, probes))
    assert loss.item() == pytest.approx(reference.item(), rel=1.0e-7, abs=1.0e-8)
    for actual, expected in zip(actual_grad, reference_grad):
        assert torch.allclose(actual, expected, atol=1.0e-8, rtol=1.0e-6)


def test_bn_running_stats_freeze_and_restoration() -> None:
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 2, 1))
    model.train()
    bn = model[0]
    before = (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
    with freeze_bn_running_stats(model):
        assert not bn.track_running_stats
        model(torch.randn(4, 3, 4, 4)).sum().backward()
    assert bn.track_running_stats
    after = (bn.running_mean, bn.running_var, bn.num_batches_tracked)
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert bn.weight.grad is not None and torch.isfinite(bn.weight.grad).all()


def test_runner_matching_schedule_and_trace_hashes() -> None:
    left = ProbeSchedule(bank_size=64, probes_per_update=4, seed=20260905)
    right = ProbeSchedule(bank_size=64, probes_per_update=4, seed=20260905)
    left_ids = np.concatenate([left.next_ids() for _ in range(16)])
    right_ids = np.concatenate([right.next_ids() for _ in range(16)])
    assert np.array_equal(left_ids, right_ids)
    assert np.array_equal(np.sort(left_ids), np.arange(64))
    assert hashlib.sha256(left_ids.tobytes()).hexdigest() == hashlib.sha256(right_ids.tobytes()).hexdigest()
