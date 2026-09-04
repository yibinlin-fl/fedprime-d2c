from __future__ import annotations

import numpy as np
import pytest
import torch
from copy import deepcopy

from fedprime.methods.cvrs import (
    ProbeSchedule,
    calibrated_regularizer_weight,
    centered_class_response,
    cvrs_loss,
    cvrs_statistics,
    pairwise_public_jsd_loss,
    compute_rahfl_augmix_dcl_loss,
)
from fedprime.methods.local_rahfl import train_local_augmix_dcl_epoch


def test_centered_response_removes_common_class_shift() -> None:
    base = torch.zeros(2, 3)
    probes = torch.tensor([[[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]])
    response = centered_class_response(base, probes)
    assert torch.allclose(response, torch.zeros_like(response))


def test_cvrs_is_high_for_persistent_and_zero_for_canceling_routing() -> None:
    base = torch.zeros(2, 2)
    persistent = torch.tensor([[[1.0, -1.0], [1.0, -1.0]]])
    canceling = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    assert cvrs_loss(base, persistent).item() == pytest.approx(1.0)
    assert cvrs_loss(base, canceling).item() == pytest.approx(0.0)


def test_cvrs_bound_and_stop_gradient_denominator() -> None:
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(5, 4, generator=generator, requires_grad=True)
    probes = torch.randn(3, 5, 4, generator=generator, requires_grad=True)
    _mu, _energy, routing = cvrs_statistics(base, probes)
    assert torch.all(routing >= -1.0e-7)
    assert torch.all(routing <= 1.0 + 1.0e-6)
    loss = cvrs_loss(base, probes)
    loss.backward()
    assert torch.isfinite(base.grad).all()
    assert torch.isfinite(probes.grad).all()


def test_public_jsd_is_zero_only_for_matching_predictions() -> None:
    base = torch.tensor([[3.0, -1.0], [-2.0, 2.0]])
    same = base.unsqueeze(0).repeat(2, 1, 1)
    changed = same.clone()
    changed[0, 0] = torch.tensor([-1.0, 3.0])
    assert pairwise_public_jsd_loss(base, same).item() == pytest.approx(0.0, abs=1.0e-7)
    assert pairwise_public_jsd_loss(base, changed).item() > 0.0


def test_gradient_ratio_calibration() -> None:
    value = calibrated_regularizer_weight(20.0, 4.0, ratio=0.1, eps=0.0)
    assert value == pytest.approx(0.5)


def test_probe_schedule_covers_bank_once_per_cycle_and_is_deterministic() -> None:
    left = ProbeSchedule(bank_size=64, probes_per_update=4, seed=20260905)
    right = ProbeSchedule(bank_size=64, probes_per_update=4, seed=20260905)
    left_ids = np.concatenate([left.next_ids() for _ in range(16)])
    right_ids = np.concatenate([right.next_ids() for _ in range(16)])
    assert np.array_equal(left_ids, right_ids)
    assert np.array_equal(np.sort(left_ids), np.arange(64))
    next_cycle = np.concatenate([left.next_ids() for _ in range(16)])
    assert np.array_equal(np.sort(next_cycle), np.arange(64))
    assert not np.array_equal(left_ids, next_cycle)


def test_private_loss_matches_original_rahfl_one_batch() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 8))
            self.linear = torch.nn.Linear(8, 3)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.linear(self.backbone(values))

    torch.manual_seed(11)
    model = TinyModel()
    reference = deepcopy(model)
    views = [torch.randn(6, 3, 4, 4) for _ in range(4)]
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    expected = compute_rahfl_augmix_dcl_loss(
        model, views, labels, device=torch.device("cpu"), lambda_jsd=12.0
    )
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.0)
    observed = train_local_augmix_dcl_epoch(
        reference,
        [(views, labels)],
        optimizer,
        torch.device("cpu"),
        lambda_jsd=12.0,
        cl_module="dcl",
    )
    assert observed == pytest.approx(float(expected.detach()), rel=1.0e-6, abs=1.0e-6)
