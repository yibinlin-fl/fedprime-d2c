from __future__ import annotations

import torch

from fedprime.methods.balanced_environment_risk import (
    balanced_environment_risk,
    soft_balanced_environment_risk,
)


def test_ber_turns_nine_to_one_group_frequency_into_equal_group_risk():
    sample_losses = torch.tensor([1.0] * 9 + [9.0], requires_grad=True)
    labels = torch.zeros(10, dtype=torch.long)
    environments = torch.tensor([0] * 9 + [1], dtype=torch.long)
    counts = torch.tensor([[9.0, 1.0]])

    loss, diagnostics = balanced_environment_risk(
        sample_losses,
        labels,
        environments,
        group_counts=counts,
        support_gamma=0.0,
    )

    assert torch.allclose(loss, torch.tensor(5.0))
    assert diagnostics["valid_groups"].item() == 2
    assert diagnostics["effective_groups_per_class"].item() == 2
    loss.backward()
    assert sample_losses.grad is not None
    assert torch.isfinite(sample_losses.grad).all()


def test_ber_does_not_fill_a_missing_environment_with_zero():
    sample_losses = torch.tensor([2.0, 4.0], requires_grad=True)
    labels = torch.zeros(2, dtype=torch.long)
    environments = torch.zeros(2, dtype=torch.long)
    counts = torch.tensor([[2.0, 0.0]])

    loss, diagnostics = balanced_environment_risk(
        sample_losses,
        labels,
        environments,
        group_counts=counts,
        support_gamma=0.0,
    )

    assert torch.allclose(loss, torch.tensor(3.0))
    assert diagnostics["valid_groups"].item() == 1


def test_soft_ber_fractionally_assigns_loss_to_multiple_environments():
    sample_losses = torch.tensor([2.0, 8.0], requires_grad=True)
    labels = torch.zeros(2, dtype=torch.long)
    probabilities = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    counts = probabilities.sum(dim=0, keepdim=True)

    loss, diagnostics = soft_balanced_environment_risk(
        sample_losses,
        labels,
        probabilities,
        group_counts=counts,
        support_gamma=0.0,
        min_group_count=0.5,
    )

    # R_e0=(2+4)/1.5=4 and R_e1=4/0.5=8; exact environment balance gives 6.
    assert torch.allclose(loss, torch.tensor(6.0))
    assert diagnostics["valid_groups"].item() == 2
    loss.backward()
    assert torch.isfinite(sample_losses.grad).all()
