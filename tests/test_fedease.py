from __future__ import annotations

import torch

from fedprime.methods.balanced_environment_risk import balanced_environment_risk
from fedprime.methods.conditional_dependence import (
    FrozenRandomProjector,
    normalized_conditional_cross_covariance,
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


def test_conditional_dependence_detects_within_class_environment_signal():
    generator = torch.Generator().manual_seed(7)
    environments = torch.arange(400) % 2
    labels = torch.zeros(400, dtype=torch.long)
    independent = torch.randn(400, 8, generator=generator)
    correlated = independent.clone()
    correlated[:, 0] = environments.float() * 2.0 - 1.0

    independent_loss, _ = normalized_conditional_cross_covariance(
        independent,
        labels,
        environments,
        num_environments=2,
    )
    correlated_loss, diagnostics = normalized_conditional_cross_covariance(
        correlated,
        labels,
        environments,
        num_environments=2,
    )

    assert correlated_loss > independent_loss * 3.0
    assert diagnostics["valid_classes"].item() == 1


def test_conditional_dependence_ignores_only_cross_class_environment_association():
    labels = torch.tensor([0] * 16 + [1] * 16)
    environments = labels.clone()
    features = torch.randn(32, 8, generator=torch.Generator().manual_seed(11))

    loss, diagnostics = normalized_conditional_cross_covariance(
        features,
        labels,
        environments,
        num_environments=2,
    )

    assert loss.item() == 0.0
    assert diagnostics["valid_classes"].item() == 0


def test_conditional_dependence_is_invariant_to_feature_scale():
    generator = torch.Generator().manual_seed(13)
    labels = torch.zeros(128, dtype=torch.long)
    environments = torch.arange(128) % 2
    features = torch.randn(128, 6, generator=generator)
    features[:, 0] += environments.float()

    loss_a, _ = normalized_conditional_cross_covariance(
        features,
        labels,
        environments,
        num_environments=2,
    )
    loss_b, _ = normalized_conditional_cross_covariance(
        features * 17.0,
        labels,
        environments,
        num_environments=2,
    )

    assert torch.allclose(loss_a, loss_b, rtol=1.0e-4, atol=1.0e-6)


def test_frozen_random_projector_is_deterministic_and_backpropagates_to_features():
    features = torch.randn(12, 10, requires_grad=True)
    projector_a = FrozenRandomProjector(output_dim=5, seed=23)
    projector_b = FrozenRandomProjector(output_dim=5, seed=23)

    projected_a = projector_a(features)
    projected_b = projector_b(features)

    assert torch.allclose(projected_a, projected_b)
    projected_a.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
