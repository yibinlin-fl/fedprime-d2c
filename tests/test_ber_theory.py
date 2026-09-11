import numpy as np
import torch

from fedprime.engine.ber_theory import (
    ber_sample_weights,
    class_balanced_sample_weights,
    class_environment_counts,
    environment_advantage_tv_bound_holds,
    maximum_within_class_support_ratio,
    pseudo_to_true_dependence_bound,
    total_variation_dependence,
    weighted_joint,
)
from fedprime.methods.balanced_environment_risk import balanced_environment_risk


def test_gamma_zero_makes_common_support_independent() -> None:
    labels = np.asarray([0] * 10 + [0] * 2 + [1] * 2 + [1] * 10)
    environments = np.asarray([0] * 10 + [1] * 2 + [0] * 2 + [1] * 10)
    counts = class_environment_counts(labels, environments, num_classes=2, num_environments=2)
    weights, _, _ = ber_sample_weights(
        labels, environments, counts,
        support_gamma=0.0, count_cap=32, min_group_count=1,
    )
    joint = weighted_joint(labels, environments, weights, num_classes=2, num_environments=2)
    assert total_variation_dependence(joint) < 1e-12


def test_half_power_contracts_support_ratio() -> None:
    labels = np.asarray([0] * 100 + [0] * 4)
    environments = np.asarray([0] * 100 + [1] * 4)
    counts = class_environment_counts(labels, environments, num_classes=1, num_environments=2)
    reference = class_balanced_sample_weights(labels, num_classes=1)
    weights, _, _ = ber_sample_weights(
        labels, environments, counts,
        support_gamma=0.5, count_cap=32, min_group_count=2,
    )
    before = weighted_joint(labels, environments, reference, num_classes=1, num_environments=2)
    after = weighted_joint(labels, environments, weights, num_classes=1, num_environments=2)
    assert np.isclose(maximum_within_class_support_ratio(before), 25.0)
    assert np.isclose(maximum_within_class_support_ratio(after), np.sqrt(32.0 / 4.0))


def test_environment_only_advantage_is_bounded_by_tv() -> None:
    joint = np.asarray([[0.4, 0.1], [0.1, 0.4]])
    assert environment_advantage_tv_bound_holds(joint)
    assert pseudo_to_true_dependence_bound(0.1, 0.2) == 0.5
    assert pseudo_to_true_dependence_bound(0.7, 0.2) == 1.0


def test_numpy_effective_distribution_matches_production_ber_loss() -> None:
    labels = np.asarray([0] * 10 + [0] * 2 + [1] * 2 + [1] * 10)
    environments = np.asarray([0] * 10 + [1] * 2 + [0] * 2 + [1] * 10)
    losses = np.linspace(0.1, 2.4, len(labels), dtype=np.float64)
    counts = class_environment_counts(labels, environments, num_classes=2, num_environments=2)
    sample_weights, _, _ = ber_sample_weights(
        labels, environments, counts,
        support_gamma=0.5, count_cap=32, min_group_count=1,
    )
    expected = float(np.sum(sample_weights * losses))
    observed, _ = balanced_environment_risk(
        torch.as_tensor(losses, dtype=torch.float64),
        torch.as_tensor(labels),
        torch.as_tensor(environments),
        group_counts=torch.as_tensor(counts, dtype=torch.float64),
        support_gamma=0.5,
        count_cap=32,
        min_group_count=1,
    )
    assert np.isclose(float(observed), expected)
