from __future__ import annotations

import numpy as np


def class_environment_counts(
    labels: np.ndarray,
    environments: np.ndarray,
    *,
    num_classes: int,
    num_environments: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    environments = np.asarray(environments, dtype=np.int64)
    if labels.shape != environments.shape or labels.ndim != 1:
        raise ValueError("labels and environments must be aligned one-dimensional arrays")
    if labels.size and ((labels < 0).any() or (labels >= num_classes).any()):
        raise ValueError("label out of range")
    if environments.size and (
        (environments < 0).any() or (environments >= num_environments).any()
    ):
        raise ValueError("environment out of range")
    flat = labels * int(num_environments) + environments
    return np.bincount(
        flat, minlength=int(num_classes) * int(num_environments)
    ).reshape(int(num_classes), int(num_environments)).astype(np.float64)


def ber_environment_weights(
    counts: np.ndarray,
    *,
    support_gamma: float,
    count_cap: int,
    min_group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact class-conditional group masses used by hard BER."""

    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or (counts < 0).any():
        raise ValueError("counts must be a non-negative matrix")
    if not 0.0 <= support_gamma <= 1.0:
        raise ValueError("support_gamma must be in [0, 1]")
    if count_cap < 1 or min_group_count < 1:
        raise ValueError("count_cap and min_group_count must be positive")
    valid = counts >= float(min_group_count)
    support = np.minimum(counts, float(count_cap)) ** float(support_gamma)
    support = np.where(valid, support, 0.0)
    denominators = support.sum(axis=1, keepdims=True)
    weights = np.divide(
        support,
        denominators,
        out=np.zeros_like(support),
        where=denominators > 0,
    )
    return weights, valid


def class_balanced_sample_weights(labels: np.ndarray, *, num_classes: int) -> np.ndarray:
    """Class-balanced ERM reference distribution over the observed samples."""

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float64)
    valid = counts > 0
    class_count = int(valid.sum())
    if class_count == 0:
        raise ValueError("no observed class")
    weights = 1.0 / (float(class_count) * counts[labels])
    return weights / weights.sum()


def ber_sample_weights(
    labels: np.ndarray,
    environments: np.ndarray,
    counts: np.ndarray,
    *,
    support_gamma: float,
    count_cap: int,
    min_group_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized per-sample mass equivalent to the dataset-level hard BER objective."""

    labels = np.asarray(labels, dtype=np.int64)
    environments = np.asarray(environments, dtype=np.int64)
    env_weights, valid = ber_environment_weights(
        counts,
        support_gamma=support_gamma,
        count_cap=count_cap,
        min_group_count=min_group_count,
    )
    valid_classes = valid.any(axis=1)
    class_count = int(valid_classes.sum())
    if class_count == 0:
        raise ValueError("BER has no valid class")
    group_counts = np.asarray(counts, dtype=np.float64)[labels, environments]
    sample_weights = np.divide(
        env_weights[labels, environments],
        group_counts * float(class_count),
        out=np.zeros(labels.shape, dtype=np.float64),
        where=group_counts > 0,
    )
    total = sample_weights.sum()
    if total <= 0:
        raise ValueError("BER sample weights have zero mass")
    return sample_weights / total, env_weights, valid


def weighted_joint(
    labels: np.ndarray,
    environments: np.ndarray,
    sample_weights: np.ndarray,
    *,
    num_classes: int,
    num_environments: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    environments = np.asarray(environments, dtype=np.int64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if labels.shape != environments.shape or labels.shape != weights.shape:
        raise ValueError("labels, environments, and weights must align")
    joint = np.zeros((int(num_classes), int(num_environments)), dtype=np.float64)
    np.add.at(joint, (labels, environments), weights)
    total = joint.sum()
    if total <= 0:
        raise ValueError("joint has zero mass")
    return joint / total


def total_variation_dependence(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=np.float64)
    product = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    return float(0.5 * np.abs(joint - product).sum())


def mutual_information(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=np.float64)
    product = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    mask = (joint > 0) & (product > 0)
    return float((joint[mask] * np.log(joint[mask] / product[mask])).sum())


def environment_only_bayes_advantage(joint: np.ndarray) -> float:
    """Bayes accuracy using E minus the best class-prior-only accuracy."""

    joint = np.asarray(joint, dtype=np.float64)
    return float(joint.max(axis=0).sum() - joint.sum(axis=1).max())


def maximum_within_class_support_ratio(joint: np.ndarray) -> float:
    ratios = []
    for row in np.asarray(joint, dtype=np.float64):
        positive = row[row > 0]
        if len(positive) >= 2:
            ratios.append(float(positive.max() / positive.min()))
    return max(ratios, default=1.0)


def pseudo_to_true_dependence_bound(pseudo_tv: float, weighted_error: float) -> float:
    """TV(Y,E) <= TV(Y,E_hat) + 2 P(E != E_hat), clipped to TV's range."""

    if pseudo_tv < 0 or not 0 <= weighted_error <= 1:
        raise ValueError("invalid dependence or error")
    return min(1.0, float(pseudo_tv) + 2.0 * float(weighted_error))


def environment_advantage_tv_bound_holds(joint: np.ndarray, *, atol: float = 1e-12) -> bool:
    return environment_only_bayes_advantage(joint) <= total_variation_dependence(joint) + atol
