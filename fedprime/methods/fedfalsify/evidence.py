from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PairedAdvantage:
    """Receiver-side class-conditional evidence for one source model."""

    class_id: int
    count: int
    source_accuracy: float
    receiver_accuracy: float
    paired_advantage: float
    paired_variance: float
    conservative_advantage: float
    shrinkage: float
    advantage_strength: float
    is_auditable: bool
    is_active: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _as_1d_int(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape={array.shape}")
    return array.astype(np.int64, copy=False)


def compute_paired_advantage(
    source_predictions,
    receiver_predictions,
    labels,
    *,
    class_id: int,
    kappa: float = 1.0,
    shrinkage_nu: float = 10.0,
    min_count: int = 5,
    eps: float = 1e-12,
) -> PairedAdvantage:
    """Estimate a conservative source-over-receiver correctness advantage.

    This is an empirical score rather than a statistical confidence interval.
    The paired outcome is in ``{-1, 0, 1}``, so it measures whether the source
    corrects more receiver mistakes than it introduces on the same examples.
    """

    source = _as_1d_int(source_predictions, "source_predictions")
    receiver = _as_1d_int(receiver_predictions, "receiver_predictions")
    target = _as_1d_int(labels, "labels")
    if source.shape != receiver.shape or source.shape != target.shape:
        raise ValueError("source predictions, receiver predictions, and labels must align")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    if kappa < 0 or shrinkage_nu < 0:
        raise ValueError("kappa and shrinkage_nu must be non-negative")

    mask = target == int(class_id)
    count = int(mask.sum())
    if count == 0:
        return PairedAdvantage(
            class_id=int(class_id),
            count=0,
            source_accuracy=float("nan"),
            receiver_accuracy=float("nan"),
            paired_advantage=float("nan"),
            paired_variance=float("nan"),
            conservative_advantage=float("nan"),
            shrinkage=0.0,
            advantage_strength=0.0,
            is_auditable=False,
            is_active=False,
        )

    class_labels = target[mask]
    source_correct = (source[mask] == class_labels).astype(np.float64)
    receiver_correct = (receiver[mask] == class_labels).astype(np.float64)
    paired = source_correct - receiver_correct
    advantage = float(paired.mean())
    variance = float(paired.var(ddof=1)) if count > 1 else 0.0
    conservative = advantage - float(kappa) * np.sqrt((variance + eps) / count)
    shrinkage = count / (count + float(shrinkage_nu))
    auditable = count >= int(min_count)
    strength = shrinkage * max(conservative, 0.0) if auditable else 0.0

    return PairedAdvantage(
        class_id=int(class_id),
        count=count,
        source_accuracy=float(source_correct.mean()),
        receiver_accuracy=float(receiver_correct.mean()),
        paired_advantage=advantage,
        paired_variance=variance,
        conservative_advantage=float(conservative),
        shrinkage=float(shrinkage),
        advantage_strength=float(strength),
        is_auditable=auditable,
        is_active=bool(strength > 0.0),
    )


def compute_classwise_paired_advantages(
    source_predictions,
    receiver_predictions,
    labels,
    *,
    num_classes: int,
    kappa: float = 1.0,
    shrinkage_nu: float = 10.0,
    min_count: int = 5,
) -> list[PairedAdvantage]:
    return [
        compute_paired_advantage(
            source_predictions,
            receiver_predictions,
            labels,
            class_id=class_id,
            kappa=kappa,
            shrinkage_nu=shrinkage_nu,
            min_count=min_count,
        )
        for class_id in range(int(num_classes))
    ]


def classwise_accuracy_tensor(
    predictions,
    labels,
    *,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute source x receiver x class accuracies from paired predictions."""

    pred = np.asarray(predictions)
    target = np.asarray(labels)
    if pred.ndim != 3:
        raise ValueError("predictions must have shape [source, receiver, sample]")
    if target.ndim != 2:
        raise ValueError("labels must have shape [receiver, sample]")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            f"prediction/label shapes do not align: {pred.shape} versus {target.shape}"
        )

    num_sources, num_receivers, _ = pred.shape
    accuracy = np.full(
        (num_sources, num_receivers, int(num_classes)),
        np.nan,
        dtype=np.float64,
    )
    counts = np.zeros((num_receivers, int(num_classes)), dtype=np.int64)
    for receiver_id in range(num_receivers):
        receiver_labels = target[receiver_id]
        for class_id in range(int(num_classes)):
            mask = receiver_labels == class_id
            count = int(mask.sum())
            counts[receiver_id, class_id] = count
            if count == 0:
                continue
            accuracy[:, receiver_id, class_id] = (
                pred[:, receiver_id, :][:, mask] == receiver_labels[mask][None, :]
            ).mean(axis=1)
    return accuracy, counts


def planned_stratified_audit_counts(
    labels,
    *,
    num_classes: int,
    audit_ratio: float,
) -> np.ndarray:
    """Project deterministic class-wise audit counts for a future train split."""

    if not 0.0 < float(audit_ratio) < 1.0:
        raise ValueError("audit_ratio must be between zero and one")
    target = _as_1d_int(labels, "labels")
    class_counts = np.bincount(target, minlength=int(num_classes)).astype(np.int64)
    audit_counts = np.zeros(int(num_classes), dtype=np.int64)
    for class_id, count in enumerate(class_counts):
        if count < 2:
            continue
        proposed = max(1, int(round(float(audit_ratio) * int(count))))
        audit_counts[class_id] = min(proposed, int(count) - 1)
    return audit_counts
