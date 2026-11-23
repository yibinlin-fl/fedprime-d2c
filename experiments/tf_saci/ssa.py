"""A client-local, out-of-fold SSA texture probe and its proposed loss."""

from dataclasses import dataclass

import numpy as np
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import logsumexp


def _moments(values):
    return [values.mean(axis=1), values.std(axis=1)]


def _quantiles(values, *quantiles):
    return list(np.quantile(values, quantiles, axis=1))


def _descriptors(images):
    rgb = images.astype(np.float64) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    dx = np.abs(np.diff(gray, axis=2))
    dy = np.abs(np.diff(gray, axis=1))
    texture = []
    for gradient in (dx, dy):
        values = gradient.reshape(len(images), -1)
        texture.extend(_moments(values) + _quantiles(values, 0.9))

    center = gray[:, 1:-1, 1:-1]
    laplacian = np.abs(
        gray[:, :-2, 1:-1] + gray[:, 2:, 1:-1]
        + gray[:, 1:-1, :-2] + gray[:, 1:-1, 2:] - 4 * center
    ).reshape(len(images), -1)
    mean, std = _moments(laplacian)
    standardized = (laplacian - mean[:, None]) / np.maximum(std[:, None], 1e-8)
    texture.extend([mean, std, (standardized**3).mean(axis=1), (standardized**4).mean(axis=1)])

    height, width = gray.shape[1:]
    block = gray.reshape(len(images), height // 4, 4, width // 4, 4)
    variance = block.var(axis=(2, 4)).reshape(len(images), -1)
    texture.extend(_moments(variance) + _quantiles(variance, 0.1, 0.5, 0.9))
    x_boundary = np.arange(1, width) % 8 == 0
    y_boundary = np.arange(1, height) % 8 == 0
    boundary = np.concatenate(
        (dx[:, :, x_boundary].reshape(len(images), -1), dy[:, y_boundary].reshape(len(images), -1)),
        axis=1,
    )
    interior = np.concatenate(
        (dx[:, :, ~x_boundary].reshape(len(images), -1), dy[:, ~y_boundary].reshape(len(images), -1)),
        axis=1,
    )
    texture.append(boundary.mean(axis=1) / np.maximum(interior.mean(axis=1), 1e-6))
    for chroma in (0.564 * (rgb[..., 2] - gray), 0.713 * (rgb[..., 0] - gray)):
        differences = [
            np.abs(np.diff(chroma, axis=axis)).reshape(len(images), -1) for axis in (1, 2)
        ]
        texture.extend(_moments(np.concatenate(differences, axis=1)))
    gray = gray.reshape(len(images), -1)
    brightness = _moments(gray) + _quantiles(gray, 0.1, 0.9) + [np.ptp(gray, axis=1)]
    return np.column_stack(texture + brightness)


def descriptors(images, chunk_size=128):
    """Return 20 texture features and those same features plus five intensity summaries."""
    if (
        images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3
        or not len(images) or any(size < 16 or size % 4 for size in images.shape[1:3])
        or chunk_size < 1
    ):
        raise ValueError("Expected nonempty uint8 NHWC RGB with spatial dimensions divisible by four")
    full = np.concatenate([
        _descriptors(images[start:start + chunk_size])
        for start in range(0, len(images), chunk_size)
    ])
    return {"phi_tex": full[:, :20], "phi_full": full}


@dataclass(frozen=True)
class ProbeResult:
    sample_ids: np.ndarray
    fold: np.ndarray
    active: np.ndarray
    raw: np.ndarray
    bias: np.ndarray
    prediction: np.ndarray
    prior_prediction: np.ndarray
    iterations: np.ndarray

    def bias_for(self, sample_ids):
        positions = np.searchsorted(self.sample_ids, sample_ids)
        if np.any(positions >= len(self.sample_ids)) or np.any(self.sample_ids[positions] != sample_ids):
            raise KeyError("SSA bias is available for fitted sample IDs only")
        return self.bias[positions]


def _linear_probe(train, targets, test, counts, l2):
    scale = train.std(axis=0)
    mean = train.mean(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    x = np.column_stack(((train - mean) / scale, np.ones(len(train))))
    evaluation = np.column_stack(((test - mean) / scale, np.ones(len(test))))
    log_prior = np.log(counts / counts.sum())
    rows = np.arange(len(targets))
    shape = (x.shape[1], counts.size)

    def objective(flat):
        weights = flat.reshape(shape)
        logits = x @ weights + log_prior
        normalizer = logsumexp(logits, axis=1)
        loss = (normalizer - logits[rows, targets]).mean() + l2 * np.square(weights[:-1]).sum() / 2
        residual = np.exp(logits - normalizer[:, None])
        residual[rows, targets] -= 1
        gradient = x.T @ residual / len(targets)
        gradient[:-1] += l2 * weights[:-1]
        return loss, gradient.ravel()

    fit = minimize(
        objective, np.zeros(np.prod(shape)), jac=True, method="L-BFGS-B",
        options={"maxiter": 600, "gtol": 1e-7, "ftol": 1e-11},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"SSA fold fit failed: {fit.message}")
    return evaluation @ fit.x.reshape(shape), fit.nit


def fit_oof(features, labels, sample_ids, source_ids, *, classes, folds, fold_seed, l2, clip):
    """Compute held-out scores. Each fold fits its own scaler and training prior."""
    labels, sample_ids, source_ids = map(np.asarray, (labels, sample_ids, source_ids))
    if (
        labels.ndim != 1 or sample_ids.shape != labels.shape or source_ids.shape != labels.shape
        or any(array.dtype.kind not in "iu" for array in (labels, sample_ids, source_ids))
        or len(np.unique(sample_ids)) != len(labels) or features.ndim != 2
        or features.shape[0] != len(labels) or not np.isfinite(features).all()
        or not len(labels) or np.any((labels < 0) | (labels >= classes))
        or folds < 2 or not np.isfinite([l2, clip]).all() or l2 <= 0 or clip <= 0
    ):
        raise ValueError("Invalid SSA probe inputs or settings")
    order = np.argsort(sample_ids, kind="stable")
    features, labels, sample_ids, source_ids = (
        values[order] for values in (features, labels, sample_ids, source_ids)
    )
    unique_sources, source_rows, group_index = np.unique(source_ids, return_index=True, return_inverse=True)
    group_labels = labels[source_rows]
    if not np.array_equal(labels, group_labels[group_index]):
        raise ValueError("A source ID has inconsistent task labels")
    support = np.bincount(group_labels, minlength=classes)
    if len(unique_sources) < folds or (support == 1).any():
        raise ValueError("OOF needs at least two source groups per observed class and enough groups for all folds")
    # Rotate the fold offset across classes so even classes with 2–4 sources
    # use distinct validation folds without producing an empty fold overall.
    rng = np.random.default_rng(fold_seed)
    group_folds = np.empty(len(unique_sources), dtype=np.int16)
    offset = 0
    for label in np.flatnonzero(support):
        groups = rng.permutation(np.flatnonzero(group_labels == label))
        group_folds[groups] = (np.arange(len(groups)) + offset) % folds
        offset = (offset + len(groups)) % folds
    row_folds = group_folds[group_index]
    raw = np.zeros((len(labels), classes))
    bias = np.zeros_like(raw)
    active = np.zeros_like(raw, dtype=bool)
    assignment = np.full(len(labels), -1, dtype=np.int16)
    prediction = np.empty(len(labels), dtype=np.int64)
    prior_prediction = np.empty_like(prediction)
    iterations = []
    for fold in range(folds):
        test = row_folds == fold
        train = ~test
        counts = np.bincount(labels[train], minlength=classes)
        present = np.flatnonzero(counts)
        if not test.any() or (counts[labels[test]] == 0).any() or (assignment[test] >= 0).any():
            raise ValueError("OOF fold is empty, overlaps another fold, or lacks held-out class support")
        scores, fitted_iterations = (
            _linear_probe(features[train], np.searchsorted(present, labels[train]),
                          features[test], counts[present], l2)
            if present.size > 1 else (np.zeros((test.sum(), 1)), 0)
        )
        rows = np.ix_(test, present)
        raw[rows] = scores
        bias[rows] = np.clip(scores - scores.mean(axis=1, keepdims=True), -clip, clip)
        active[rows] = True
        assignment[test] = fold
        prediction[test] = present[scores.argmax(axis=1)]
        prior_prediction[test] = counts.argmax()
        iterations.append(fitted_iterations)
    if (assignment < 0).any():
        raise RuntimeError("OOF did not score every sample")
    return ProbeResult(
        sample_ids, assignment, active, raw, bias, prediction, prior_prediction, np.asarray(iterations)
    )


def ssa_loss(logits, labels, bias, *, strength, poe_weight=1.0):
    """Apply a frozen local training bias. The model forward path stays unchanged."""
    if logits.shape != bias.shape or not np.isfinite(strength) or strength < 0 or not 0 <= poe_weight <= 1:
        raise ValueError("Invalid SSA loss inputs")
    if strength == 0 or poe_weight == 0:
        return F.cross_entropy(logits, labels)
    poe = F.cross_entropy(logits + strength * bias.detach(), labels)
    if poe_weight == 1:
        return poe
    return (1 - poe_weight) * F.cross_entropy(logits, labels) + poe_weight * poe
