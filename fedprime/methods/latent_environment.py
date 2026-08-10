from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from fedprime.data.corruptions import apply_corruption


@dataclass(frozen=True)
class InterventionProgram:
    """A label-free intervention program used only to construct paired views."""

    steps: tuple[tuple[str, int], ...]

    @property
    def severity(self) -> float:
        if not self.steps:
            return 0.0
        return float(sum(severity for _, severity in self.steps) / len(self.steps))


def _program_id(program: InterventionProgram, operators: Sequence[str]) -> int:
    if not program.steps:
        return 0
    lookup = {name: index for index, name in enumerate(operators)}
    base = len(operators) * 5 + 1
    value = 1
    for operator, severity in program.steps:
        value = value * base + lookup[operator] * 5 + int(severity)
    return int(value)


class PairedInterventionDataset(data.Dataset):
    """Build cross-content pairs that share a corruption mechanism.

    The returned training contract contains no family label. Concrete operator
    names are used only inside the synthetic intervention generator and are
    represented by an opaque program id after image generation.
    """

    def __init__(
        self,
        images: np.ndarray,
        indices: np.ndarray,
        *,
        operators: Sequence[str],
        seed: int,
        labels: np.ndarray | None = None,
        max_chain_length: int = 2,
        clean_fraction: float = 0.1,
    ) -> None:
        self.images = np.asarray(images, dtype=np.uint8)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.operators = tuple(dict.fromkeys(str(name) for name in operators))
        self.seed = int(seed)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.max_chain_length = int(max_chain_length)
        self.clean_fraction = float(clean_fraction)
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError("images must have shape [sample, height, width, 3]")
        if self.indices.ndim != 1 or len(self.indices) < 2:
            raise ValueError("paired intervention data requires at least two indices")
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("paired intervention indices must be unique")
        if int(self.indices.min()) < 0 or int(self.indices.max()) >= len(self.images):
            raise ValueError("paired intervention indices are out of image bounds")
        if not self.operators:
            raise ValueError("at least one concrete corruption operator is required")
        if self.labels is not None and len(self.labels) != len(self.images):
            raise ValueError("labels and images must have the same length")
        if self.max_chain_length < 1:
            raise ValueError("max_chain_length must be positive")
        if not 0.0 <= self.clean_fraction < 1.0:
            raise ValueError("clean_fraction must be in [0, 1)")

    def __len__(self) -> int:
        return int(len(self.indices))

    def _program(self, item: int) -> InterventionProgram:
        rng = np.random.default_rng(self.seed + int(item) * 104729)
        if float(rng.random()) < self.clean_fraction:
            return InterventionProgram(())
        length = int(rng.integers(1, min(self.max_chain_length, len(self.operators)) + 1))
        selected = rng.choice(len(self.operators), size=length, replace=False)
        severities = rng.integers(1, 6, size=length)
        return InterventionProgram(
            tuple(
                (self.operators[int(operator_index)], int(severity))
                for operator_index, severity in zip(selected, severities)
            )
        )

    def _paired_positions(self, item: int) -> tuple[int, int]:
        first = int(item) % len(self.indices)
        offset = 1 + ((self.seed + int(item) * 15485863) % (len(self.indices) - 1))
        return first, int((first + offset) % len(self.indices))

    @staticmethod
    def _apply(
        image: np.ndarray,
        program: InterventionProgram,
        *,
        seed: int,
    ) -> np.ndarray:
        result = np.asarray(image, dtype=np.uint8)
        rng = np.random.default_rng(seed)
        for operator, severity in program.steps:
            result = apply_corruption(result, operator, severity, rng)
        return result

    @staticmethod
    def _tensor(image: np.ndarray) -> torch.Tensor:
        chw = np.ascontiguousarray(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
        return torch.from_numpy(chw).float() / 255.0

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        first_position, second_position = self._paired_positions(item)
        first_index = int(self.indices[first_position])
        second_index = int(self.indices[second_position])
        program = self._program(item)
        first = self._apply(
            self.images[first_index],
            program,
            seed=self.seed + int(item) * 1000003 + 17,
        )
        second = self._apply(
            self.images[second_index],
            program,
            seed=self.seed + int(item) * 1000003 + 31,
        )
        label_a = -1 if self.labels is None else int(self.labels[first_index])
        label_b = -1 if self.labels is None else int(self.labels[second_index])
        return {
            "view_a": self._tensor(first),
            "view_b": self._tensor(second),
            "program_id": torch.tensor(_program_id(program, self.operators), dtype=torch.long),
            "severity": torch.tensor(program.severity, dtype=torch.float32),
            "chain_length": torch.tensor(len(program.steps), dtype=torch.long),
            "content_a": torch.tensor(label_a, dtype=torch.long),
            "content_b": torch.tensor(label_b, dtype=torch.long),
            "source_index_a": torch.tensor(first_index, dtype=torch.long),
            "source_index_b": torch.tensor(second_index, dtype=torch.long),
        }


class PairedInterventionEncoder(nn.Module):
    """Small continuous degradation encoder with no environment classifier."""

    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, self.embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - 0.5) / 0.5
        return self.projector(self.encoder(normalized).flatten(1))


def _multi_positive_nce(
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    program_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    anchors = F.normalize(anchors, dim=1)
    candidates = F.normalize(candidates, dim=1)
    similarities = anchors @ candidates.T / float(temperature)
    positives = program_ids[:, None].eq(program_ids[None, :])
    positive_logits = similarities.masked_fill(~positives, float("-inf"))
    return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(similarities, dim=1)).mean()


def _variance_covariance_regularizer(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1.0e-4)
    variance = F.relu(1.0 - standard_deviation).mean()
    if len(embeddings) < 2:
        return variance, embeddings.new_zeros(())
    covariance = centered.T @ centered / float(len(embeddings) - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().sum() / max(embeddings.shape[1], 1)
    return variance, covariance_loss


def paired_intervention_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    program_ids: torch.Tensor,
    *,
    temperature: float = 0.1,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if embedding_a.shape != embedding_b.shape or embedding_a.ndim != 2:
        raise ValueError("paired embeddings must have the same [batch, dimension] shape")
    if program_ids.shape != (len(embedding_a),):
        raise ValueError("program_ids must have shape [batch]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    contrastive = 0.5 * (
        _multi_positive_nce(embedding_a, embedding_b, program_ids, temperature)
        + _multi_positive_nce(embedding_b, embedding_a, program_ids, temperature)
    )
    variance, covariance = _variance_covariance_regularizer(
        torch.cat([embedding_a, embedding_b], dim=0)
    )
    total = contrastive + float(variance_weight) * variance + float(covariance_weight) * covariance
    return total, {
        "contrastive_loss": contrastive,
        "variance_loss": variance,
        "covariance_loss": covariance,
    }


def train_paired_intervention_encoder(
    model: PairedInterventionEncoder,
    loader: data.DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    temperature: float = 0.1,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
    max_batches: int | None = None,
) -> list[dict[str, float]]:
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    if float(learning_rate) <= 0:
        raise ValueError("learning_rate must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, float]] = []
    model.to(device)
    for epoch in range(int(epochs)):
        model.train()
        collected: dict[str, list[float]] = {
            "loss": [],
            "contrastive_loss": [],
            "variance_loss": [],
            "covariance_loss": [],
        }
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            view_a = batch["view_a"].to(device, non_blocking=True)
            view_b = batch["view_b"].to(device, non_blocking=True)
            program_ids = batch["program_id"].to(device, non_blocking=True)
            joined_embeddings = model(torch.cat([view_a, view_b], dim=0))
            embedding_a, embedding_b = joined_embeddings.chunk(2, dim=0)
            loss, diagnostics = paired_intervention_loss(
                embedding_a,
                embedding_b,
                program_ids,
                temperature=temperature,
                variance_weight=variance_weight,
                covariance_weight=covariance_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            collected["loss"].append(float(loss.detach().cpu()))
            for name, value in diagnostics.items():
                collected[name].append(float(value.detach().cpu()))
        if not collected["loss"]:
            raise RuntimeError("paired intervention loader produced no training batches")
        row = {"epoch": float(epoch)}
        row.update({name: sum(values) / len(values) for name, values in collected.items()})
        history.append(row)
        print(
            f"[heartbeat] PIE epoch={epoch:03d} loss={row['loss']:.4f} "
            f"nce={row['contrastive_loss']:.4f} var={row['variance_loss']:.4f} "
            f"cov={row['covariance_loss']:.4f}",
            flush=True,
        )
    model.eval()
    return history


@dataclass
class RepresentationAuditMetrics:
    samples: int
    retrieval_recall_at_one: float
    retrieval_chance: float
    retrieval_lift: float
    severity_spearman: float
    mean_dimension_std: float
    active_dimension_fraction: float
    content_probe_accuracy: float
    content_probe_chance: float
    content_probe_lift: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class RepresentationAuditThresholds:
    min_seen_retrieval_lift: float = 5.0
    min_heldout_retrieval_lift: float = 3.0
    min_severity_spearman: float = 0.5
    min_active_dimension_fraction: float = 0.75
    max_content_accuracy_floor: float = 0.05
    max_content_lift: float = 2.0


def _content_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float]:
    valid = labels >= 0
    embeddings = embeddings[valid]
    labels = labels[valid]
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or len(labels) < 20 or int(counts.min()) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in unique:
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        class_test_size = min(
            max(1, int(round(0.3 * len(class_indices)))), len(class_indices) - 1
        )
        test_indices.extend(int(index) for index in class_indices[:class_test_size])
        train_indices.extend(int(index) for index in class_indices[class_test_size:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    train_x = embeddings[train_indices].astype(np.float64)
    test_x = embeddings[test_indices].astype(np.float64)
    train_y_np = labels[train_indices]
    test_y = labels[test_indices]
    mean = train_x.mean(axis=0, keepdims=True)
    standard_deviation = np.maximum(train_x.std(axis=0, keepdims=True), 1.0e-6)
    train_x = (train_x - mean) / standard_deviation
    test_x = (test_x - mean) / standard_deviation
    centroids = np.stack(
        [train_x[train_y_np == label].mean(axis=0) for label in unique], axis=0
    )
    squared_distances = ((test_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predicted_labels = unique[np.argmin(squared_distances, axis=1)]
    accuracy = float(np.mean(predicted_labels == test_y))
    _, test_counts = np.unique(test_y, return_counts=True)
    chance = float(test_counts.max() / len(test_y))
    return accuracy, chance


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = float(
        np.sqrt(
            np.sum(first_centered**2)
            * np.sum(second_centered**2)
        )
    )
    if denominator <= 0:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def _collect_pair_embeddings(
    model: PairedInterventionEncoder,
    loader: data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[torch.Tensor]] = {
        "embedding_a": [],
        "embedding_b": [],
        "program_id": [],
        "severity": [],
        "content_a": [],
        "content_b": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            joined_views = torch.cat([batch["view_a"], batch["view_b"]], dim=0).to(device)
            embedding_a, embedding_b = model(joined_views).chunk(2, dim=0)
            collected["embedding_a"].append(embedding_a.cpu())
            collected["embedding_b"].append(embedding_b.cpu())
            for name in ("program_id", "severity", "content_a", "content_b"):
                collected[name].append(batch[name].cpu())
    if not collected["embedding_a"]:
        raise RuntimeError("paired intervention audit loader is empty")
    return {
        name: torch.cat(values).numpy()
        for name, values in collected.items()
    }


def audit_representation(
    model: PairedInterventionEncoder,
    loader: data.DataLoader,
    device: torch.device,
    *,
    seed: int = 0,
) -> RepresentationAuditMetrics:
    values = _collect_pair_embeddings(model, loader, device)
    embedding_a = values["embedding_a"]
    embedding_b = values["embedding_b"]
    program_ids = values["program_id"].astype(np.int64)
    normalized_a = embedding_a / np.clip(np.linalg.norm(embedding_a, axis=1, keepdims=True), 1e-12, None)
    normalized_b = embedding_b / np.clip(np.linalg.norm(embedding_b, axis=1, keepdims=True), 1e-12, None)
    similarities = torch.from_numpy(normalized_a) @ torch.from_numpy(normalized_b).T
    nearest = torch.argmax(similarities, dim=1).numpy()
    retrieval = float(np.mean(program_ids[nearest] == program_ids))
    _, program_counts = np.unique(program_ids, return_counts=True)
    chance = float(np.sum(program_counts.astype(np.float64) ** 2) / len(program_ids) ** 2)
    embeddings = np.concatenate([embedding_a, embedding_b], axis=0)
    dimension_std = embeddings.std(axis=0)
    clean = values["severity"] <= 0
    if bool(clean.any()) and bool((~clean).any()):
        clean_embeddings = 0.5 * (embedding_a[clean] + embedding_b[clean])
        clean_center = clean_embeddings.mean(axis=0, keepdims=True)
        paired = 0.5 * (embedding_a + embedding_b)
        distances = np.linalg.norm(paired - clean_center, axis=1)
        severity_spearman = _spearman_correlation(distances, values["severity"])
    else:
        severity_spearman = 0.0
    content_embeddings = np.concatenate([embedding_a, embedding_b], axis=0)
    content_labels = np.concatenate([values["content_a"], values["content_b"]], axis=0)
    content_accuracy, content_chance = _content_probe(
        content_embeddings,
        content_labels,
        seed=seed,
    )
    return RepresentationAuditMetrics(
        samples=int(len(program_ids)),
        retrieval_recall_at_one=retrieval,
        retrieval_chance=chance,
        retrieval_lift=retrieval / max(chance, 1.0e-12),
        severity_spearman=severity_spearman,
        mean_dimension_std=float(dimension_std.mean()),
        active_dimension_fraction=float(np.mean(dimension_std >= 0.05)),
        content_probe_accuracy=content_accuracy,
        content_probe_chance=content_chance,
        content_probe_lift=(
            content_accuracy / max(content_chance, 1.0e-12)
            if content_chance > 0
            else 0.0
        ),
    )


def representation_audit_gates(
    seen: RepresentationAuditMetrics,
    heldout: RepresentationAuditMetrics,
    thresholds: RepresentationAuditThresholds | None = None,
) -> dict[str, bool | float]:
    thresholds = thresholds or RepresentationAuditThresholds()
    seen_content_limit = max(
        float(thresholds.max_content_accuracy_floor),
        float(thresholds.max_content_lift) * seen.content_probe_chance,
    )
    heldout_content_limit = max(
        float(thresholds.max_content_accuracy_floor),
        float(thresholds.max_content_lift) * heldout.content_probe_chance,
    )
    gates: dict[str, bool | float] = {
        "seen_retrieval_lift": seen.retrieval_lift >= thresholds.min_seen_retrieval_lift,
        "heldout_retrieval_lift": heldout.retrieval_lift >= thresholds.min_heldout_retrieval_lift,
        "seen_severity_ordering": seen.severity_spearman >= thresholds.min_severity_spearman,
        "heldout_severity_ordering": (
            heldout.severity_spearman >= thresholds.min_severity_spearman
        ),
        "noncollapsed_dimensions": (
            seen.active_dimension_fraction >= thresholds.min_active_dimension_fraction
        ),
        "seen_content_leakage": seen.content_probe_accuracy <= seen_content_limit,
        "heldout_content_leakage": (
            heldout.content_probe_accuracy <= heldout_content_limit
        ),
        "seen_content_accuracy_limit": seen_content_limit,
        "heldout_content_accuracy_limit": heldout_content_limit,
    }
    gates["pass"] = all(
        bool(value)
        for name, value in gates.items()
        if not name.endswith("_limit")
    )
    return gates


def verify_operator_partition(
    training_operators: Iterable[str],
    heldout_operators: Iterable[str],
) -> dict[str, object]:
    training = tuple(dict.fromkeys(str(name) for name in training_operators))
    heldout = tuple(dict.fromkeys(str(name) for name in heldout_operators))
    overlap = sorted(set(training) & set(heldout))
    if overlap:
        raise ValueError(f"training and heldout operators overlap: {overlap}")
    if not training or not heldout:
        raise ValueError("training and heldout operator sets must both be non-empty")
    return {
        "training_operators": list(training),
        "heldout_operators": list(heldout),
        "overlap": overlap,
        "taxonomy_labels_used": False,
    }
