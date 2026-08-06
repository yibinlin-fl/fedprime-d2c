from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from fedprime.data.corruptions import CORRUPTION_GROUPS, apply_corruption, sample_corruption_from_group
from fedprime.data.loaders import _cifar10_train_from_tar, _cifar100_train_from_tar


PEW_ENVIRONMENT_NAMES = ("clean", *CORRUPTION_GROUPS.keys(), "unknown")


class PublicEnvironmentDataset(data.Dataset):
    """Synthetic environment supervision built only from unlabeled public images."""

    def __init__(self, images: np.ndarray, indices: np.ndarray, seed: int) -> None:
        self.images = np.asarray(images, dtype=np.uint8)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        image = self.images[int(self.indices[item])]
        environment_id = int(item % len(PEW_ENVIRONMENT_NAMES))
        severity = int((item // len(PEW_ENVIRONMENT_NAMES)) % 5) + 1
        rng = np.random.default_rng(self.seed + item * 104729)

        if environment_id == 0:
            corrupted = image
            severity_target = 0
        elif environment_id == len(PEW_ENVIRONMENT_NAMES) - 1:
            first_group = list(CORRUPTION_GROUPS)[item % len(CORRUPTION_GROUPS)]
            second_group = list(CORRUPTION_GROUPS)[(item + 1) % len(CORRUPTION_GROUPS)]
            corrupted = apply_corruption(
                image,
                sample_corruption_from_group(first_group, rng),
                severity,
                rng,
            )
            corrupted = apply_corruption(
                corrupted,
                sample_corruption_from_group(second_group, rng),
                severity,
                rng,
            )
            severity_target = severity
        else:
            group = list(CORRUPTION_GROUPS)[environment_id - 1]
            corrupted = apply_corruption(
                image,
                sample_corruption_from_group(group, rng),
                severity,
                rng,
            )
            severity_target = severity

        tensor = torch.from_numpy(np.ascontiguousarray(corrupted.transpose(2, 0, 1))).float() / 255.0
        return tensor, environment_id, severity_target


class PublicEnvironmentWitness(nn.Module):
    """Small corruption-only network trained without private class labels."""

    def __init__(self, embedding_dim: int = 32, num_environments: int = 6, severity_levels: int = 5) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_environments = int(num_environments)
        self.severity_levels = int(severity_levels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(128, self.embedding_dim)
        self.environment_head = nn.Linear(self.embedding_dim, self.num_environments)
        self.severity_head = nn.Linear(self.embedding_dim, self.severity_levels)

    def forward(self, images: torch.Tensor):
        normalized = (images - 0.5) / 0.5
        encoded = self.encoder(normalized).flatten(1)
        embedding = self.embedding(encoded)
        return self.environment_head(embedding), self.severity_head(embedding), embedding


@dataclass
class WitnessReport:
    environment_accuracy: float
    severity_accuracy: float
    clean_accuracy: float
    unknown_accuracy: float
    mean_confidence: float
    expected_calibration_error: float
    negative_log_likelihood: float
    unknown_auroc: float
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict[str, object]:
        return {
            "environment_accuracy": self.environment_accuracy,
            "severity_accuracy": self.severity_accuracy,
            "clean_accuracy": self.clean_accuracy,
            "unknown_accuracy": self.unknown_accuracy,
            "mean_confidence": self.mean_confidence,
            "expected_calibration_error": self.expected_calibration_error,
            "negative_log_likelihood": self.negative_log_likelihood,
            "unknown_auroc": self.unknown_auroc,
            "confusion_matrix": self.confusion_matrix,
        }


def _binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    positives = int(targets.sum().item())
    negatives = int(targets.numel() - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[targets.bool()].sum().item()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def build_public_environment_loaders(
    public_root: str | Path,
    *,
    public_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    validation_fraction: float = 0.2,
    public_dataset: str = "cifar100",
) -> tuple[data.DataLoader, data.DataLoader]:
    dataset_name = str(public_dataset).lower()
    if dataset_name == "cifar100":
        images, _ = _cifar100_train_from_tar(public_root)
    elif dataset_name == "cifar10":
        images, _ = _cifar10_train_from_tar(public_root)
    else:
        raise ValueError(f"Unsupported PEW public dataset: {public_dataset}")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(images), size=min(int(public_size), len(images)), replace=False)
    split = max(1, min(len(selected) - 1, int(round(len(selected) * (1.0 - validation_fraction)))))
    train_ds = PublicEnvironmentDataset(images, selected[:split], seed=seed)
    val_ds = PublicEnvironmentDataset(images, selected[split:], seed=seed + 1_000_003)
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        data.DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        data.DataLoader(val_ds, shuffle=False, drop_last=False, **common),
    )


def evaluate_environment_witness(
    model: PublicEnvironmentWitness,
    loader: data.DataLoader,
    device: torch.device,
) -> WitnessReport:
    model.eval()
    environment_correct = 0
    environment_total = 0
    severity_correct = 0
    severity_total = 0
    clean_correct = 0
    clean_total = 0
    unknown_correct = 0
    unknown_total = 0
    confidences = []
    all_probabilities = []
    all_targets = []
    unknown_id = len(PEW_ENVIRONMENT_NAMES) - 1
    with torch.no_grad():
        for images, environments, severities in loader:
            images = images.to(device, non_blocking=True)
            environments = environments.to(device).long()
            severities = severities.to(device).long()
            environment_logits, severity_logits, _ = model(images)
            probabilities = environment_logits.softmax(dim=1)
            predictions = probabilities.argmax(dim=1)
            environment_correct += int(predictions.eq(environments).sum().item())
            environment_total += int(environments.numel())
            confidences.append(probabilities.max(dim=1).values.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())
            all_targets.append(environments.detach().cpu())

            severity_mask = severities > 0
            if bool(severity_mask.any()):
                severity_predictions = severity_logits.argmax(dim=1) + 1
                severity_correct += int(severity_predictions[severity_mask].eq(severities[severity_mask]).sum().item())
                severity_total += int(severity_mask.sum().item())
            clean_mask = environments == 0
            unknown_mask = environments == unknown_id
            clean_correct += int(predictions[clean_mask].eq(0).sum().item())
            clean_total += int(clean_mask.sum().item())
            unknown_correct += int(predictions[unknown_mask].eq(unknown_id).sum().item())
            unknown_total += int(unknown_mask.sum().item())

    confidence = torch.cat(confidences).mean().item() if confidences else 0.0
    probabilities = torch.cat(all_probabilities) if all_probabilities else torch.empty(0, len(PEW_ENVIRONMENT_NAMES))
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    predictions = probabilities.argmax(dim=1) if targets.numel() else targets
    confusion = torch.zeros(len(PEW_ENVIRONMENT_NAMES), len(PEW_ENVIRONMENT_NAMES), dtype=torch.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1
    ece = 0.0
    if targets.numel():
        max_probabilities = probabilities.max(dim=1).values
        correctness = predictions.eq(targets).float()
        for lower in torch.linspace(0.0, 0.9, 10):
            upper = lower + 0.1
            mask = (max_probabilities > lower) & (max_probabilities <= upper)
            if bool(mask.any()):
                ece += float(mask.float().mean() * (correctness[mask].mean() - max_probabilities[mask].mean()).abs())
        nll = float(nn.functional.nll_loss(probabilities.clamp_min(1e-12).log(), targets).item())
        unknown_auroc = _binary_auroc(probabilities[:, unknown_id], targets.eq(unknown_id))
    else:
        nll = 0.0
        unknown_auroc = 0.0
    return WitnessReport(
        environment_accuracy=100.0 * environment_correct / max(environment_total, 1),
        severity_accuracy=100.0 * severity_correct / max(severity_total, 1),
        clean_accuracy=100.0 * clean_correct / max(clean_total, 1),
        unknown_accuracy=100.0 * unknown_correct / max(unknown_total, 1),
        mean_confidence=float(confidence),
        expected_calibration_error=ece,
        negative_log_likelihood=nll,
        unknown_auroc=unknown_auroc,
        confusion_matrix=confusion.tolist(),
    )


def train_environment_witness(
    model: PublicEnvironmentWitness,
    train_loader: data.DataLoader,
    validation_loader: data.DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    severity_weight: float = 0.25,
    max_batches: int | None = None,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history = []
    best_state = None
    best_epoch = -1
    best_environment_accuracy = float("-inf")
    model.to(device)
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for batch_idx, (images, environments, severities) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            environments = environments.to(device).long()
            severities = severities.to(device).long()
            environment_logits, severity_logits, _ = model(images)
            environment_loss = nn.functional.cross_entropy(environment_logits, environments)
            severity_mask = severities > 0
            if bool(severity_mask.any()):
                severity_loss = nn.functional.cross_entropy(
                    severity_logits[severity_mask],
                    severities[severity_mask] - 1,
                )
            else:
                severity_loss = environment_loss.new_zeros(())
            loss = environment_loss + float(severity_weight) * severity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        report = evaluate_environment_witness(model, validation_loader, device)
        row = {"epoch": float(epoch), "loss": sum(losses) / max(len(losses), 1), **report.as_dict()}
        history.append(row)
        if report.environment_accuracy > best_environment_accuracy:
            best_environment_accuracy = float(report.environment_accuracy)
            best_epoch = int(epoch)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"[heartbeat] PEW epoch={epoch:03d} loss={row['loss']:.4f} "
            f"env_acc={row['environment_accuracy']:.2f} severity_acc={row['severity_accuracy']:.2f} "
            f"unknown_acc={row['unknown_accuracy']:.2f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        for index, row in enumerate(history):
            row["is_best"] = float(index == best_epoch)
        print(
            f"[setup] restored best PEW checkpoint from epoch={best_epoch:03d} "
            f"env_acc={best_environment_accuracy:.2f}",
            flush=True,
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return history


def select_unknown_threshold(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    unknown_id: int,
    thresholds: torch.Tensor | None = None,
) -> dict[str, float]:
    """Choose a rejection threshold using only synthetic public validation labels."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [sample, environment]")
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("targets must have shape [sample]")
    if thresholds is None:
        thresholds = torch.linspace(0.0, 0.95, 96)

    confidence, native_prediction = probabilities.max(dim=1)
    best = None
    for raw_threshold in thresholds:
        threshold = float(raw_threshold.item())
        prediction = torch.where(
            confidence >= threshold,
            native_prediction,
            torch.full_like(native_prediction, int(unknown_id)),
        )
        accuracy = float(prediction.eq(targets).float().mean().item())
        unknown_rate = float(prediction.eq(int(unknown_id)).float().mean().item())
        candidate = {
            "threshold": threshold,
            "accuracy": accuracy,
            "unknown_rate": unknown_rate,
        }
        # Prefer the less aggressive rejection threshold when accuracy ties.
        if best is None or (accuracy, -threshold) > (best["accuracy"], -best["threshold"]):
            best = candidate
    assert best is not None
    best["native_accuracy"] = float(native_prediction.eq(targets).float().mean().item())
    return best


def calibrate_unknown_threshold(
    model: PublicEnvironmentWitness,
    loader: data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    probabilities = []
    targets = []
    with torch.no_grad():
        for images, environments, _ in loader:
            logits, _, _ = model(images.to(device, non_blocking=True))
            probabilities.append(logits.softmax(dim=1).cpu())
            targets.append(environments.long().cpu())
    if not probabilities:
        raise ValueError("cannot calibrate PEW unknown threshold on an empty loader")
    result = select_unknown_threshold(
        torch.cat(probabilities),
        torch.cat(targets),
        unknown_id=len(PEW_ENVIRONMENT_NAMES) - 1,
    )
    print(
        f"[setup] calibrated PEW unknown threshold={result['threshold']:.2f} "
        f"validation_acc={100.0 * result['accuracy']:.2f} "
        f"unknown_rate={result['unknown_rate']:.3f}",
        flush=True,
    )
    return result


def infer_environment_annotations(
    model: PublicEnvironmentWitness,
    images: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    confidence_threshold: float,
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    raw = np.asarray(images, dtype=np.uint8)
    if max_samples is not None:
        raw = raw[: int(max_samples)]
    tensor = torch.from_numpy(np.ascontiguousarray(raw.transpose(0, 3, 1, 2))).float() / 255.0
    loader = data.DataLoader(data.TensorDataset(tensor), batch_size=int(batch_size), shuffle=False)
    environment_ids = []
    confidences = []
    embeddings = []
    unknown_id = len(PEW_ENVIRONMENT_NAMES) - 1
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits, _, embedding = model(batch.to(device, non_blocking=True))
            probabilities = logits.softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            prediction = torch.where(
                confidence >= float(confidence_threshold),
                prediction,
                torch.full_like(prediction, unknown_id),
            )
            environment_ids.append(prediction.cpu())
            confidences.append(confidence.cpu())
            embeddings.append(embedding.cpu())
    return {
        "environment_ids": torch.cat(environment_ids).numpy().astype(np.int64),
        "confidence": torch.cat(confidences).numpy().astype(np.float32),
        "embedding": torch.cat(embeddings).numpy().astype(np.float32),
    }


def save_environment_witness(model: PublicEnvironmentWitness, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "num_environments": model.num_environments,
            "severity_levels": model.severity_levels,
            "environment_names": PEW_ENVIRONMENT_NAMES,
        },
        path,
    )


def load_environment_witness(path: str | Path, device: torch.device) -> PublicEnvironmentWitness:
    payload = torch.load(path, map_location=device)
    model = PublicEnvironmentWitness(
        embedding_dim=int(payload["embedding_dim"]),
        num_environments=int(payload["num_environments"]),
        severity_levels=int(payload["severity_levels"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model
