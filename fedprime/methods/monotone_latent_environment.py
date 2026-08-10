from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from fedprime.methods.latent_environment import (
    InterventionProgram,
    PairedInterventionDataset,
    PairedInterventionEncoder,
    RepresentationAuditMetrics,
    RepresentationAuditThresholds,
    _multi_positive_nce,
    _variance_covariance_regularizer,
    representation_audit_gates,
)


CONFIRMATORY_HELDOUT_OPERATORS: tuple[str, ...] = (
    "shot_noise",
    "motion_blur",
    "frost",
    "jpeg_compression",
)


def _mechanism_id(steps: Sequence[str], operators: Sequence[str]) -> int:
    lookup = {name: index for index, name in enumerate(operators)}
    base = len(operators) + 1
    value = 1
    for operator in steps:
        value = value * base + lookup[operator] + 1
    return int(value)


def _ordered_program_id(mechanism_id: int, severity: int) -> int:
    return 0 if int(severity) == 0 else int(mechanism_id) * 6 + int(severity)


class OrderedPairedInterventionDataset(data.Dataset):
    """Four-view public data with a known low-to-high intervention order."""

    def __init__(
        self,
        images: np.ndarray,
        indices: np.ndarray,
        *,
        operators: Sequence[str],
        seed: int,
        max_chain_length: int = 2,
    ) -> None:
        self.base = PairedInterventionDataset(
            images,
            indices,
            operators=operators,
            seed=seed,
            labels=None,
            max_chain_length=max_chain_length,
            clean_fraction=0.0,
        )
        self.operators = self.base.operators
        self.seed = int(seed)
        self.max_chain_length = int(max_chain_length)

    def __len__(self) -> int:
        return len(self.base)

    def _ordered_program(self, item: int) -> tuple[tuple[str, ...], int, int]:
        rng = np.random.default_rng(self.seed + int(item) * 104729)
        length = int(rng.integers(1, min(self.max_chain_length, len(self.operators)) + 1))
        selected = rng.choice(len(self.operators), size=length, replace=False)
        steps = tuple(self.operators[int(index)] for index in selected)
        low_severity = int(rng.integers(0, 5))
        high_severity = int(rng.integers(low_severity + 1, 6))
        return steps, low_severity, high_severity

    def _apply(
        self,
        image: np.ndarray,
        steps: Sequence[str],
        severity: int,
        *,
        seed: int,
    ) -> np.ndarray:
        if int(severity) == 0:
            return np.asarray(image, dtype=np.uint8)
        program = InterventionProgram(tuple((operator, int(severity)) for operator in steps))
        return self.base._apply(image, program, seed=seed)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        first_position, second_position = self.base._paired_positions(item)
        first_index = int(self.base.indices[first_position])
        second_index = int(self.base.indices[second_position])
        steps, low_severity, high_severity = self._ordered_program(item)
        mechanism_id = _mechanism_id(steps, self.operators)
        seed_base = self.seed + int(item) * 1000003
        views = {
            "view_a_low": self._apply(
                self.base.images[first_index], steps, low_severity, seed=seed_base + 17
            ),
            "view_b_low": self._apply(
                self.base.images[second_index], steps, low_severity, seed=seed_base + 31
            ),
            "view_a_high": self._apply(
                self.base.images[first_index], steps, high_severity, seed=seed_base + 47
            ),
            "view_b_high": self._apply(
                self.base.images[second_index], steps, high_severity, seed=seed_base + 61
            ),
        }
        return {
            name: self.base._tensor(view) for name, view in views.items()
        } | {
            "mechanism_id": torch.tensor(mechanism_id, dtype=torch.long),
            "program_id_low": torch.tensor(
                _ordered_program_id(mechanism_id, low_severity), dtype=torch.long
            ),
            "program_id_high": torch.tensor(
                _ordered_program_id(mechanism_id, high_severity), dtype=torch.long
            ),
            "low_severity": torch.tensor(low_severity, dtype=torch.float32),
            "high_severity": torch.tensor(high_severity, dtype=torch.float32),
            "source_index_a": torch.tensor(first_index, dtype=torch.long),
            "source_index_b": torch.tensor(second_index, dtype=torch.long),
        }


class MonotonePairedInterventionEncoder(nn.Module):
    """Radial environment code: direction is mechanism and radius is intensity."""

    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")
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
        self.direction_projector = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, self.embedding_dim),
        )
        self.radius_projector = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )

    def decompose(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = (images - 0.5) / 0.5
        features = self.encoder(normalized).flatten(1)
        direction = F.normalize(self.direction_projector(features), dim=1)
        radius = F.softplus(self.radius_projector(features)).squeeze(1) + 1.0e-6
        embedding = direction * radius[:, None]
        return embedding, direction, radius

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embedding, _, _ = self.decompose(images)
        return embedding


def _contrastive_pair_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    identifiers: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return 0.5 * (
        _multi_positive_nce(embedding_a, embedding_b, identifiers, temperature)
        + _multi_positive_nce(embedding_b, embedding_a, identifiers, temperature)
    )


def _valid_contrastive_views(
    low_a: torch.Tensor,
    low_b: torch.Tensor,
    high_a: torch.Tensor,
    high_b: torch.Tensor,
    low_identifiers: torch.Tensor,
    high_identifiers: torch.Tensor,
    low_severity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nonclean = low_severity > 0
    anchors = torch.cat([high_a, low_a[nonclean]], dim=0)
    candidates = torch.cat([high_b, low_b[nonclean]], dim=0)
    identifiers = torch.cat([high_identifiers, low_identifiers[nonclean]], dim=0)
    return anchors, candidates, identifiers


def unordered_ordered_pair_loss(
    low_a: torch.Tensor,
    low_b: torch.Tensor,
    high_a: torch.Tensor,
    high_b: torch.Tensor,
    program_id_low: torch.Tensor,
    program_id_high: torch.Tensor,
    low_severity: torch.Tensor,
    *,
    temperature: float = 0.1,
    covariance_weight: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    anchors, candidates, identifiers = _valid_contrastive_views(
        low_a,
        low_b,
        high_a,
        high_b,
        program_id_low,
        program_id_high,
        low_severity,
    )
    contrastive = _contrastive_pair_loss(anchors, candidates, identifiers, temperature)
    variance, covariance = _variance_covariance_regularizer(
        torch.cat([low_a, low_b, high_a, high_b], dim=0)
    )
    total = contrastive + variance + float(covariance_weight) * covariance
    return total, {
        "contrastive_loss": contrastive,
        "variance_loss": variance,
        "covariance_loss": covariance,
    }


def monotone_paired_intervention_loss(
    low_a: torch.Tensor,
    low_b: torch.Tensor,
    high_a: torch.Tensor,
    high_b: torch.Tensor,
    mechanism_ids: torch.Tensor,
    low_severity: torch.Tensor,
    *,
    temperature: float = 0.1,
    ordinal_margin: float = 0.25,
    covariance_weight: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    anchors, candidates, identifiers = _valid_contrastive_views(
        low_a,
        low_b,
        high_a,
        high_b,
        mechanism_ids,
        mechanism_ids,
        low_severity,
    )
    contrastive = _contrastive_pair_loss(anchors, candidates, identifiers, temperature)
    radius_low_a = low_a.norm(dim=1)
    radius_low_b = low_b.norm(dim=1)
    radius_high_a = high_a.norm(dim=1)
    radius_high_b = high_b.norm(dim=1)
    radius_low = 0.5 * (radius_low_a + radius_low_b)
    radius_high = 0.5 * (radius_high_a + radius_high_b)
    ordinal = F.softplus(float(ordinal_margin) + radius_low - radius_high).mean()
    radial_consistency = 0.5 * (
        F.smooth_l1_loss(radius_low_a, radius_low_b)
        + F.smooth_l1_loss(radius_high_a, radius_high_b)
    )
    clean = low_severity <= 0
    clean_anchor = (
        0.5 * (radius_low_a[clean].square() + radius_low_b[clean].square()).mean()
        if bool(clean.any())
        else low_a.new_zeros(())
    )
    variance, covariance = _variance_covariance_regularizer(
        torch.cat([low_a, low_b, high_a, high_b], dim=0)
    )
    total = (
        contrastive
        + variance
        + float(covariance_weight) * covariance
        + ordinal
        + radial_consistency
        + clean_anchor
    )
    return total, {
        "contrastive_loss": contrastive,
        "variance_loss": variance,
        "covariance_loss": covariance,
        "ordinal_loss": ordinal,
        "radial_consistency_loss": radial_consistency,
        "clean_anchor_loss": clean_anchor,
        "mean_low_radius": radius_low.mean(),
        "mean_high_radius": radius_high.mean(),
    }


def _train_ordered_encoder(
    model: nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    temperature: float,
    monotone: bool,
    ordinal_margin: float = 0.25,
    max_batches: int | None = None,
) -> list[dict[str, float]]:
    if int(epochs) < 1 or float(learning_rate) <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    model.to(device)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        collected: dict[str, list[float]] = {}
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            names = ("view_a_low", "view_b_low", "view_a_high", "view_b_high")
            joined = torch.cat([batch[name] for name in names], dim=0).to(device)
            low_a, low_b, high_a, high_b = model(joined).chunk(4, dim=0)
            low_severity = batch["low_severity"].to(device)
            if monotone:
                loss, diagnostics = monotone_paired_intervention_loss(
                    low_a,
                    low_b,
                    high_a,
                    high_b,
                    batch["mechanism_id"].to(device),
                    low_severity,
                    temperature=temperature,
                    ordinal_margin=ordinal_margin,
                )
            else:
                loss, diagnostics = unordered_ordered_pair_loss(
                    low_a,
                    low_b,
                    high_a,
                    high_b,
                    batch["program_id_low"].to(device),
                    batch["program_id_high"].to(device),
                    low_severity,
                    temperature=temperature,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            collected.setdefault("loss", []).append(float(loss.detach().cpu()))
            for name, value in diagnostics.items():
                collected.setdefault(name, []).append(float(value.detach().cpu()))
        if not collected.get("loss"):
            raise RuntimeError("ordered intervention loader produced no training batches")
        row = {"epoch": float(epoch)}
        row.update({name: sum(values) / len(values) for name, values in collected.items()})
        history.append(row)
        tag = "MPIE" if monotone else "PIE-matched"
        message = (
            f"[heartbeat] {tag} epoch={epoch:03d} loss={row['loss']:.4f} "
            f"nce={row['contrastive_loss']:.4f}"
        )
        if monotone:
            message += (
                f" ord={row['ordinal_loss']:.4f} "
                f"r={row['mean_low_radius']:.3f}->{row['mean_high_radius']:.3f}"
            )
        print(message, flush=True)
    model.eval()
    return history


def train_matched_unordered_encoder(
    model: PairedInterventionEncoder,
    loader: data.DataLoader,
    device: torch.device,
    **kwargs,
) -> list[dict[str, float]]:
    return _train_ordered_encoder(model, loader, device, monotone=False, **kwargs)


def train_monotone_encoder(
    model: MonotonePairedInterventionEncoder,
    loader: data.DataLoader,
    device: torch.device,
    **kwargs,
) -> list[dict[str, float]]:
    return _train_ordered_encoder(model, loader, device, monotone=True, **kwargs)


@dataclass(frozen=True)
class ConfirmatoryAttributionThresholds:
    min_heldout_severity_delta: float = 0.02
    min_seen_retrieval_lift_delta: float = -0.5
    min_heldout_retrieval_lift_delta: float = -0.5


def confirmatory_audit_gates(
    control_seen: RepresentationAuditMetrics,
    control_heldout: RepresentationAuditMetrics,
    candidate_seen: RepresentationAuditMetrics,
    candidate_heldout: RepresentationAuditMetrics,
    *,
    absolute_thresholds: RepresentationAuditThresholds | None = None,
    attribution_thresholds: ConfirmatoryAttributionThresholds | None = None,
) -> dict[str, object]:
    attribution_thresholds = attribution_thresholds or ConfirmatoryAttributionThresholds()
    absolute = representation_audit_gates(
        candidate_seen, candidate_heldout, absolute_thresholds
    )
    deltas = {
        "heldout_severity": (
            candidate_heldout.severity_spearman - control_heldout.severity_spearman
        ),
        "seen_retrieval_lift": (
            candidate_seen.retrieval_lift - control_seen.retrieval_lift
        ),
        "heldout_retrieval_lift": (
            candidate_heldout.retrieval_lift - control_heldout.retrieval_lift
        ),
    }
    attribution = {
        "heldout_severity_improvement": (
            deltas["heldout_severity"]
            >= attribution_thresholds.min_heldout_severity_delta
        ),
        "seen_retrieval_noninferiority": (
            deltas["seen_retrieval_lift"]
            >= attribution_thresholds.min_seen_retrieval_lift_delta
        ),
        "heldout_retrieval_noninferiority": (
            deltas["heldout_retrieval_lift"]
            >= attribution_thresholds.min_heldout_retrieval_lift_delta
        ),
    }
    attribution["pass"] = all(bool(value) for value in attribution.values())
    return {
        "absolute": absolute,
        "deltas": deltas,
        "attribution": attribution,
        "pass": bool(absolute["pass"]) and bool(attribution["pass"]),
    }
