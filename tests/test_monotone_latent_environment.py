from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedprime.methods import latent_environment as latent
from fedprime.methods import monotone_latent_environment as monotone


def _images(count: int = 10) -> np.ndarray:
    return np.random.default_rng(11).integers(
        0, 256, size=(count, 32, 32, 3), dtype=np.uint8
    )


def test_ordered_dataset_is_deterministic_and_strictly_ordered(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def identity(image, name, severity, rng):
        calls.append((str(name), int(severity)))
        return np.asarray(image, dtype=np.uint8)

    monkeypatch.setattr(latent, "apply_corruption", identity)
    dataset = monotone.OrderedPairedInterventionDataset(
        _images(),
        np.arange(10),
        operators=("brightness", "contrast"),
        seed=1,
        max_chain_length=2,
    )
    first = dataset[3]
    repeated = dataset[3]
    assert first["high_severity"].item() > first["low_severity"].item()
    assert first["source_index_a"].item() != first["source_index_b"].item()
    assert torch.equal(first["view_a_high"], repeated["view_a_high"])
    assert first["program_id_high"].item() != first["program_id_low"].item()
    assert {name for name, _ in calls}.issubset({"brightness", "contrast"})


def test_radial_encoder_has_positive_radius() -> None:
    model = monotone.MonotonePairedInterventionEncoder(embedding_dim=8)
    embedding, direction, radius = model.decompose(torch.rand(5, 3, 32, 32))
    assert embedding.shape == (5, 8)
    assert torch.all(radius > 0)
    assert torch.allclose(direction.norm(dim=1), torch.ones(5), atol=1.0e-5)
    assert torch.allclose(embedding.norm(dim=1), radius, atol=1.0e-5)


def test_ordinal_diagnostic_prefers_correct_radius_order() -> None:
    directions = torch.eye(4)
    low = directions * 0.2
    high = directions * 1.5
    identifiers = torch.arange(4)
    severities = torch.tensor([0.0, 1.0, 2.0, 3.0])
    _, correct = monotone.monotone_paired_intervention_loss(
        low,
        low,
        high,
        high,
        identifiers,
        severities,
    )
    _, reversed_order = monotone.monotone_paired_intervention_loss(
        high,
        high,
        low,
        low,
        identifiers,
        severities,
    )
    assert correct["ordinal_loss"] < reversed_order["ordinal_loss"]


def test_matched_control_and_candidate_training_run(monkeypatch) -> None:
    monkeypatch.setattr(
        latent,
        "apply_corruption",
        lambda image, name, severity, rng: np.asarray(image, dtype=np.uint8),
    )
    dataset = monotone.OrderedPairedInterventionDataset(
        _images(8),
        np.arange(8),
        operators=("brightness", "contrast"),
        seed=1,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    common = {
        "epochs": 1,
        "learning_rate": 1.0e-3,
        "temperature": 0.1,
        "ordinal_margin": 0.25,
        "max_batches": 1,
    }
    control_history = monotone.train_matched_unordered_encoder(
        latent.PairedInterventionEncoder(embedding_dim=4),
        loader,
        torch.device("cpu"),
        **common,
    )
    candidate_history = monotone.train_monotone_encoder(
        monotone.MonotonePairedInterventionEncoder(embedding_dim=4),
        loader,
        torch.device("cpu"),
        **common,
    )
    assert np.isfinite(control_history[0]["loss"])
    assert np.isfinite(candidate_history[0]["loss"])
    assert "ordinal_loss" in candidate_history[0]


def _metrics(retrieval: float, severity: float) -> latent.RepresentationAuditMetrics:
    return latent.RepresentationAuditMetrics(
        samples=100,
        retrieval_recall_at_one=0.5,
        retrieval_chance=0.1,
        retrieval_lift=retrieval,
        severity_spearman=severity,
        mean_dimension_std=0.5,
        active_dimension_fraction=1.0,
        content_probe_accuracy=0.02,
        content_probe_chance=0.02,
        content_probe_lift=1.0,
    )


def test_confirmatory_gate_requires_absolute_and_attribution_pass() -> None:
    gates = monotone.confirmatory_audit_gates(
        _metrics(5.5, 0.55),
        _metrics(4.0, 0.51),
        _metrics(5.2, 0.60),
        _metrics(3.8, 0.55),
    )
    assert gates["absolute"]["pass"] is True
    assert gates["attribution"]["pass"] is True
    assert gates["pass"] is True
