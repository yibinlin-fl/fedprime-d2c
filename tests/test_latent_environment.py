from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fedprime.methods import latent_environment as latent


def _images(count: int = 12) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(count, 32, 32, 3), dtype=np.uint8)


def test_paired_dataset_is_deterministic_cross_content_and_label_free(monkeypatch) -> None:
    calls: list[str] = []

    def identity(image, name, severity, rng):
        calls.append(str(name))
        return np.asarray(image, dtype=np.uint8)

    monkeypatch.setattr(latent, "apply_corruption", identity)
    dataset = latent.PairedInterventionDataset(
        _images(),
        np.arange(12),
        operators=("brightness",),
        seed=3,
        labels=None,
        clean_fraction=0.0,
    )
    first = dataset[4]
    repeated = dataset[4]
    assert first["source_index_a"].item() != first["source_index_b"].item()
    assert torch.equal(first["view_a"], repeated["view_a"])
    assert first["content_a"].item() == -1
    assert first["content_b"].item() == -1
    assert set(calls) == {"brightness"}
    assert first["view_a"].shape == (3, 32, 32)


def test_encoder_and_loss_are_finite_and_differentiable() -> None:
    model = latent.PairedInterventionEncoder(embedding_dim=8)
    embedding_a = model(torch.rand(6, 3, 32, 32))
    embedding_b = model(torch.rand(6, 3, 32, 32))
    loss, diagnostics = latent.paired_intervention_loss(
        embedding_a, embedding_b, torch.arange(6)
    )
    loss.backward()
    assert embedding_a.shape == (6, 8)
    assert torch.isfinite(loss)
    assert set(diagnostics) == {
        "contrastive_loss",
        "variance_loss",
        "covariance_loss",
    }
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tiny_training_loop_runs(monkeypatch) -> None:
    monkeypatch.setattr(
        latent,
        "apply_corruption",
        lambda image, name, severity, rng: np.asarray(image, dtype=np.uint8),
    )
    dataset = latent.PairedInterventionDataset(
        _images(8),
        np.arange(8),
        operators=("brightness", "contrast"),
        seed=0,
        clean_fraction=0.0,
    )
    model = latent.PairedInterventionEncoder(embedding_dim=4)
    history = latent.train_paired_intervention_encoder(
        model,
        DataLoader(dataset, batch_size=4, shuffle=False),
        torch.device("cpu"),
        epochs=1,
        learning_rate=1.0e-3,
        max_batches=1,
    )
    assert len(history) == 1
    assert np.isfinite(history[0]["loss"])


def test_audit_gates_and_operator_partition() -> None:
    good = latent.RepresentationAuditMetrics(
        samples=100,
        retrieval_recall_at_one=0.6,
        retrieval_chance=0.1,
        retrieval_lift=6.0,
        severity_spearman=0.7,
        mean_dimension_std=0.3,
        active_dimension_fraction=0.9,
        content_probe_accuracy=0.15,
        content_probe_chance=0.1,
        content_probe_lift=1.5,
    )
    heldout = latent.RepresentationAuditMetrics(
        samples=100,
        retrieval_recall_at_one=0.4,
        retrieval_chance=0.1,
        retrieval_lift=4.0,
        severity_spearman=0.6,
        mean_dimension_std=0.2,
        active_dimension_fraction=0.8,
        content_probe_accuracy=0.15,
        content_probe_chance=0.1,
        content_probe_lift=1.5,
    )
    assert latent.representation_audit_gates(good, heldout)["pass"] is True
    partition = latent.verify_operator_partition(("brightness",), ("fog",))
    assert partition["taxonomy_labels_used"] is False
    with pytest.raises(ValueError, match="overlap"):
        latent.verify_operator_partition(("brightness",), ("brightness",))


def test_representation_audit_runs_on_structured_pairs() -> None:
    class MeanEncoder(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            mean = images.mean(dim=(2, 3))
            return torch.cat([mean, mean[:, :1].square()], dim=1)

    rows = []
    for index in range(30):
        program = index % 5
        severity = float(program)
        base = torch.tensor(
            [(program + 1) / 6, (program + 2) / 7, (program + 3) / 8]
        ).view(3, 1, 1)
        rows.append(
            {
                "view_a": base.expand(3, 8, 8).clone(),
                "view_b": base.expand(3, 8, 8).clone(),
                "program_id": torch.tensor(program),
                "severity": torch.tensor(severity),
                "content_a": torch.tensor(index % 3),
                "content_b": torch.tensor((index + 1) % 3),
            }
        )
    metrics = latent.audit_representation(
        MeanEncoder(), DataLoader(rows, batch_size=10), torch.device("cpu"), seed=0
    )
    assert metrics.samples == 30
    assert np.isfinite(metrics.retrieval_lift)
    assert np.isfinite(metrics.severity_spearman)
    assert metrics.content_probe_chance > 0
