from __future__ import annotations

import torch

import numpy as np

from fedprime.methods.environment_witness import (
    PublicEnvironmentDataset,
    PublicEnvironmentWitness,
    infer_environment_annotations,
    select_unknown_threshold,
)


def test_public_environment_witness_outputs_environment_severity_and_embedding():
    model = PublicEnvironmentWitness(embedding_dim=16, num_environments=6, severity_levels=5)
    environment, severity, embedding = model(torch.rand(4, 3, 32, 32))
    assert environment.shape == (4, 6)
    assert severity.shape == (4, 5)
    assert embedding.shape == (4, 16)


def test_unknown_threshold_calibration_prefers_public_validation_accuracy():
    probabilities = torch.tensor(
        [
            [0.60, 0.10, 0.10, 0.10, 0.05, 0.05],
            [0.24, 0.20, 0.16, 0.15, 0.13, 0.12],
            [0.05, 0.05, 0.05, 0.05, 0.10, 0.70],
        ]
    )
    targets = torch.tensor([0, 5, 5])
    result = select_unknown_threshold(
        probabilities,
        targets,
        unknown_id=5,
        thresholds=torch.tensor([0.0, 0.3, 0.8]),
    )
    assert result["threshold"] == torch.tensor(0.3).item()
    assert result["accuracy"] == 1.0


def test_multilabel_pew_exposes_composite_corruption_as_soft_memberships():
    images = np.zeros((6, 32, 32, 3), dtype=np.uint8)
    dataset = PublicEnvironmentDataset(
        images,
        np.arange(6),
        seed=0,
        label_mode="multi_label",
    )

    _, target, severity = dataset[5]

    assert target.shape == (6,)
    assert torch.allclose(target.sum(), torch.tensor(2.0))
    assert int((target > 0).sum()) == 2
    assert target[-1].item() == 0.0
    assert severity > 0


def test_multilabel_inference_returns_normalized_soft_ber_responsibilities():
    class FixedWitness(torch.nn.Module):
        def forward(self, images):
            logits = torch.tensor(
                [[0.0, 2.0, 2.0, 0.0, 0.0, -4.0], [0.0] * 6],
                device=images.device,
            )[: len(images)]
            return logits, logits[:, :5], torch.zeros(len(images), 3, device=images.device)

    annotations = infer_environment_annotations(
        FixedWitness(),
        np.zeros((2, 32, 32, 3), dtype=np.uint8),
        torch.device("cpu"),
        batch_size=2,
        confidence_threshold=0.6,
        include_probabilities=True,
    )

    probabilities = torch.from_numpy(annotations["environment_probabilities"])
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert probabilities[0, 1] > 0 and probabilities[0, 2] > 0
    assert probabilities[1, -1] == 1.0
