from __future__ import annotations

import torch

from fedprime.methods.environment_witness import PublicEnvironmentWitness, select_unknown_threshold


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
