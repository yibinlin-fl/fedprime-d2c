from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fedprime.data.loaders import build_public_loader
from fedprime.methods.pccd import (
    leave_one_out_consensus_teacher,
    log_opinion_consensus,
    normalized_entropy_confidence,
    paired_counterfactual_distillation,
    probability_view_disagreement,
)


class PCCDCoreTests(unittest.TestCase):
    def test_log_opinion_requires_cross_view_evidence(self):
        stable_a = torch.tensor([[0.8, 0.2]])
        stable_b = torch.tensor([[0.7, 0.3]])
        stable = log_opinion_consensus([stable_a, stable_b])
        conflict = log_opinion_consensus([
            torch.tensor([[0.9, 0.1]]),
            torch.tensor([[0.1, 0.9]]),
        ])
        self.assertGreater(float(stable[0, 0]), 0.7)
        self.assertTrue(torch.allclose(conflict, torch.tensor([[0.5, 0.5]]), atol=1e-6))

    def test_entropy_confidence_is_bounded(self):
        confidence = normalized_entropy_confidence(torch.tensor([
            [0.5, 0.5],
            [0.99, 0.01],
        ]))
        self.assertAlmostEqual(float(confidence[0]), 0.0, places=6)
        self.assertGreater(float(confidence[1]), 0.9)
        self.assertTrue(((confidence >= 0.0) & (confidence <= 1.0)).all())

    def test_leave_one_out_excludes_receiver_and_ignores_uniform_sender(self):
        consensuses = {
            0: torch.tensor([[0.99, 0.01]]),
            1: torch.tensor([[0.80, 0.20]]),
            2: torch.tensor([[0.50, 0.50]]),
        }
        confidences = {
            client_id: normalized_entropy_confidence(probabilities)
            for client_id, probabilities in consensuses.items()
        }
        teacher, weight = leave_one_out_consensus_teacher(consensuses, confidences, receiver_id=0)
        self.assertTrue(torch.allclose(teacher, consensuses[1], atol=1e-6))
        self.assertGreater(float(weight[0]), 0.0)

    def test_pccd_loss_has_finite_gradients(self):
        teacher = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        weights = torch.tensor([0.8, 0.6])
        first = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], requires_grad=True)
        second = torch.tensor([[0.5, -0.5], [0.0, 0.0]], requires_grad=True)
        result = paired_counterfactual_distillation([first, second], teacher, weights)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreaterEqual(
            float(result.worst_view_kl.detach()),
            float(result.mean_kl.detach()),
        )
        result.loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_view_disagreement_detects_conflict(self):
        same = probability_view_disagreement([
            torch.tensor([[0.8, 0.2]]),
            torch.tensor([[0.8, 0.2]]),
        ])
        conflict = probability_view_disagreement([
            torch.tensor([[0.99, 0.01]]),
            torch.tensor([[0.01, 0.99]]),
        ])
        self.assertAlmostEqual(float(same), 0.0, places=6)
        self.assertGreater(float(conflict), float(same))

    def test_cifar10_npy_public_loader_ignores_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = np.arange(4 * 32 * 32 * 3, dtype=np.uint8).reshape(4, 32, 32, 3)
            np.save(root / "public_images.npy", images)
            loader = build_public_loader(
                cifar100_root=root,
                public_size=4,
                batch_size=2,
                num_workers=0,
                seed=0,
                download=False,
                public_dataset="cifar10_npy",
            )
            batch_images, batch_targets = next(iter(loader))
            self.assertEqual(tuple(batch_images.shape), (2, 3, 32, 32))
            self.assertTrue(torch.equal(batch_targets, torch.zeros_like(batch_targets)))


if __name__ == "__main__":
    unittest.main()
