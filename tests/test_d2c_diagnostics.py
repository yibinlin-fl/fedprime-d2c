from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from fedprime.engine.prior_diagnostics import PriorDiagnosticsRecorder, compare_priors
from fedprime.methods.d2c import D2CServer, entropy


def legacy_build_teacher(server: D2CServer, logits: torch.Tensor, oracle_prior=None):
    num_clients, _, num_classes = logits.shape
    probs = F.softmax(logits / server.temperature, dim=-1)
    prior = oracle_prior.to(logits.device) if oracle_prior is not None else probs.mean(dim=1)
    prior = prior.clamp(min=server.p_min, max=1.0)
    prior = prior / prior.sum(dim=-1, keepdim=True)
    beta = server._client_beta(prior, num_classes).view(num_clients, 1, 1)
    debiased_logits = logits - beta * torch.log(prior[:, None, :] + server.eps)
    debiased_probs = F.softmax(debiased_logits / server.temperature, dim=-1)
    class_weight = (prior + server.eps).pow(server.eta)
    class_weight = class_weight / class_weight.sum(dim=0, keepdim=True).clamp_min(server.eps)
    sample_conf = 1.0 - entropy(debiased_probs, dim=-1) / math.log(num_classes)
    sample_conf = sample_conf.clamp(min=0.0, max=1.0)
    weight = class_weight[:, None, :] * sample_conf[:, :, None]
    scores = (weight * debiased_probs).sum(dim=0)
    teacher = scores / scores.sum(dim=-1, keepdim=True).clamp_min(server.eps)
    return teacher.detach(), prior.detach()


class D2CDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.logits = torch.randn(4, 8, 10)

    def test_predicted_path_matches_legacy_formula(self):
        server = D2CServer()
        expected_teacher, expected_prior = legacy_build_teacher(server, self.logits)
        teacher, prior, diagnostics = server.build_teacher_with_diagnostics(self.logits)
        torch.testing.assert_close(teacher, expected_teacher, rtol=0, atol=0)
        torch.testing.assert_close(prior, expected_prior, rtol=0, atol=0)
        torch.testing.assert_close(diagnostics.predicted_prior, expected_prior, rtol=0, atol=0)

    def test_oracle_path_matches_legacy_formula_and_keeps_predicted_prior(self):
        oracle = torch.rand(4, 10)
        oracle = oracle / oracle.sum(dim=-1, keepdim=True)
        server = D2CServer()
        expected_teacher, expected_prior = legacy_build_teacher(server, self.logits, oracle)
        teacher, prior, diagnostics = server.build_teacher_with_diagnostics(
            self.logits,
            oracle_prior=oracle,
        )
        torch.testing.assert_close(teacher, expected_teacher, rtol=0, atol=0)
        torch.testing.assert_close(prior, expected_prior, rtol=0, atol=0)
        self.assertFalse(torch.equal(diagnostics.predicted_prior, diagnostics.used_prior))

    def test_prior_metrics_and_recorder_outputs(self):
        oracle = torch.eye(3)
        predicted = oracle.clone()
        metrics = compare_priors(predicted, oracle)
        torch.testing.assert_close(metrics["prior_l1"], torch.zeros(3))
        torch.testing.assert_close(metrics["prior_kl"], torch.zeros(3))
        torch.testing.assert_close(metrics["top_class_match"], torch.ones(3))

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            recorder = PriorDiagnosticsRecorder(
                output_dir=output_dir,
                oracle_prior=oracle,
                prior_source="oracle",
                save_rounds=[3],
            )
            recorder.record(3, 0, predicted, oracle)
            recorder.finalize()
            self.assertTrue((output_dir / "prior_diagnostics.csv").exists())
            self.assertTrue((output_dir / "priors" / "round_003.npz").exists())
            summary = json.loads((output_dir / "prior_summary.json").read_text())
            self.assertEqual(summary["num_records"], 3)
            self.assertEqual(summary["metrics"]["prior_l1"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
