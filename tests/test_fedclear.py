from __future__ import annotations

import unittest

import torch

from fedprime.augmentations.counterfactual import build_counterfactual_views, select_operators
from fedprime.methods.ccre import class_conditional_counterfactual_risk
from fedprime.methods.ird import (
    invariant_anchor,
    leave_one_out_median,
    smooth_worst_view_distillation,
    standardize_logits,
)


class FedClearCoreTests(unittest.TestCase):
    def test_counterfactual_views_are_deterministic_and_bounded(self):
        images = torch.full((2, 3, 8, 8), 0.5)
        config = {
            "num_views": 3,
            "operators": ["identity", "gaussian_noise", "blur", "pixelate"],
            "noise_std": 0.1,
            "pixelate_size": 4,
        }
        views_a, names_a = build_counterfactual_views(images, config, seed=17)
        views_b, names_b = build_counterfactual_views(images, config, seed=17)
        self.assertEqual(names_a, names_b)
        self.assertEqual(names_a[0], "identity")
        self.assertEqual(len(views_a), 3)
        for left, right in zip(views_a, views_b):
            self.assertTrue(torch.equal(left, right))
            self.assertGreaterEqual(float(left.min()), 0.0)
            self.assertLessEqual(float(left.max()), 1.0)

    def test_operator_selection_cycles_when_more_views_are_requested(self):
        names = select_operators(["identity", "blur"], num_views=4, seed=0)
        self.assertEqual(names, ["identity", "blur", "blur", "blur"])

    def test_ccre_balances_classes_and_focuses_worst_view(self):
        labels = torch.tensor([0, 0, 0, 1])
        easy = torch.tensor([
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
        ], requires_grad=True)
        hard = torch.tensor([
            [0.0, 4.0],
            [4.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
        ], requires_grad=True)
        result = class_conditional_counterfactual_risk([easy, hard], labels, temperature=0.2)
        self.assertEqual(result.num_present_classes, 2)
        self.assertGreater(
            float(result.mean_worst_view_risk.detach()),
            float(result.mean_view_risk.detach()),
        )
        result.loss.backward()
        self.assertTrue(torch.isfinite(easy.grad).all())
        self.assertTrue(torch.isfinite(hard.grad).all())

    def test_ccre_accepts_local_presence_correction_weights(self):
        labels = torch.tensor([0, 1])
        logits = torch.tensor([[4.0, 0.0], [4.0, 0.0]], requires_grad=True)
        unweighted = class_conditional_counterfactual_risk([logits], labels, temperature=0.5)
        weighted = class_conditional_counterfactual_risk(
            [logits],
            labels,
            temperature=0.5,
            class_weights=torch.tensor([1.0, 10.0]),
        )
        self.assertGreater(float(weighted.loss.detach()), float(unweighted.loss.detach()))
        weighted.loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_logit_standardization_and_anchor(self):
        first = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
        second = first * 10.0 + 7.0
        standardized = standardize_logits(first)
        self.assertTrue(torch.allclose(standardized[0].mean(), torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(standardized[0].std(unbiased=False), torch.tensor(1.0), atol=1e-5))
        self.assertTrue(torch.allclose(invariant_anchor([first, second]), standardized, atol=1e-5))

    def test_leave_one_out_median_excludes_receiver(self):
        anchors = {
            0: torch.tensor([[100.0, 100.0]]),
            1: torch.tensor([[1.0, 4.0]]),
            2: torch.tensor([[2.0, 3.0]]),
            3: torch.tensor([[3.0, 2.0]]),
        }
        teacher = leave_one_out_median(anchors, receiver_id=0)
        self.assertTrue(torch.equal(teacher, torch.tensor([[2.0, 3.0]])))

    def test_ird_has_finite_gradients_and_reports_worst_view(self):
        teacher = torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
        good = torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]], requires_grad=True)
        bad = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], requires_grad=True)
        result = smooth_worst_view_distillation(
            [good, bad],
            teacher,
            distill_temperature=2.0,
            smooth_temperature=0.5,
        )
        self.assertGreater(float(result.worst_view_kl.detach()), float(result.mean_kl.detach()))
        result.loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())
        self.assertTrue(torch.isfinite(bad.grad).all())


if __name__ == "__main__":
    unittest.main()
