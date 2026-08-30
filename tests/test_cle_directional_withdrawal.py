import numpy as np
import torch

import scripts.analyze_cle_public_canonicalization_phase_b0 as phase_b0

from fedprime.engine.cle_directional_withdrawal import (
    confidence_calibrated_scdw_loss,
    decide_bridge_only_gate,
    directional_withdrawal,
    family_aggregated_withdrawal,
    family_separability_accuracy,
    within_source_variance_contraction,
)
from fedprime.engine.cle_probe_directional_promotion import score_binding_retrieval
from fedprime.engine.cle_shortcut_alignment import (
    OPERATOR_FAMILY_IDS,
    historical_family_binding,
)
from fedprime.models.public_canonicalizer import PublicNuisanceCanonicalizer


def _paired_probabilities(strength: float, sources_per_class: int = 12):
    labels = np.repeat(np.arange(10), sources_per_class)
    binding = historical_family_binding()
    original = np.empty((4, labels.size, 16, 10), dtype=np.float64)
    canonical = np.empty_like(original)
    for client_id in range(4):
        for source_id, label in enumerate(labels):
            for probe_id, family_id in enumerate(OPERATOR_FAMILY_IDS):
                neutral = np.full(10, 0.02, dtype=np.float64)
                neutral[label] = 0.82
                canonical[client_id, source_id, probe_id] = neutral / neutral.sum()
                shifted = neutral.copy()
                bound = np.flatnonzero(binding[client_id] == family_id)
                shifted[bound] *= 1.0 + strength
                original[client_id, source_id, probe_id] = shifted / shifted.sum()
    return original, canonical, labels, binding


def test_public_canonicalizer_preserves_shape_range_and_gradient():
    model = PublicNuisanceCanonicalizer(base_channels=8, residual_scale=0.25)
    inputs = torch.rand(3, 3, 32, 32, requires_grad=True)
    outputs = model(inputs)
    assert outputs.shape == inputs.shape
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
    outputs.mean().backward()
    assert inputs.grad is not None


def test_withdrawal_recovers_hidden_binding_only_after_oracle_stratification():
    original, canonical, labels, binding = _paired_probabilities(5.0)
    result = directional_withdrawal(original, canonical, labels)
    assert result.matrix.shape == (4, 16, 10)
    family_matrix = family_aggregated_withdrawal(result.matrix, OPERATOR_FAMILY_IDS)
    metrics = score_binding_retrieval(family_matrix, binding, np.arange(4))
    assert metrics["mean_average_precision"] > 0.99
    assert metrics["class_to_probe_family_hit_rate"] == 1.0
    assert result.pooled > 0.0


def test_identity_bridge_has_zero_directional_withdrawal():
    original, _, labels, _ = _paired_probabilities(5.0)
    result = directional_withdrawal(original, original.copy(), labels)
    assert np.allclose(result.matrix, 0.0)
    assert result.pooled == 0.0


def test_confidence_loss_stops_canonical_gradient():
    logits_original = torch.randn(40, 4, requires_grad=True)
    logits_canonical = torch.randn(40, 4, requires_grad=True)
    original = torch.softmax(logits_original, dim=1)
    canonical = torch.softmax(logits_canonical, dim=1)
    labels = torch.arange(40) % 4
    loss, means, lower = confidence_calibrated_scdw_loss(
        original, canonical, labels, z_alpha=0.0
    )
    assert means.shape == (4,)
    assert lower.shape == (4,)
    loss.backward()
    assert logits_original.grad is not None
    assert logits_canonical.grad is None


def test_nuisance_contraction_and_family_separability_detect_neutralization():
    rng = np.random.default_rng(7)
    clean = rng.integers(48, 208, size=(20, 32, 32, 3), dtype=np.uint8)
    family_offsets = np.asarray(
        [[20, 0, 0], [0, 20, 0], [0, 0, 20], [-20, -20, -20]], dtype=np.int16
    )
    images = np.empty((20, 16, 32, 32, 3), dtype=np.uint8)
    for probe_id, family_id in enumerate(OPERATOR_FAMILY_IDS):
        images[:, probe_id] = np.clip(
            clean.astype(np.int16) + family_offsets[family_id], 0, 255
        ).astype(np.uint8)
    neutral = np.broadcast_to(clean[:, None], images.shape).copy()
    assert within_source_variance_contraction(images, neutral) > 0.999
    base_accuracy = family_separability_accuracy(images, clean, OPERATOR_FAMILY_IDS)
    neutral_accuracy = family_separability_accuracy(neutral, clean, OPERATOR_FAMILY_IDS)
    assert base_accuracy > 0.99
    assert neutral_accuracy <= 0.30


def test_bridge_gate_requires_all_seven_groups():
    metrics = {
        "semantic_accuracy_delta_min": -0.005,
        "variance_contraction": 0.40,
        "separability_relative_reduction": 0.50,
        "hfl_gamma9_map": 0.80,
        "hfl_map_delta": 0.30,
        "hfl_hit_rate": 0.80,
        "hfl_positive_clients": 4,
        "local_gamma9_map": 0.82,
        "local_map_delta": 0.31,
        "local_hit_rate": 0.85,
        "local_positive_clients": 4,
        "canonical_vs_overlay_contraction_margin": 0.20,
        "clean_scdw_max": 0.01,
    }
    decision = decide_bridge_only_gate(metrics)
    assert decision["verdict"] == "GO_TO_12ROUND_ABC"
    metrics["semantic_accuracy_delta_min"] = -0.02
    failed = decide_bridge_only_gate(metrics)
    assert failed["verdict"] == "NO_GO_PNCB_BRIDGE"
    assert not failed["gates"]["G1_semantic_preservation"]


def test_infer_all_arms_handles_factory_client_dictionary(tmp_path, monkeypatch):
    class FakeModel:
        def load_state_dict(self, state, strict):
            assert state == {}
            assert strict is True

    monkeypatch.setattr(
        phase_b0,
        "build_models",
        lambda names, num_classes: {client_id: FakeModel() for client_id in range(4)},
    )
    monkeypatch.setattr(phase_b0, "load_state", lambda path: {})
    monkeypatch.setattr(
        phase_b0,
        "infer_model",
        lambda model, images, **kwargs: np.zeros((images.shape[0] * images.shape[1], 10)),
    )
    for arm in phase_b0.ARMS:
        arm_root = tmp_path / arm
        arm_root.mkdir()
        for client_id in range(4):
            (arm_root / f"client_{client_id}.pt").touch()
    images = np.zeros((2, 3, 32, 32, 3), dtype=np.uint8)
    result = phase_b0.infer_all_arms(
        tmp_path,
        {"identity": images},
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert set(result) == set(phase_b0.ARMS)
    assert result["h0"]["identity"].shape == (4, 2, 3, 10)
