from __future__ import annotations

import torch
from torch import nn

from fedprime.engine.cle_crsf_surgery import (
    direct_rawspec_gradients,
    direct_response_gradients,
    gradient_agreement,
    rawspec_loss_from_moments,
    response_loss_from_moments,
    two_pass_rawspec_gradients,
    two_pass_response_gradients,
)


def _fixture():
    torch.manual_seed(20260913)
    layer = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 4)).double()
    base = torch.randn(9, 5, dtype=torch.float64)
    probes = base[:, None] + 0.2 * torch.randn(9, 4, 5, dtype=torch.float64)
    return layer, base, probes


def test_crsf_is_scale_invariant() -> None:
    torch.manual_seed(4)
    mean = torch.randn(6, 5, dtype=torch.float64)
    energy = mean.square().sum(dim=1) + 0.5
    first = response_loss_from_moments(mean, energy, "crsf")
    second = response_loss_from_moments(7.0 * mean, 49.0 * energy, "crsf")
    torch.testing.assert_close(first, second, rtol=1.0e-10, atol=1.0e-12)


def test_two_pass_crsf_matches_direct_full_graph() -> None:
    layer, base, probes = _fixture()
    parameters = list(layer.parameters())
    direct = direct_response_gradients(parameters, layer, base, probes, "crsf")
    two_pass = two_pass_response_gradients(parameters, layer, base, probes, "crsf")
    agreement = gradient_agreement(direct, two_pass)
    assert agreement.relative_error <= 1.0e-5
    assert agreement.cosine >= 0.99999


def test_two_pass_shared_mean_and_gi_match_direct_full_graph() -> None:
    for objective in ("shared_mean", "generic_invariance"):
        layer, base, probes = _fixture()
        parameters = list(layer.parameters())
        direct = direct_response_gradients(parameters, layer, base, probes, objective)
        two_pass = two_pass_response_gradients(parameters, layer, base, probes, objective)
        agreement = gradient_agreement(direct, two_pass)
        assert agreement.relative_error <= 1.0e-5
        assert agreement.cosine >= 0.99999


def test_two_pass_rawspec_matches_direct_full_graph() -> None:
    layer, base, _ = _fixture()
    parameters = list(layer.parameters())
    direct = direct_rawspec_gradients(parameters, layer, base)
    two_pass = two_pass_rawspec_gradients(parameters, layer, base)
    agreement = gradient_agreement(direct, two_pass)
    assert agreement.relative_error <= 1.0e-5
    assert agreement.cosine >= 0.99999


def test_rawspec_is_scale_invariant() -> None:
    torch.manual_seed(7)
    features = torch.randn(20, 6, dtype=torch.float64)
    mean = features.mean(0)
    second = features.T @ features / features.shape[0]
    first = rawspec_loss_from_moments(mean, second)
    second_loss = rawspec_loss_from_moments(3.0 * mean, 9.0 * second)
    torch.testing.assert_close(first, second_loss, rtol=1.0e-10, atol=1.0e-12)


def test_apply_state_delta_changes_only_named_parameter() -> None:
    from fedprime.engine.cle_crsf_surgery import apply_state_delta

    model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    apply_state_delta(model, {"0.weight": torch.ones_like(model[0].weight)})
    torch.testing.assert_close(model[0].weight, before["0.weight"] + 1.0)
    torch.testing.assert_close(model[0].bias, before["0.bias"])
    torch.testing.assert_close(model[1].weight, before["1.weight"])


def test_k1c_gate_aggregation_keeps_rawspec_specificity() -> None:
    from scripts.run_cle_k1_c_crsf_surgery import aggregate_stage1

    rows = []
    for system in ("h9", "l9"):
        for client in range(4):
            for fold in ("ab", "ba"):
                for arm, chi, energy in (
                    ("frozen", 1.0, 1.0),
                    ("crsf", 0.60, 0.70),
                    ("shared_mean", 0.80, 0.50),
                    ("generic_invariance", 0.85, 0.40),
                    ("rawspec", 0.90, 0.95),
                ):
                    rows.append(
                        {
                            "system": system,
                            "client": client,
                            "fold": fold,
                            "arm": arm,
                            "chi_unseen": chi,
                            "response_energy": energy,
                        }
                    )
    result = aggregate_stage1(rows)
    assert result["h9"]["pass"]
    assert result["l9"]["pass"]
    assert result["h9"]["combined_reduction"]["crsf"] == 0.4


def test_generic_dominance_requires_same_control_in_both_systems() -> None:
    from scripts.run_cle_k1_c_crsf_surgery import baseline_dominance

    effects = {
        "crsf": {
            "dsa_reduction": 0.10,
            "wcca_improvement": 2.0,
            "cfg_reduction": 2.0,
            "avg_loss": 0.2,
            "worst_loss": 0.2,
            "clean_loss": 0.2,
        },
        "shared_mean": {
            "dsa_reduction": 0.10,
            "wcca_improvement": 2.0,
            "cfg_reduction": 2.0,
            "avg_loss": 0.2,
            "worst_loss": 0.2,
            "clean_loss": 0.2,
        },
        "generic_invariance": {
            "dsa_reduction": 0.0,
            "wcca_improvement": 0.0,
            "cfg_reduction": 0.0,
            "avg_loss": 2.0,
            "worst_loss": 2.0,
            "clean_loss": 2.0,
        },
    }
    stage2 = {"h9": {"effects": effects}, "l9": {"effects": effects}}
    result = baseline_dominance(stage2)
    assert result["generic_baseline_dominates"]
    assert result["global_dominators"] == ["shared_mean"]
