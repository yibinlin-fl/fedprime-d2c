from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from fedprime.engine.cle_crsf_surgery import (
    assign_exact_response_gradient,
    direct_rawspec_gradients,
    direct_response_gradients,
    evaluate_objective,
    gradient_agreement,
    prepare_exact_surgery,
    public_anchor_kl,
    public_anchor_probabilities,
    rawspec_loss_from_moments,
    response_moments_from_prefix,
    response_loss_from_moments,
    run_exact_surgery,
    two_pass_rawspec_gradients,
    two_pass_response_gradients,
)


class _ToyAdapter:
    def __init__(self, layer: nn.Linear, head: nn.Linear):
        self.layer = layer
        self.head = head

    def feature_from_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.layer(prefix))

    def logits_from_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_from_prefix(prefix))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.layer.parameters())

    def clone(self, device: torch.device):
        replica = copy.deepcopy(self)
        replica.layer.to(device)
        replica.head.to(device)
        return replica


def _legacy_exact_surgery(adapter, base, probes, *, learning_rate, accepted_steps, anchor_limit):
    device = torch.device("cpu")
    reference = public_anchor_probabilities(adapter, base, device=device, batch_size=4)
    initial = evaluate_objective(adapter, "crsf", base, probes, device=device, batch_size=4)
    optimizer = torch.optim.Adam(adapter.trainable_parameters(), lr=learning_rate, weight_decay=0.0)
    normalized = [1.0]
    decisions = []
    for _step in range(accepted_steps):
        moments, _base_feature = response_moments_from_prefix(
            adapter, base, probes, device=device, batch_size=4
        )
        assign_exact_response_gradient(
            adapter,
            base,
            probes,
            moments,
            "crsf",
            normalizer=initial,
            device=device,
            batch_size=4,
        )
        optimizer.step()
        after = evaluate_objective(adapter, "crsf", base, probes, device=device, batch_size=4)
        anchor = public_anchor_kl(adapter, base, reference, device=device, batch_size=4)
        accepted = bool(after / (initial + 1.0e-12) <= normalized[-1] + 1.0e-6 and anchor <= anchor_limit)
        decisions.append(accepted)
        if not accepted:
            raise AssertionError("toy legacy fixture unexpectedly required rollback")
        normalized.append(after / (initial + 1.0e-12))
    return normalized, decisions


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


def test_optimized_exact_surgery_matches_frozen_legacy_contract() -> None:
    torch.manual_seed(20260903)
    device = torch.device("cpu")
    original = _ToyAdapter(nn.Linear(5, 4), nn.Linear(4, 3))
    for parameter in original.head.parameters():
        parameter.requires_grad_(False)
    base = np.random.default_rng(20260903).standard_normal((12, 5)).astype(np.float32)
    probes = [
        base + np.random.default_rng(100 + probe_id).normal(0.0, 0.05, base.shape).astype(np.float32)
        for probe_id in range(4)
    ]

    legacy = original.clone(device)
    legacy_losses, legacy_decisions = _legacy_exact_surgery(
        legacy,
        base,
        probes,
        learning_rate=1.0e-5,
        accepted_steps=3,
        anchor_limit=0.02,
    )

    prepared_on = original.clone(device)
    preparation = prepare_exact_surgery(
        prepared_on,
        "crsf",
        base,
        probes,
        device=device,
        batch_size=4,
    )
    independent_gradient_adapter = original.clone(device)
    independent_moments, independent_base_feature = response_moments_from_prefix(
        independent_gradient_adapter, base, probes, device=device, batch_size=4
    )
    assign_exact_response_gradient(
        independent_gradient_adapter,
        base,
        probes,
        independent_moments,
        "crsf",
        normalizer=preparation.initial_raw_loss,
        device=device,
        batch_size=4,
        base_feature=independent_base_feature,
    )
    gradient_match = gradient_agreement(
        [parameter.grad for parameter in independent_gradient_adapter.trainable_parameters()],
        preparation.gradients,
    )
    assert gradient_match.relative_error <= 1.0e-5
    assert gradient_match.cosine >= 0.99999

    optimized = original.clone(device)
    trace = run_exact_surgery(
        optimized,
        "crsf",
        base,
        probes,
        device=device,
        batch_size=4,
        learning_rate=1.0e-5,
        accepted_steps=3,
        anchor_limit=0.02,
        initial_preparation=preparation,
    )

    np.testing.assert_allclose(trace.accepted_normalized_losses, legacy_losses, rtol=1.0e-6, atol=1.0e-8)
    assert [bool(row["accepted"]) for row in trace.attempts] == legacy_decisions
    for legacy_parameter, optimized_parameter in zip(
        legacy.trainable_parameters(), optimized.trainable_parameters()
    ):
        torch.testing.assert_close(legacy_parameter, optimized_parameter, rtol=1.0e-5, atol=1.0e-7)
    assert all("post_update_exact_objective_seconds" in row for row in trace.attempts)


def test_rejected_candidates_keep_exact_post_eval_and_restore_state() -> None:
    torch.manual_seed(20260904)
    device = torch.device("cpu")
    adapter = _ToyAdapter(nn.Linear(5, 4), nn.Linear(4, 3))
    for parameter in adapter.head.parameters():
        parameter.requires_grad_(False)
    before = [parameter.detach().clone() for parameter in adapter.trainable_parameters()]
    base = np.random.default_rng(7).standard_normal((8, 5)).astype(np.float32)
    probes = [base + 0.05, base - 0.04]
    trace = run_exact_surgery(
        adapter,
        "crsf",
        base,
        probes,
        device=device,
        batch_size=4,
        learning_rate=1.0e-5,
        accepted_steps=1,
        anchor_limit=-1.0,
        maximum_backtracks=1,
    )
    assert trace.contract_failure
    assert trace.accepted_steps == 0
    assert [bool(row["accepted"]) for row in trace.attempts] == [False, False]
    assert all(float(row["post_update_exact_objective_seconds"]) > 0.0 for row in trace.attempts)
    for actual, expected in zip(adapter.trainable_parameters(), before):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_shared_prime_input_cache_is_exact_and_reusable(tmp_path) -> None:
    from fedprime.augmentations.frozen_prime import apply_frozen_prime_recipe
    from scripts.run_cle_k1_c_crsf_surgery import build_transformed_input_cache, load_banks

    images = np.random.default_rng(20260905).integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)
    recipe = load_banks()["a"]["recipes"][0]
    base, banks, manifest = build_transformed_input_cache(
        images,
        {"a": [recipe]},
        cache_dir=tmp_path,
        device=torch.device("cpu"),
        batch_size=2,
    )
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.0)
    direct = apply_frozen_prime_recipe(tensor, recipe).numpy()
    np.testing.assert_array_equal(np.asarray(banks["a"][0]), direct)
    np.testing.assert_array_equal(np.asarray(base), tensor.numpy())

    base_again, banks_again, manifest_again = build_transformed_input_cache(
        images,
        {"a": [recipe]},
        cache_dir=tmp_path,
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert manifest_again["signature"] == manifest["signature"]
    np.testing.assert_array_equal(np.asarray(base_again), np.asarray(base))
    np.testing.assert_array_equal(np.asarray(banks_again["a"][0]), direct)
