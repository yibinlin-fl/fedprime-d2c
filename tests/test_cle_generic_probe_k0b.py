from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from fedprime.augmentations.frozen_prime import (
    apply_frozen_prime_recipe,
    load_frozen_prime_bank,
    sample_frozen_prime_bank,
    save_frozen_prime_bank,
)
from fedprime.engine.cle_generic_probe_gate import (
    decide_generic_probe_gate,
    generic_probe_statistics,
    paired_bootstrap_generic_deltas,
)


def test_frozen_prime_bank_is_deterministic_and_serializes_complete_state(tmp_path) -> None:
    first = sample_frozen_prime_bank(seed=20260902, count=2)
    second = sample_frozen_prime_bank(seed=20260902, count=2)
    different = sample_frozen_prime_bank(seed=20260903, count=2)
    assert first["bank_sha256"] == second["bank_sha256"]
    assert first["bank_sha256"] != different["bank_sha256"]
    assert [row["recipe_sha256"] for row in first["recipes"]] == [
        row["recipe_sha256"] for row in second["recipes"]
    ]

    manifest = save_frozen_prime_bank(
        first,
        state_path=tmp_path / "states.npz",
        manifest_path=tmp_path / "manifest.json",
    )
    assert manifest["count"] == 2
    assert manifest["bank_sha256"] == first["bank_sha256"]
    assert all(row["recipe_sha256"] for row in manifest["recipes"])
    with np.load(tmp_path / "states.npz", allow_pickle=False) as archive:
        assert "recipe_000_weights" in archive.files
        assert any(name.endswith("_dx") for name in archive.files)
        assert any(name.endswith("_coefficients") for name in archive.files)
        assert any(name.endswith("_kernel") for name in archive.files)
    loaded = load_frozen_prime_bank(
        state_path=tmp_path / "states.npz",
        manifest_path=tmp_path / "manifest.json",
    )
    assert loaded["bank_sha256"] == first["bank_sha256"]


def test_frozen_prime_recipe_reuses_identical_state_for_all_carriers() -> None:
    recipe = sample_frozen_prime_bank(seed=17, count=1)["recipes"][0]
    image = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    batch = image.repeat(2, 1, 1, 1)
    transformed = apply_frozen_prime_recipe(batch, recipe)
    repeated = apply_frozen_prime_recipe(batch, recipe)
    assert torch.equal(transformed[0], transformed[1])
    assert torch.equal(transformed, repeated)
    assert torch.isfinite(transformed).all()
    assert float(transformed.min()) >= 0.0
    assert float(transformed.max()) <= 1.0


def test_bundled_formal_banks_match_preregistered_hashes() -> None:
    root = Path("fedprime/augmentations/assets/cle_generic_probe_k0b")
    bank_a = load_frozen_prime_bank(
        state_path=root / "bank_a_states.npz",
        manifest_path=root / "bank_a_manifest.json",
    )
    bank_b = load_frozen_prime_bank(
        state_path=root / "bank_b_states.npz",
        manifest_path=root / "bank_b_manifest.json",
    )
    assert bank_a["bank_sha256"] == "6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01"
    assert bank_b["bank_sha256"] == "4A53497EC5DB6EC05C312E6166109FA4B52A5CC402CCE74E6EDB1253D913BF4E"


def test_rho_rejects_stable_but_nonselective_direction() -> None:
    selective = np.zeros((1, 20, 2, 3), dtype=np.float64)
    selective[:, :, 0] = np.asarray([1.0, -0.5, -0.5])
    selective[:, :, 1] = np.asarray([0.7, -0.35, -0.35])
    nonselective = np.zeros_like(selective)
    nonselective[:, :, 0] = np.asarray([0.5, 0.5, -1.0])
    nonselective[:, :, 1] = np.asarray([0.4, 0.4, -0.8])

    selective_result = generic_probe_statistics(selective)
    nonselective_result = generic_probe_statistics(nonselective)
    assert np.all(selective_result.kappa_cf > 0.99)
    assert np.all(nonselective_result.kappa_cf > 0.99)
    assert np.all(selective_result.rho > 0.5)
    assert np.allclose(nonselective_result.rho, 0.0, atol=1.0e-12)


def test_active_probe_rule_excludes_low_energy_half() -> None:
    response = np.zeros((1, 20, 4, 3), dtype=np.float64)
    response[:, :, 0] = np.asarray([0.01, -0.005, -0.005])
    response[:, :, 1] = np.asarray([0.02, -0.01, -0.01])
    response[:, :, 2] = np.asarray([1.0, -0.5, -0.5])
    response[:, :, 3] = np.asarray([2.0, -1.0, -1.0])
    result = generic_probe_statistics(response)
    assert result.active.shape == (1, 4)
    assert result.active.sum() == 2
    assert not result.active[0, 0]
    assert result.active[0, 2]


def test_paired_bootstrap_detects_selective_risk_increase() -> None:
    rng = np.random.default_rng(8)
    zero = rng.normal(scale=0.05, size=(2, 80, 6, 4))
    zero -= zero.mean(axis=-1, keepdims=True)
    nine = zero.copy()
    nine[:, :, :4] += np.asarray([0.7, -0.2, -0.2, -0.3])
    result = paired_bootstrap_generic_deltas(zero, nine, samples=100, seed=9)
    assert result["Dcf_delta"]["ci95"][0] > 0.0
    assert result["K_delta"]["ci95"][0] > 0.0
    assert result["R_delta"]["ci95"][0] > 0.0


def _arm(S: float, Dcf: float, K: float, R: float) -> dict[str, object]:
    summary = {
        "S": S,
        "Dcf": Dcf,
        "K": K,
        "R": R,
        "R_client": [R, R * 1.01, R * 0.99, R * 1.02],
    }
    return {
        "combined": summary,
        "bank_a": {**summary, "R": R * 0.98},
        "bank_b": {**summary, "R": R * 1.02},
    }


def _bootstrap(low: float = 0.01) -> dict[str, object]:
    return {
        "S_delta": {"ci95": [low, low + 0.1]},
        "Dcf_delta": {"ci95": [low, low + 0.1]},
        "K_delta": {"ci95": [low, low + 0.1]},
        "R_delta": {"ci95": [low, low + 0.1]},
    }


def test_gate_requires_combined_and_both_independent_banks() -> None:
    arms = {
        "h0": _arm(1.0, 0.1, 0.10, 0.10),
        "h9": _arm(1.3, 0.3, 0.15, 0.14),
        "l0": _arm(1.0, 0.1, 0.10, 0.10),
        "l9": _arm(1.3, 0.3, 0.15, 0.14),
    }
    decision = decide_generic_probe_gate(
        arms, {"hfl": _bootstrap(), "local": _bootstrap()}
    )
    assert decision["verdict"] == "GO_TO_K1_CHECKPOINT_SURGERY"

    arms["l9"]["bank_b"]["R"] = 0.105
    decision = decide_generic_probe_gate(
        arms, {"hfl": _bootstrap(), "local": _bootstrap()}
    )
    assert decision["verdict"] == "NO_GO_GENERIC_DIRECTIONAL_SIGNAL"
    assert not decision["systems"]["local"]["gates"]["G8_bank_b_replication"]


def test_generic_fragility_cannot_pass_without_K_or_R() -> None:
    arms = {
        "h0": _arm(1.0, 0.1, 0.10, 0.10),
        "h9": _arm(2.0, 0.2, 0.11, 0.105),
        "l0": _arm(1.0, 0.1, 0.10, 0.10),
        "l9": _arm(2.0, 0.2, 0.11, 0.105),
    }
    decision = decide_generic_probe_gate(
        arms, {"hfl": _bootstrap(), "local": _bootstrap()}
    )
    assert decision["verdict"] == "NO_GO_GENERIC_DIRECTIONAL_SIGNAL"
    assert decision["generic_fragility_kill_triggered"]


def test_primary_analyzer_does_not_import_cle_taxonomy() -> None:
    import scripts.analyze_cle_generic_probe_k0b as analyzer

    source = inspect.getsource(analyzer)
    assert "historical_family_binding" not in source
    assert "OPERATOR_FAMILY_IDS" not in source
    assert "apply_corruption" not in source
