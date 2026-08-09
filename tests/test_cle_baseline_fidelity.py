from __future__ import annotations

from scripts.openi_cle_baseline_fidelity_entry import (
    ARM_ORDER,
    build_configs,
    fidelity_manifest,
    select_arms,
)


def test_fidelity_entry_keeps_repaired_and_anchor_arms_separate() -> None:
    configs = build_configs()
    assert tuple(configs) == ARM_ORDER
    assert configs["aughfl_fidelity"]["method"]["communication"] == "aughfl_fidelity"
    assert configs["feddf_fidelity"]["method"]["communication"] == "feddf_fidelity"
    assert configs["kt_pfl_fidelity"]["method"]["communication"] == "kt_pfl_fidelity"
    assert configs["rahfl"]["method"]["communication"] == "asymhfl_val"
    assert "cdep" not in configs["pew_ber"]["method"]["fedease"]


def test_fidelity_entry_preserves_common_cle_screening_contract() -> None:
    for config in build_configs().values():
        assert config["seed"] == 0
        assert config["data"]["scenario"] == "cle_hfl_v2"
        assert config["method"]["strict_fit_audit"]["enabled"] is True
        assert config["train"]["rounds"] == 12
        assert config["train"]["local_epochs"] == 1
        assert config["train"]["public_batches_per_round"] == 4
        assert config["checkpoints"]["save_final"] is False


def test_fidelity_manifest_does_not_claim_full_official_recipe() -> None:
    manifest = fidelity_manifest()
    assert "not an untouched official recipe" in manifest["scope"]
    assert manifest["historical_adapters_preserved"] == ["aughfl", "feddf", "kt_pfl"]
    assert manifest["matched_budget"]["public_batches_per_round"] == 4


def test_fidelity_arm_parser_accepts_comma_separated_subset() -> None:
    assert select_arms("feddf_fidelity,kt_pfl_fidelity") == [
        "feddf_fidelity",
        "kt_pfl_fidelity",
    ]
