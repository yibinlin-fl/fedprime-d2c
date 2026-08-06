from __future__ import annotations

from scripts.openi_cle_local_ablation_entry import ARM_ORDER, build_arm_configs, selected_arms


def test_local_ablation_matrix_has_frozen_seven_arms() -> None:
    configs = build_arm_configs()
    assert tuple(configs) == ARM_ORDER
    assert selected_arms("all") == list(ARM_ORDER)

    assert configs["a0_rahfl"]["method"]["cl_module"] == "dcl"
    assert configs["a1_ber"]["method"]["fedease"]["ber"]["enabled"] is True
    assert configs["a1_ber"]["method"]["fedease"]["cdep"]["enabled"] is False
    assert configs["a2_cdep"]["method"]["fedease"]["ber"]["enabled"] is False
    assert configs["a2_cdep"]["method"]["fedease"]["cdep"]["enabled"] is True
    assert configs["a3_full"]["method"]["fedease"]["ber"]["enabled"] is True
    assert configs["a3_full"]["method"]["fedease"]["cdep"]["enabled"] is True
    assert configs["a4_uncalibrated"]["method"]["fedease"]["pew"]["unknown_threshold"] == 0.55
    assert configs["a5_shuffled"]["method"]["fedease"]["environment_mode"] == "learned_shuffled"
    assert configs["a6_oracle_family"]["method"]["fedease"]["environment_mode"] == "oracle_family"


def test_ablation_configs_keep_protocol_and_communication_frozen() -> None:
    configs = build_arm_configs()
    for config in configs.values():
        assert config["data"]["scenario"] == "cle_hfl_v2"
        assert config["train"]["rounds"] == 12
        assert config["method"]["communication"] == "asymhfl_val"
        assert config["method"]["strict_fit_audit"]["enabled"] is True
        assert config["checkpoints"]["save_final"] is False
