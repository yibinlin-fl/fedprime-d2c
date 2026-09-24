from pathlib import Path

from scripts.analyze_cle_submission_multiseed import (
    aggregate_cifar100,
    aggregate_local_first,
    aggregate_map2,
)
from scripts.openi_cle_v2_factorial_entry import selected_train_seeds as factorial_seeds
from scripts.openi_cle_v2_spurious_final_entry import selected_train_seeds as map2_seeds
from scripts.run_cle_cifar100_submission import ARMS as CIFAR100_ARMS, submission_arm_config
from scripts.run_cle_hfl_context import ARMS as CONTEXT_ARMS, context_arm_config
from scripts.run_cle_taxonomy_stress import ARMS as STRESS_ARMS, stress_arm_config


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "package"
    for relative in (
        "data/gamma00",
        "data/gamma09",
        "public",
        "initial_states",
        "splits",
        "pew_standard/annotations/gamma09",
        "pew_loo_motion_blur/annotations/gamma09",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def test_pending_seed_selectors_do_not_repeat_seed0() -> None:
    assert map2_seeds("all") == (1, 2)
    assert factorial_seeds("all") == (1, 2)


def test_cifar100_submission_configs_keep_roles_and_seed_matched(tmp_path: Path) -> None:
    package = _package(tmp_path)
    configs = {
        arm: submission_arm_config(
            arm,
            package_root=package,
            mode="formal",
            train_seed=2,
            device="cpu",
            output_root=tmp_path / "outputs",
        )
        for arm in CIFAR100_ARMS
    }
    assert configs["erm_gamma0"]["data"]["private_root"].endswith("gamma00")
    assert all(config["data"]["num_classes"] == 100 for config in configs.values())
    assert all(config["data"]["public_dataset"] == "cifar10" for config in configs.values())
    assert all(config["seed"] == 2 for config in configs.values())
    assert configs["pew_ber"]["method"]["fedease"]["objective"] == "ce_ber"


def test_taxonomy_stress_uses_loo_asset_only_for_candidate(tmp_path: Path) -> None:
    package = _package(tmp_path)
    configs = {
        arm: stress_arm_config(
            arm,
            package_root=package,
            mode="formal",
            device="cpu",
            output_root=tmp_path / "outputs",
        )
        for arm in STRESS_ARMS
    }
    assert "fedease" not in configs["erm"]["method"]
    assert "pew_loo_motion_blur" in configs["pew_ber_loo"]["method"]["fedease"]["pew"]["checkpoint"]
    assert all(config["train"]["rounds"] == 40 for config in configs.values())


def test_hfl_context_is_standalone_not_plugin_comparison(tmp_path: Path) -> None:
    package = _package(tmp_path)
    configs = {
        arm: context_arm_config(
            arm,
            package_root=package,
            mode="formal",
            output_root=tmp_path / "outputs",
            device="cpu",
        )
        for arm in CONTEXT_ARMS
    }
    assert configs["local_erm"]["method"]["communication"] == "none"
    assert configs["fedmd_adapter"]["method"]["communication"] == "fedmd"
    assert configs["fedproto_adapter"]["method"]["communication"] == "fedproto"
    assert configs["fedtgp_adapter"]["method"]["communication"] == "fedtgp"
    assert configs["feddf_fidelity"]["method"]["communication"] == "feddf_fidelity"
    assert configs["kt_pfl_fidelity"]["method"]["communication"] == "kt_pfl_fidelity"
    assert configs["fccl_adapter"]["method"]["communication"] == "fccl"
    assert configs["rhfl_adapter"]["method"]["communication"] == "rhfl"
    assert configs["aughfl_fidelity"]["method"]["communication"] == "aughfl_fidelity"
    assert configs["rahfl_fidelity"]["method"]["communication"] == "asymhfl_val"
    assert all("fedease" not in config["method"] for config in configs.values())
    assert all(
        config["data"]["scenario_id"] == "cle_hfl_v2_cross_map2_seed0_split0"
        for config in configs.values()
    )


def test_multiseed_aggregators_encode_claim_boundaries() -> None:
    map2 = {
        seed: {"key_deltas": {
            "erm_minus_pew_ber_dsa": 0.1,
            "pew_groupdro_minus_pew_ber_dsa": 0.05,
            "pew_ber_minus_cvar_dsa": 0.01,
            "pew_ber_minus_cvar_operator_accuracy_pp": 1.0,
            "pew_ber_minus_erm_operator_accuracy_pp": 2.0,
            "pew_ber_minus_erm_last10_avg_pp": 2.0,
        }} for seed in (0, 1, 2)
    }
    assert aggregate_map2(map2)["verdict"] == "GO_MAP2_TRAINING_SEED_STABILITY"
    local = {
        seed: {"estimands": {
            "hfl_cle_effect_base": 0.12,
            "local_cle_effect_base": 0.10,
            "communication_amplification_base": 0.02,
        }} for seed in (0, 1, 2)
    }
    assert aggregate_local_first(local)["verdict"] == "PREDOMINANTLY_LOCAL_FIRST"
    cifar100 = {
        seed: {"key_deltas": {
            "erm_cle_effect": 0.1,
            "erm_minus_pew_ber_dsa": 0.05,
            "cvar_minus_pew_ber_dsa": 0.01,
            "pew_ber_minus_erm_operator_accuracy_pp": 1.0,
            "pew_ber_minus_erm_last10_avg_pp": 1.0,
        }} for seed in (0, 1, 2)
    }
    assert aggregate_cifar100(cifar100)["verdict"] == "GO_SECOND_CONTROLLED_DATASET"
