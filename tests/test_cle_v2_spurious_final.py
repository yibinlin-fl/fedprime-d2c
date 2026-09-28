from pathlib import Path

import pytest

from scripts.run_cle_v2_spurious_final import (
    ARMS,
    LOCAL_BATCH_CAP,
    ROUND_BUDGET,
    final_arm_config,
)


def _fake_package(tmp_path: Path) -> Path:
    package = tmp_path / "cle_hfl_v2_cross_map2_seed0_split0"
    (package / "data/gamma09").mkdir(parents=True)
    (package / "public").mkdir()
    (package / "initial_states").mkdir()
    (package / "splits").mkdir()
    (package / "pew/frozen_annotations").mkdir(parents=True)
    (package / "manifest.json").write_text(
        '{"protocol":"unit","scenario_seed":0,"split_seed":0}', encoding="utf-8"
    )
    return package


@pytest.mark.parametrize("mode", ["smoke", "benchmark", "formal"])
def test_final_configs_are_four_matched_objectives(tmp_path: Path, mode: str) -> None:
    package = _fake_package(tmp_path)
    configs = {
        arm: final_arm_config(
            arm,
            package_root=package,
            mode=mode,
            device="cpu",
            output_root=tmp_path / "outputs",
            train_seed=2,
        )
        for arm in ARMS
    }
    assert ARMS == ("erm", "cvar_dro", "pew_groupdro", "pew_ber")
    assert all(config["train"]["rounds"] == ROUND_BUDGET[mode] for config in configs.values())
    assert all(
        config["train"]["max_local_batches"] == LOCAL_BATCH_CAP[mode]
        for config in configs.values()
    )
    assert all(
        config["data"]["scenario_id"] == "cle_hfl_v2_cross_map2_seed0_split0"
        for config in configs.values()
    )
    assert configs["erm"]["method"]["cl_module"] == "none"
    assert configs["cvar_dro"]["method"]["cl_module"] == "cvar_dro"
    assert configs["pew_groupdro"]["method"]["fedease"]["objective"] == "pew_groupdro"
    assert configs["pew_ber"]["method"]["fedease"]["objective"] == "ce_ber"
    assert not any("cdep" in str(config).lower() for config in configs.values())
    assert all(config["seed"] == 2 for config in configs.values())
    assert all(config["experiment_name"].endswith("trainseed2") for config in configs.values())


def test_formal_budget_is_frozen() -> None:
    assert ROUND_BUDGET["formal"] == 40
    assert LOCAL_BATCH_CAP["formal"] is None


def test_full_epoch_modes_restore_july_training_intensity(tmp_path: Path) -> None:
    package = _fake_package(tmp_path)
    for mode in ("benchmark", "formal"):
        config = final_arm_config(
            "erm",
            package_root=package,
            mode=mode,
            device="cpu",
            output_root=tmp_path / "outputs",
        )
        assert config["train"]["pretrain_epochs"] == 0
        assert config["train"]["local_epochs"] == 1
        assert config["train"]["max_local_batches"] is None
        assert config["train"]["batch_size"] == 64
        assert config["train"]["public_batches_per_round"] == 4
        assert config["train"]["skip_nonfinite"] is False
        assert config["train"]["max_test_batches"] is None
        assert config["method"]["strict_fit_audit"]["max_audit_batches"] is None
