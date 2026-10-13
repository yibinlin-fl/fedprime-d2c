from __future__ import annotations

import json
from pathlib import Path

from scripts.run_cle_v2_moderate import (
    AXIS2_ARMS,
    DOSE_ARMS,
    moderate_arm_config,
    verify_moderate_package,
)


PACKAGE = Path(
    "local_runs/cle_v2_moderate_alpha10_rho05/"
    "cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0"
)


def test_moderate_package_contract() -> None:
    manifest = verify_moderate_package(PACKAGE)
    assert manifest["dirichlet_alpha"] == 1.0
    assert manifest["corruption_rate"] == 0.5
    assert manifest["gammas"] == {"gamma00": 0.0, "gamma05": 0.5, "gamma09": 0.9}


def test_dose_configs_use_frozen_conditions(tmp_path: Path) -> None:
    expected = {
        "erm_g00": "gamma00",
        "erm_g05": "gamma05",
        "erm_g09": "gamma09",
        "pew_ber_g09": "gamma09",
    }
    for arm in DOSE_ARMS:
        config = moderate_arm_config(
            arm, package_root=PACKAGE.resolve(), mode="smoke", output_root=tmp_path, device="cpu"
        )
        condition = expected[arm]
        assert Path(config["data"]["private_root"]).name == condition
        assert config["data"]["scenario"] == "cle_hfl_v2"
        assert config["data"]["private_corrupt_rate"] == 0.5
        assert config["train"]["max_local_batches"] == 1
        if arm == "pew_ber_g09":
            assert config["method"]["fedease"]["objective"] == "ce_ber"
            assert Path(config["method"]["fedease"]["pew"]["annotation_root"]).name == condition


def test_axis2_configs_are_protocol_matched(tmp_path: Path) -> None:
    configs = {
        arm: moderate_arm_config(
            arm, package_root=PACKAGE.resolve(), mode="formal", output_root=tmp_path, device="cuda"
        )
        for arm in AXIS2_ARMS
    }
    for arm, config in configs.items():
        assert config["train"]["rounds"] == 40
        assert config["train"]["max_local_batches"] == 64
        assert config["train"]["batch_size"] == 64
        assert config["seed"] == 0
        assert Path(config["data"]["private_root"]).name == "gamma09"
        assert config["train"]["skip_nonfinite"] is False
    assert configs["local_erm"]["method"]["communication"] == "none"
    assert configs["fedmd_adapter"]["method"]["communication"] == "fedmd"
    assert configs["asymhfl_erm"]["method"]["communication"] == "asymhfl_val"
    assert configs["asymhfl_pew_ber"]["method"]["fedease"]["objective"] == "ce_ber"
    assert "cdep" not in json.dumps(configs["asymhfl_pew_ber"]).lower()
