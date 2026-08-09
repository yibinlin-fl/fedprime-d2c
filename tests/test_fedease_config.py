from __future__ import annotations

from pathlib import Path

import pytest

from fedprime.methods.fedease import FedEASEExperiment
from fedprime.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "debug_fedease_oracle_control.yaml",
    "debug_fedease_oracle_ber.yaml",
]


@pytest.mark.parametrize("config_name", CONFIGS)
def test_fedease_phase_one_configs_are_local_only_oracle_experiments(config_name: str, tmp_path):
    config = load_config(ROOT / "configs" / config_name)
    config["output_root"] = str(tmp_path)
    experiment = FedEASEExperiment(config)

    assert config["method_name"] == "fedease"
    assert config["data"]["scenario"] == "cle_hfl"
    assert config["method"]["cl_module"] == "fedease"
    assert config["method"]["communication"] == "none"
    assert config["method"]["fedease"]["environment_mode"] == "oracle"
    assert experiment._use_no_communication(config["method"])


def test_fedease_rejects_unknown_communication(tmp_path):
    config = load_config(ROOT / "configs" / CONFIGS[-1])
    config["output_root"] = str(tmp_path)
    config["method"]["communication"] = "public_magic"

    with pytest.raises(ValueError, match="communication"):
        FedEASEExperiment(config)
