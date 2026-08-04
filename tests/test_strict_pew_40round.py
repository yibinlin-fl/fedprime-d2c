from __future__ import annotations

import copy

import pandas as pd
import pytest

from fedprime.utils.config import load_config
from scripts.analyze_strict_pew_asymhfl_40round import build_payload, summarize
from scripts.openi_strict_pew_asymhfl_40round_entry import (
    ARCHIVE_NAME,
    COMPARISON,
    CONFIGS,
    EXPERIMENTS,
)


def _without_identity_and_rounds(config: dict) -> dict:
    cloned = copy.deepcopy(config)
    cloned.pop("experiment_name")
    cloned["train"].pop("rounds")
    return cloned


def test_40round_configs_preserve_seed0_protocol() -> None:
    control = load_config(CONFIGS["control"])
    candidate = load_config(CONFIGS["candidate"])
    base_control = load_config("configs/openi_v100_rahfl_val_cle_v2_probe.yaml")
    base_candidate = load_config(
        "configs/openi_v100_fedease_pew_asymhfl_val_cle_v2_probe.yaml"
    )

    assert control["seed"] == candidate["seed"] == 0
    assert control["train"]["rounds"] == candidate["train"]["rounds"] == 40
    assert control["data"] == candidate["data"]
    assert control["models"] == candidate["models"]
    assert control["checkpoints"] == candidate["checkpoints"]
    assert control["method"]["strict_fit_audit"] == candidate["method"]["strict_fit_audit"]
    assert control["method"]["strict_fit_audit"]["seed"] == 0
    assert control["method"]["communication"] == candidate["method"]["communication"]
    assert control["method"]["cl_module"] == "dcl"
    assert candidate["method"]["cl_module"] == "fedease"
    assert candidate["method"]["fedease"]["preserve_dcl"] is True
    assert candidate["method"]["fedease"]["ber"]["enabled"] is True
    assert candidate["method"]["fedease"]["cdep"]["enabled"] is True
    assert _without_identity_and_rounds(control) == _without_identity_and_rounds(base_control)
    assert _without_identity_and_rounds(candidate) == _without_identity_and_rounds(base_candidate)


def test_40round_output_names_are_isolated() -> None:
    assert EXPERIMENTS["control"].startswith("durability40_")
    assert EXPERIMENTS["candidate"].startswith("durability40_")
    assert COMPARISON.endswith("40round_seed0_comparison.json")
    assert ARCHIVE_NAME == "strict_pew_asymhfl_val_40round_seed0_outputs.tar.gz"


def test_durability_gate_go() -> None:
    control = {
        scope: {"avg_acc": 30.0, "worst_acc": 25.0, "wcca": 1.0, "cfg": 30.0}
        for scope in ("final", "last_ten", "last_five")
    }
    candidate = {
        "final": {"avg_acc": 34.0, "worst_acc": 28.0, "wcca": 4.0, "cfg": 25.0},
        "last_ten": {"avg_acc": 33.0, "worst_acc": 27.0, "wcca": 3.0, "cfg": 27.0},
        "last_five": {"avg_acc": 31.0, "worst_acc": 25.5, "wcca": 1.0, "cfg": 29.5},
    }

    payload = build_payload(control, candidate)
    assert payload["verdict"] == "GO"
    assert all(payload["frozen_durability_gates"].values())


def test_durability_gate_rejects_late_collapse() -> None:
    control = {
        scope: {"avg_acc": 30.0, "worst_acc": 25.0, "wcca": 1.0, "cfg": 30.0}
        for scope in ("final", "last_ten", "last_five")
    }
    candidate = {
        "final": {"avg_acc": 29.0, "worst_acc": 24.0, "wcca": 0.0, "cfg": 31.0},
        "last_ten": {"avg_acc": 33.0, "worst_acc": 27.0, "wcca": 3.0, "cfg": 27.0},
        "last_five": {"avg_acc": 29.5, "worst_acc": 24.5, "wcca": 0.5, "cfg": 30.5},
    }

    payload = build_payload(control, candidate)
    assert payload["verdict"] == "NO-GO"
    assert payload["frozen_durability_gates"]["last_five_avg_acc_positive"] is False
    assert payload["frozen_durability_gates"]["last_five_cfg_negative"] is False


def test_summarize_requires_all_40_rounds(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "round": list(range(39)),
            "avg_acc": [1.0] * 39,
            "worst_acc": [1.0] * 39,
            "wcca": [1.0] * 39,
            "cfg": [1.0] * 39,
        }
    )
    frame.to_csv(tmp_path / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="Expected rounds 0-39"):
        summarize(tmp_path)
