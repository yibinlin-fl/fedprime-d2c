from __future__ import annotations

import copy

from fedprime.utils.config import load_config
from scripts.analyze_strict_pew_asymhfl_multiseed import aggregate
from scripts.openi_strict_pew_asymhfl_entry import (
    CONFIGS,
    EXPERIMENTS,
    archive_name,
    comparison_path,
)


def _without_identity(config: dict) -> dict:
    cloned = copy.deepcopy(config)
    cloned.pop("experiment_name")
    cloned.pop("seed")
    return cloned


def test_train_seed_configs_preserve_matched_protocol() -> None:
    assert sorted(CONFIGS) == [0, 1, 2]
    assert sorted(EXPERIMENTS) == [0, 1, 2]

    loaded: dict[int, dict[str, dict]] = {}
    for train_seed in (1, 2):
        control = load_config(CONFIGS[train_seed]["control"])
        candidate = load_config(CONFIGS[train_seed]["candidate"])
        loaded[train_seed] = {"control": control, "candidate": candidate}

        assert control["seed"] == candidate["seed"] == train_seed
        assert control["device"] == candidate["device"]
        assert control["num_workers"] == candidate["num_workers"]
        assert control["output_root"] == candidate["output_root"]
        assert control["data"] == candidate["data"]
        assert control["models"] == candidate["models"]
        assert control["train"] == candidate["train"]
        assert control["checkpoints"] == candidate["checkpoints"]
        assert control["method"]["strict_fit_audit"] == candidate["method"]["strict_fit_audit"]
        assert control["method"]["strict_fit_audit"]["seed"] == 0
        assert control["method"]["strict_fit_audit"]["split_path"].endswith(
            "strict_cle_v2_alpha05_gamma09_seed0_split0.npz"
        )
        assert "seed0_split0" in control["data"]["private_root"]
        assert control["method"]["communication"] == "asymhfl_val"
        assert candidate["method"]["communication"] == "asymhfl_val"
        assert control["method"]["cl_module"] == "dcl"
        assert candidate["method"]["cl_module"] == "fedease"
        assert candidate["method"]["fedease"]["preserve_dcl"] is True
        assert candidate["method"]["fedease"]["ber"]["enabled"] is True
        assert "cdep" not in candidate["method"]["fedease"]

    assert _without_identity(loaded[1]["control"]) == _without_identity(loaded[2]["control"])
    assert _without_identity(loaded[1]["candidate"]) == _without_identity(loaded[2]["candidate"])


def test_seed_zero_output_names_remain_backward_compatible() -> None:
    assert comparison_path(0) == "outputs/strict_pew_asymhfl_val_comparison.json"
    assert archive_name(0) == "strict_pew_asymhfl_val_probe_outputs.tar.gz"
    assert comparison_path(1).endswith("trainseed1_comparison.json")
    assert archive_name(2).endswith("trainseed2_probe_outputs.tar.gz")


def test_multiseed_gate_go() -> None:
    payload = aggregate(
        {
            0: {"avg_acc": 3.9377, "worst_acc": 3.9040, "wcca": 5.05, "cfg": -6.32},
            1: {"avg_acc": 2.0, "worst_acc": 1.5, "wcca": 0.5, "cfg": -2.0},
            2: {"avg_acc": 0.5, "worst_acc": 0.4, "wcca": 0.1, "cfg": -0.5},
        }
    )

    assert payload["verdict"] == "GO"
    assert payload["full_gate_pass_count"] == 2
    assert all(payload["frozen_multiseed_gates"].values())


def test_multiseed_gate_rejects_directional_regression() -> None:
    payload = aggregate(
        {
            0: {"avg_acc": 5.0, "worst_acc": 4.0, "wcca": 5.0, "cfg": -6.0},
            1: {"avg_acc": 4.0, "worst_acc": 3.0, "wcca": 2.0, "cfg": -3.0},
            2: {"avg_acc": -0.1, "worst_acc": 0.5, "wcca": 0.1, "cfg": -0.5},
        }
    )

    assert payload["verdict"] == "NO-GO"
    assert payload["frozen_multiseed_gates"]["every_seed_avg_acc_positive"] is False


def test_multiseed_requires_exact_seed_set() -> None:
    try:
        aggregate(
            {
                0: {"avg_acc": 1.0, "worst_acc": 1.0, "wcca": 1.0, "cfg": -1.0},
                1: {"avg_acc": 1.0, "worst_acc": 1.0, "wcca": 1.0, "cfg": -1.0},
            }
        )
    except ValueError as exc:
        assert "Expected exactly training seeds" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("aggregate accepted an incomplete seed set")
