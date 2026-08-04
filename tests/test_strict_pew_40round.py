from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fedprime.utils.config import load_config
import scripts.openi_strict_pew_asymhfl_40round_entry as entry
from scripts.analyze_strict_pew_asymhfl_40round import build_payload, summarize
from scripts.openi_strict_pew_asymhfl_40round_entry import (
    CONFIGS,
    EXPERIMENTS,
    archive_name,
    comparison_path,
    selected_train_seeds,
)


def _without_identity_and_rounds(config: dict) -> dict:
    cloned = copy.deepcopy(config)
    cloned.pop("experiment_name")
    cloned["train"].pop("rounds")
    return cloned


def test_40round_configs_preserve_matched_protocol() -> None:
    assert sorted(CONFIGS) == [0, 1, 2]
    assert sorted(EXPERIMENTS) == [0, 1, 2]

    for train_seed in (0, 1, 2):
        control = load_config(CONFIGS[train_seed]["control"])
        candidate = load_config(CONFIGS[train_seed]["candidate"])
        suffix = "" if train_seed == 0 else f"_trainseed{train_seed}"
        base_control = load_config(
            f"configs/openi_v100_rahfl_val_cle_v2{suffix}_probe.yaml"
        )
        base_candidate = load_config(
            f"configs/openi_v100_fedease_pew_asymhfl_val_cle_v2{suffix}_probe.yaml"
        )

        assert control["seed"] == candidate["seed"] == train_seed
        assert control["train"]["rounds"] == candidate["train"]["rounds"] == 40
        assert control["data"] == candidate["data"]
        assert control["models"] == candidate["models"]
        assert control["train"] == candidate["train"]
        assert control["checkpoints"] == candidate["checkpoints"]
        assert control["method"]["strict_fit_audit"] == candidate["method"]["strict_fit_audit"]
        assert control["method"]["strict_fit_audit"]["seed"] == 0
        assert control["method"]["strict_fit_audit"]["split_path"].endswith(
            "strict_cle_v2_alpha05_gamma09_seed0_split0.npz"
        )
        assert control["method"]["communication"] == candidate["method"]["communication"]
        assert control["method"]["cl_module"] == "dcl"
        assert candidate["method"]["cl_module"] == "fedease"
        assert candidate["method"]["fedease"]["preserve_dcl"] is True
        assert candidate["method"]["fedease"]["ber"]["enabled"] is True
        assert candidate["method"]["fedease"]["cdep"]["enabled"] is True
        assert _without_identity_and_rounds(control) == _without_identity_and_rounds(base_control)
        assert _without_identity_and_rounds(candidate) == _without_identity_and_rounds(base_candidate)


def test_40round_output_names_are_isolated() -> None:
    for train_seed in (0, 1, 2):
        assert EXPERIMENTS[train_seed]["control"].startswith("durability40_")
        assert EXPERIMENTS[train_seed]["candidate"].startswith("durability40_")
    assert comparison_path(0).endswith("40round_seed0_comparison.json")
    assert comparison_path(1).endswith("40round_trainseed1_comparison.json")
    assert comparison_path(2).endswith("40round_trainseed2_comparison.json")
    assert archive_name(0) == "strict_pew_asymhfl_val_40round_seed0_outputs.tar.gz"
    assert archive_name(1) == "strict_pew_asymhfl_val_40round_trainseed1_outputs.tar.gz"
    assert archive_name(2) == "strict_pew_asymhfl_val_40round_trainseed2_outputs.tar.gz"


def test_overnight_mode_runs_only_pending_training_seeds_in_order() -> None:
    assert selected_train_seeds("all") == [1, 2]
    assert selected_train_seeds("0") == [0]
    assert selected_train_seeds("1") == [1]
    assert selected_train_seeds("2") == [2]


def test_overnight_main_packages_and_uploads_each_seed_before_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    packaged: list[int] = []
    uploaded: list[str] = []
    args = SimpleNamespace(
        mode="both",
        train_seed="all",
        data_source="",
        skip_install=True,
        skip_import=True,
        skip_train=False,
        skip_summary=True,
        no_upload=False,
    )
    monkeypatch.setattr(entry, "parse_args", lambda: args)
    monkeypatch.setattr(entry, "prepare_c2net", lambda: object())
    monkeypatch.setattr(entry, "run", lambda command, environment: commands.append(command))

    def fake_package(methods: list[str], train_seed: int) -> Path:
        assert methods == ["control", "candidate"]
        packaged.append(train_seed)
        return Path(f"seed{train_seed}.tar.gz")

    def fake_upload(context: object, archive: Path, comparison: str) -> None:
        uploaded.append(f"{archive.name}:{comparison}")

    monkeypatch.setattr(entry, "package_outputs", fake_package)
    monkeypatch.setattr(entry, "upload_outputs", fake_upload)
    entry.main()

    run_configs = [
        command[-1]
        for command in commands
        if "scripts/run_experiment.py" in command
    ]
    assert run_configs == [
        CONFIGS[1]["control"],
        CONFIGS[1]["candidate"],
        CONFIGS[2]["control"],
        CONFIGS[2]["candidate"],
    ]
    assert packaged == [1, 2]
    assert uploaded == [
        f"seed1.tar.gz:{comparison_path(1)}",
        f"seed2.tar.gz:{comparison_path(2)}",
    ]


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
