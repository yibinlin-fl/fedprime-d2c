from pathlib import Path
import copy

import numpy as np
import pytest

from scripts.openi_cle_cdep_v2_paired_entry import (
    SHARED_CHECKPOINT,
    build_configs,
    compare_last_five,
    verify_shared_pew,
)


def test_paired_configs_change_only_cdep_and_experiment_name():
    configs = build_configs()
    control = configs["control"]
    candidate = configs["candidate"]
    assert control["seed"] == candidate["seed"] == 0
    assert control["train"] == candidate["train"]
    assert control["data"] == candidate["data"]
    assert control["models"] == candidate["models"]
    assert control["method"]["communication"] == candidate["method"]["communication"]
    assert control["method"]["strict_fit_audit"] == candidate["method"]["strict_fit_audit"]
    assert control["method"]["fedease"]["pew"] == candidate["method"]["fedease"]["pew"]
    assert control["method"]["fedease"]["pew"]["checkpoint"] == SHARED_CHECKPOINT
    assert control["method"]["fedease"]["cdep"]["enabled"] is False
    assert candidate["method"]["fedease"]["cdep"]["version"] == "v2"
    stripped_control = copy.deepcopy(control)
    stripped_candidate = copy.deepcopy(candidate)
    stripped_control.pop("experiment_name")
    stripped_candidate.pop("experiment_name")
    stripped_control["method"]["fedease"].pop("cdep")
    stripped_candidate["method"]["fedease"].pop("cdep")
    assert stripped_control == stripped_candidate


def test_paired_gate_uses_candidate_minus_control():
    result = compare_last_five(
        {"avg_acc": 30.0, "worst_acc": 20.0, "wcca": 5.0, "cfg": 12.0},
        {"avg_acc": 30.1, "worst_acc": 20.2, "wcca": 5.0, "cfg": 11.4},
    )
    assert result["candidate_minus_control"] == pytest.approx({
        "avg_acc": 0.1,
        "worst_acc": 0.2,
        "wcca": 0.0,
        "cfg": -0.6,
    })
    assert result["pass"] is True


def test_shared_pew_verification_requires_byte_identical_files(tmp_path: Path):
    outputs = {}
    for arm in ("control", "candidate"):
        output = tmp_path / arm
        predictions = output / "pew_predictions"
        predictions.mkdir(parents=True)
        for client_id in range(2):
            np.savez_compressed(
                predictions / f"client_{client_id}.npz",
                environment_ids=np.array([0, 1, 2]),
                confidence=np.array([0.8, 0.7, 0.9]),
            )
        outputs[arm] = output
    result = verify_shared_pew(outputs, num_clients=2)
    assert result["byte_identical"] is True
