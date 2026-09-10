from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fedprime.methods.fedease import shuffle_values_within_class
from scripts.analyze_cle_v2_oracle_granularity import evaluate_gates
from scripts.run_cle_v2_oracle_granularity import oracle_arm_config


def _package_root() -> Path:
    return Path("local_runs/cle_v2_factorial/cle_hfl_v2_paired_factorial_seed0_split0").resolve()


def test_within_class_random_control_preserves_every_class_group_count() -> None:
    labels = np.repeat(np.arange(3), 20)
    values = np.tile(np.arange(5), 12)
    shuffled = shuffle_values_within_class(values, labels, seed=20260911)
    assert not np.array_equal(values, shuffled)
    assert np.array_equal(shuffled, shuffle_values_within_class(values, labels, seed=20260911))
    for class_id in range(3):
        mask = labels == class_id
        assert np.array_equal(
            np.bincount(values[mask], minlength=5),
            np.bincount(shuffled[mask], minlength=5),
        )


@pytest.mark.parametrize(
    ("arm", "environment_mode", "num_environments"),
    [
        ("h9_of", "oracle_family", 6),
        ("h9_oo", "oracle_operator", 15),
        ("h9_ro", "oracle_operator_shuffled", 15),
    ],
)
def test_oracle_configs_are_frozen_and_non_deployable(
    tmp_path: Path, arm: str, environment_mode: str, num_environments: int
) -> None:
    config = oracle_arm_config(
        arm,
        package_root=_package_root(),
        mode="formal",
        device="cuda",
        output_root=tmp_path,
    )
    fedease = config["method"]["fedease"]
    assert fedease["environment_mode"] == environment_mode
    assert fedease["num_environments"] == num_environments
    assert fedease["preserve_dcl"] is True
    assert fedease["ber"] == {
        "enabled": True,
        "support_gamma": 0.5,
        "count_cap": 32,
        "min_group_count": 2,
    }
    assert "pew" not in fedease
    assert "cdep" not in json.dumps(config).lower()
    assert config["train"]["rounds"] == 12
    assert config["train"]["max_local_batches"] == 16
    assert config["train"]["max_test_batches"] is None


def test_oracle_granularity_gates_require_gap_association_and_utility() -> None:
    dsa = {"h9_of": 0.08, "h9_oo": 0.04, "h9_ro": 0.09}
    ci = {"family_minus_operator": [0.01, 0.07], "random_minus_operator": [0.02, 0.08]}
    accuracy = {arm: {"pooled": value} for arm, value in {
        "h9_of": 23.0, "h9_oo": 22.5, "h9_ro": 21.0
    }.items()}
    metrics = {arm: {"avg_acc": value} for arm, value in {
        "h9_of": 25.0, "h9_oo": 24.5, "h9_ro": 23.0
    }.items()}
    zeros = {"h9_of": [1, 1, 2, 1], "h9_oo": [1, 0, 1, 1], "h9_ro": [2, 2, 2, 2]}
    gates = evaluate_gates(
        dsa=dsa,
        confidence_intervals=ci,
        accuracies=accuracy,
        metrics=metrics,
        zero_recall=zeros,
        traces={"matched": True},
    )
    assert all(gates.values())
    dsa["h9_of"] = 0.055
    failed = evaluate_gates(
        dsa=dsa,
        confidence_intervals=ci,
        accuracies=accuracy,
        metrics=metrics,
        zero_recall=zeros,
        traces={"matched": True},
    )
    assert failed["G2_operator_granularity_gap"] is False
