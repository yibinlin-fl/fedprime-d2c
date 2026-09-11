from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_cle_v2_cross_scenario import evaluate_cross_gates
from scripts.run_cle_v2_cross_scenario import cross_arm_config


def _package_root() -> Path:
    return Path(
        "local_runs/cle_v2_factorial/cle_hfl_v2_paired_factorial_seed0_split0"
    ).resolve()


def _flatten(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, item in value.items():
            output.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
    else:
        output[prefix] = value
    return output


def test_cross_map_pair_is_pure_pew_ber_and_matched(tmp_path: Path) -> None:
    base = cross_arm_config(
        "h9_b",
        package_root=_package_root(),
        map_seed=1,
        mode="formal",
        device="cuda",
        output_root=tmp_path,
    )
    plugin = cross_arm_config(
        "h9_p",
        package_root=_package_root(),
        map_seed=1,
        mode="formal",
        device="cuda",
        output_root=tmp_path,
    )
    assert "cdep" not in json.dumps(plugin).lower()
    assert base["method"]["communication"] == plugin["method"]["communication"]
    assert plugin["method"]["fedease"]["ber"]["enabled"] is True
    flat = {"base": _flatten(base), "plugin": _flatten(plugin)}
    differing = {
        key
        for key in set(flat["base"]) | set(flat["plugin"])
        if flat["base"].get(key) != flat["plugin"].get(key)
    }
    assert differing == {
        "experiment_name",
        "method_name",
        "method.cl_module",
        "method.fedease.environment_mode",
        "method.fedease.num_environments",
        "method.fedease.preserve_dcl",
        "method.fedease.pew.annotation_root",
        "method.fedease.pew.checkpoint",
        "method.fedease.pew.unknown_threshold",
        "method.fedease.ber.enabled",
        "method.fedease.ber.support_gamma",
        "method.fedease.ber.count_cap",
        "method.fedease.ber.min_group_count",
    }


def test_cross_map_benchmark_budget() -> None:
    config = cross_arm_config(
        "h9_p",
        package_root=_package_root(),
        map_seed=2,
        mode="benchmark",
        device="cuda",
        output_root=Path("outputs"),
    )
    assert config["train"]["rounds"] == 1
    assert config["train"]["max_local_batches"] == 8
    assert config["train"]["max_test_batches"] == 1
    assert config["num_workers"] == 0


def test_cross_map_gates_focus_on_dsa_not_utility() -> None:
    gates = evaluate_cross_gates(
        0.12,
        {"null_p95": 0.03, "p_value": 0.001},
        0.06,
        np.asarray([0.03, 0.04, 0.05, 0.02]),
        [0.01, 0.08],
        {"h9_b": {"pooled": 22.0}, "h9_p": {"pooled": 21.0}},
        {"matched": True},
    )
    assert all(gates.values())
