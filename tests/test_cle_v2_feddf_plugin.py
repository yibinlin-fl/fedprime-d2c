from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fedprime.methods.fedease import FedEASEExperiment
from scripts.analyze_cle_v2_feddf_plugin import evaluate_gates
from scripts.run_cle_v2_feddf_plugin import ARMS, FEDDF_CONFIG, feddf_plugin_arm_config


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


def test_feddf_pair_is_native_ce_vs_pew_ber_ce(tmp_path: Path) -> None:
    configs = {
        arm: feddf_plugin_arm_config(
            arm,
            package_root=_package_root(),
            mode="formal",
            device="cpu",
            output_root=tmp_path,
        )
        for arm in ARMS
    }
    base, plugin = configs["fd_b"], configs["fd_p"]
    assert base["method"]["communication"] == "feddf_fidelity"
    assert plugin["method"]["communication"] == "feddf_fidelity"
    assert base["method"]["baseline"] == FEDDF_CONFIG
    assert plugin["method"]["baseline"] == FEDDF_CONFIG
    assert base["method"]["local_loader_mode"] == "standard"
    assert plugin["method"]["local_loader_mode"] == "standard"
    assert base["method"]["cl_module"] == "none"
    assert plugin["method"]["fedease"]["objective"] == "ce_ber"
    assert plugin["method"]["fedease"]["preserve_dcl"] is False
    assert base["method"]["lambda_jsd"] == plugin["method"]["lambda_jsd"] == 0.0
    assert "cdep" not in json.dumps(plugin).lower()


def test_feddf_pair_diff_is_only_plugin_fields(tmp_path: Path) -> None:
    base = feddf_plugin_arm_config(
        "fd_b", package_root=_package_root(), mode="formal", device="cuda", output_root=tmp_path
    )
    plugin = feddf_plugin_arm_config(
        "fd_p", package_root=_package_root(), mode="formal", device="cuda", output_root=tmp_path
    )
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
        "method.fedease.objective",
        "method.fedease.preserve_dcl",
        "method.fedease.pew.annotation_root",
        "method.fedease.pew.checkpoint",
        "method.fedease.pew.unknown_threshold",
        "method.fedease.ber.enabled",
        "method.fedease.ber.support_gamma",
        "method.fedease.ber.count_cap",
        "method.fedease.ber.min_group_count",
    }


def test_fedease_accepts_only_repaired_feddf_adapter(tmp_path: Path) -> None:
    config = feddf_plugin_arm_config(
        "fd_p", package_root=_package_root(), mode="smoke", device="cpu", output_root=tmp_path
    )
    experiment = FedEASEExperiment(config)
    assert experiment._communication_strategy.name == "feddf_fidelity"
    config["method"]["communication"] = "feddf"
    with pytest.raises(ValueError, match="communication"):
        FedEASEExperiment(config)


def test_feddf_formal_gates_require_dsa_and_mean_utility() -> None:
    accuracies = {"fd_b": {"pooled": 25.0}, "fd_p": {"pooled": 24.5}}
    gates = evaluate_gates(
        reduction=0.03,
        client_reduction=np.asarray([0.01, 0.02, 0.03, 0.01]),
        confidence_interval=[0.01, 0.05],
        accuracies=accuracies,
        utility_delta={"avg_acc": 0.1, "worst_acc": -2.0, "wcca": 0.0, "cfg": 0.0},
        trace_audit={"matched": True},
    )
    assert all(gates.values())
    gates["P2_mean_utility"] = evaluate_gates(
        reduction=0.03,
        client_reduction=np.asarray([0.01, 0.02, 0.03, 0.01]),
        confidence_interval=[0.01, 0.05],
        accuracies=accuracies,
        utility_delta={"avg_acc": -0.01, "worst_acc": 2.0, "wcca": 0.0, "cfg": 0.0},
        trace_audit={"matched": True},
    )["P2_mean_utility"]
    assert gates["P2_mean_utility"] is False
