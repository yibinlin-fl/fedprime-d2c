from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fedprime.methods.fedease import FedEASEExperiment
from scripts.analyze_cle_v2_kt_fccl_plugin import evaluate_pair_gates
from scripts.run_cle_v2_kt_fccl_plugin import ARMS, BASES, kt_fccl_plugin_arm_config


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


def test_existing_communication_adapters_are_composed_without_local_coupling(tmp_path: Path) -> None:
    for key, pair in {"kt": ("kt_b", "kt_p"), "fc": ("fc_b", "fc_p")}.items():
        configs = {
            arm: kt_fccl_plugin_arm_config(
                arm,
                package_root=_package_root(),
                mode="formal",
                device="cpu",
                output_root=tmp_path,
            )
            for arm in pair
        }
        base, plugin = configs[pair[0]], configs[pair[1]]
        assert base["method"]["communication"] == plugin["method"]["communication"]
        assert base["method"]["communication"] == BASES[key]["communication"]
        assert base["method"]["baseline"] == plugin["method"]["baseline"]
        assert base["method"]["local_loader_mode"] == plugin["method"]["local_loader_mode"] == "standard"
        assert base["method"]["cl_module"] == "none"
        assert plugin["method"]["fedease"]["objective"] == "ce_ber"
        assert plugin["method"]["fedease"]["preserve_dcl"] is False
        assert base["method"]["lambda_jsd"] == plugin["method"]["lambda_jsd"] == 0.0
        assert "cdep" not in json.dumps(plugin).lower()


def test_pair_diff_is_only_independent_plugin_fields(tmp_path: Path) -> None:
    expected = {
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
    for pair in (("kt_b", "kt_p"), ("fc_b", "fc_p")):
        configs = [
            kt_fccl_plugin_arm_config(
                arm,
                package_root=_package_root(),
                mode="formal",
                device="cuda",
                output_root=tmp_path,
            )
            for arm in pair
        ]
        flat = [_flatten(config) for config in configs]
        differing = {
            key
            for key in set(flat[0]) | set(flat[1])
            if flat[0].get(key) != flat[1].get(key)
        }
        assert differing == expected


def test_plugin_runner_accepts_existing_kt_and_fccl_adapters(tmp_path: Path) -> None:
    names = {}
    for arm in ("kt_p", "fc_p"):
        config = kt_fccl_plugin_arm_config(
            arm,
            package_root=_package_root(),
            mode="smoke",
            device="cpu",
            output_root=tmp_path,
        )
        names[arm] = FedEASEExperiment(config)._communication_strategy.name
    assert names == {"kt_p": "kt_pfl_fidelity", "fc_p": "fccl"}


def test_all_four_arms_and_separated_gates() -> None:
    assert ARMS == ("kt_b", "kt_p", "fc_b", "fc_p")
    pair = ("kt_b", "kt_p")
    accuracies = {"kt_b": {"pooled": 25.0}, "kt_p": {"pooled": 24.5}}
    gates = evaluate_pair_gates(
        base_dsa=0.1,
        null={"null_p95": 0.02, "p_value": 0.001},
        reduction=0.03,
        client_reduction=np.asarray([0.01, 0.02, 0.03, 0.01]),
        confidence_interval=[0.01, 0.05],
        accuracies=accuracies,
        utility_delta={"avg_acc": 0.1, "worst_acc": -2.0, "wcca": 0.0, "cfg": 0.0},
        trace_audit={"matched": True},
        pair=pair,
    )
    assert all(gates.values())
    gates = evaluate_pair_gates(
        base_dsa=0.1,
        null={"null_p95": 0.02, "p_value": 0.001},
        reduction=0.03,
        client_reduction=np.asarray([0.01, 0.02, 0.03, 0.01]),
        confidence_interval=[0.01, 0.05],
        accuracies=accuracies,
        utility_delta={"avg_acc": -0.01, "worst_acc": 2.0, "wcca": 0.0, "cfg": 0.0},
        trace_audit={"matched": True},
        pair=pair,
    )
    assert gates["C2_dsa_reduction"] is True
    assert gates["U0_mean_utility"] is False
