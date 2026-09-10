from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_cle_v2_plugin_stage2 import evaluate_gates
from scripts.openi_cle_v2_plugin_stage2_entry import package_light_outputs
from scripts.run_cle_v2_plugin_stage2 import stage2_arm_config


def _package_root() -> Path:
    return Path("local_runs/cle_v2_factorial/cle_hfl_v2_paired_factorial_seed0_split0").resolve()


def _flatten(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, item in value.items():
            output.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
    else:
        output[prefix] = value
    return output


def test_stage2_pair_is_pure_pew_ber_and_matched(tmp_path: Path) -> None:
    package_root = _package_root()
    baseline = stage2_arm_config(
        "h9_b", package_root=package_root, mode="formal", device="cuda", output_root=tmp_path
    )
    plugin = stage2_arm_config(
        "h9_p", package_root=package_root, mode="formal", device="cuda", output_root=tmp_path
    )
    serialized = json.dumps(plugin).lower()
    assert "cdep" not in serialized
    assert plugin["method"]["fedease"]["environment_mode"] == "learned"
    assert plugin["method"]["fedease"]["ber"]["enabled"] is True
    assert plugin["method"]["fedease"]["preserve_dcl"] is True
    assert baseline["train"]["max_test_batches"] is None
    assert plugin["train"]["max_test_batches"] is None
    flat = {"base": _flatten(baseline), "plugin": _flatten(plugin)}
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


def test_stage2_rejects_non_stage2_arm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not allow"):
        stage2_arm_config(
            "h0_p", package_root=_package_root(), mode="smoke", device="cpu", output_root=tmp_path
        )


def test_frozen_gates_pass_only_for_full_success() -> None:
    accuracies = {arm: {"pooled": 25.0} for arm in ("h9_b", "h9_p")}
    utility = {"avg_acc": 2.0, "worst_acc": 1.5, "wcca": 0.2, "cfg": -1.2}
    gates = evaluate_gates(
        0.03,
        np.asarray([0.03, 0.02, 0.04, 0.025]),
        [0.01, 0.05],
        accuracies,
        utility,
        {"matched": True},
    )
    assert all(gates.values())
    failed = evaluate_gates(
        0.019,
        np.asarray([0.03, 0.02, 0.04, 0.025]),
        [0.01, 0.05],
        accuracies,
        utility,
        {"matched": True},
    )
    assert failed["P1_dsa_reduction"] is False


def test_light_package_excludes_checkpoints(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    configs = tmp_path / "configs"
    analysis = tmp_path / "analysis"
    for root in (outputs, configs, analysis):
        root.mkdir()
        (root / "keep.txt").write_text("ok", encoding="utf-8")
    checkpoints = outputs / "arm/checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "client_0.pt").write_bytes(b"large")
    archive = package_light_outputs("smoke", outputs, configs, analysis, tmp_path / "result.tar.gz")
    import tarfile

    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert any(name.endswith("keep.txt") for name in names)
    assert not any("checkpoints" in Path(name).parts for name in names)
