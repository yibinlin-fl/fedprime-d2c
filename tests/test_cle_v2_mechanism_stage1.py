import json
import tarfile

import numpy as np

from fedprime.engine.cle_v2_factorial import OperatorDSAResult
from scripts.analyze_cle_v2_mechanism_stage1 import (
    ARMS,
    audit_traces,
    balanced_source_indices,
    evaluate_gates,
    mechanism_estimands,
)
from scripts.openi_cle_v2_mechanism_stage1_entry import package_light_outputs
from scripts.run_cle_v2_mechanism_stage1 import stage1_arm_config


def fake_result(pooled: float, client: list[float]) -> OperatorDSAResult:
    values = np.asarray(client, dtype=np.float64)
    return OperatorDSAResult(
        source_effects=np.zeros((4, 2, 3), dtype=np.float64),
        client_operator=np.zeros((4, 2), dtype=np.float64),
        client=values,
        pooled=float(pooled),
    )


def test_stage1_configs_are_baseline_only_and_frozen(tmp_path):
    for mode, rounds, batches in (("smoke", 1, 1), ("benchmark", 1, 16), ("formal", 12, 16)):
        for arm in ARMS:
            config = stage1_arm_config(
                arm,
                package_root=tmp_path,
                mode=mode,
                device="cpu",
                output_root=tmp_path / "outputs",
            )
            assert config["method_name"] == "rahfl"
            assert "fedease" not in config["method"]
            assert config["train"]["rounds"] == rounds
            assert config["train"]["max_local_batches"] == batches
            assert config["train"]["max_test_batches"] == 1
            assert config["num_workers"] == 0
            assert config["checkpoints"]["save_rounds"] == []
            assert config["method"]["strict_fit_audit"]["max_audit_batches"] == (
                1 if mode == "smoke" else None
            )


def test_mechanism_estimands_use_only_four_baseline_arms():
    results = {
        "h0_b": fake_result(0.05, [0.05] * 4),
        "h9_b": fake_result(0.20, [0.20] * 4),
        "l0_b": fake_result(0.04, [0.04] * 4),
        "l9_b": fake_result(0.12, [0.12] * 4),
    }
    summary = mechanism_estimands(results)
    assert np.isclose(summary["estimands"]["hfl_cle_effect_base"], 0.15)
    assert np.isclose(summary["estimands"]["local_cle_effect_base"], 0.08)
    assert np.isclose(summary["estimands"]["communication_amplification_base"], 0.07)


def test_stage1_gates_include_learning_floor():
    estimands = {"hfl_cle_effect_base": 0.15, "local_cle_effect_base": 0.09}
    client = {
        "hfl_cle_effect_base": [0.10, 0.12, 0.16, 0.22],
        "local_cle_effect_base": [0.06, 0.07, 0.10, 0.13],
    }
    bootstrap = {
        "hfl_cle_effect_base": {"ci95": [0.11, 0.19]},
        "local_cle_effect_base": {"ci95": [0.06, 0.12]},
    }
    shuffled = {"observed": 0.20, "null_p95": 0.08, "p_value": 0.01}
    accuracy = {arm: {"pooled": 25.0} for arm in ARMS}
    gates, share = evaluate_gates(estimands, client, bootstrap, shuffled, accuracy)
    assert all(gates.values())
    assert np.isclose(share, 0.6)
    accuracy["l0_b"]["pooled"] = 19.99
    gates, _ = evaluate_gates(estimands, client, bootstrap, shuffled, accuracy)
    assert gates["L0_learning_floor"] is False


def test_smoke_source_subset_is_class_balanced():
    labels = np.repeat(np.arange(10), 5)
    selected = balanced_source_indices(labels, 20)
    assert np.array_equal(np.bincount(labels[selected], minlength=10), np.full(10, 2))


def test_trace_audit_and_light_archive_exclude_checkpoints(tmp_path):
    outputs = tmp_path / "outputs"
    for arm in ARMS:
        root = outputs / f"cle_v2_mechanism_stage1_{arm}_trainseed0"
        root.mkdir(parents=True)
        rows = [json.dumps({"round": 0, "client": 0, "sha256": "g0" if "0_" in arm else "g9"})]
        (root / "local_batch_trace.jsonl").write_text("\n".join(rows), encoding="utf-8")
        checkpoint = root / "checkpoints"
        checkpoint.mkdir()
        (checkpoint / "client_0.pt").write_bytes(b"model")
        (root / "metrics.csv").write_text("round\n0\n", encoding="utf-8")
    assert audit_traces(outputs)["gamma00"]["matched"] is True
    configs = tmp_path / "configs"
    analysis = tmp_path / "analysis"
    configs.mkdir()
    analysis.mkdir()
    (configs / "config.json").write_text("{}", encoding="utf-8")
    (analysis / "RESULT_SUMMARY.json").write_text("{}", encoding="utf-8")
    archive = package_light_outputs(
        "benchmark", outputs, configs, analysis, tmp_path / "stage1.tar.gz"
    )
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert any(name.endswith("metrics.csv") for name in names)
    assert any(name.endswith("RESULT_SUMMARY.json") for name in names)
    assert not any("checkpoints" in name for name in names)
