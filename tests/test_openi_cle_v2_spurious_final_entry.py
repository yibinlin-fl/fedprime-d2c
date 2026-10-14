from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from scripts.openi_cle_v2_spurious_final_entry import (
    package_arm_outputs,
    validate_execution_request,
    write_arm_timing,
)


def _build_arm_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    outputs = tmp_path / "outputs"
    configs = tmp_path / "configs"
    analysis = tmp_path / "analysis"
    arm_root = outputs / "cle_v2_spurious_final_map2_erm_trainseed0"
    checkpoint = arm_root / "checkpoints" / "client_0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (arm_root / "metrics.csv").write_text("round,avg_acc\n0,1\n", encoding="utf-8")
    outputs.mkdir(exist_ok=True)
    (outputs / "INPUT_AUDIT.json").write_text('{"status":"PASS"}', encoding="utf-8")
    configs.mkdir()
    for name in (
        "map2_erm_trainseed0.json",
        "SPURIOUS_FINAL_MAP2_CONTRACT_TRAINSEED0_erm.json",
        "SPURIOUS_FINAL_MAP2_COMPLETION_TRAINSEED0_erm.json",
    ):
        (configs / name).write_text("{}", encoding="utf-8")
    analysis.mkdir()
    (analysis / "RESULT_SUMMARY.json").write_text("{}", encoding="utf-8")
    return outputs, configs, analysis


def test_arm_recovery_archive_keeps_checkpoint_and_merge_layout(tmp_path: Path) -> None:
    outputs, configs, _ = _build_arm_tree(tmp_path)
    archive = package_arm_outputs(
        outputs_root=outputs,
        configs_root=configs,
        analysis_root=None,
        arm="erm",
        train_seed=0,
        archive_path=tmp_path / "recovery.tar.gz",
        include_checkpoints=True,
    )

    with tarfile.open(archive, "r:gz") as handle:
        members = set(handle.getnames())
    assert "outputs/INPUT_AUDIT.json" in members
    assert (
        "outputs/cle_v2_spurious_final_map2_erm_trainseed0/checkpoints/client_0.pt"
        in members
    )
    assert "configs/SPURIOUS_FINAL_MAP2_CONTRACT_TRAINSEED0_erm.json" in members


def test_arm_final_archive_excludes_checkpoints_and_keeps_analysis(tmp_path: Path) -> None:
    outputs, configs, analysis = _build_arm_tree(tmp_path)
    archive = package_arm_outputs(
        outputs_root=outputs,
        configs_root=configs,
        analysis_root=analysis,
        arm="erm",
        train_seed=0,
        archive_path=tmp_path / "final.tar.gz",
        include_checkpoints=False,
    )

    with tarfile.open(archive, "r:gz") as handle:
        members = set(handle.getnames())
    assert "analysis/RESULT_SUMMARY.json" in members
    assert "outputs/cle_v2_spurious_final_map2_erm_trainseed0/metrics.csv" in members
    assert not any("checkpoints" in name for name in members)


def test_failed_axis1_timing_is_never_scientific_evidence(tmp_path: Path) -> None:
    timing = tmp_path / "RUN_TIMING_erm.json"
    write_arm_timing(
        timing,
        mode="formal",
        arm="erm",
        train_seed=0,
        training_seconds=10.0,
        analysis_seconds=None,
        status="failed_with_recovery_artifact",
        error="RuntimeError: platform stopped",
    )

    payload = json.loads(timing.read_text(encoding="utf-8"))
    assert payload["scientific_evidence"] is False
    assert payload["status"] == "failed_with_recovery_artifact"
    assert "platform stopped" in payload["warning"]


@pytest.mark.parametrize(
    ("arm", "train_seed"),
    (("all", "0"), ("erm", "all"), ("all", "all")),
)
def test_formal_rejects_multiplexed_long_jobs(arm: str, train_seed: str) -> None:
    with pytest.raises(PermissionError, match="exactly one arm and one train_seed"):
        validate_execution_request(mode="formal", arm=arm, train_seed=train_seed)


def test_benchmark_may_still_run_all_arms() -> None:
    validate_execution_request(mode="benchmark", arm="all", train_seed="0")

