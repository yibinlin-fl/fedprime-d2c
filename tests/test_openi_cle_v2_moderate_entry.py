from __future__ import annotations

import json
import tarfile
from pathlib import Path

from scripts.openi_cle_v2_moderate_entry import package_recovery_outputs, write_timing


def test_recovery_archive_keeps_checkpoints(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    configs = tmp_path / "configs"
    checkpoint = outputs / "arm" / "checkpoints" / "client_0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    configs.mkdir()
    (configs / "CONTRACT.json").write_text("{}", encoding="utf-8")

    archive = package_recovery_outputs(outputs, configs, tmp_path / "recovery.tar.gz")

    with tarfile.open(archive, "r:gz") as handle:
        members = set(handle.getnames())
    assert "outputs/arm/checkpoints/client_0.pt" in members
    assert "configs/CONTRACT.json" in members


def test_failed_timing_is_never_scientific_evidence(tmp_path: Path) -> None:
    timing = tmp_path / "RUN_TIMING.json"
    write_timing(
        timing,
        mode="formal",
        shard="asymhfl_pew_ber",
        training_seconds=12.0,
        analysis_seconds=None,
        status="failed_with_recovery_artifact",
        error="RuntimeError: CUDA requested but unavailable",
    )

    payload = json.loads(timing.read_text(encoding="utf-8"))
    assert payload["scientific_evidence"] is False
    assert payload["status"] == "failed_with_recovery_artifact"
    assert "CUDA requested" in payload["warning"]


def test_training_pending_timing_is_not_scientific_evidence(tmp_path: Path) -> None:
    timing = tmp_path / "RUN_TIMING.json"
    write_timing(
        timing,
        mode="formal",
        shard="feddf_fidelity",
        training_seconds=100.0,
        analysis_seconds=None,
        status="training_complete_analysis_pending",
    )

    payload = json.loads(timing.read_text(encoding="utf-8"))
    assert payload["scientific_evidence"] is False
    assert payload["analysis_seconds"] is None
