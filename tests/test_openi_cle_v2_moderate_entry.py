from __future__ import annotations

import json
import sys
import tarfile
from types import SimpleNamespace
from pathlib import Path

import pytest

import scripts.openi_cle_v2_moderate_entry as entry
from scripts.openi_cle_v2_moderate_entry import (
    package_recovery_outputs,
    require_upload_context,
    validate_request,
    write_timing,
)


def test_formal_rejects_multi_arm_shard() -> None:
    with pytest.raises(ValueError, match="exactly one arm"):
        validate_request(mode="formal", shard="axis2_asym", confirm_formal="true")


def test_formal_allows_single_arm_shard() -> None:
    validate_request(mode="formal", shard="asymhfl_erm", confirm_formal="true")


def test_upload_context_is_required() -> None:
    with pytest.raises(RuntimeError, match="refusing to start training"):
        require_upload_context(None)
    with pytest.raises(RuntimeError, match="refusing to start training"):
        require_upload_context(SimpleNamespace(output_path=""))
    require_upload_context(SimpleNamespace(output_path="/tmp/output"))


def test_missing_upload_context_stops_before_install_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry.os, "chdir", lambda path: None)
    monkeypatch.setattr(entry, "prepare_c2net", lambda: None)
    monkeypatch.setattr(entry, "run", lambda command, environment: commands.append(command))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openi_cle_v2_moderate_entry.py",
            "--mode=formal",
            "--shard=asymhfl_erm",
            "--confirm_formal=true",
        ],
    )

    with pytest.raises(RuntimeError, match="refusing to start training"):
        entry.main()

    assert commands == []


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


def test_analysis_failure_uploads_status_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / entry.PACKAGE_NAME
    package_root.mkdir()
    uploaded: list[list[str]] = []
    calls = 0

    def fake_run(command, environment):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("analysis failed")

    def fake_upload(context, paths):
        uploaded.append([Path(path).name for path in paths])

    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        entry,
        "prepare_c2net",
        lambda: SimpleNamespace(output_path=str(tmp_path / "openi-output")),
    )
    monkeypatch.setattr(entry, "find_archive", lambda roots: tmp_path / entry.INPUT_ARCHIVE)
    monkeypatch.setattr(entry, "extract_archive", lambda source, destination: package_root)
    monkeypatch.setattr(entry, "run", fake_run)
    monkeypatch.setattr(entry, "upload", fake_upload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openi_cle_v2_moderate_entry.py",
            "--mode=formal",
            "--shard=asymhfl_erm",
            "--confirm_formal=true",
            "--skip_install=true",
        ],
    )

    with pytest.raises(RuntimeError, match="analysis failed"):
        entry.main()

    assert len(uploaded) == 2
    assert any(name.endswith("_recovery_outputs.tar.gz") for name in uploaded[0])
    assert uploaded[1] == ["RUN_TIMING.json"]
    timing = json.loads(
        (
            tmp_path
            / "outputs"
            / "cle_v2_moderate_asymhfl_erm_formal"
            / "RUN_TIMING.json"
        ).read_text(encoding="utf-8")
    )
    assert timing["status"] == "failed_with_recovery_artifact"
    assert timing["scientific_evidence"] is False
