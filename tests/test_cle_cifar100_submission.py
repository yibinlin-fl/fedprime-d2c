from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.openi_cle_cifar100_submission_entry as entry
from scripts.openi_cle_cifar100_submission_entry import (
    selected_train_seeds,
    validate_request,
    verify_input_archive,
    write_timing,
)
from scripts.run_cle_cifar100_submission import submission_arm_config


def test_seed_selection_and_formal_lock() -> None:
    assert selected_train_seeds("all") == (0, 1, 2)
    assert selected_train_seeds("2") == (2,)
    with pytest.raises(PermissionError, match="Formal is locked"):
        validate_request(mode="formal", confirm_formal="false")
    validate_request(mode="formal", confirm_formal="true")
    validate_request(mode="benchmark", confirm_formal="false")


def test_input_archive_requires_exact_size_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / entry.INPUT_ARCHIVE
    archive.write_bytes(b"frozen-input")
    monkeypatch.setattr(entry, "INPUT_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(entry, "INPUT_ARCHIVE_SHA256", entry.sha256_file(archive))
    verify_input_archive(archive)
    archive.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="archive size"):
        verify_input_archive(archive)


def test_four_arm_configs_keep_data_roles_and_training_budget_matched(tmp_path: Path) -> None:
    configs = {
        arm: submission_arm_config(
            arm,
            package_root=tmp_path / "package",
            mode="formal",
            train_seed=1,
            device="cpu",
            output_root=tmp_path / "outputs",
        )
        for arm in ("erm_gamma0", "erm", "cvar_dro", "pew_ber")
    }
    for config in configs.values():
        assert config["data"]["private_dataset"] == "cifar100"
        assert config["data"]["public_dataset"] == "cifar10"
        assert config["data"]["num_classes"] == 100
        assert config["train"]["rounds"] == 40
        assert config["train"]["max_local_batches"] == 16
        assert config["method"]["strict_fit_audit"]["loader_seed"] == 20260918
        assert config["checkpoints"]["require_all"] is True
        assert config["checkpoints"]["strict"] is True
    assert configs["erm_gamma0"]["data"]["private_root"].endswith("gamma00")
    for arm in ("erm", "cvar_dro", "pew_ber"):
        assert configs[arm]["data"]["private_root"].endswith("gamma09")
    assert configs["erm"]["method"]["lambda_jsd"] == 0.0
    assert configs["cvar_dro"]["method"]["lambda_jsd"] == 0.0
    assert configs["pew_ber"]["method"]["fedease"]["preserve_dcl"] is False
    assert "cdep" not in json.dumps(configs).lower()


def test_failed_timing_is_not_scientific_evidence(tmp_path: Path) -> None:
    timing = tmp_path / "RUN_TIMING.json"
    write_timing(
        timing,
        mode="formal",
        train_seed=0,
        pew_seconds=10.0,
        training_seconds=20.0,
        analysis_seconds=None,
        status="failed_with_recovery_artifact",
        error="RuntimeError: analysis failed",
    )
    payload = json.loads(timing.read_text(encoding="utf-8"))
    assert payload["scientific_evidence"] is False
    assert payload["status"] == "failed_with_recovery_artifact"


def test_analysis_failure_preserves_and_uploads_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / entry.PACKAGE_DIRECTORY
    package_root.mkdir()
    source = tmp_path / entry.INPUT_ARCHIVE
    source.write_bytes(b"input")
    uploaded: list[list[str]] = []
    calls: list[list[str]] = []

    def fake_run(command, environment):
        calls.append(command)
        # install skipped; preflight, PEW, input audit, training, output audit,
        # then analysis. Simulate only the final analysis failure.
        if any("analyze_cle_cifar100_submission.py" in token for token in command):
            raise RuntimeError("analysis failed")
        if any("run_cle_cifar100_submission.py" in token for token in command):
            checkpoint = (
                tmp_path
                / "outputs/cle_cifar100_submission_trainseed0_formal"
                / "arm/checkpoints/client_0.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        entry,
        "prepare_c2net",
        lambda: SimpleNamespace(output_path=str(tmp_path / "openi-output")),
    )
    monkeypatch.setattr(entry, "find_archive", lambda roots: source)
    monkeypatch.setattr(entry, "verify_input_archive", lambda path: None)
    monkeypatch.setattr(entry, "safe_extract", lambda source, destination: package_root)
    monkeypatch.setattr(entry, "run", fake_run)
    monkeypatch.setattr(
        entry,
        "upload",
        lambda context, paths: uploaded.append([Path(path).name for path in paths]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openi_cle_cifar100_submission_entry.py",
            "--mode=formal",
            "--train_seed=0",
            "--confirm_formal=true",
            "--skip_install=true",
        ],
    )

    with pytest.raises(RuntimeError, match="analysis failed"):
        entry.main()

    assert len(uploaded) == 2
    assert any(name.endswith("_recovery_outputs.tar.gz") for name in uploaded[0])
    assert uploaded[1] == ["RUN_TIMING.json"]
    recovery = tmp_path / "cle_cifar100_submission_trainseed0_formal_recovery_outputs.tar.gz"
    with tarfile.open(recovery, "r:gz") as archive:
        assert "outputs/arm/checkpoints/client_0.pt" in archive.getnames()

