from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.openi_cle_v2_spurious_final_entry as entry

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


@pytest.mark.parametrize("context", (None, SimpleNamespace(output_path="")))
def test_missing_upload_context_stops_before_install_or_training(monkeypatch, context) -> None:
    monkeypatch.setattr(entry, "parse_args", lambda: SimpleNamespace(
        mode="formal", arm="erm", train_seed="0", confirm_formal="true",
    ))
    monkeypatch.setattr(entry.os, "chdir", lambda path: None)
    monkeypatch.setattr(entry, "prepare_c2net", lambda: context)
    calls = []
    monkeypatch.setattr(entry, "run", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="output context is required before training"):
        entry.main()
    assert calls == []


@pytest.mark.parametrize("failure", (None, "analysis", "recovery_upload", "second_arm"))
def test_per_arm_publish_order_and_failure_recovery(tmp_path, monkeypatch, failure) -> None:
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry.os, "chdir", lambda path: None)
    monkeypatch.setattr(entry, "parse_args", lambda: SimpleNamespace(
        mode="benchmark", arm="all", train_seed="0", confirm_formal="false", skip_install="true",
    ))
    monkeypatch.setattr(entry, "prepare_c2net", lambda: SimpleNamespace(output_path=str(tmp_path)))
    monkeypatch.setattr(entry, "candidate_roots", lambda *args: [])
    monkeypatch.setattr(entry, "find_input_archive", lambda roots: tmp_path / "input.tar.gz")
    bundle = tmp_path / "bundle"
    (bundle / "cle_hfl_v2_cross_map2_seed0_split0").mkdir(parents=True)
    monkeypatch.setattr(entry, "safe_extract_bundle", lambda *args: bundle)
    monkeypatch.setattr(entry, "selected_arms", lambda arm: ("erm", "pew_ber"))
    events = []

    def fake_run(command, environment):
        def value(flag):
            return command[command.index(flag) + 1]

        if command[2].endswith("audit_cle_v2_factorial.py"):
            Path(value("--output")).write_text('{"status":"PASS"}', encoding="utf-8")
        elif command[2].endswith("run_cle_v2_spurious_final.py"):
            arm = value("--arm")
            events.append((arm, "train"))
            if failure == "second_arm" and arm == "pew_ber":
                raise RuntimeError("second arm stopped")
            checkpoint = Path(value("--output-root")) / entry._arm_output_name(arm, 0) / "checkpoints/client_0.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
        else:
            arm = value("--arm")
            assert value("--device") == "auto"
            events.append((arm, "analyze"))
            if failure == "analysis":
                raise RuntimeError("analysis stopped")
            (Path(value("--output-dir")) / "RESULT_SUMMARY.json").write_text("{}", encoding="utf-8")

    def fake_upload(context, paths):
        for path in paths:
            assert path.is_file()
            if path.suffix != ".gz":
                continue
            arm = "pew_ber" if "pew_ber" in path.name else "erm"
            kind = "recovery" if "recovery" in path.name else "final"
            with tarfile.open(path, "r:gz") as archive:
                has_checkpoint = any("checkpoints/client_0.pt" in name for name in archive.getnames())
            assert has_checkpoint == (kind == "recovery")
            if failure == "recovery_upload" and kind == "recovery":
                raise RuntimeError("upload stopped")
            events.append((arm, kind))

    monkeypatch.setattr(entry, "run", fake_run)
    monkeypatch.setattr(entry, "upload", fake_upload)
    if failure is None:
        entry.main()
        assert events == [(arm, stage) for arm in ("erm", "pew_ber") for stage in (
            "train", "recovery", "analyze", "final",
        )]
    else:
        with pytest.raises(RuntimeError, match="stopped"):
            entry.main()
        failed_arm = "pew_ber" if failure == "second_arm" else "erm"
        timing = next(tmp_path.glob(f"outputs/*/RUN_TIMING_{failed_arm}.json"))
        assert json.loads(timing.read_text(encoding="utf-8"))["scientific_evidence"] is False
        if failure == "analysis":
            assert events == [("erm", stage) for stage in ("train", "recovery", "analyze")]
        elif failure == "recovery_upload":
            assert events == [("erm", "train")]
        else:
            assert events == [("erm", stage) for stage in ("train", "recovery", "analyze", "final")] + [
                ("pew_ber", "train"),
            ]
            assert next(tmp_path.glob("*_erm_benchmark_recovery_outputs.tar.gz")).is_file()
            assert next(tmp_path.glob("*_erm_benchmark_outputs.tar.gz")).is_file()
