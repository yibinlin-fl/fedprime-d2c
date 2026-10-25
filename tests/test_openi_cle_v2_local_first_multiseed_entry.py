from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.openi_cle_v2_local_first_multiseed_entry import (
    selected_train_seeds,
    validate_request,
    write_timing,
)
from scripts.openi_cle_v2_moderate_entry import require_upload_context


def test_all_selects_only_pending_seeds() -> None:
    assert selected_train_seeds("all") == (1, 2)
    assert selected_train_seeds("1") == (1,)
    assert selected_train_seeds("2") == (2,)


def test_formal_requires_explicit_confirmation() -> None:
    with pytest.raises(PermissionError, match="Formal is locked"):
        validate_request(mode="formal", confirm_formal="false")
    validate_request(mode="formal", confirm_formal="true")
    validate_request(mode="benchmark", confirm_formal="false")


def test_upload_context_is_required_before_training() -> None:
    with pytest.raises(RuntimeError, match="refusing to start training"):
        require_upload_context(None)
    require_upload_context(SimpleNamespace(output_path="/tmp/output"))


def test_incomplete_timing_is_not_scientific_evidence(tmp_path: Path) -> None:
    path = tmp_path / "RUN_TIMING.json"
    write_timing(
        path,
        mode="formal",
        train_seed=1,
        training_seconds=100.0,
        analysis_seconds=None,
        status="failed_with_recovery_artifact",
        error="RuntimeError: analysis failed",
    )
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert payload["scientific_evidence"] is False
    assert payload["rounds"] == 12
    assert payload["local_batches_per_client_round"] == 16
