from __future__ import annotations

import json
from pathlib import Path

from scripts.run_cle_k1_c_minimal import (
    ARMS,
    CLIENTS,
    CONFIG_PATH,
    SYSTEMS,
    aggregate_stage1,
    aggregate_stage2,
    decide,
    freeze_selection,
    load_config,
    parse_bool as parse_runner_bool,
)
from scripts.openi_cle_k1_c_minimal_entry import parse_bool as parse_openi_bool


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_selection_reconstructs_expected_hashes() -> None:
    config = load_config()
    selected = freeze_selection(50000, config)
    manifest = selected["manifest"]
    assert len(selected["correction"]) == 512
    assert len(selected["holdout"]) == 2000
    assert len(selected["probe_ids"]) == 16
    assert manifest["carrier_global_index_sha256"] == config["selection"]["carrier_global_index_sha256"]
    assert manifest["probe_ids"] == config["selection"]["probe_ids"]
    assert manifest["disjoint"] is True


def stage1_rows() -> list[dict[str, object]]:
    rows = []
    for system in SYSTEMS:
        for client, architecture in CLIENTS:
            for arm, chi, energy in (
                ("frozen", 0.40, 1.0),
                ("crsf", 0.30, 0.8),
                ("rawspec", 0.38, 0.9),
            ):
                rows.append(
                    {
                        "system": system,
                        "client": client,
                        "architecture": architecture,
                        "arm": arm,
                        "chi_unseen": chi,
                        "response_energy": energy,
                    }
                )
    return rows


def stage2_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    oracle = []
    task = []
    for system in SYSTEMS:
        for arm, dsa, client in (
            ("frozen", 0.20, [0.20, 0.20]),
            ("crsf", 0.12, [0.12, 0.12]),
            ("rawspec", 0.18, [0.18, 0.18]),
        ):
            oracle.append(
                {"system": system, "arm": arm, "dsa": dsa, "dsa_client": json.dumps(client)}
            )
            task.append(
                {
                    "system": system,
                    "arm": arm,
                    "avg_acc": 50.0,
                    "worst_acc": 45.0,
                    "wcca": 30.0,
                    "cfg": 5.0,
                    "clean_avg": 60.0,
                    "clean_worst": 55.0,
                }
            )
    return oracle, task


def test_minimal_gate_requires_transfer_and_rawspec_specificity() -> None:
    config = load_config()
    stage1 = aggregate_stage1(stage1_rows(), config)
    oracle, task = stage2_rows()
    stage2 = aggregate_stage2(oracle, task, config)
    result = decide(stage1, stage2)
    assert all(stage1[system]["pass"] for system in SYSTEMS)
    assert all(stage2[system]["pass"] for system in SYSTEMS)
    assert result["verdict"] == "GO_TO_K1_C_MINIMAL_REPLICATION"
    assert result["full_training_authorized"] is False

    failed = stage1_rows()
    for row in failed:
        if row["system"] == "h9" and row["arm"] == "rawspec":
            row["chi_unseen"] = 0.29
    failed_stage1 = aggregate_stage1(failed, config)
    assert failed_stage1["h9"]["gates"]["crsf_minus_rawspec_chi_advantage_ge_10pp"] is False
    assert decide(failed_stage1, stage2)["verdict"] == "NO_GO_CRSF_INTERVENTION"


def test_full_protocol_is_retained_but_superseded() -> None:
    old_spec = (ROOT / "docs/experiments/current/CLE_K1_C_CRSF_SURGERY_OPENI_ZH.md").read_text(
        encoding="utf-8"
    )
    new_spec = (ROOT / "docs/experiments/current/CLE_K1_C_MINIMAL_CAUSAL_GATE_OPENI_ZH.md").read_text(
        encoding="utf-8"
    )
    entry = (ROOT / "scripts/openi_cle_k1_c_minimal_entry.py").read_text(encoding="utf-8")
    assert "SUPERSEDED_BEFORE_FORMAL" in old_spec
    assert "512" in new_spec and "2,000" in new_spec and "64" in new_spec
    assert 'choices=("smoke", "benchmark", "formal")' in entry
    assert '"calibration"' not in entry
    assert CONFIG_PATH.is_file()
    assert ARMS == ("frozen", "crsf", "rawspec")


def test_confirm_formal_accepts_openi_explicit_boolean_values() -> None:
    for parser in (parse_runner_bool, parse_openi_bool):
        assert parser(True) is True
        assert parser("true") is True
        assert parser("1") is True
        assert parser("false") is False
        assert parser("0") is False
