from __future__ import annotations

import numpy as np
import torch
import inspect
import json
from pathlib import Path

from fedprime.engine.cle_public_carrier_moment import class_vs_rest_evidence
from fedprime.engine.cle_sdmn_headonly import (
    centered_response_from_features,
    class_vs_rest_evidence_torch,
    make_direction_sham,
    match_random_probes,
    run_head_surgery,
    select_high_risk_probes,
)
from scripts.run_cle_k1_sdmn_headonly import public_split, sha256_array
from scripts.run_cle_k1_sdmn_formal import aggregate_primary, load_frozen_contract


def test_torch_class_vs_rest_matches_frozen_numpy_definition() -> None:
    rng = np.random.default_rng(31)
    logits = rng.normal(size=(7, 4, 5)).astype(np.float64)
    actual = class_vs_rest_evidence_torch(torch.from_numpy(logits)).numpy()
    expected = class_vs_rest_evidence(logits)
    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_high_risk_selection_reuses_active_top_twenty_percent() -> None:
    response = np.zeros((20, 10, 3), dtype=np.float64)
    for probe_id in range(10):
        scale = float(probe_id + 1)
        response[:, probe_id] = scale * np.asarray([1.0, -0.4, -0.6])
    selection = select_high_risk_probes(response)
    assert selection.active.sum() == 5
    assert selection.selected_probe_ids.tolist() == [9]
    assert np.isclose(selection.weights.sum(), 1.0)
    assert np.isclose(np.linalg.norm(selection.directions[0]), 1.0)


def test_direction_sham_and_random_probe_controls_are_matched() -> None:
    response = np.zeros((20, 20, 5), dtype=np.float64)
    rng = np.random.default_rng(7)
    for probe_id in range(20):
        direction = rng.normal(size=5)
        direction -= direction.mean()
        response[:, probe_id] = float(probe_id + 1) * direction
    selection = select_high_risk_probes(response)
    sham, permutations = make_direction_sham(selection.directions, seed=20260907)
    cosine = np.sum(selection.directions * sham, axis=1) / (
        np.linalg.norm(selection.directions, axis=1) * np.linalg.norm(sham, axis=1)
    )
    assert np.all(np.abs(cosine) <= 0.20 + 1.0e-12)
    assert len(permutations) == selection.selected_probe_ids.size
    random_ids = match_random_probes(selection)
    assert random_ids.size == selection.selected_probe_ids.size
    assert len(np.unique(random_ids)) == random_ids.size
    assert np.all(selection.active[random_ids])
    assert not np.any(np.isin(random_ids, selection.selected_probe_ids))


def test_exact_full_carrier_targeted_surgery_reduces_detected_moment() -> None:
    torch.manual_seed(11)
    base = torch.randn(80, 6)
    offsets = torch.tensor(
        [[1.2, -0.3, 0.2, 0.1, -0.4, 0.5], [0.2, 0.8, -0.5, 0.3, 0.1, -0.4]]
    )
    probes = base[:, None, :] + offsets[None, :, :]
    head = torch.nn.Linear(6, 4)
    with torch.no_grad():
        before_response = centered_response_from_features(head, base, probes)
    directions = before_response.mean(dim=0).numpy()
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    weights = np.asarray([0.6, 0.4], dtype=np.float64)
    before_alignment = np.sum(before_response.mean(dim=0).numpy() * directions, axis=1)

    repaired, trace = run_head_surgery(
        head,
        base,
        probes,
        arm="targeted",
        directions=directions,
        weights=weights,
        learning_rate=1.0e-3,
        steps=20,
        anchor_limit=0.02,
    )
    with torch.no_grad():
        after_response = centered_response_from_features(repaired, base, probes)
    after_alignment = np.sum(after_response.mean(dim=0).numpy() * directions, axis=1)
    assert np.sum(weights * np.square(after_alignment)) < np.sum(
        weights * np.square(before_alignment)
    )
    assert max(trace.anchor_kl) <= 0.02 + 1.0e-8
    assert all(np.isfinite(trace.objective))


def test_public_three_way_split_preserves_k0b_discover_and_is_disjoint() -> None:
    split = public_split(50000, discover_count=1000, surgery_count=2000, holdout_count=2000)
    assert sha256_array(split["discover"]) == (
        "731B8CFFDCBD241474D33B261E323F9EC11C2EA59BC7705261140A3B8572F6CA"
    )
    assert np.unique(split["discover"]).size == 1000
    assert np.unique(split["surgery"]).size == 2000
    assert np.unique(split["holdout"]).size == 2000
    assert np.intersect1d(split["discover"], split["surgery"]).size == 0
    assert np.intersect1d(split["discover"], split["holdout"]).size == 0
    assert np.intersect1d(split["surgery"], split["holdout"]).size == 0
    assert sha256_array(split["surgery"]) == (
        "B5441E50539085299F81CD1291636C84A18BA2894BA57D8CB2631D6DF905334A"
    )
    assert sha256_array(split["holdout"]) == (
        "321C0910E8AA376B10D04D1319F24917EE91EABD25BCC8C31A0BDE66F8E240EE"
    )


def test_formal_calibration_contract_contains_only_scalar_learning_rates() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/cle_k1_sdmn_headonly_calibration_seed0.json"
    contract = load_frozen_contract(path)
    for system in ("h9", "l9"):
        for fold in ("ab", "ba"):
            values = contract["learning_rates"][system][fold]
            assert len(values) == 4
            assert all(isinstance(value, float) for value in values)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["optimizer_contract"]["formal_steps"] == 10
    assert raw["schema_recovery"]["rerun_required"] is False


def test_formal_primary_gate_requires_directional_specificity() -> None:
    rows = []
    values = {
        "frozen": 10.0,
        "targeted": 6.0,
        "direction_sham": 8.0,
        "random_probe": 9.0,
        "generic_invariance": 7.0,
    }
    for system in ("h9", "l9"):
        for fold in ("ab", "ba"):
            for client in range(4):
                rows.append(
                    {
                        "checkpoint_arm": system,
                        "fold": fold,
                        "client": client,
                        "metrics": {arm: {"R": value} for arm, value in values.items()},
                    }
                )
    summary = aggregate_primary(rows)
    assert summary["h9"]["pass"] is True
    assert summary["l9"]["pass"] is True
    assert np.isclose(summary["h9"]["combined_relative_R_reduction"]["targeted"], 0.4)


def test_formal_module_has_no_eager_cle_taxonomy_import() -> None:
    import scripts.run_cle_k1_sdmn_formal as formal

    source = inspect.getsource(formal)
    assert "from fedprime.engine.cle_shortcut_alignment import" not in source


def test_primary_surgery_analyzer_does_not_import_cle_taxonomy() -> None:
    import scripts.run_cle_k1_sdmn_headonly as analyzer

    source = inspect.getsource(analyzer)
    assert "historical_family_binding" not in source
    assert "OPERATOR_FAMILY_IDS" not in source
    assert "apply_corruption" not in source
    assert "_cifar100_train_from_tar" not in source
    assert "fine_labels" not in source
