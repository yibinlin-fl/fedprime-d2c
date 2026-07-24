from __future__ import annotations

import numpy as np

from fedprime.data.corruptions import (
    CIFAR_C_CORE_CORRUPTIONS,
    DEFAULT_UNSEEN_CORRUPTIONS,
    apply_corruption,
)
from fedprime.engine.operator_metrics import summarize_operator_splits
from scripts.prepare_cle_v2_data import (
    build_class_operator_map,
    parse_operator_split,
    sample_operator_for_label,
)


def test_operator_split_is_disjoint_and_complete():
    seen, unseen = parse_operator_split(",".join(DEFAULT_UNSEEN_CORRUPTIONS))
    assert set(seen).isdisjoint(unseen)
    assert set(seen).union(unseen) == set(CIFAR_C_CORE_CORRUPTIONS)


def test_client_class_mapping_is_deterministic_and_seen_only():
    seen, _ = parse_operator_split(",".join(DEFAULT_UNSEEN_CORRUPTIONS))
    first = build_class_operator_map(4, 10, seen, seed=7)
    second = build_class_operator_map(4, 10, seen, seed=7)
    assert first == second
    assert all(operator in seen for mapping in first.values() for operator in mapping.values())
    assert all(len(set(mapping.values())) == 10 for mapping in first.values())


def test_gamma_one_always_selects_dominant_operator():
    seen, _ = parse_operator_split(",".join(DEFAULT_UNSEEN_CORRUPTIONS))
    mapping = build_class_operator_map(1, 10, seen, seed=3)
    rng = np.random.default_rng(19)
    selected = {
        sample_operator_for_label(
            label=2,
            client_id=0,
            class_operator_map=mapping,
            seen_operators=seen,
            gamma=1.0,
            rng=rng,
        )
        for _ in range(50)
    }
    assert selected == {mapping[0][2]}


def test_every_core_corruption_returns_uint8_image():
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    for offset, operator in enumerate(CIFAR_C_CORE_CORRUPTIONS):
        result = apply_corruption(
            image,
            operator,
            severity=3,
            rng=np.random.default_rng(100 + offset),
        )
        assert result.shape == image.shape
        assert result.dtype == np.uint8


def test_seen_unseen_summary_uses_operator_metadata():
    summary = {
        "groups": {"op_a": 80.0, "op_b": 60.0, "op_c": 40.0},
        "clients": {
            0: {"op_a": 90.0, "op_b": 70.0, "op_c": 50.0},
            1: {"op_a": 70.0, "op_b": 50.0, "op_c": 30.0},
        },
        "class_corruption_rows": [
            {"client": 0, "class_id": 0, "group": "op_a", "acc": 100.0, "total": 10},
            {"client": 1, "class_id": 0, "group": "op_a", "acc": 80.0, "total": 10},
            {"client": 0, "class_id": 0, "group": "op_b", "acc": 60.0, "total": 10},
            {"client": 1, "class_id": 0, "group": "op_b", "acc": 40.0, "total": 10},
            {"client": 0, "class_id": 0, "group": "op_c", "acc": 50.0, "total": 10},
            {"client": 1, "class_id": 0, "group": "op_c", "acc": 30.0, "total": 10},
        ],
    }
    metadata = {
        "operator_splits": {"op_a": "seen", "op_b": "seen", "op_c": "unseen"}
    }
    result = summarize_operator_splits(summary, metadata)
    assert result["seen"]["avg_acc"] == 70.0
    assert result["seen"]["worst_acc"] == 60.0
    assert result["seen"]["wcca"] == 50.0
    assert result["seen"]["cfg"] == 40.0
    assert result["unseen"]["avg_acc"] == 40.0
    assert result["unseen"]["wcca"] == 40.0
