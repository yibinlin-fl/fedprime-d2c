from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fedprime.data.corruptions import CORRUPTION_GROUPS, DEFAULT_UNSEEN_CORRUPTIONS
from fedprime.methods import environment_witness
from fedprime.methods.environment_witness import (
    PublicEnvironmentDataset,
    PublicEnvironmentWitness,
    load_environment_witness,
    resolve_public_corruption_groups,
    save_environment_witness,
)
from scripts.openi_cle_pew_loo_entry import (
    ARM_ORDER,
    STRICT_LOO_OPERATORS,
    audit_private_fit_holdout,
    build_configs,
    compare_last_five,
)


def test_default_public_corruption_pools_are_unchanged() -> None:
    assert resolve_public_corruption_groups() == CORRUPTION_GROUPS


def test_strict_loo_removes_one_operator_from_each_family() -> None:
    pools = resolve_public_corruption_groups(DEFAULT_UNSEEN_CORRUPTIONS)
    assert set(DEFAULT_UNSEEN_CORRUPTIONS).isdisjoint(
        operator for operators in pools.values() for operator in operators
    )
    assert {group: len(operators) for group, operators in pools.items()} == {
        "noise": 3,
        "blur": 3,
        "weather": 3,
        "digital": 3,
    }


def test_public_environment_dataset_never_samples_excluded_operator(monkeypatch) -> None:
    sampled: list[str] = []

    def record_corruption(image, name, severity, rng):
        sampled.append(str(name))
        return np.asarray(image, dtype=np.uint8)

    monkeypatch.setattr(environment_witness, "apply_corruption", record_corruption)
    images = np.zeros((72, 32, 32, 3), dtype=np.uint8)
    dataset = PublicEnvironmentDataset(
        images,
        np.arange(len(images)),
        seed=7,
        excluded_operators=STRICT_LOO_OPERATORS,
    )
    for index in range(len(dataset)):
        dataset[index]

    assert sampled
    assert set(sampled).isdisjoint(STRICT_LOO_OPERATORS)


def test_invalid_public_operator_exclusions_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unknown PEW excluded operators"):
        resolve_public_corruption_groups(["not_a_corruption"])
    with pytest.raises(ValueError, match="emptied corruption groups"):
        resolve_public_corruption_groups(CORRUPTION_GROUPS["noise"])


def test_checkpoint_records_operator_exclusion_protocol(tmp_path) -> None:
    path = tmp_path / "strict_loo.pt"
    model = PublicEnvironmentWitness()
    save_environment_witness(
        model,
        path,
        excluded_operators=STRICT_LOO_OPERATORS,
    )
    restored = load_environment_witness(path, torch.device("cpu"))
    assert tuple(restored.training_excluded_operators) == STRICT_LOO_OPERATORS


def test_pew_loo_entry_keeps_standard_and_strict_variants() -> None:
    configs = build_configs()
    assert tuple(configs) == ARM_ORDER
    assert configs["rahfl"]["method_name"] == "rahfl"

    standard = configs["standard_pew_ber"]
    strict = configs["strict_loo_pew_ber"]
    assert standard["method"]["fedease"]["ber"]["enabled"] is True
    assert "cdep" not in standard["method"]["fedease"]
    assert "cdep" not in strict["method"]["fedease"]
    assert standard["method"]["fedease"]["pew"]["exclude_operators"] == []
    assert strict["method"]["fedease"]["pew"]["exclude_operators"] == list(
        DEFAULT_UNSEEN_CORRUPTIONS
    )
    assert (
        standard["method"]["fedease"]["pew"]["checkpoint"]
        != strict["method"]["fedease"]["pew"]["checkpoint"]
    )


def test_private_fit_holdout_audit_rejects_leakage(tmp_path) -> None:
    operator_to_id = {
        "gaussian_noise": 0,
        "impulse_noise": 1,
        "zoom_blur": 2,
        "fog": 3,
        "pixelate": 4,
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps({"operator_to_id": operator_to_id}),
        encoding="utf-8",
    )
    client = tmp_path / "client_0"
    client.mkdir()
    np.save(client / "train_corruption_ids.npy", np.array([0, 0], dtype=np.int64))
    assert audit_private_fit_holdout(tmp_path)["passed"] is True

    np.save(client / "train_corruption_ids.npy", np.array([0, 1], dtype=np.int64))
    with pytest.raises(RuntimeError, match="private fit leakage"):
        audit_private_fit_holdout(tmp_path)


def test_strict_loo_uses_original_candidate_gate() -> None:
    report = {
        "runs": {
            "rahfl": {"last_five": {"avg_acc": 30, "worst_acc": 25, "wcca": 1, "cfg": 30}},
            "standard_pew_ber": {
                "last_five": {"avg_acc": 34, "worst_acc": 29, "wcca": 7, "cfg": 24}
            },
            "strict_loo_pew_ber": {
                "last_five": {"avg_acc": 32, "worst_acc": 27, "wcca": 2, "cfg": 28}
            },
        }
    }
    decision = compare_last_five(report)
    assert decision["strict_loo_minus_rahfl"] == {
        "avg_acc": 2.0,
        "worst_acc": 2.0,
        "wcca": 1.0,
        "cfg": -2.0,
    }
    assert decision["pass"] is True
