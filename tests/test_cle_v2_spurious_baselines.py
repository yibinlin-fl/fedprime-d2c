from pathlib import Path

import numpy as np
import pytest
import torch

from fedprime.methods.spurious_baselines import (
    GroupDROState,
    cvar_dro_loss,
    group_dro_loss,
    jtt_reweighted_loss,
)
from scripts.run_cle_v2_spurious_baselines import ARMS, spurious_arm_config


def test_jtt_matches_explicit_example_repetition() -> None:
    losses = torch.tensor([1.0, 2.0, 5.0])
    errors = torch.tensor([False, True, False])
    actual = jtt_reweighted_loss(losses, errors, upweight=3.0)
    expected = torch.tensor([1.0, 2.0, 2.0, 2.0, 5.0]).mean()
    torch.testing.assert_close(actual, expected)


def test_cvar_uses_upper_loss_tail() -> None:
    losses = torch.tensor([1.0, 4.0, 2.0, 8.0, 3.0])
    torch.testing.assert_close(cvar_dro_loss(losses, tail_fraction=0.4), torch.tensor(6.0))


def test_groupdro_updates_only_observed_groups_and_backpropagates() -> None:
    losses = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    environments = torch.tensor([0, 1, 0])
    state = GroupDROState.uniform(2, 2, torch.device("cpu"))
    before = state.weights.clone()
    objective, stats = group_dro_loss(
        losses, labels, environments, state, num_environments=2, step_size=0.1
    )
    objective.backward()
    assert torch.isfinite(losses.grad).all()
    assert state.weights[3] < before[3]
    assert stats["observed_groups"].item() == 3.0
    assert state.weights.sum().item() == pytest.approx(1.0)


def _fake_package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    (package / "data/gamma09").mkdir(parents=True)
    (package / "public").mkdir()
    (package / "initial_states").mkdir()
    (package / "splits").mkdir()
    (package / "pew/frozen_annotations").mkdir(parents=True)
    (package / "manifest.json").write_text(
        '{"protocol":"unit","scenario_seed":0,"split_seed":0}', encoding="utf-8"
    )
    return package


def test_spurious_configs_keep_objectives_separate(tmp_path: Path) -> None:
    package = _fake_package(tmp_path)
    output = tmp_path / "outputs"
    annotations = tmp_path / "jtt"
    configs = {
        arm: spurious_arm_config(
            arm,
            package_root=package,
            mode="smoke",
            device="cpu",
            output_root=output,
            jtt_annotation_root=annotations,
        )
        for arm in ARMS
    }
    assert configs["erm"]["method"]["cl_module"] == "none"
    assert configs["jtt"]["method"]["cl_module"] == "jtt"
    assert "fedease" not in configs["jtt"]["method"]
    assert configs["cvar_dro"]["method"]["cl_module"] == "cvar_dro"
    assert "fedease" not in configs["cvar_dro"]["method"]
    assert configs["pew_groupdro"]["method"]["fedease"]["objective"] == "pew_groupdro"
    assert "ber" not in configs["pew_groupdro"]["method"]["fedease"]
    assert configs["pew_ber"]["method"]["fedease"]["objective"] == "ce_ber"
    assert all(config["method"]["communication"] == "asymhfl_val" for config in configs.values())
    assert all(config["method"]["local_loader_mode"] == "standard" for config in configs.values())
    assert all(config["method"]["lambda_jsd"] == 0.0 for config in configs.values())
    assert not any("cdep" in str(config).lower() for config in configs.values())


def test_invalid_objective_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        cvar_dro_loss(torch.ones(4), tail_fraction=0.0)
    with pytest.raises(ValueError):
        jtt_reweighted_loss(torch.ones(2), torch.zeros(2), upweight=0.5)
