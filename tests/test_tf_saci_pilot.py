import io
import tarfile

import numpy as np
import pytest
import torch

from experiments.tf_saci.gate0 import sha256, write_json
from experiments.tf_saci.pilot_data import load_aligned_cache
from experiments.tf_saci.pilot_evaluate import evaluate, group_metrics
from experiments.tf_saci.ssa import ssa_loss
from scripts.openi_tf_saci_pilot_entry import extract_input


@pytest.mark.parametrize("eta", [1.0, 0.5])
def test_zero_strength_is_erm_gradient(eta):
    logits = torch.randn(5, 3, requires_grad=True)
    bias = torch.randn_like(logits, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1])
    vanilla = torch.nn.functional.cross_entropy(logits, labels)
    out = ssa_loss(logits, labels, bias, strength=0.0, poe_weight=eta)
    torch.testing.assert_close(out, vanilla, atol=0, rtol=0)
    (g_ref,) = torch.autograd.grad(vanilla, logits)
    g_ssa, g_bias = torch.autograd.grad(out, (logits, bias), allow_unused=True)
    torch.testing.assert_close(g_ssa, g_ref, atol=0, rtol=0)
    assert g_bias is None


def test_cache_alignment_uses_frozen_fit_and_source_identity(tmp_path):
    path = tmp_path / "cache.npz"
    ids = np.array([0, 2])
    source_ids = np.array([5, 7, 6])
    labels = np.array([2, 7, 1])
    np.savez(path, sample_ids=ids, source_ids=source_ids[ids], labels=labels[ids], bias=np.zeros((2, 10)))
    fit, bias = load_aligned_cache(path, ids[::-1], source_ids, labels)
    np.testing.assert_array_equal(fit, ids)
    assert bias.dtype == np.float32
    with pytest.raises(ValueError, match="alignment"):
        load_aligned_cache(path, ids, source_ids[::-1], labels)
    with pytest.raises(ValueError, match="alignment"):
        load_aligned_cache(path, np.array([0, 1]), source_ids, labels)


def test_operator_metrics_match_class_operator_pooling():
    labels = np.array([0, 0, 1, 1])
    operators = np.array([0, 1, 0, 1])
    correct = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=bool)
    result = group_metrics(correct, labels, operators)
    assert result["wcca"] == 50
    assert result["cfg"] == 50
    assert result["class_operator"][0][0]["n"] == 2


def test_openi_archive_rejects_development_or_sealed_paths(tmp_path):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("evaluation/paired_dsa/test_labels.npy")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unexpected paths"):
        extract_input(path, tmp_path / "stage", sha256(path))
    assert not (tmp_path / "stage").exists()


def test_joint_evaluation_rejects_smoke_before_opening_sealed_input(tmp_path):
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    write_json(smoke / "TRAINING_COMPLETE.json", {"arm": "ssa_poe", "mode": "smoke", "completed_rounds": 2})
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="frozen Pilot"):
        evaluate(tmp_path / "absent_sealed_input", (smoke, tmp_path / "absent_anchor"),
                 (), output, torch.device("cpu"))
    assert not output.exists()
