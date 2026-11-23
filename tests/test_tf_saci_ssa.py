import numpy as np
import pytest
import torch
from torch.nn.functional import cross_entropy

from experiments.tf_saci.gate0 import load_fit, score, sha256
from experiments.tf_saci.ssa import descriptors, fit_oof, ssa_loss


def make_probe_data():
    rng = np.random.default_rng(42)
    sources = np.repeat(np.arange(101, 125), 2)
    labels = np.repeat(np.tile([0, 2], 12), 2)
    samples = 5 * np.arange(labels.size) + 9
    features = rng.normal(size=(len(labels), 4))
    features[:, 0] += labels
    return features, labels, samples, sources


def run_probe(data):
    return fit_oof(*data, classes=4, folds=5, fold_seed=20261123, l2=0.01, clip=0.5)


def test_descriptor_subsets_bounds_and_stability():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
    images[0] = 160
    image_features = descriptors(images, chunk_size=2)
    assert image_features["phi_tex"].shape == (4, 20)
    assert image_features["phi_full"].shape == (4, 25)
    np.testing.assert_array_equal(image_features["phi_tex"], image_features["phi_full"][:, :20])
    np.testing.assert_allclose(image_features["phi_tex"][0], 0, atol=1e-10)
    np.testing.assert_allclose(image_features["phi_full"][0, [20, 22, 23]], 160 / 255)
    np.testing.assert_allclose(descriptors(images[::-1])["phi_full"][::-1], image_features["phi_full"])


def test_source_folds_active_classes_and_stable_cache_lookup():
    data = make_probe_data()
    result = run_probe(data)
    for source in np.unique(data[3]):
        assert np.unique(result.fold[data[3] == source]).size == 1
    assert set(result.fold) == set(range(5))
    assert np.all(result.active[:, [0, 2]])
    assert not result.active[:, [1, 3]].any()
    np.testing.assert_array_equal(result.bias[:, [1, 3]], 0)
    active = result.raw[:, [0, 2]]
    np.testing.assert_allclose(result.bias[:, [0, 2]], np.clip(active - active.mean(axis=1)[:, None], -0.5, 0.5))
    shuffled = np.random.default_rng(1).permutation(len(result.fold))
    refit = run_probe(tuple(values[shuffled] for values in data))
    np.testing.assert_array_equal(result.fold, refit.fold)
    np.testing.assert_array_equal(refit.bias_for(data[2][shuffled]), result.bias[shuffled])
    with pytest.raises(KeyError):
        result.bias_for([99999])


def test_held_out_features_do_not_change_fitted_scores_for_same_fold():
    features, labels, samples, sources = make_probe_data()
    before = run_probe((features, labels, samples, sources))
    in_fold = np.flatnonzero(before.fold == 0)
    features[in_fold[1:]] += 10000
    after = run_probe((features, labels, samples, sources))
    np.testing.assert_array_equal(before.raw[in_fold[0]], after.raw[in_fold[0]])


def test_failed_group_identity_and_low_class_support():
    features, labels, samples, sources = make_probe_data()
    labels[0] = 2
    with pytest.raises(ValueError, match="inconsistent task labels"):
        run_probe((features, labels, samples, sources))
    labels[:2] = 3
    with pytest.raises(ValueError, match="source group"):
        run_probe((features, labels, samples, sources))


def test_single_active_class_has_zero_bias():
    features, labels, samples, sources = make_probe_data()
    labels[:] = 2
    result = run_probe((features, labels, samples, sources))
    np.testing.assert_array_equal(result.prediction, labels)
    np.testing.assert_array_equal(result.bias, 0)
    assert np.all(result.active[:, 2])


def test_rare_class_remains_available_in_every_training_fold():
    features, labels, samples, sources = make_probe_data()
    labels[:6] = 3
    result = run_probe((features, labels, samples, sources))
    assert np.unique(result.fold[:6]).size == 3
    assert np.all(result.active[:, 3])


def test_fit_reader_requires_disjoint_groups_and_checks_integrity(tmp_path):
    images = np.arange(4 * 16 * 16 * 3, dtype=np.uint8).reshape(4, 16, 16, 3)
    labels = np.array([3, 4, 7, 8])
    metadata = {}
    for name, data in {"images": images, "labels": labels, "source_indices": [51, 52, 53, 54]}.items():
        file = tmp_path / f"train_{name}.npy"
        np.save(file, data)
        metadata[file.name] = {"sha256": sha256(file)}
    fit, audit = np.array([0, 2]), np.array([1, 3])
    x, y, ids = load_fit(tmp_path, fit, audit, metadata)
    np.testing.assert_array_equal(x, images[fit])
    np.testing.assert_array_equal(y, [3, 7])
    np.testing.assert_array_equal(ids, [51, 53])
    with pytest.raises(ValueError, match="disjoint fit/audit"):
        load_fit(tmp_path, fit, np.array([0, 1]), metadata)
    source_file = tmp_path / "train_source_indices.npy"
    np.save(source_file, [51, 51, 53, 54])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_fit(tmp_path, fit, audit, metadata)
    metadata[source_file.name]["sha256"] = sha256(source_file)
    with pytest.raises(ValueError, match="cross fit/audit"):
        load_fit(tmp_path, fit, audit, metadata)


def test_slice_balances_supported_classes_and_preserves_counts():
    labels = np.array([0, 0, 0, 1, 2])
    selection = np.array([True, True, True, True, False])
    result = score(labels, np.array([0, 0, 0, 0, 2]), np.zeros(5), selection)
    assert result["balanced_accuracy"] == 0.5
    assert result["per_class"]["0"]["n"] == 3
    assert result["class_count"] == 2
    assert score(labels, labels, labels, ~np.ones(5, bool))["balanced_accuracy"] is None


@pytest.mark.parametrize("poe_weight", [0.0, 0.5, 1.0])
def test_loss_anchor_and_bias_have_no_effect_on_inference_or_bias_gradient(poe_weight):
    model = torch.nn.Linear(3, 2)
    x = torch.tensor([[1.0, 0.3, -1.0], [0.1, 0.2, -0.4]])
    y = torch.tensor([0, 1])
    logits = model(x)
    baseline = logits.detach().clone()
    bias = torch.tensor([[1.0, -1.0], [0.3, -0.3]], requires_grad=True)
    pure = cross_entropy(logits, y)
    assert torch.equal(ssa_loss(logits, y, bias, strength=0), pure)
    loss = ssa_loss(logits, y, bias, strength=0.5, poe_weight=poe_weight)
    expected = (1 - poe_weight) * pure + poe_weight * cross_entropy(logits + 0.5 * bias.detach(), y)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert bias.grad is None
    assert model.weight.grad is not None
    torch.testing.assert_close(model(x), baseline)
