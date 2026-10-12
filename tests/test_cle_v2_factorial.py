import numpy as np

from fedprime.engine.cle_v2_factorial import (
    compute_operator_dsa,
    factorial_estimands,
    paired_bootstrap_estimand,
    shuffled_binding_null,
)
from scripts.prepare_cle_v2_factorial_data import generate_client_condition, make_test_arrays
from scripts.run_cle_v2_factorial import ARMS, arm_config


def synthetic_dsa(strength: float):
    clients, sources, operators, classes = 2, 60, 4, 3
    labels = np.arange(sources) % classes
    binding = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    probabilities = np.full((clients, sources, operators, classes), 0.1, dtype=np.float64)
    for client in range(clients):
        for source in range(sources):
            for operator in range(operators):
                raw = np.full(classes, 0.1)
                raw[labels[source]] += 0.5
                for bound_class in np.flatnonzero(binding[client] == operator):
                    raw[bound_class] += strength
                probabilities[client, source, operator] = raw / raw.sum()
    return probabilities, labels, binding


def test_operator_dsa_detects_directional_binding():
    null, labels, binding = synthetic_dsa(0.0)
    aligned, _, _ = synthetic_dsa(0.4)
    null_result = compute_operator_dsa(null, labels, binding)
    aligned_result = compute_operator_dsa(aligned, labels, binding)
    assert aligned_result.pooled > null_result.pooled + 0.1
    bootstrap = paired_bootstrap_estimand(
        {"aligned": aligned_result, "null": null_result},
        {"aligned": 1.0, "null": -1.0},
        samples=50,
        seed=3,
    )
    assert np.quantile(bootstrap, 0.025) > 0.0


def test_shuffled_binding_breaks_alignment():
    aligned, labels, binding = synthetic_dsa(0.5)
    shuffled = shuffled_binding_null(aligned, labels, binding, permutations=100, seed=8)
    assert shuffled["observed"] > shuffled["null_p95"]


def test_factorial_estimands_have_expected_signs():
    probabilities, labels, binding = synthetic_dsa(0.0)
    base = compute_operator_dsa(probabilities, labels, binding)
    results = {name: base for name in ARMS}
    aligned, _, _ = synthetic_dsa(0.5)
    results["h9_b"] = compute_operator_dsa(aligned, labels, binding)
    results["l9_b"] = compute_operator_dsa(aligned, labels, binding)
    weaker, _, _ = synthetic_dsa(0.2)
    results["h9_p"] = compute_operator_dsa(weaker, labels, binding)
    results["l9_p"] = compute_operator_dsa(weaker, labels, binding)
    summary = factorial_estimands(results)
    assert summary["estimands"]["hfl_cle_effect_base"] > 0
    assert summary["estimands"]["local_plugin_mitigation"] > 0
    assert abs(summary["estimands"]["communication_amplification_base"]) < 1e-10


def test_fixed_test_grid_is_operator_complete_and_severity_matched():
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    labels = np.asarray([0, 1, 2, 3])
    source_indices = np.arange(4)
    operators = ["brightness", "contrast"]
    arrays = make_test_arrays(
        images,
        labels,
        source_indices,
        operators,
        {name: index for index, name in enumerate(operators)},
        fixed_severity=3,
    )
    assert arrays["test_images"].shape == (8, 32, 32, 3)
    assert np.all(arrays["test_severity_ids"] == 3)
    assert np.array_equal(np.bincount(arrays["test_source_ids"]), np.full(4, 2))


def test_mixed_corruption_mask_is_paired_across_gamma():
    images = np.full((40, 32, 32, 3), 127, dtype=np.uint8)
    labels = np.arange(40, dtype=np.int64) % 10
    sources = np.arange(40, dtype=np.int64)
    seen = ["gaussian_noise", "shot_noise"]
    binding = {0: {label: seen[label % 2] for label in range(10)}}
    operator_to_id = {name: index for index, name in enumerate(seen)}
    null = generate_client_condition(
        images,
        labels,
        sources,
        client_id=0,
        gamma=0.0,
        binding=binding,
        seen=seen,
        operator_to_id=operator_to_id,
        corruption_rate=0.5,
    )
    strong = generate_client_condition(
        images,
        labels,
        sources,
        client_id=0,
        gamma=0.9,
        binding=binding,
        seen=seen,
        operator_to_id=operator_to_id,
        corruption_rate=0.5,
    )
    assert np.array_equal(null["train_is_corrupted"], strong["train_is_corrupted"])
    assert np.array_equal(null["train_severity_ids"], strong["train_severity_ids"])
    assert np.array_equal(null["train_source_indices"], strong["train_source_indices"])
    clean = ~null["train_is_corrupted"]
    assert np.all(null["train_corruption_ids"][clean] == 255)
    assert np.all(null["train_severity_ids"][clean] == 0)
    assert np.array_equal(null["train_images"][clean], images[sources][clean])


def test_arm_configs_share_data_and_only_plugin_uses_annotations(tmp_path):
    configs = {
        arm: arm_config(
            arm,
            package_root=tmp_path,
            train_seed=0,
            rounds=12,
            device="cpu",
            output_root=tmp_path / "outputs",
            smoke=False,
        )
        for arm in ARMS
    }
    assert configs["h9_b"]["data"]["private_root"] == configs["h9_p"]["data"]["private_root"]
    assert configs["h9_b"]["method"]["communication"] == configs["h9_p"]["method"]["communication"]
    assert "fedease" not in configs["h9_b"]["method"]
    assert configs["h9_p"]["method"]["fedease"]["ber"]["enabled"] is True
    assert configs["l9_p"]["method"]["communication"] == "none"
    assert all(config["method"]["paired_local_rng"]["enabled"] for config in configs.values())
    assert all(config["num_workers"] == 0 for config in configs.values())

    benchmark = arm_config(
        "h9_p",
        package_root=tmp_path,
        train_seed=0,
        rounds=1,
        device="cpu",
        output_root=tmp_path / "outputs",
        smoke=False,
        benchmark=True,
    )
    assert benchmark["train"]["batch_size"] == 64
    assert benchmark["train"]["max_local_batches"] == 8
    assert benchmark["train"]["max_test_batches"] == 1
    assert benchmark["num_workers"] == 0
