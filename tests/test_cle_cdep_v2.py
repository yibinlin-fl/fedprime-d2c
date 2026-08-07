from scripts.openi_cle_cdep_v2_entry import build_config, compare_with_reference


def test_cdep_v2_openi_config_is_single_frozen_arm():
    config = build_config()
    cdep = config["method"]["fedease"]["cdep"]
    assert config["experiment_name"] == "cle_cdep_v2_seed0_12round"
    assert config["seed"] == 0
    assert config["train"]["rounds"] == 12
    assert cdep == {
        "enabled": True,
        "version": "v2",
        "projection_dim": 64,
        "lambda": 1.0,
        "buffer_size_per_group": 64,
        "min_confidence": 0.20,
        "min_group_count": 4,
        "min_environments": 2,
        "warmup_rounds": 2,
        "ramp_rounds": 3,
        "eps": 1.0e-6,
    }


def test_cdep_v2_decision_requires_all_pre_registered_gates():
    passing = compare_with_reference(
        {"avg_acc": 34.7, "worst_acc": 29.5, "wcca": 7.3, "cfg": 24.0}
    )
    assert passing["pass"] is True

    weak_cfg = compare_with_reference(
        {"avg_acc": 34.7, "worst_acc": 29.5, "wcca": 7.3, "cfg": 24.3}
    )
    assert weak_cfg["pre_registered_gates"]["cfg_improvement"] is False
    assert weak_cfg["pass"] is False
