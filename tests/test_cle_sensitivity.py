from scripts.openi_cle_sensitivity_entry import ARMS, build_configs, select_arms


def test_sensitivity_is_one_factor_at_a_time():
    configs = build_configs()
    base = configs["base"]["method"]["fedease"]
    assert list(configs) == list(ARMS)
    assert configs["pew_t045"]["method"]["fedease"]["pew"]["unknown_threshold"] == 0.45
    assert configs["ber_g000"]["method"]["fedease"]["ber"]["support_gamma"] == 0.0
    assert configs["cdep_l010"]["method"]["fedease"]["cdep"]["lambda"] == 0.10
    assert base["pew"]["unknown_threshold"] == "auto"
    assert all(config["train"]["rounds"] == 12 for config in configs.values())


def test_sensitivity_arm_selection():
    assert select_arms("base,cdep_l001") == ["base", "cdep_l001"]
