from scripts.openi_cle_communication_factorial_entry import ARM_ORDER, build_arm_configs


def test_factorial_contains_all_two_by_three_arms() -> None:
    configs = build_arm_configs()
    assert tuple(configs) == ARM_ORDER
    assert len(configs) == 6
    for name, config in configs.items():
        local, communication = name.split("_", maxsplit=1)
        assert config["method"]["communication"] == communication
        assert config["method"]["strict_fit_audit"]["enabled"] is True
        assert config["train"]["rounds"] == 12
        assert config["method"]["cl_module"] == ("dcl" if local == "l0" else "fedease")
