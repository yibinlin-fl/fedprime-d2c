from scripts.openi_cle_stress_grid_entry import GRID, build_configs, dataset_archive_name


def test_stress_grid_is_frozen_two_by_three() -> None:
    assert {(alpha, gamma) for alpha, gamma, _ in GRID.values()} == {
        (0.1, 0.5), (0.1, 0.9), (0.5, 0.5),
        (0.5, 0.9), (1.0, 0.5), (1.0, 0.9),
    }
    for cell, (_, _, scenario) in GRID.items():
        configs = build_configs(cell)
        assert dataset_archive_name(cell) == f"cle_hfl_v2_prepared_{scenario}.tar.gz"
        for config in configs.values():
            assert config["seed"] == 0
            assert config["data"]["private_root"].endswith(scenario)
            assert config["train"]["rounds"] == 12
            assert config["method"]["strict_fit_audit"]["enabled"] is True
