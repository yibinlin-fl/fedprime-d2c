from pathlib import Path

from scripts.openi_cle_cross_scenario_40round_entry import SCENARIOS, build_configs, dataset_archive_name


def test_cross_scenario_configs_change_only_scenario_and_freeze_training_seed() -> None:
    for seed, scenario in SCENARIOS.items():
        configs = build_configs(seed)
        assert dataset_archive_name(seed) == f"cle_hfl_v2_prepared_{scenario}.tar.gz"
        for config in configs.values():
            assert config["seed"] == 0
            assert config["data"]["private_root"].endswith(scenario)
            assert config["method"]["strict_fit_audit"]["seed"] == 0
            assert scenario in config["method"]["strict_fit_audit"]["split_path"]
            assert config["train"]["rounds"] == 40
            assert config["checkpoints"]["save_final"] is False
            assert "trainseed0" in config["experiment_name"]
        assert configs["control"]["method"]["communication"] == "asymhfl_val"
        assert configs["candidate"]["method"]["communication"] == "asymhfl_val"
