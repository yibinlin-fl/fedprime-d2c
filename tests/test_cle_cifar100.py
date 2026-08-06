from pathlib import Path

from fedprime.data.loaders import CIFAR100_MEAN, _prepared_private_dataset_name
from scripts.openi_cle_cifar100_entry import build_configs


def test_cifar100_configs_keep_protocol_roles_disjoint():
    configs = build_configs()
    for config in configs.values():
        data = config["data"]
        assert data["private_dataset"] == "cifar100"
        assert data["public_dataset"] == "cifar10"
        assert data["num_classes"] == 100
        assert config["train"]["rounds"] == 12
        assert config["method"]["strict_fit_audit"]["enabled"] is True
    assert "fedease" not in configs["control"]["method"]
    assert configs["candidate"]["method"]["cl_module"] == "fedease"


def test_prepared_dataset_name_reads_metadata(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(
        '{"private_dataset":"cifar100"}', encoding="utf-8"
    )
    assert _prepared_private_dataset_name(tmp_path) == "cifar100"
    assert len(CIFAR100_MEAN) == 3
