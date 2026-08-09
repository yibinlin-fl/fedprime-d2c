from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedprime.data.strict_fit_audit import (
    StrictClientSplit,
    build_client_audit_loaders,
)
from fedprime.methods.fedease import FedEASEExperiment
from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment
from fedprime.utils.config import load_config


class _IdentityClassifier(torch.nn.Module):
    def forward(self, images):
        return images


def test_client_audit_loader_contains_only_audit_indices() -> None:
    images = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    labels = torch.arange(6, dtype=torch.long) % 2
    dataset = TensorDataset(images, labels)
    split = StrictClientSplit(
        client_id=0,
        fit_indices=np.asarray([0, 2, 4]),
        audit_indices=np.asarray([1, 3, 5]),
        labels=labels.numpy(),
        fit_loader=DataLoader(dataset, batch_size=2),
        probe_dataset=dataset,
    )

    loader = build_client_audit_loaders({0: split}, batch_size=8, num_workers=0)[0]
    loaded_images, loaded_labels = next(iter(loader))

    assert torch.equal(loaded_images, images[[1, 3, 5]])
    assert torch.equal(loaded_labels, labels[[1, 3, 5]])


def test_private_audit_evaluation_uses_each_clients_own_loader(tmp_path) -> None:
    config = {
        "experiment_name": "strict-routing-test",
        "output_root": str(tmp_path),
        "device": "cpu",
        "seed": 0,
        "train": {},
        "method": {"strict_fit_audit": {"enabled": True}},
    }
    experiment = AsymHFLExperiment(config)
    models = {0: _IdentityClassifier(), 1: _IdentityClassifier()}
    loaders = {
        0: DataLoader(TensorDataset(torch.tensor([[5.0, 0.0]]), torch.tensor([0]))),
        1: DataLoader(TensorDataset(torch.tensor([[0.0, 5.0]]), torch.tensor([1]))),
    }

    assert experiment._evaluate_private_loaders(models, loaders) == [100.0, 100.0]


def test_fedease_accepts_cle_v2_strict_asymhfl_val(tmp_path) -> None:
    config = load_config("configs/debug_fedease_pew_asymhfl_val_cle_v2.yaml")
    config["output_root"] = str(tmp_path)

    experiment = FedEASEExperiment(config)

    assert config["data"]["scenario"] == "cle_hfl_v2"
    assert config["method"]["communication"] == "asymhfl_val"
    assert config["method"]["strict_fit_audit"]["enabled"] is True
    assert not experiment._use_no_communication(config["method"])


def test_cle_v2_operator_ids_are_mapped_to_pew_families_for_diagnostics(tmp_path) -> None:
    root = tmp_path / "cle_v2"
    root.mkdir()
    (root / "metadata.json").write_text(
        """
        {
          "operator_to_id": {"gaussian_noise": 0, "motion_blur": 5, "snow": 7, "jpeg_compression": 14},
          "operator_families": {"gaussian_noise": "noise", "motion_blur": "blur", "snow": "weather", "jpeg_compression": "digital"}
        }
        """,
        encoding="utf-8",
    )
    config = load_config("configs/debug_fedease_pew_asymhfl_val_cle_v2.yaml")
    config["output_root"] = str(tmp_path / "outputs")
    experiment = FedEASEExperiment(config)

    mapped = experiment._diagnostic_environment_ids(np.asarray([0, 5, 7, 14]), root)

    assert mapped.tolist() == [1, 2, 3, 4]
