from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedprime.data.fedease import FedEASEEvaluationDataset
from fedprime.models.factory import build_models, forward_logits


def _clean_state_dict(state) -> dict[str, torch.Tensor]:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint payload is not a state_dict")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def _torch_load_weights(file_obj, device: torch.device):
    try:
        return torch.load(file_obj, map_location=device, weights_only=True)
    except TypeError:
        file_obj.seek(0)
        return torch.load(file_obj, map_location=device)


def load_models_from_archive(
    *,
    checkpoint_archive: str | Path,
    experiment_name: str,
    model_names: list[str],
    num_classes: int,
    device: torch.device,
) -> dict[int, torch.nn.Module]:
    """Load model state_dict objects directly from an experiment tar archive."""

    models = build_models(model_names, int(num_classes))
    archive_path = Path(checkpoint_archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint archive: {archive_path}")

    with tarfile.open(archive_path, mode="r:*") as archive:
        member_names = set(archive.getnames())
        for client_id, model in models.items():
            expected_suffix = (
                f"{experiment_name}/checkpoints/client_{int(client_id)}.pt"
            )
            matches = [
                name
                for name in member_names
                if name.replace("\\", "/").endswith(expected_suffix)
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one archive member ending in {expected_suffix}, "
                    f"found {len(matches)}"
                )
            extracted = archive.extractfile(matches[0])
            if extracted is None:
                raise OSError(f"Could not read archive member: {matches[0]}")
            payload = _torch_load_weights(io.BytesIO(extracted.read()), device)
            model.load_state_dict(_clean_state_dict(payload), strict=True)
            model.to(device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
    return models


def evaluate_all_models_on_receiver(
    *,
    models: dict[int, torch.nn.Module],
    dataset_directory: str | Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions [source, sample] and labels for one receiver split."""

    dataset = FedEASEEvaluationDataset(dataset_directory)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    predictions = {client_id: [] for client_id in models}
    labels = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            images, batch_labels, _ = batch
            images = images.to(device, non_blocking=True)
            labels.append(batch_labels.cpu().numpy().astype(np.int64, copy=False))
            for client_id, model in models.items():
                pred = forward_logits(model, images).argmax(dim=1)
                predictions[client_id].append(
                    pred.cpu().numpy().astype(np.int16, copy=False)
                )

    if not labels:
        raise RuntimeError(f"No audit batches were read from {dataset_directory}")
    ordered_ids = sorted(models)
    prediction_array = np.stack(
        [np.concatenate(predictions[client_id]) for client_id in ordered_ids],
        axis=0,
    )
    return prediction_array, np.concatenate(labels)
