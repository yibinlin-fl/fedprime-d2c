from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class FrozenRandomProjector:
    """A deterministic, parameter-free projection that cannot learn to collapse."""

    def __init__(self, output_dim: int = 64, seed: int = 0) -> None:
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self._input_dim: int | None = None
        self._matrix_cpu: torch.Tensor | None = None

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, dimension]")
        input_dim = int(features.shape[1])
        if self._matrix_cpu is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            matrix = torch.randn(input_dim, self.output_dim, generator=generator)
            matrix = F.normalize(matrix, dim=0) / math.sqrt(max(self.output_dim, 1))
            self._matrix_cpu = matrix
            self._input_dim = input_dim
        elif input_dim != self._input_dim:
            raise ValueError(
                f"projector input dimension changed from {self._input_dim} to {input_dim}"
            )
        matrix = self._matrix_cpu.to(device=features.device, dtype=features.dtype)
        return features @ matrix


def normalized_conditional_cross_covariance(
    features: torch.Tensor,
    labels: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    num_environments: int,
    environment_features: torch.Tensor | None = None,
    min_class_count: int = 3,
    eps: float = 1.0e-5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Measure normalized feature/environment dependence within each class."""

    if features.ndim != 2:
        raise ValueError("features must have shape [batch, dimension]")
    if labels.ndim != 1 or environment_ids.ndim != 1:
        raise ValueError("labels and environment_ids must have shape [batch]")
    if features.shape[0] != labels.shape[0] or labels.shape != environment_ids.shape:
        raise ValueError("features, labels, and environment_ids must share the batch dimension")
    if num_environments < 2:
        raise ValueError("num_environments must be at least 2")
    if min_class_count < 2:
        raise ValueError("min_class_count must be at least 2")

    labels = labels.long()
    environment_ids = environment_ids.long()
    if bool(((environment_ids < 0) | (environment_ids >= num_environments)).any()):
        raise ValueError("environment_ids contain an out-of-range value")

    if environment_features is None:
        environment = F.one_hot(environment_ids, num_classes=num_environments).to(features.dtype)
    else:
        if environment_features.ndim != 2 or environment_features.shape[0] != features.shape[0]:
            raise ValueError("environment_features must have shape [batch, dimension]")
        environment = environment_features.to(device=features.device, dtype=features.dtype).detach()
    class_losses = []
    mean_abs_covariances = []

    for class_id in torch.unique(labels, sorted=True):
        mask = labels == class_id
        count = int(mask.sum().item())
        if count < min_class_count:
            continue
        if environment_features is None and int(torch.unique(environment_ids[mask]).numel()) < 2:
            continue

        class_features = features[mask]
        class_environment = environment[mask]
        feature_std = class_features.std(dim=0, unbiased=False)
        environment_std = class_environment.std(dim=0, unbiased=False)
        varying_environment = environment_std > eps
        if not bool(varying_environment.any()):
            continue

        class_features = (
            class_features - class_features.mean(dim=0, keepdim=True)
        ) / feature_std.clamp_min(eps)
        class_environment = (
            class_environment[:, varying_environment]
            - class_environment[:, varying_environment].mean(dim=0, keepdim=True)
        ) / environment_std[varying_environment].clamp_min(eps)

        covariance = class_features.transpose(0, 1) @ class_environment
        covariance = covariance / max(count - 1, 1)
        class_losses.append(covariance.square().mean())
        mean_abs_covariances.append(covariance.abs().mean())

    if class_losses:
        loss = torch.stack(class_losses).mean()
        mean_abs_covariance = torch.stack(mean_abs_covariances).mean()
    else:
        loss = features.sum() * 0.0
        mean_abs_covariance = features.new_zeros(())

    diagnostics = {
        "valid_classes": features.new_tensor(float(len(class_losses))),
        "mean_abs_covariance": mean_abs_covariance,
    }
    return loss, diagnostics
