from __future__ import annotations

import math
from collections import deque

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


def cdep_v2_ramp(round_idx: int, *, warmup_rounds: int, ramp_rounds: int) -> float:
    """Return the pre-registered CDep-v2 activation multiplier for a round."""

    if warmup_rounds < 0 or ramp_rounds < 1:
        raise ValueError("warmup_rounds must be non-negative and ramp_rounds must be positive")
    if int(round_idx) < int(warmup_rounds):
        return 0.0
    return min(1.0, (int(round_idx) - int(warmup_rounds) + 1) / float(ramp_rounds))


class BufferedConditionalMomentAlignment:
    """Client-local, confidence-gated class/environment feature memory.

    CDep-v1 estimates feature/environment cross-covariance from one Non-IID
    mini-batch. CDep-v2 instead aligns environment-specific feature centroids
    inside each class using a bounded cross-batch memory. Stored features are
    detached and never leave the client; gradients flow only through current
    fit-batch features.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        num_environments: int,
        max_size_per_group: int = 64,
    ) -> None:
        if num_classes < 1 or num_environments < 2 or max_size_per_group < 1:
            raise ValueError("invalid CDep-v2 memory dimensions")
        self.num_classes = int(num_classes)
        self.num_environments = int(num_environments)
        self.max_size_per_group = int(max_size_per_group)
        self._queues = [
            [deque(maxlen=self.max_size_per_group) for _ in range(self.num_environments)]
            for _ in range(self.num_classes)
        ]

    @property
    def sample_count(self) -> int:
        return sum(len(queue) for class_queues in self._queues for queue in class_queues)

    def _memory_group(
        self,
        class_id: int,
        environment_id: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        queue = self._queues[class_id][environment_id]
        if not queue:
            return None, None
        features = torch.stack([item[0] for item in queue]).to(device=device, dtype=dtype)
        confidence = torch.tensor(
            [item[1] for item in queue],
            device=device,
            dtype=dtype,
        )
        return features, confidence

    def enqueue(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        environment_ids: torch.Tensor,
        confidence: torch.Tensor,
        *,
        min_confidence: float,
    ) -> None:
        normalized = F.normalize(features.detach(), dim=1).cpu()
        labels_cpu = labels.detach().long().cpu()
        environments_cpu = environment_ids.detach().long().cpu()
        confidence_cpu = confidence.detach().float().cpu()
        for feature, label, environment, score in zip(
            normalized,
            labels_cpu,
            environments_cpu,
            confidence_cpu,
        ):
            class_id = int(label.item())
            environment_id = int(environment.item())
            score_value = float(score.item())
            if not 0 <= class_id < self.num_classes:
                continue
            if not 0 <= environment_id < self.num_environments:
                continue
            if score_value < float(min_confidence):
                continue
            self._queues[class_id][environment_id].append((feature.clone(), score_value))

    def loss_and_update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        environment_ids: torch.Tensor,
        confidence: torch.Tensor,
        *,
        min_confidence: float = 0.20,
        min_group_count: int = 4,
        min_environments: int = 2,
        eps: float = 1.0e-6,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Align buffered within-class environment centroids, then update memory."""

        if features.ndim != 2:
            raise ValueError("features must have shape [batch, dimension]")
        if labels.shape != environment_ids.shape or labels.shape != confidence.shape:
            raise ValueError("labels, environment_ids, and confidence must share shape [batch]")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and annotations must share the batch dimension")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if min_group_count < 1 or min_environments < 2:
            raise ValueError("invalid CDep-v2 support thresholds")

        labels = labels.long()
        environment_ids = environment_ids.long()
        confidence = confidence.to(device=features.device, dtype=features.dtype).detach()
        normalized = F.normalize(features, dim=1)
        class_losses = []
        group_shifts = []
        valid_group_count = 0

        for raw_class_id in torch.unique(labels, sorted=True):
            class_id = int(raw_class_id.item())
            if not 0 <= class_id < self.num_classes:
                continue
            centroids = []
            class_has_current = False
            for environment_id in range(self.num_environments):
                current_mask = (
                    labels.eq(class_id)
                    & environment_ids.eq(environment_id)
                    & confidence.ge(float(min_confidence))
                )
                current_features = normalized[current_mask]
                current_confidence = confidence[current_mask]
                memory_features, memory_confidence = self._memory_group(
                    class_id,
                    environment_id,
                    device=features.device,
                    dtype=features.dtype,
                )
                support = int(current_features.shape[0])
                if memory_features is not None:
                    support += int(memory_features.shape[0])
                if support < int(min_group_count):
                    continue
                group_features = current_features
                group_confidence = current_confidence
                if memory_features is not None and memory_confidence is not None:
                    group_features = torch.cat([group_features, memory_features.detach()], dim=0)
                    group_confidence = torch.cat([group_confidence, memory_confidence.detach()], dim=0)
                weights = group_confidence.clamp_min(float(eps))
                centroid = (group_features * weights.unsqueeze(1)).sum(dim=0) / weights.sum().clamp_min(
                    float(eps)
                )
                centroids.append(centroid)
                class_has_current = class_has_current or bool(current_mask.any())

            if len(centroids) < int(min_environments) or not class_has_current:
                continue
            stacked = torch.stack(centroids)
            balanced_center = stacked.mean(dim=0, keepdim=True)
            squared_shift = (stacked - balanced_center).square().sum(dim=1)
            class_losses.append(squared_shift.mean())
            group_shifts.append(squared_shift.sqrt().mean())
            valid_group_count += len(centroids)

        if class_losses:
            loss = torch.stack(class_losses).mean()
            mean_group_shift = torch.stack(group_shifts).mean()
        else:
            loss = features.sum() * 0.0
            mean_group_shift = features.new_zeros(())

        self.enqueue(
            normalized,
            labels,
            environment_ids,
            confidence,
            min_confidence=float(min_confidence),
        )
        diagnostics = {
            "valid_classes": features.new_tensor(float(len(class_losses))),
            "valid_groups": features.new_tensor(float(valid_group_count)),
            "mean_group_shift": mean_group_shift,
            "buffer_samples": features.new_tensor(float(self.sample_count)),
        }
        return loss, diagnostics


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
