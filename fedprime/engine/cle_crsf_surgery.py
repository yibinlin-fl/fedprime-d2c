from __future__ import annotations

import copy
import dataclasses
import hashlib
import time
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import torch
from torch import nn


EPS = 1.0e-12
ObjectiveName = Literal["crsf", "shared_mean", "generic_invariance", "rawspec"]
ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclasses.dataclass(frozen=True)
class GradientAgreement:
    relative_error: float
    cosine: float


@dataclasses.dataclass(frozen=True)
class ResponseMoments:
    mean: torch.Tensor
    energy: torch.Tensor
    count: int


@dataclasses.dataclass(frozen=True)
class RawMoments:
    mean: torch.Tensor
    second: torch.Tensor
    count: int


@dataclasses.dataclass(frozen=True)
class LateBlockAudit:
    architecture: str
    trainable_stage: str
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    total_parameter_count: int


@dataclasses.dataclass(frozen=True)
class SurgeryTrace:
    objective: str
    initial_raw_loss: float
    accepted_raw_losses: tuple[float, ...]
    accepted_normalized_losses: tuple[float, ...]
    anchor_kl: tuple[float, ...]
    accepted_learning_rates: tuple[float, ...]
    attempts: tuple[dict[str, object], ...]
    accepted_steps: int
    contract_failure: bool
    timings_seconds: dict[str, float]


@dataclasses.dataclass(frozen=True)
class ExactSurgeryPreparation:
    """Reusable exact state at the common pre-update checkpoint.

    The preparation contains only quantities that are identical for learning-rate
    candidates starting from the same checkpoint.  Post-update objective and KL
    evaluation are deliberately *not* reusable and remain mandatory.
    """

    objective: str
    initial_raw_loss: float
    reference_probability: np.ndarray
    moments: ResponseMoments | RawMoments
    base_feature: np.ndarray | None
    gradients: tuple[torch.Tensor, ...]
    timings_seconds: dict[str, float]


def _emit(progress: ProgressCallback | None, event: str, **payload: object) -> None:
    if progress is not None:
        progress(event, payload)


def response_loss_from_moments(
    mean: torch.Tensor,
    energy: torch.Tensor,
    objective: ObjectiveName,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute one of the frozen response objectives from exact moments.

    ``mean`` has shape [Q, D] and ``energy`` has shape [Q].  Float64 is
    recommended for the sufficient-statistic graph.
    """
    if mean.ndim != 2 or energy.ndim != 1 or mean.shape[0] != energy.shape[0]:
        raise ValueError("invalid response-moment shapes")
    if objective == "shared_mean":
        return mean.square().sum(dim=1).mean()
    if objective == "generic_invariance":
        return energy.mean()
    if objective != "crsf":
        raise ValueError(f"unsupported response objective: {objective}")
    response = mean / (torch.sqrt(torch.clamp_min(energy, 0.0))[:, None] + float(eps))
    gram = response @ response.T
    numerator = gram.square().sum()
    denominator = torch.trace(gram).square() + float(eps)
    return numerator / denominator


def rawspec_loss_from_moments(
    mean: torch.Tensor,
    second: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    if mean.ndim != 1 or second.ndim != 2 or second.shape != (mean.numel(), mean.numel()):
        raise ValueError("invalid raw-feature moment shapes")
    covariance = second - torch.outer(mean, mean)
    numerator = torch.trace(covariance @ covariance)
    denominator = torch.trace(covariance).square() + float(eps)
    return numerator / denominator


def response_cotangents(
    moments: ResponseMoments,
    objective: ObjectiveName,
    *,
    normalizer: float = 1.0,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    mean = moments.mean.detach().to(dtype=torch.float64).requires_grad_(True)
    energy = moments.energy.detach().to(dtype=torch.float64).requires_grad_(True)
    raw = response_loss_from_moments(mean, energy, objective)
    normalized = raw / (float(normalizer) + EPS)
    g_mean, g_energy = torch.autograd.grad(normalized, (mean, energy), allow_unused=True)
    if g_mean is None:
        g_mean = torch.zeros_like(mean)
    if g_energy is None:
        g_energy = torch.zeros_like(energy)
    return float(raw.detach()), g_mean.detach(), g_energy.detach()


def rawspec_cotangents(
    moments: RawMoments,
    *,
    normalizer: float = 1.0,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    mean = moments.mean.detach().to(dtype=torch.float64).requires_grad_(True)
    second = moments.second.detach().to(dtype=torch.float64).requires_grad_(True)
    raw = rawspec_loss_from_moments(mean, second)
    normalized = raw / (float(normalizer) + EPS)
    g_mean, g_second = torch.autograd.grad(normalized, (mean, second))
    return float(raw.detach()), g_mean.detach(), g_second.detach()


def exact_response_moments(delta_batches: Iterable[torch.Tensor]) -> ResponseMoments:
    total = None
    energy = None
    count = 0
    for delta in delta_batches:
        values = delta.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim != 3:
            raise ValueError("delta batches must have shape [N,Q,D]")
        batch_total = values.sum(dim=0)
        batch_energy = values.square().sum(dim=-1).sum(dim=0)
        total = batch_total if total is None else total + batch_total
        energy = batch_energy if energy is None else energy + batch_energy
        count += int(values.shape[0])
    if count == 0 or total is None or energy is None:
        raise ValueError("empty response stream")
    return ResponseMoments(total / count, energy / count, count)


def exact_raw_moments(feature_batches: Iterable[torch.Tensor]) -> RawMoments:
    total = None
    second = None
    count = 0
    for feature in feature_batches:
        values = feature.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim != 2:
            raise ValueError("feature batches must have shape [N,D]")
        batch_total = values.sum(dim=0)
        batch_second = values.T @ values
        total = batch_total if total is None else total + batch_total
        second = batch_second if second is None else second + batch_second
        count += int(values.shape[0])
    if count == 0 or total is None or second is None:
        raise ValueError("empty feature stream")
    return RawMoments(total / count, second / count, count)


def gradient_agreement(reference: Iterable[torch.Tensor], candidate: Iterable[torch.Tensor]) -> GradientAgreement:
    first = torch.cat([value.detach().reshape(-1).to(dtype=torch.float64, device="cpu") for value in reference])
    second = torch.cat([value.detach().reshape(-1).to(dtype=torch.float64, device="cpu") for value in candidate])
    difference = torch.linalg.vector_norm(first - second)
    denominator = torch.clamp_min(torch.linalg.vector_norm(first), EPS)
    relative = float(difference / denominator)
    cosine = float(torch.dot(first, second) / torch.clamp_min(torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second), EPS))
    return GradientAgreement(relative, cosine)


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def _is_normalization(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LocalResponseNorm,
        ),
    )


class LateBlockAdapter:
    """Actual-graph adapter for the four frozen heterogeneous architectures."""

    def __init__(self, model: nn.Module, architecture: str):
        self.model = model
        self.architecture = architecture
        if architecture in ("ResNet10", "ResNet12"):
            self.stage_name = "layer4"
            self.stage = model.layer4
            self._kind = "resnet"
        elif architecture == "ShuffleNet":
            self.stage_name = "layer3"
            self.stage = model.layer3
            self._kind = "shuffle"
        elif architecture == "Mobilenetv2":
            if not isinstance(model.layers, nn.Sequential) or len(model.layers) < 1:
                raise TypeError("unexpected MobileNetV2 graph")
            self.stage_name = f"layers.{len(model.layers) - 1}"
            self.stage = model.layers[-1]
            self._kind = "mobile"
        else:
            raise ValueError(f"unsupported architecture: {architecture}")
        if not hasattr(model, "linear") or not isinstance(model.linear, nn.Linear):
            raise TypeError("classifier must be model.linear")

    def configure(self, device: torch.device) -> LateBlockAudit:
        self.model.to(device).eval()
        for module in self.model.modules():
            module.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        norm_prefixes = {
            name for name, module in self.stage.named_modules() if name and _is_normalization(module)
        }
        for relative_name, parameter in self.stage.named_parameters():
            parent_name = relative_name.rsplit(".", 1)[0] if "." in relative_name else ""
            if not any(parent_name == prefix or parent_name.startswith(prefix + ".") for prefix in norm_prefixes):
                parameter.requires_grad_(True)
        names = dict(self.model.named_parameters())
        trainable = tuple(name for name, value in names.items() if value.requires_grad)
        frozen = tuple(name for name, value in names.items() if not value.requires_grad)
        expected_prefix = self.stage_name + "."
        if not trainable or any(not name.startswith(expected_prefix) for name in trainable):
            raise AssertionError(f"trainable scope escaped {self.stage_name}: {trainable}")
        if any("bn" in name.lower() for name in trainable):
            raise AssertionError(f"normalization parameter became trainable: {trainable}")
        return LateBlockAudit(
            architecture=self.architecture,
            trainable_stage=self.stage_name,
            trainable_parameter_names=trainable,
            frozen_parameter_names=frozen,
            trainable_parameter_count=sum(names[name].numel() for name in trainable),
            total_parameter_count=sum(value.numel() for value in names.values()),
        )

    def prefix(self, images: torch.Tensor) -> torch.Tensor:
        if self._kind == "resnet":
            out = self.model.conv1(images)
            out = self.model.bn1(out)
            out = torch.relu(out)
            out = self.model.layer1(out)
            out = self.model.layer2(out)
            return self.model.layer3(out)
        if self._kind == "shuffle":
            out = self.model.conv1(images)
            out = self.model.bn1(out)
            out = torch.relu(out)
            out = self.model.layer1(out)
            return self.model.layer2(out)
        out = self.model.conv1(images)
        out = self.model.bn1(out)
        out = torch.relu(out)
        for layer in list(self.model.layers.children())[:-1]:
            out = layer(out)
        return out

    def feature_from_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        out = self.stage(prefix)
        if self._kind == "mobile":
            out = self.model.conv2(out)
            out = self.model.bn2(out)
            out = torch.relu(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        return out.flatten(1)

    def logits_from_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.model.linear(self.feature_from_prefix(prefix))

    def clone(self, device: torch.device) -> "LateBlockAdapter":
        replica = LateBlockAdapter(copy.deepcopy(self.model).to(device), self.architecture)
        replica.configure(device)
        return replica

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def approved_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.model.named_parameters() if parameter.requires_grad)

    def state_delta(self, original: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        current = self.model.state_dict()
        return {name: current[name].detach().cpu() - original[name].detach().cpu() for name in self.approved_parameter_names()}


def changed_state_keys(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> tuple[str, ...]:
    return tuple(sorted(name for name in before if not torch.equal(before[name].cpu(), after[name].cpu())))


def apply_state_delta(model: nn.Module, delta: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in delta.items():
            if name not in parameters:
                raise KeyError(name)
            parameter = parameters[name]
            parameter.add_(value.to(dtype=parameter.dtype, device=parameter.device))


def _feature_batches_from_prefix(
    adapter: LateBlockAdapter,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    grad: bool,
):
    context = torch.enable_grad if grad else torch.no_grad
    with context():
        for start in range(0, int(values.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(values.shape[0]))
            prefix = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device=device, dtype=torch.float32)
            yield start, stop, adapter.feature_from_prefix(prefix)


def response_moments_from_prefix(
    adapter: LateBlockAdapter,
    base_prefix: np.ndarray,
    probe_prefixes: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> tuple[ResponseMoments, np.ndarray]:
    _emit(progress, "pass_start", pass_name="exact_response_moments", probes=len(probe_prefixes))
    base_parts = []
    for _start, _stop, feature in _feature_batches_from_prefix(
        adapter, base_prefix, device=device, batch_size=batch_size, grad=False
    ):
        base_parts.append(feature.detach().cpu().numpy().astype(np.float32, copy=False))
    base_feature = np.concatenate(base_parts, axis=0)
    means = []
    energies = []
    for probe_id, probe in enumerate(probe_prefixes):
        total = np.zeros(base_feature.shape[1], dtype=np.float64)
        energy = 0.0
        for start, stop, feature in _feature_batches_from_prefix(
            adapter, probe, device=device, batch_size=batch_size, grad=False
        ):
            delta = feature.detach().cpu().numpy().astype(np.float64) - base_feature[start:stop]
            total += delta.sum(axis=0)
            energy += float(np.square(delta).sum())
        means.append(total / base_feature.shape[0])
        energies.append(energy / base_feature.shape[0])
        if probe_id == 0 or (probe_id + 1) % 8 == 0 or probe_id + 1 == len(probe_prefixes):
            _emit(
                progress,
                "probe_progress",
                pass_name="exact_response_moments",
                probe=probe_id + 1,
                probes=len(probe_prefixes),
            )
    moments = ResponseMoments(
        torch.from_numpy(np.stack(means)),
        torch.from_numpy(np.asarray(energies, dtype=np.float64)),
        int(base_feature.shape[0]),
    )
    _emit(progress, "pass_complete", pass_name="exact_response_moments", probes=len(probe_prefixes))
    return moments, base_feature


def features_from_prefix_numpy(
    adapter: LateBlockAdapter,
    prefix_values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    parts = []
    for _start, _stop, feature in _feature_batches_from_prefix(
        adapter, prefix_values, device=device, batch_size=batch_size, grad=False
    ):
        parts.append(feature.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def raw_moments_from_prefix(
    adapter: LateBlockAdapter,
    base_prefix: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> RawMoments:
    return exact_raw_moments(
        feature for _start, _stop, feature in _feature_batches_from_prefix(
            adapter, base_prefix, device=device, batch_size=batch_size, grad=False
        )
    )


def assign_exact_response_gradient(
    adapter: LateBlockAdapter,
    base_prefix: np.ndarray,
    probe_prefixes: list[np.ndarray],
    moments: ResponseMoments,
    objective: ObjectiveName,
    *,
    normalizer: float,
    device: torch.device,
    batch_size: int,
    base_feature: np.ndarray | None = None,
    progress: ProgressCallback | None = None,
) -> float:
    _emit(progress, "pass_start", pass_name="exact_response_gradient_vjp", probes=len(probe_prefixes))
    raw_loss, g_mean, g_energy = response_cotangents(moments, objective, normalizer=normalizer)
    parameters = adapter.trainable_parameters()
    for parameter in parameters:
        parameter.grad = None
    if base_feature is None:
        base_feature = features_from_prefix_numpy(
            adapter, base_prefix, device=device, batch_size=batch_size
        )
    base_cotangent = np.zeros_like(base_feature, dtype=np.float32)
    count = float(moments.count)
    for q, probe in enumerate(probe_prefixes):
        gm = g_mean[q].to(device=device, dtype=torch.float32)
        ge = g_energy[q].to(device=device, dtype=torch.float32)
        for start, stop, feature in _feature_batches_from_prefix(
            adapter, probe, device=device, batch_size=batch_size, grad=True
        ):
            base = torch.from_numpy(base_feature[start:stop]).to(device=device)
            delta = feature - base
            surrogate = (delta @ gm).sum() / count + ge * delta.square().sum() / count
            surrogate.backward()
            cotangent = (-gm[None] - 2.0 * ge * delta.detach()) / count
            base_cotangent[start:stop] += cotangent.cpu().numpy()
        if q == 0 or (q + 1) % 8 == 0 or q + 1 == len(probe_prefixes):
            _emit(
                progress,
                "probe_progress",
                pass_name="exact_response_gradient_vjp",
                probe=q + 1,
                probes=len(probe_prefixes),
            )
    for start, stop, feature in _feature_batches_from_prefix(
        adapter, base_prefix, device=device, batch_size=batch_size, grad=True
    ):
        cotangent = torch.from_numpy(base_cotangent[start:stop]).to(device=device)
        (feature * cotangent).sum().backward()
    _emit(progress, "pass_complete", pass_name="exact_response_gradient_vjp", probes=len(probe_prefixes))
    return raw_loss


def assign_exact_rawspec_gradient(
    adapter: LateBlockAdapter,
    base_prefix: np.ndarray,
    moments: RawMoments,
    *,
    normalizer: float,
    device: torch.device,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> float:
    _emit(progress, "pass_start", pass_name="exact_rawspec_gradient_vjp", probes=0)
    raw_loss, g_mean, g_second = rawspec_cotangents(moments, normalizer=normalizer)
    for parameter in adapter.trainable_parameters():
        parameter.grad = None
    gm = g_mean.to(device=device, dtype=torch.float32)
    gq = g_second.to(device=device, dtype=torch.float32)
    count = float(moments.count)
    for _start, _stop, feature in _feature_batches_from_prefix(
        adapter, base_prefix, device=device, batch_size=batch_size, grad=True
    ):
        surrogate = ((feature @ gm) + torch.einsum("nd,df,nf->n", feature, gq, feature)).sum() / count
        surrogate.backward()
    _emit(progress, "pass_complete", pass_name="exact_rawspec_gradient_vjp", probes=0)
    return raw_loss


def public_anchor_probabilities(
    adapter: LateBlockAdapter,
    base_prefix: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, int(base_prefix.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(base_prefix.shape[0]))
            prefix = torch.from_numpy(np.array(base_prefix[start:stop], copy=True)).to(device=device, dtype=torch.float32)
            parts.append(torch.softmax(adapter.logits_from_prefix(prefix), dim=-1).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def public_anchor_kl(
    adapter: LateBlockAdapter,
    base_prefix: np.ndarray,
    reference_probabilities: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, int(base_prefix.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(base_prefix.shape[0]))
            prefix = torch.from_numpy(np.array(base_prefix[start:stop], copy=True)).to(device=device, dtype=torch.float32)
            log_probability = torch.log_softmax(adapter.logits_from_prefix(prefix), dim=-1)
            reference = torch.from_numpy(reference_probabilities[start:stop]).to(device=device)
            kl = (reference * (torch.log(torch.clamp_min(reference, 1.0e-12)) - log_probability)).sum(dim=-1)
            total += float(kl.sum().cpu())
            count += int(stop - start)
    return total / count


def _optimizer_snapshot(optimizer: torch.optim.Optimizer) -> dict:
    return copy.deepcopy(optimizer.state_dict())


def _parameter_snapshot(adapter: LateBlockAdapter) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in adapter.trainable_parameters()]


def _restore_parameters(adapter: LateBlockAdapter, values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(adapter.trainable_parameters(), values):
            parameter.copy_(value)


def evaluate_objective(
    adapter: LateBlockAdapter,
    objective: ObjectiveName,
    base_prefix: np.ndarray,
    probe_prefixes: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> float:
    raw, _moments, _base_feature = evaluate_objective_state(
        adapter,
        objective,
        base_prefix,
        probe_prefixes,
        device=device,
        batch_size=batch_size,
        progress=progress,
    )
    return raw


def evaluate_objective_state(
    adapter: LateBlockAdapter,
    objective: ObjectiveName,
    base_prefix: np.ndarray,
    probe_prefixes: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> tuple[float, ResponseMoments | RawMoments, np.ndarray | None]:
    """Run the mandatory exact full-carrier objective evaluation.

    Returning the exact post-update moments lets the next accepted step reuse
    them as its pre-update sufficient statistics.  The evaluation itself is
    never skipped, so accept/rollback semantics are unchanged.
    """
    if objective == "rawspec":
        _emit(progress, "pass_start", pass_name="exact_post_update_objective", probes=0)
        moments = raw_moments_from_prefix(adapter, base_prefix, device=device, batch_size=batch_size)
        raw = float(rawspec_loss_from_moments(moments.mean, moments.second))
        _emit(progress, "pass_complete", pass_name="exact_post_update_objective", probes=0)
        return raw, moments, None
    moments, base_feature = response_moments_from_prefix(
        adapter,
        base_prefix,
        probe_prefixes,
        device=device,
        batch_size=batch_size,
        progress=progress,
    )
    raw = float(response_loss_from_moments(moments.mean, moments.energy, objective))
    return raw, moments, base_feature


def _capture_gradients(adapter: LateBlockAdapter) -> tuple[torch.Tensor, ...]:
    values = []
    for parameter in adapter.trainable_parameters():
        if parameter.grad is None:
            raise RuntimeError("exact preparation produced a missing gradient")
        values.append(parameter.grad.detach().cpu().clone())
    return tuple(values)


def _assign_captured_gradients(adapter: LateBlockAdapter, gradients: tuple[torch.Tensor, ...]) -> None:
    parameters = adapter.trainable_parameters()
    if len(parameters) != len(gradients):
        raise ValueError("captured-gradient parameter count mismatch")
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype).clone()


def prepare_exact_surgery(
    adapter: LateBlockAdapter,
    objective: ObjectiveName,
    base_prefix: np.ndarray,
    probe_prefixes: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> ExactSurgeryPreparation:
    """Compute the common step-1 exact state once for matched LR candidates."""

    started = time.perf_counter()
    anchor_started = time.perf_counter()
    reference_probability = public_anchor_probabilities(
        adapter, base_prefix, device=device, batch_size=batch_size
    )
    anchor_seconds = time.perf_counter() - anchor_started
    _emit(progress, "pass_complete", pass_name="reference_anchor", seconds=anchor_seconds)

    moments_started = time.perf_counter()
    initial, moments, base_feature = evaluate_objective_state(
        adapter,
        objective,
        base_prefix,
        probe_prefixes,
        device=device,
        batch_size=batch_size,
        progress=progress,
    )
    moments_seconds = time.perf_counter() - moments_started

    gradient_started = time.perf_counter()
    if objective == "rawspec":
        assert isinstance(moments, RawMoments)
        assign_exact_rawspec_gradient(
            adapter,
            base_prefix,
            moments,
            normalizer=initial,
            device=device,
            batch_size=batch_size,
            progress=progress,
        )
    else:
        assert isinstance(moments, ResponseMoments)
        assign_exact_response_gradient(
            adapter,
            base_prefix,
            probe_prefixes,
            moments,
            objective,
            normalizer=initial,
            device=device,
            batch_size=batch_size,
            base_feature=base_feature,
            progress=progress,
        )
    gradient_seconds = time.perf_counter() - gradient_started
    gradients = _capture_gradients(adapter)
    for parameter in adapter.trainable_parameters():
        parameter.grad = None
    return ExactSurgeryPreparation(
        objective=objective,
        initial_raw_loss=initial,
        reference_probability=reference_probability,
        moments=moments,
        base_feature=base_feature,
        gradients=gradients,
        timings_seconds={
            "reference_anchor": anchor_seconds,
            "initial_exact_moments": moments_seconds,
            "initial_exact_gradient_vjp": gradient_seconds,
            "total": time.perf_counter() - started,
        },
    )


def run_exact_surgery(
    adapter: LateBlockAdapter,
    objective: ObjectiveName,
    base_prefix: np.ndarray,
    probe_prefixes: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    accepted_steps: int,
    anchor_limit: float,
    maximum_backtracks: int = 12,
    initial_preparation: ExactSurgeryPreparation | None = None,
    progress: ProgressCallback | None = None,
) -> SurgeryTrace:
    if objective != "rawspec" and not probe_prefixes:
        raise ValueError("response objectives require probes")
    run_started = time.perf_counter()
    preparation = initial_preparation or prepare_exact_surgery(
        adapter,
        objective,
        base_prefix,
        probe_prefixes,
        device=device,
        batch_size=batch_size,
        progress=progress,
    )
    if preparation.objective != objective:
        raise ValueError("exact-surgery preparation objective mismatch")
    reference_probability = preparation.reference_probability
    initial = preparation.initial_raw_loss
    current_moments = preparation.moments
    current_base_feature = preparation.base_feature
    prepared_gradients: tuple[torch.Tensor, ...] | None = preparation.gradients
    optimizer = torch.optim.Adam(adapter.trainable_parameters(), lr=float(learning_rate), weight_decay=0.0)
    accepted_raw = [initial]
    accepted_norm = [1.0]
    accepted_kl = [0.0]
    accepted_lr = []
    attempts: list[dict[str, object]] = []
    current_lr = float(learning_rate)
    contract_failure = False
    while len(accepted_lr) < int(accepted_steps):
        step_id = len(accepted_lr)
        _emit(progress, "step_start", step=step_id, learning_rate=current_lr)
        accepted = False
        parameter_state = _parameter_snapshot(adapter)
        optimizer_state = _optimizer_snapshot(optimizer)
        gradient_started = time.perf_counter()
        if prepared_gradients is None:
            if objective == "rawspec":
                assert isinstance(current_moments, RawMoments)
                assign_exact_rawspec_gradient(
                    adapter,
                    base_prefix,
                    current_moments,
                    normalizer=initial,
                    device=device,
                    batch_size=batch_size,
                    progress=progress,
                )
            else:
                assert isinstance(current_moments, ResponseMoments)
                assign_exact_response_gradient(
                    adapter,
                    base_prefix,
                    probe_prefixes,
                    current_moments,
                    objective,
                    normalizer=initial,
                    device=device,
                    batch_size=batch_size,
                    base_feature=current_base_feature,
                    progress=progress,
                )
            prepared_gradients = _capture_gradients(adapter)
        gradient_seconds = time.perf_counter() - gradient_started
        finite_gradient = all(torch.isfinite(value).all() for value in prepared_gradients)
        gradient_norm = float(
            torch.sqrt(sum(value.detach().float().square().sum() for value in prepared_gradients)).cpu()
        ) if finite_gradient else float("nan")
        for retry in range(int(maximum_backtracks) + 1):
            _restore_parameters(adapter, parameter_state)
            optimizer.load_state_dict(optimizer_state)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            _assign_captured_gradients(adapter, prepared_gradients)
            _emit(progress, "candidate_start", step=step_id, retry=retry, learning_rate=current_lr)
            if finite_gradient:
                optimizer.step()
                parameter_delta_norm = float(
                    torch.sqrt(
                        sum(
                            (parameter.detach() - before).float().square().sum()
                            for parameter, before in zip(adapter.trainable_parameters(), parameter_state)
                        )
                    ).cpu()
                )
                post_started = time.perf_counter()
                raw_after, post_moments, post_base_feature = evaluate_objective_state(
                    adapter,
                    objective,
                    base_prefix,
                    probe_prefixes,
                    device=device,
                    batch_size=batch_size,
                    progress=progress,
                )
                post_seconds = time.perf_counter() - post_started
                normalized_after = raw_after / (initial + EPS)
                anchor_started = time.perf_counter()
                anchor = public_anchor_kl(
                    adapter,
                    base_prefix,
                    reference_probability,
                    device=device,
                    batch_size=batch_size,
                )
                anchor_seconds = time.perf_counter() - anchor_started
            else:
                parameter_delta_norm = float("nan")
                raw_after = float("nan")
                normalized_after = float("nan")
                anchor = float("nan")
                post_moments = current_moments
                post_base_feature = current_base_feature
                post_seconds = 0.0
                anchor_seconds = 0.0
            accepted = bool(
                finite_gradient
                and np.isfinite(raw_after)
                and np.isfinite(anchor)
                and anchor <= float(anchor_limit)
                and normalized_after <= accepted_norm[-1] + 1.0e-6
            )
            attempts.append(
                {
                    "step": len(accepted_lr),
                    "retry": retry,
                    "learning_rate": current_lr,
                    "raw_objective": raw_after,
                    "normalized_objective": normalized_after,
                    "anchor_kl": anchor,
                    "finite_gradient": finite_gradient,
                    "gradient_norm": gradient_norm,
                    "parameter_delta_norm": parameter_delta_norm,
                    "accepted": accepted,
                    "pre_step_exact_gradient_seconds": gradient_seconds if retry == 0 else 0.0,
                    "post_update_exact_objective_seconds": post_seconds,
                    "post_update_anchor_kl_seconds": anchor_seconds,
                }
            )
            _emit(
                progress,
                "candidate_complete",
                step=step_id,
                retry=retry,
                learning_rate=current_lr,
                accepted=accepted,
                normalized_objective=normalized_after,
                anchor_kl=anchor,
                gradient_seconds=gradient_seconds,
                post_objective_seconds=post_seconds,
                anchor_seconds=anchor_seconds,
            )
            if accepted:
                accepted_raw.append(raw_after)
                accepted_norm.append(normalized_after)
                accepted_kl.append(anchor)
                accepted_lr.append(current_lr)
                current_moments = post_moments
                current_base_feature = post_base_feature
                prepared_gradients = None
                break
            current_lr *= 0.5
        if not accepted:
            _restore_parameters(adapter, parameter_state)
            optimizer.load_state_dict(optimizer_state)
            contract_failure = True
            break
    return SurgeryTrace(
        objective=objective,
        initial_raw_loss=initial,
        accepted_raw_losses=tuple(accepted_raw),
        accepted_normalized_losses=tuple(accepted_norm),
        anchor_kl=tuple(accepted_kl),
        accepted_learning_rates=tuple(accepted_lr),
        attempts=tuple(attempts),
        accepted_steps=len(accepted_lr),
        contract_failure=contract_failure,
        timings_seconds={
            "preparation_total": float(preparation.timings_seconds.get("total", 0.0)),
            "run_total": time.perf_counter() - run_started,
            "exact_gradient_total": float(sum(float(row["pre_step_exact_gradient_seconds"]) for row in attempts)),
            "post_update_exact_objective_total": float(
                sum(float(row["post_update_exact_objective_seconds"]) for row in attempts)
            ),
            "post_update_anchor_kl_total": float(
                sum(float(row["post_update_anchor_kl_seconds"]) for row in attempts)
            ),
        },
    )


def direct_response_gradients(
    parameters: list[nn.Parameter],
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    probes: torch.Tensor,
    objective: ObjectiveName,
) -> tuple[torch.Tensor, ...]:
    base_feature = feature_fn(base)
    deltas = torch.stack([feature_fn(probes[:, q]) - base_feature for q in range(probes.shape[1])], dim=1)
    mean = deltas.mean(dim=0).to(dtype=torch.float64)
    energy = deltas.square().sum(dim=-1).mean(dim=0).to(dtype=torch.float64)
    loss = response_loss_from_moments(mean, energy, objective)
    return tuple(torch.autograd.grad(loss, parameters))


def two_pass_response_gradients(
    parameters: list[nn.Parameter],
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    probes: torch.Tensor,
    objective: ObjectiveName,
) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        base_feature = feature_fn(base)
        delta = torch.stack([feature_fn(probes[:, q]) - base_feature for q in range(probes.shape[1])], dim=1)
    moments = exact_response_moments([delta])
    _, g_mean, g_energy = response_cotangents(moments, objective)
    for parameter in parameters:
        parameter.grad = None
    base_detached = base_feature.detach()
    base_cotangent = torch.zeros_like(base_detached)
    count = float(base.shape[0])
    for q in range(probes.shape[1]):
        transformed = feature_fn(probes[:, q])
        difference = transformed - base_detached
        gm = g_mean[q].to(device=difference.device, dtype=difference.dtype)
        ge = g_energy[q].to(device=difference.device, dtype=difference.dtype)
        surrogate = (difference @ gm).sum() / count + ge * difference.square().sum() / count
        surrogate.backward()
        base_cotangent += (-gm[None] - 2.0 * ge * difference.detach()) / count
    base_again = feature_fn(base)
    (base_again * base_cotangent).sum().backward()
    return tuple(parameter.grad.detach().clone() for parameter in parameters)


def direct_rawspec_gradients(
    parameters: list[nn.Parameter], feature_fn: Callable[[torch.Tensor], torch.Tensor], base: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    feature = feature_fn(base)
    mean = feature.mean(dim=0).to(dtype=torch.float64)
    second = (feature.to(dtype=torch.float64).T @ feature.to(dtype=torch.float64)) / feature.shape[0]
    return tuple(torch.autograd.grad(rawspec_loss_from_moments(mean, second), parameters))


def two_pass_rawspec_gradients(
    parameters: list[nn.Parameter], feature_fn: Callable[[torch.Tensor], torch.Tensor], base: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        feature = feature_fn(base)
    moments = exact_raw_moments([feature])
    _, g_mean, g_second = rawspec_cotangents(moments)
    for parameter in parameters:
        parameter.grad = None
    feature_again = feature_fn(base)
    gm = g_mean.to(device=feature_again.device, dtype=feature_again.dtype)
    gq = g_second.to(device=feature_again.device, dtype=feature_again.dtype)
    surrogate = ((feature_again @ gm) + torch.einsum("nd,df,nf->n", feature_again, gq, feature_again)).mean()
    surrogate.backward()
    return tuple(parameter.grad.detach().clone() for parameter in parameters)
