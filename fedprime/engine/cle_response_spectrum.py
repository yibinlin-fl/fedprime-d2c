from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


EPS_ENERGY = 1.0e-12
EPS_SPECTRUM = 1.0e-12


@dataclass(frozen=True)
class SpectrumStatistics:
    concentration: float
    effective_rank: float
    top1_share: float
    top3_share: float
    trace: float
    eigenvalues: np.ndarray


@dataclass(frozen=True)
class ResponseSpectrum:
    gram: np.ndarray
    mean_response_energy: float
    statistics: SpectrumStatistics


def _float64(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def spectrum_from_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    epsilon: float = EPS_SPECTRUM,
) -> SpectrumStatistics:
    values = _float64(eigenvalues).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("finite eigenvalues are required")
    # Symmetric eigensolvers can return tiny negative round-off values for PSD
    # matrices. Large negative values remain an implementation failure.
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(values.min()) < -1.0e-8 * scale:
        raise ValueError("spectrum is not positive semidefinite")
    values = np.maximum(values, 0.0)
    values = np.sort(values)[::-1]
    trace = float(values.sum())
    concentration = float(np.square(values).sum() / (trace * trace + float(epsilon)))
    effective_rank = float(1.0 / (concentration + float(epsilon)))
    top1 = float(values[0] / (trace + float(epsilon)))
    top3 = float(values[: min(3, values.size)].sum() / (trace + float(epsilon)))
    return SpectrumStatistics(concentration, effective_rank, top1, top3, trace, values)


def spectrum_from_gram(
    gram: np.ndarray,
    *,
    epsilon: float = EPS_SPECTRUM,
) -> SpectrumStatistics:
    matrix = _float64(gram)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gram matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Gram matrix must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-8):
        raise ValueError("Gram matrix must be symmetric")
    symmetric = torch.as_tensor((matrix + matrix.T) * 0.5, dtype=torch.float64)
    values = torch.linalg.eigvalsh(symmetric).cpu().numpy()
    return spectrum_from_eigenvalues(values, epsilon=epsilon)


def response_spectrum(
    delta: np.ndarray,
    *,
    eps_energy: float = EPS_ENERGY,
    eps_spectrum: float = EPS_SPECTRUM,
) -> ResponseSpectrum:
    """Compute the all-probe response spectrum for one carrier half.

    ``delta`` has shape ``[carrier, probe, feature]``.  The returned Gram
    matrix is in probe space and therefore remains 64x64 across heterogeneous
    feature dimensions.
    """

    values = _float64(delta)
    if values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError("delta must have shape [carrier, probe, feature]")
    if not np.all(np.isfinite(values)):
        raise ValueError("delta must be finite")
    means = values.mean(axis=0)
    energies = np.square(values).sum(axis=-1).mean(axis=0)
    standardized = means / (np.sqrt(energies)[:, None] + float(eps_energy))
    standardized_t = torch.as_tensor(standardized, dtype=torch.float64)
    gram = (standardized_t @ standardized_t.T).cpu().numpy()
    statistics = spectrum_from_gram(gram, epsilon=eps_spectrum)
    return ResponseSpectrum(
        gram=gram,
        mean_response_energy=float(energies.mean()),
        statistics=statistics,
    )


def clean_spectrum(
    features: np.ndarray,
    *,
    epsilon: float = EPS_SPECTRUM,
) -> SpectrumStatistics:
    values = _float64(features)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("features must have shape [carrier, feature]")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    centered = values - values.mean(axis=0, keepdims=True)
    n = float(values.shape[0])
    centered_t = torch.as_tensor(centered, dtype=torch.float64)
    if values.shape[1] <= values.shape[0]:
        covariance = (centered_t.T @ centered_t / n).cpu().numpy()
    else:
        covariance = (centered_t @ centered_t.T / n).cpu().numpy()
    return spectrum_from_gram(covariance, epsilon=epsilon)


def bootstrap_count_matrix(indices: np.ndarray, carriers: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] <= 0:
        raise ValueError("indices must have shape [replicate, draw]")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= int(carriers)):
        raise ValueError("bootstrap index is outside the carrier range")
    counts = np.zeros((values.shape[0], int(carriers)), dtype=np.float64)
    for replicate in range(values.shape[0]):
        np.add.at(counts[replicate], values[replicate], 1.0)
    return counts


def _resolve_compute_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA bootstrap requested but unavailable")
    return resolved


def bootstrap_response_concentration(
    delta: np.ndarray,
    counts: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 16,
    eps_energy: float = EPS_ENERGY,
    eps_spectrum: float = EPS_SPECTRUM,
) -> np.ndarray:
    """Recompute μ, E, S, K and chi for every carrier bootstrap draw."""

    values = np.asarray(delta, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("delta must have shape [carrier, probe, feature]")
    if weights.ndim != 2 or weights.shape[1] != values.shape[0]:
        raise ValueError("count matrix must match the carrier dimension")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    draws = weights.sum(axis=1)
    if not np.all(draws > 0.0) or not np.allclose(draws, draws[0]):
        raise ValueError("each bootstrap replicate must contain the same positive draw count")

    compute = _resolve_compute_device(device)
    delta_t = torch.as_tensor(values, dtype=torch.float64, device=compute)
    flat_t = delta_t.reshape(values.shape[0], -1)
    squared_t = delta_t.square().sum(dim=-1)
    output = np.empty(weights.shape[0], dtype=np.float64)
    for start in range(0, weights.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), weights.shape[0])
        count_t = torch.as_tensor(weights[start:stop], dtype=torch.float64, device=compute)
        draw_t = count_t.sum(dim=1)
        means = torch.matmul(count_t, flat_t).reshape(
            stop - start, values.shape[1], values.shape[2]
        ) / draw_t[:, None, None]
        energy = torch.matmul(count_t, squared_t) / draw_t[:, None]
        standardized = means / (torch.sqrt(energy)[:, :, None] + float(eps_energy))
        gram = torch.bmm(standardized, standardized.transpose(1, 2))
        trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
        chi = gram.square().sum(dim=(1, 2)) / (trace.square() + float(eps_spectrum))
        output[start:stop] = chi.detach().cpu().numpy()
    return output


def bootstrap_clean_concentration(
    features: np.ndarray,
    counts: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 32,
    eps_spectrum: float = EPS_SPECTRUM,
) -> np.ndarray:
    """Exactly recompute centered clean-covariance chi for bootstrap draws.

    The implementation uses the carrier Gram matrix and algebraically exact
    weighted covariance trace identities.  It avoids materializing one dxd
    covariance matrix per replicate.
    """

    values = np.asarray(features, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must have shape [carrier, feature]")
    if weights.ndim != 2 or weights.shape[1] != values.shape[0]:
        raise ValueError("count matrix must match the carrier dimension")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    compute = _resolve_compute_device(device)
    feature_t = torch.as_tensor(values, dtype=torch.float64, device=compute)
    gram = feature_t @ feature_t.T
    gram_squared = gram.square()
    diagonal = torch.diagonal(gram)
    output = np.empty(weights.shape[0], dtype=np.float64)
    for start in range(0, weights.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), weights.shape[0])
        w = torch.as_tensor(weights[start:stop], dtype=torch.float64, device=compute)
        n = w.sum(dim=1)
        wg = w @ gram
        mean_norm_sq = (wg * w).sum(dim=1) / n.square()
        trace_a = (w * diagonal).sum(dim=1) / n
        trace_c = trace_a - mean_norm_sq
        trace_a_sq = ((w @ gram_squared) * w).sum(dim=1) / n.square()
        mean_a_mean = (w * wg.square()).sum(dim=1) / n.pow(3)
        trace_c_sq = trace_a_sq - 2.0 * mean_a_mean + mean_norm_sq.square()
        trace_c_sq = torch.clamp(trace_c_sq, min=0.0)
        chi = trace_c_sq / (trace_c.square() + float(eps_spectrum))
        output[start:stop] = chi.detach().cpu().numpy()
    return output


__all__ = [
    "EPS_ENERGY",
    "EPS_SPECTRUM",
    "ResponseSpectrum",
    "SpectrumStatistics",
    "bootstrap_clean_concentration",
    "bootstrap_count_matrix",
    "bootstrap_response_concentration",
    "clean_spectrum",
    "response_spectrum",
    "spectrum_from_eigenvalues",
    "spectrum_from_gram",
]
