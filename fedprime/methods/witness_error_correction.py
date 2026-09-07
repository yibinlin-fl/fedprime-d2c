from __future__ import annotations

from itertools import combinations

import numpy as np
import torch


def _as_matrix(value: np.ndarray, *, name: str) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    return torch.as_tensor(array, dtype=torch.float64)


def _as_vector(value: np.ndarray, *, length: int, name: str) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return torch.as_tensor(array, dtype=torch.float64)


def normalize_confusion_counts(counts: np.ndarray) -> np.ndarray:
    """Column-normalize observed-by-true confusion counts."""

    matrix = _as_matrix(counts, name="counts")
    if bool((matrix < 0).any()):
        raise ValueError("counts cannot be negative")
    totals = matrix.sum(dim=0)
    if bool((totals <= 0).any()):
        raise ValueError("every true-environment column must have positive support")
    return (matrix / totals.unsqueeze(0)).numpy()


def channel_diagnostics(channel: np.ndarray, *, tolerance: float = 1.0e-10) -> dict:
    """Return stable SVD diagnostics for a rectangular observed-by-latent channel."""

    matrix = _as_matrix(channel, name="channel")
    singular_values = torch.linalg.svdvals(matrix)
    rank = int((singular_values > float(tolerance)).sum().item())
    minimum = float(singular_values[-1].item()) if singular_values.numel() else 0.0
    maximum = float(singular_values[0].item()) if singular_values.numel() else 0.0
    condition = float(maximum / minimum) if minimum > 0 else float("inf")
    return {
        "singular_values": [float(value) for value in singular_values.tolist()],
        "rank": rank,
        "condition_number": condition,
        "min_singular_value": minimum,
    }


def _subsets(size: int):
    for subset_size in range(1, size + 1):
        yield from combinations(range(size), subset_size)


def solve_simplex_ridge(
    channel: np.ndarray,
    observed: np.ndarray,
    regularization: float,
    *,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Solve non-negative ridge least squares on the probability simplex.

    The latent dimension is deliberately small in WEC-BER Phase-0. Enumerating
    active sets makes the solution deterministic and avoids an optimizer whose
    stopping tolerance could change the frozen gate.
    """

    matrix = _as_matrix(channel, name="channel")
    target = _as_vector(observed, length=matrix.shape[0], name="observed")
    ridge = float(regularization)
    if ridge < 0:
        raise ValueError("regularization cannot be negative")

    latent = int(matrix.shape[1])
    best = None
    best_objective = float("inf")
    for subset in _subsets(latent):
        columns = matrix[:, list(subset)]
        hessian = columns.T @ columns + ridge * torch.eye(len(subset), dtype=torch.float64)
        linear = columns.T @ target
        ones = torch.ones(len(subset), dtype=torch.float64)
        kkt = torch.zeros((len(subset) + 1, len(subset) + 1), dtype=torch.float64)
        kkt[:-1, :-1] = hessian
        kkt[:-1, -1] = ones
        kkt[-1, :-1] = ones
        rhs = torch.cat([linear, torch.ones(1, dtype=torch.float64)])
        try:
            solution = torch.linalg.solve(kkt, rhs)[:-1]
        except RuntimeError:
            solution = torch.linalg.lstsq(kkt, rhs.unsqueeze(1)).solution[:-1, 0]
        if bool((solution < -float(tolerance)).any()):
            continue
        solution = solution.clamp_min(0.0)
        solution = solution / solution.sum().clamp_min(torch.finfo(solution.dtype).eps)
        full = torch.zeros(latent, dtype=torch.float64)
        full[list(subset)] = solution
        residual = matrix @ full - target
        objective = float((residual.square().sum() + ridge * full.square().sum()).item())
        if objective < best_objective - 1.0e-15:
            best_objective = objective
            best = full
    if best is None:
        raise RuntimeError("simplex ridge solver found no feasible active set")
    return best.numpy()


def solve_nonnegative_ridge(
    channel: np.ndarray,
    observed_mass: np.ndarray,
    regularization: float,
    *,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Solve non-negative ridge least squares for unnormalized loss mass."""

    matrix = _as_matrix(channel, name="channel")
    target = _as_vector(observed_mass, length=matrix.shape[0], name="observed_mass")
    ridge = float(regularization)
    if ridge < 0:
        raise ValueError("regularization cannot be negative")

    latent = int(matrix.shape[1])
    best = torch.zeros(latent, dtype=torch.float64)
    best_objective = float(target.square().sum().item())
    for subset in _subsets(latent):
        columns = matrix[:, list(subset)]
        hessian = columns.T @ columns + ridge * torch.eye(len(subset), dtype=torch.float64)
        linear = columns.T @ target
        try:
            solution = torch.linalg.solve(hessian, linear)
        except RuntimeError:
            solution = torch.linalg.lstsq(hessian, linear.unsqueeze(1)).solution[:, 0]
        if bool((solution < -float(tolerance)).any()):
            continue
        solution = solution.clamp_min(0.0)
        full = torch.zeros(latent, dtype=torch.float64)
        full[list(subset)] = solution
        residual = matrix @ full - target
        objective = float((residual.square().sum() + ridge * full.square().sum()).item())
        if objective < best_objective - 1.0e-15:
            best_objective = objective
            best = full
    return best.numpy()


def select_public_regularization(
    folds: list[dict[str, np.ndarray]],
    candidates: tuple[float, ...],
) -> tuple[float, list[dict[str, float]]]:
    """Choose lambda using public folds only; private inputs are not accepted."""

    if not folds:
        raise ValueError("at least one public fold is required")
    rows = []
    for candidate in candidates:
        errors = []
        for fold in folds:
            corrected = solve_simplex_ridge(
                fold["channel"], fold["observed"], float(candidate)
            )
            truth = np.asarray(fold["truth"], dtype=np.float64)
            errors.append(float(np.abs(corrected - truth).sum()))
        rows.append(
            {
                "lambda": float(candidate),
                "median_l1": float(np.median(errors)),
                "mean_l1": float(np.mean(errors)),
            }
        )
    # A larger lambda wins an exact error tie, as frozen in the protocol.
    selected = min(rows, key=lambda row: (row["median_l1"], row["mean_l1"], -row["lambda"]))
    return float(selected["lambda"]), rows
