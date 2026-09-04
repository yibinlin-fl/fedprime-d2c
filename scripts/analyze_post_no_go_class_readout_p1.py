from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("h9", "l9")
ARMS = ("frozen", "crsf", "rawspec")
INTERVENTIONS = ("crsf", "rawspec")
CLIENTS = ((0, "ResNet10", 512), (3, "Mobilenetv2", 1280))
HALVES = ("u1", "u2")
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only P1 class-readout routing audit.")
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_k1_c_minimal_formal_seed0/extracted/outputs/cle_k1_c_minimal_seed0_formal",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT
        / "local_runs/cle_public_canonicalization_phase_b0/cle_public_canonicalization_phase_b0_seed0_inputs/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "deliverables/post_no_go_class_readout_audit_20260904",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.reshape(-1).to(dtype=torch.float64)
    y = right.reshape(-1).to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float(torch.dot(x, y) / torch.clamp_min(denominator, EPS))


def vector_cosines(left: torch.Tensor, right: torch.Tensor, *, axis: int) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=axis)
    denominator = torch.linalg.vector_norm(left, dim=axis) * torch.linalg.vector_norm(right, dim=axis)
    return numerator / torch.clamp_min(denominator, EPS)


def top_margin(matrix: torch.Tensor) -> torch.Tensor:
    values = torch.topk(matrix, k=2, dim=0).values
    return values[0] - values[1]


def ratio(values: torch.Tensor, k: int) -> float:
    return float(values[:k].sum() / torch.clamp_min(values.sum(), EPS))


def rankdata(values: Iterable[float]) -> torch.Tensor:
    data = torch.tensor(tuple(values), dtype=torch.float64)
    order = torch.argsort(data, stable=True)
    ranks = torch.empty_like(data)
    start = 0
    while start < data.numel():
        end = start + 1
        while end < data.numel() and float(data[order[end]]) == float(data[order[start]]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(left: Iterable[float], right: Iterable[float]) -> float:
    x = torch.tensor(tuple(left), dtype=torch.float64)
    y = torch.tensor(tuple(right), dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if x.numel() < 2 or float(denominator) <= EPS:
        return float("nan")
    return float(torch.dot(x, y) / denominator)


def load_classifier(
    checkpoint: Path,
    *,
    expected_dim: int,
    expected_checkpoint_sha256: str,
    expected_classifier_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != expected_checkpoint_sha256:
        raise AssertionError(f"checkpoint hash mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    classifier_keys = sorted(name for name in state if name.startswith("linear."))
    if classifier_keys != ["linear.bias", "linear.weight"]:
        raise ValueError(f"non-single-linear classifier structure: {classifier_keys}")
    classifier = {name: state[name] for name in classifier_keys}
    classifier_hash = state_dict_sha256(classifier)
    if classifier_hash != expected_classifier_sha256:
        raise AssertionError(f"classifier hash mismatch: {checkpoint}")
    weight = state["linear.weight"].detach().to(dtype=torch.float64, device="cpu")
    bias = state["linear.bias"].detach().to(dtype=torch.float64, device="cpu")
    if tuple(weight.shape) != (10, expected_dim) or tuple(bias.shape) != (10,):
        raise ValueError(f"classifier shape mismatch: {checkpoint} -> {tuple(weight.shape)}")
    centered = weight - weight.mean(dim=0, keepdim=True)
    return centered, bias, {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "classifier_keys": classifier_keys,
        "classifier_sha256": classifier_hash,
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "bias_used_in_response": False,
    }


def spectrum_bundle(
    mean: np.ndarray,
    energy: np.ndarray,
    saved_gram: np.ndarray,
    centered_weight: torch.Tensor,
) -> dict[str, object]:
    mean_t = torch.from_numpy(np.asarray(mean, dtype=np.float64))
    energy_t = torch.from_numpy(np.asarray(energy, dtype=np.float64))
    response_qd = mean_t / torch.sqrt(torch.clamp_min(energy_t, 0.0))[:, None].clamp_min(EPS)
    response = response_qd.T.contiguous()
    gram = response.T @ response
    expected_gram = torch.from_numpy(np.asarray(saved_gram, dtype=np.float64))
    gram_error = float(torch.max(torch.abs(gram - expected_gram)))
    if gram_error > 1.0e-10:
        raise AssertionError(f"normalized response does not reconstruct saved Gram: {gram_error}")

    u, singular, vh = torch.linalg.svd(response, full_matrices=False)
    eigenvalues = singular.square()
    chi = float(eigenvalues.square().sum() / torch.clamp_min(eigenvalues.sum().square(), EPS))
    gain_by_mode = (centered_weight @ u).square().sum(dim=0)
    weighted = eigenvalues * gain_by_mode
    weighted_sorted = torch.sort(weighted, descending=True).values
    chi_rw = float(weighted.square().sum() / torch.clamp_min(weighted.sum().square(), EPS))
    coupling = centered_weight @ u

    routing = centered_weight @ response
    raw_routing = centered_weight @ mean_t.T
    routing_singular = torch.linalg.svdvals(routing)
    per_class_norm = torch.linalg.vector_norm(routing, dim=1)
    margins = top_margin(routing)
    return {
        "response": response,
        "u": u,
        "singular": singular,
        "vh": vh,
        "eigenvalues": eigenvalues,
        "gain_by_mode": gain_by_mode,
        "weighted": weighted,
        "coupling": coupling,
        "routing": routing,
        "raw_routing": raw_routing,
        "routing_singular": routing_singular,
        "per_class_norm": per_class_norm,
        "top_class": torch.argmax(routing, dim=0),
        "top_abs_class": torch.argmax(torch.abs(routing), dim=0),
        "margins": margins,
        "metrics": {
            "gram_max_abs_error": gram_error,
            "chi": chi,
            "chi_rw": chi_rw,
            "response_trace": float(eigenvalues.sum()),
            "weighted_trace": float(weighted.sum()),
            "mode1_readout_gain": float(gain_by_mode[0]),
            "mean_readout_gain": float(gain_by_mode.mean()),
            "response_energy_weighted_readout_gain": float(
                weighted.sum() / torch.clamp_min(eigenvalues.sum(), EPS)
            ),
            "mode1_weighted_lambda": float(weighted[0]),
            "top1_weighted_share": ratio(weighted_sorted, 1),
            "top2_weighted_share": ratio(weighted_sorted, 2),
            "top4_weighted_share": ratio(weighted_sorted, 4),
            "top8_weighted_share": ratio(weighted_sorted, 8),
            "leading1_response_mode_weighted_share": ratio(weighted, 1),
            "leading2_response_modes_weighted_share": ratio(weighted, 2),
            "leading4_response_modes_weighted_share": ratio(weighted, 4),
            "leading8_response_modes_weighted_share": ratio(weighted, 8),
            "routing_frobenius": float(torch.linalg.vector_norm(routing)),
            "raw_routing_frobenius": float(torch.linalg.vector_norm(raw_routing)),
            "mean_top1_top2_margin": float(margins.mean()),
            "routing_top1_share": ratio(routing_singular.square(), 1),
            "routing_top2_share": ratio(routing_singular.square(), 2),
            "routing_top4_share": ratio(routing_singular.square(), 4),
            "routing_top8_share": ratio(routing_singular.square(), 8),
        },
    }


def aligned_mode_metrics(reference: dict[str, object], current: dict[str, object]) -> dict[str, float]:
    u0 = reference["u"]
    u1 = current["u"]
    mode_dots = torch.sum(u0 * u1, dim=0)
    signs = torch.where(mode_dots >= 0.0, 1.0, -1.0)
    aligned_u1 = u1 * signs[None, :]
    aligned_coupling = current["coupling"] * signs[None, :]
    mode_cosines = torch.abs(torch.sum(u0 * aligned_u1, dim=0))
    coupling_cosines = vector_cosines(reference["coupling"], aligned_coupling, axis=0)
    result: dict[str, float] = {
        "mode1_abs_cosine": float(mode_cosines[0]),
        "top4_mode_abs_cosine_mean": float(mode_cosines[:4].mean()),
        "top8_mode_abs_cosine_mean": float(mode_cosines[:8].mean()),
        "mode1_readout_coupling_cosine": float(coupling_cosines[0]),
        "top4_readout_coupling_cosine_mean": float(coupling_cosines[:4].mean()),
        "top8_readout_coupling_cosine_mean": float(coupling_cosines[:8].mean()),
    }
    for k in (2, 4, 8):
        cross = u0[:, :k].T @ u1[:, :k]
        result[f"top{k}_subspace_overlap"] = float(cross.square().sum() / k)
    return result


def routing_comparison(reference: dict[str, object], current: dict[str, object]) -> dict[str, float]:
    row_cos = vector_cosines(reference["routing"], current["routing"], axis=1)
    column_cos = vector_cosines(reference["routing"], current["routing"], axis=0)
    reference_class_norm = reference["per_class_norm"]
    current_class_norm = current["per_class_norm"]
    frozen_order = torch.argsort(reference_class_norm, descending=True)
    current_order = torch.argsort(current_class_norm, descending=True)
    return {
        "routing_matrix_cosine": tensor_cosine(reference["routing"], current["routing"]),
        "raw_routing_matrix_cosine": tensor_cosine(reference["raw_routing"], current["raw_routing"]),
        "row_cosine_mean": float(row_cos.mean()),
        "row_cosine_min": float(row_cos.min()),
        "column_cosine_mean": float(column_cos.mean()),
        "column_cosine_min": float(column_cos.min()),
        "top_class_retention": float((reference["top_class"] == current["top_class"]).to(torch.float64).mean()),
        "top_abs_class_retention": float(
            (reference["top_abs_class"] == current["top_abs_class"]).to(torch.float64).mean()
        ),
        "class_norm_vector_cosine": tensor_cosine(reference_class_norm, current_class_norm),
        "top3_class_norm_overlap": float(
            len(set(frozen_order[:3].tolist()) & set(current_order[:3].tolist())) / 3.0
        ),
        "top5_class_norm_overlap": float(
            len(set(frozen_order[:5].tolist()) & set(current_order[:5].tolist())) / 5.0
        ),
    }


def mean_by(rows: Iterable[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return float(sum(values) / len(values))


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    moments_path = formal_root / "unseen_response_moments_and_grams.npz"
    checkpoint_manifest = json.loads((formal_root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    manifest_lookup = {(row["system"], int(row["client"])): row for row in checkpoint_manifest}
    delta_root = formal_root / "surgery_block_deltas"
    dsa_rows = read_csv(formal_root / "oracle_dsa_metrics.csv")
    dsa_lookup: dict[tuple[str, str, int], float] = {}
    for row in dsa_rows:
        values = json.loads(row["dsa_client"])
        for local_client, (client_id, _architecture, _dim) in enumerate(CLIENTS):
            dsa_lookup[(row["system"], row["arm"], client_id)] = float(values[local_client])

    with np.load(moments_path, allow_pickle=False) as archive:
        moments = {name: archive[name] for name in archive.files}

    classifier_manifest: list[dict[str, object]] = []
    weighted_modes: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    class_norms: list[dict[str, object]] = []
    routing_spectrum: list[dict[str, object]] = []
    bundles: dict[tuple[str, int, str, str], dict[str, object]] = {}

    for system in SYSTEMS:
        for client_id, architecture, dimension in CLIENTS:
            manifest = manifest_lookup[(system, client_id)]
            delta_hashes: set[str] = set()
            delta_parameter_names: set[tuple[str, ...]] = set()
            for arm in INTERVENTIONS:
                delta = torch.load(delta_root / f"{system}_ab_client{client_id}_{arm}.pt", map_location="cpu")
                delta_hashes.add(str(delta["classifier_sha256"]))
                delta_parameter_names.add(tuple(sorted(delta["delta"])))
                if any(name.startswith("linear.") for name in delta["delta"]):
                    raise AssertionError("intervention delta modifies classifier")
            if len(delta_hashes) != 1:
                raise AssertionError("arm classifier hashes disagree")
            checkpoint = checkpoint_root / system / f"client_{client_id}.pt"
            centered_weight, bias, audit = load_classifier(
                checkpoint,
                expected_dim=dimension,
                expected_checkpoint_sha256=str(manifest["sha256"]),
                expected_classifier_sha256=next(iter(delta_hashes)),
            )
            audit.update(
                {
                    "system": system,
                    "client": client_id,
                    "architecture": architecture,
                    "feature_dimension": dimension,
                    "centered_weight_row_sum_max_abs": float(torch.abs(centered_weight.sum(dim=0)).max()),
                    "intervention_parameter_names": [list(names) for names in sorted(delta_parameter_names)],
                    "classifier_unchanged_across_arms": True,
                }
            )
            classifier_manifest.append(audit)

            for half in HALVES:
                for arm in ARMS:
                    prefix = f"{system}_ab_client{client_id}_{arm}"
                    bundle = spectrum_bundle(
                        moments[f"{prefix}_mean_{half}"],
                        moments[f"{prefix}_energy_{half}"],
                        moments[f"{prefix}_gram_{half}"],
                        centered_weight,
                    )
                    bundles[(system, client_id, half, arm)] = bundle
                    row = {
                        "system": system,
                        "client": client_id,
                        "architecture": architecture,
                        "half": half,
                        "arm": arm,
                        **bundle["metrics"],
                    }
                    if arm == "frozen":
                        row.update(
                            {
                                "mode1_abs_cosine": 1.0,
                                "top4_mode_abs_cosine_mean": 1.0,
                                "top8_mode_abs_cosine_mean": 1.0,
                                "mode1_readout_coupling_cosine": 1.0,
                                "top4_readout_coupling_cosine_mean": 1.0,
                                "top8_readout_coupling_cosine_mean": 1.0,
                                "top2_subspace_overlap": 1.0,
                                "top4_subspace_overlap": 1.0,
                                "top8_subspace_overlap": 1.0,
                                "routing_matrix_cosine": 1.0,
                                "raw_routing_matrix_cosine": 1.0,
                                "row_cosine_mean": 1.0,
                                "row_cosine_min": 1.0,
                                "column_cosine_mean": 1.0,
                                "column_cosine_min": 1.0,
                                "top_class_retention": 1.0,
                                "top_abs_class_retention": 1.0,
                                "class_norm_vector_cosine": 1.0,
                                "top3_class_norm_overlap": 1.0,
                                "top5_class_norm_overlap": 1.0,
                            }
                        )
                    else:
                        reference = bundles[(system, client_id, half, "frozen")]
                        row.update(aligned_mode_metrics(reference, bundle))
                        row.update(routing_comparison(reference, bundle))
                    summaries.append(row)

                    for mode_id in range(bundle["eigenvalues"].numel()):
                        weighted_modes.append(
                            {
                                "system": system,
                                "client": client_id,
                                "architecture": architecture,
                                "half": half,
                                "arm": arm,
                                "mode": mode_id + 1,
                                "lambda": float(bundle["eigenvalues"][mode_id]),
                                "readout_gain": float(bundle["gain_by_mode"][mode_id]),
                                "weighted_lambda": float(bundle["weighted"][mode_id]),
                            }
                        )
                    for class_id in range(10):
                        class_norms.append(
                            {
                                "system": system,
                                "client": client_id,
                                "architecture": architecture,
                                "half": half,
                                "arm": arm,
                                "class": class_id,
                                "routing_norm": float(bundle["per_class_norm"][class_id]),
                            }
                        )
                    routing_values = bundle["routing_singular"].square()
                    for mode_id in range(routing_values.numel()):
                        routing_spectrum.append(
                            {
                                "system": system,
                                "client": client_id,
                                "architecture": architecture,
                                "half": half,
                                "arm": arm,
                                "routing_mode": mode_id + 1,
                                "routing_eigenvalue": float(routing_values[mode_id]),
                            }
                        )

    aggregate_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for client_id, architecture, _dimension in CLIENTS:
            for arm in ARMS:
                selected = [
                    row
                    for row in summaries
                    if row["system"] == system and int(row["client"]) == client_id and row["arm"] == arm
                ]
                aggregate_rows.append(
                    {
                        "system": system,
                        "client": client_id,
                        "architecture": architecture,
                        "arm": arm,
                        **{field: mean_by(selected, field) for field in (
                            "chi",
                            "chi_rw",
                            "response_trace",
                            "weighted_trace",
                            "mode1_readout_gain",
                            "mean_readout_gain",
                            "response_energy_weighted_readout_gain",
                            "mode1_weighted_lambda",
                            "top1_weighted_share",
                            "top2_weighted_share",
                            "top4_weighted_share",
                            "top8_weighted_share",
                            "leading1_response_mode_weighted_share",
                            "leading2_response_modes_weighted_share",
                            "leading4_response_modes_weighted_share",
                            "leading8_response_modes_weighted_share",
                            "routing_frobenius",
                            "raw_routing_frobenius",
                            "mean_top1_top2_margin",
                            "routing_top1_share",
                            "routing_top2_share",
                            "routing_top4_share",
                            "routing_top8_share",
                            "routing_matrix_cosine",
                            "raw_routing_matrix_cosine",
                            "row_cosine_mean",
                            "row_cosine_min",
                            "column_cosine_mean",
                            "column_cosine_min",
                            "top_class_retention",
                            "top_abs_class_retention",
                            "class_norm_vector_cosine",
                            "top3_class_norm_overlap",
                            "top5_class_norm_overlap",
                            "mode1_abs_cosine",
                            "mode1_readout_coupling_cosine",
                            "top4_subspace_overlap",
                            "top8_subspace_overlap",
                        )},
                        "dsa": dsa_lookup[(system, arm, client_id)],
                    }
                )

    aggregate_lookup = {(row["system"], int(row["client"]), row["arm"]): row for row in aggregate_rows}
    change_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for client_id, architecture, _dimension in CLIENTS:
            frozen = aggregate_lookup[(system, client_id, "frozen")]
            for arm in INTERVENTIONS:
                current = aggregate_lookup[(system, client_id, arm)]
                change_rows.append(
                    {
                        "system": system,
                        "client": client_id,
                        "architecture": architecture,
                        "arm": arm,
                        "relative_delta_chi": (frozen["chi"] - current["chi"]) / max(abs(frozen["chi"]), EPS),
                        "relative_delta_chi_rw": (frozen["chi_rw"] - current["chi_rw"])
                        / max(abs(frozen["chi_rw"]), EPS),
                        "absolute_delta_dsa": frozen["dsa"] - current["dsa"],
                        "relative_delta_dsa": (frozen["dsa"] - current["dsa"])
                        / max(abs(frozen["dsa"]), EPS),
                        "chi_rw_to_dsa_efficiency": (
                            (frozen["dsa"] - current["dsa"]) / max(abs(frozen["dsa"]), EPS)
                        )
                        / max(
                            abs((frozen["chi_rw"] - current["chi_rw"]) / max(abs(frozen["chi_rw"]), EPS)),
                            EPS,
                        ),
                        "relative_delta_mode1_readout_gain": (
                            frozen["mode1_readout_gain"] - current["mode1_readout_gain"]
                        )
                        / max(abs(frozen["mode1_readout_gain"]), EPS),
                        "relative_delta_weighted_mean_readout_gain": (
                            frozen["response_energy_weighted_readout_gain"]
                            - current["response_energy_weighted_readout_gain"]
                        )
                        / max(abs(frozen["response_energy_weighted_readout_gain"]), EPS),
                        "routing_frobenius_ratio": current["routing_frobenius"]
                        / max(abs(frozen["routing_frobenius"]), EPS),
                        "raw_routing_frobenius_ratio": current["raw_routing_frobenius"]
                        / max(abs(frozen["raw_routing_frobenius"]), EPS),
                        "routing_matrix_cosine": current["routing_matrix_cosine"],
                        "row_cosine_mean": current["row_cosine_mean"],
                        "column_cosine_mean": current["column_cosine_mean"],
                        "top_class_retention": current["top_class_retention"],
                        "class_norm_vector_cosine": current["class_norm_vector_cosine"],
                        "top3_class_norm_overlap": current["top3_class_norm_overlap"],
                        "top5_class_norm_overlap": current["top5_class_norm_overlap"],
                        "mode1_abs_cosine": current["mode1_abs_cosine"],
                        "mode1_readout_coupling_cosine": current["mode1_readout_coupling_cosine"],
                    }
                )

    correlation_rows: list[dict[str, object]] = []
    selections = {
        "all": change_rows,
        "all_functionally_unique": [
            row for row in change_rows if not (row["system"] == "l9" and int(row["client"]) == 0)
        ],
        "crsf": [row for row in change_rows if row["arm"] == "crsf"],
        "crsf_functionally_unique": [
            row
            for row in change_rows
            if row["arm"] == "crsf" and not (row["system"] == "l9" and int(row["client"]) == 0)
        ],
        "rawspec": [row for row in change_rows if row["arm"] == "rawspec"],
        "rawspec_functionally_unique": [
            row
            for row in change_rows
            if row["arm"] == "rawspec" and not (row["system"] == "l9" and int(row["client"]) == 0)
        ],
        "resnet10": [row for row in change_rows if row["architecture"] == "ResNet10"],
        "mobilenetv2": [row for row in change_rows if row["architecture"] == "Mobilenetv2"],
    }
    for scope, selected in selections.items():
        dsa_values = [float(row["relative_delta_dsa"]) for row in selected]
        for predictor in ("relative_delta_chi", "relative_delta_chi_rw"):
            predictor_values = [float(row[predictor]) for row in selected]
            correlation_rows.append(
                {
                    "scope": scope,
                    "predictor": predictor,
                    "n": len(selected),
                    "pearson": correlation(predictor_values, dsa_values),
                    "spearman": correlation(rankdata(predictor_values), rankdata(dsa_values)),
                }
            )

    write_csv(output_dir / "readout_weighted_modes.csv", weighted_modes)
    write_csv(output_dir / "readout_weighted_summary_by_half.csv", summaries)
    write_csv(output_dir / "readout_weighted_summary.csv", aggregate_rows)
    write_csv(output_dir / "chi_chirw_dsa_changes.csv", change_rows)
    write_csv(output_dir / "chi_chirw_dsa_correlations.csv", correlation_rows)
    write_csv(output_dir / "class_routing_norms.csv", class_norms)
    write_csv(output_dir / "routing_singular_spectrum.csv", routing_spectrum)
    (output_dir / "classifier_manifest.json").write_text(
        json.dumps(classifier_manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "contexts": len(classifier_manifest),
                "summary_rows": len(aggregate_rows),
                "change_rows": change_rows,
                "gpu_used": False,
                "checkpoint_updated": False,
                "model_forward_backward": False,
                "prime_generated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
