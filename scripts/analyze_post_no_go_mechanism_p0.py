from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYSTEMS = ("h9", "l9")
ARMS = ("frozen", "crsf", "rawspec")
INTERVENTIONS = ("crsf", "rawspec")
CLIENTS = ((0, "ResNet10"), (3, "Mobilenetv2"))
EPS = 1.0e-12


@dataclass(frozen=True)
class DSAResult:
    family_effects: np.ndarray
    client_family: np.ndarray
    client: np.ndarray
    pooled: float


def family_class_contrast(probabilities: np.ndarray, operator_family_ids: np.ndarray) -> np.ndarray:
    """NumPy-only copy of the frozen Phase-A0 family contrast."""

    probs = np.asarray(probabilities, dtype=np.float64)
    family_ids = np.asarray(operator_family_ids, dtype=np.int64)
    num_families = int(family_ids.max()) + 1
    family_means = np.stack(
        [probs[:, :, family_ids == family_id, :].mean(axis=2) for family_id in range(num_families)],
        axis=1,
    )
    other_means = (family_means.sum(axis=1, keepdims=True) - family_means) / max(num_families - 1, 1)
    return family_means - other_means


def compute_dsa(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
    operator_family_ids: np.ndarray,
) -> DSAResult:
    """NumPy-only copy of the frozen Phase-A0 DSA definition."""

    contrast = family_class_contrast(probabilities, operator_family_ids)
    labels = np.asarray(labels, dtype=np.int64)
    binding = np.asarray(binding, dtype=np.int64)
    num_clients, num_families, num_sources, _num_classes = contrast.shape
    effects = np.full((num_clients, num_families, num_sources), np.nan, dtype=np.float64)
    client_family = np.full((num_clients, num_families), np.nan, dtype=np.float64)
    for client_id in range(num_clients):
        for family_id in range(num_families):
            bound_classes = np.flatnonzero(binding[client_id] == family_id)
            if bound_classes.size == 0:
                continue
            valid = ~np.isin(labels, bound_classes)
            values = contrast[client_id, family_id][:, bound_classes].sum(axis=1)
            effects[client_id, family_id, valid] = values[valid]
            if bool(valid.any()):
                client_family[client_id, family_id] = float(values[valid].mean())
    client = np.nanmean(client_family, axis=1)
    return DSAResult(effects, client_family, client, float(np.nanmean(client)))


def secondary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
    operator_family_ids: np.ndarray,
) -> dict[str, float | list[float]]:
    """NumPy-only copy of the frozen task-side reporting metrics used here."""

    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probs.argmax(axis=-1)
    client_acc = 100.0 * (predictions == labels[None, :, None]).mean(axis=(1, 2))
    num_classes = probs.shape[-1]
    num_families = int(np.max(operator_family_ids)) + 1
    cell_acc = np.full((num_classes, num_families), np.nan, dtype=np.float64)
    for class_id in range(num_classes):
        source_mask = labels == class_id
        for family_id in range(num_families):
            operator_mask = np.asarray(operator_family_ids) == family_id
            cell = predictions[:, source_mask][:, :, operator_mask]
            if cell.size:
                cell_acc[class_id, family_id] = 100.0 * (cell == class_id).mean()
    return {
        "avg_acc": float(client_acc.mean()),
        "worst_acc": float(client_acc.min()),
        "client_acc": client_acc.tolist(),
        "wcca": float(np.nanmin(cell_acc)),
        "cfg": float(np.nanmean(np.nanmax(cell_acc, axis=1) - np.nanmin(cell_acc, axis=1))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only Post-NO-GO mechanism audit P0.")
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_k1_c_minimal_formal_seed0/extracted/outputs/cle_k1_c_minimal_seed0_formal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "deliverables/post_no_go_mechanism_audit_20260904",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.dot(left, right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return numerator / max(denominator, EPS)


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(tuple(left), dtype=np.float64)
    y = np.asarray(tuple(right), dtype=np.float64)
    if x.size < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return float(
        np.dot(x_centered, y_centered)
        / max(float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered)), EPS)
    )


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    return pearson(average_ranks(np.asarray(tuple(left))), average_ranks(np.asarray(tuple(right))))


def class_routing(probabilities: np.ndarray, labels: np.ndarray, binding: np.ndarray, family_ids: np.ndarray) -> np.ndarray:
    """Return additive [client,class] terms that reproduce family-level DSA."""

    contrast = family_class_contrast(probabilities, family_ids)
    output = np.empty(binding.shape, dtype=np.float64)
    for local_client in range(binding.shape[0]):
        for class_id in range(binding.shape[1]):
            family_id = int(binding[local_client, class_id])
            family_classes = np.flatnonzero(binding[local_client] == family_id)
            valid = ~np.isin(labels, family_classes)
            output[local_client, class_id] = float(
                contrast[local_client, family_id, valid, class_id].mean()
            )

        check = compute_dsa(probabilities[None, local_client], labels, binding[None, local_client], family_ids)
        for family_id in np.unique(binding[local_client]):
            reconstructed = float(output[local_client, binding[local_client] == family_id].sum())
            expected = float(check.client_family[0, family_id])
            if not np.isclose(reconstructed, expected, atol=1.0e-12, rtol=1.0e-12):
                raise AssertionError("class-wise DSA terms do not reconstruct family DSA")
    return output


def load_predictions(formal_root: Path) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], dict[tuple[str, str], object]]:
    payloads: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    dsa: dict[tuple[str, str], object] = {}
    for system in SYSTEMS:
        for arm in ARMS:
            path = formal_root / "oracle_predictions" / f"{system}_ab_{arm}.npz"
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            payloads[(system, arm)] = payload
            dsa[(system, arm)] = compute_dsa(
                payload["probabilities"], payload["labels"], payload["binding"], payload["operator_family_ids"]
            )
    return payloads, dsa


def class_routing_audit(
    payloads: dict[tuple[str, str], dict[str, np.ndarray]],
    dsa: dict[tuple[str, str], object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    class_values: dict[tuple[str, str], np.ndarray] = {}
    for key, payload in payloads.items():
        class_values[key] = class_routing(
            payload["probabilities"], payload["labels"], payload["binding"], payload["operator_family_ids"]
        )

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for system in SYSTEMS:
        frozen = class_values[(system, "frozen")]
        binding = payloads[(system, "frozen")]["binding"]
        for arm in INTERVENTIONS:
            current = class_values[(system, arm)]
            frozen_ranks = average_ranks(-frozen.reshape(-1))
            current_ranks = average_ranks(-current.reshape(-1))
            for local_client, (client_id, architecture) in enumerate(CLIENTS):
                for class_id in range(frozen.shape[1]):
                    flat_id = local_client * frozen.shape[1] + class_id
                    rows.append(
                        {
                            "system": system,
                            "client": client_id,
                            "architecture": architecture,
                            "class": class_id,
                            "bound_family": int(binding[local_client, class_id]),
                            "intervention": arm,
                            "frozen_class_dsa": float(frozen[local_client, class_id]),
                            "intervention_class_dsa": float(current[local_client, class_id]),
                            "delta_class_dsa": float(frozen[local_client, class_id] - current[local_client, class_id]),
                            "frozen_desc_rank": float(frozen_ranks[flat_id]),
                            "intervention_desc_rank": float(current_ranks[flat_id]),
                        }
                    )
            frozen_vector = frozen.reshape(-1)
            current_vector = current.reshape(-1)
            pooled_frozen = frozen.mean(axis=0)
            pooled_current = current.mean(axis=0)
            for scope, left, right in (
                ("client_binding_20d", frozen_vector, current_vector),
                ("semantic_class_10d", pooled_frozen, pooled_current),
            ):
                order_left = np.argsort(-left)
                order_right = np.argsort(-right)
                summaries.append(
                    {
                        "system": system,
                        "intervention": arm,
                        "scope": scope,
                        "vector_cosine": cosine(left, right),
                        "rank_spearman": spearman(left, right),
                        "magnitude_ratio": float(np.linalg.norm(right) / max(np.linalg.norm(left), EPS)),
                        "top3_overlap": len(set(order_left[:3]).intersection(order_right[:3])) / 3.0,
                        "top5_overlap": len(set(order_left[:5]).intersection(order_right[:5])) / 5.0,
                        "pooled_dsa_frozen": float(dsa[(system, "frozen")].pooled),
                        "pooled_dsa_intervention": float(dsa[(system, arm)].pooled),
                    }
                )
    return rows, summaries, class_values


def coupling_audit(
    formal_root: Path,
    dsa: dict[tuple[str, str], object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = read_csv(formal_root / "taxonomy_free_metrics.csv")
    lookup = {(row["system"], int(row["client"]), row["arm"]): row for row in metrics}
    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for local_client, (client_id, architecture) in enumerate(CLIENTS):
            chi_f = float(lookup[(system, client_id, "frozen")]["chi_unseen"])
            dsa_f = float(dsa[(system, "frozen")].client[local_client])
            for arm in INTERVENTIONS:
                chi_i = float(lookup[(system, client_id, arm)]["chi_unseen"])
                dsa_i = float(dsa[(system, arm)].client[local_client])
                relative_delta_chi = (chi_f - chi_i) / max(abs(chi_f), EPS)
                relative_delta_dsa = (dsa_f - dsa_i) / max(abs(dsa_f), EPS)
                rows.append(
                    {
                        "level": "client",
                        "system": system,
                        "client": client_id,
                        "architecture": architecture,
                        "intervention": arm,
                        "relative_delta_chi": relative_delta_chi,
                        "absolute_delta_dsa": dsa_f - dsa_i,
                        "relative_delta_dsa": relative_delta_dsa,
                        "chi_to_dsa_efficiency": relative_delta_dsa / max(abs(relative_delta_chi), EPS),
                    }
                )
        for arm in INTERVENTIONS:
            chi_f = np.mean([float(lookup[(system, c, "frozen")]["chi_unseen"]) for c, _ in CLIENTS])
            chi_i = np.mean([float(lookup[(system, c, arm)]["chi_unseen"]) for c, _ in CLIENTS])
            dsa_f = float(dsa[(system, "frozen")].pooled)
            dsa_i = float(dsa[(system, arm)].pooled)
            relative_delta_chi = (chi_f - chi_i) / max(abs(chi_f), EPS)
            relative_delta_dsa = (dsa_f - dsa_i) / max(abs(dsa_f), EPS)
            rows.append(
                {
                    "level": "pooled",
                    "system": system,
                    "client": -1,
                    "architecture": "Pooled",
                    "intervention": arm,
                    "relative_delta_chi": relative_delta_chi,
                    "absolute_delta_dsa": dsa_f - dsa_i,
                    "relative_delta_dsa": relative_delta_dsa,
                    "chi_to_dsa_efficiency": relative_delta_dsa / max(abs(relative_delta_chi), EPS),
                }
            )

    client_rows = [row for row in rows if row["level"] == "client"]
    correlation_rows: list[dict[str, object]] = []
    selections = {
        "client_all": client_rows,
        "client_crsf": [row for row in client_rows if row["intervention"] == "crsf"],
        "client_rawspec": [row for row in client_rows if row["intervention"] == "rawspec"],
        "client_resnet10": [row for row in client_rows if row["architecture"] == "ResNet10"],
        "client_mobilenetv2": [row for row in client_rows if row["architecture"] == "Mobilenetv2"],
        "pooled_system_arm": [row for row in rows if row["level"] == "pooled"],
    }
    for scope, selected in selections.items():
        x = [float(row["relative_delta_chi"]) for row in selected]
        y = [float(row["relative_delta_dsa"]) for row in selected]
        correlation_rows.append(
            {"scope": scope, "n": len(selected), "pearson": pearson(x, y), "spearman": spearman(x, y)}
        )
    return rows, correlation_rows


def spectrum_metrics(gram: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    matrix = 0.5 * (np.asarray(gram, dtype=np.float64) + np.asarray(gram, dtype=np.float64).T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values[::-1], 0.0)
    vector = vectors[:, -1]
    trace = float(values.sum())
    chi = float(np.square(values).sum() / (trace * trace + EPS))
    result = {
        "trace": trace,
        "lambda1": float(values[0]),
        "lambda1_over_trace": float(values[0] / max(trace, EPS)),
        "top2_over_trace": float(values[:2].sum() / max(trace, EPS)),
        "top4_over_trace": float(values[:4].sum() / max(trace, EPS)),
        "top8_over_trace": float(values[:8].sum() / max(trace, EPS)),
        "effective_rank_participation": float(1.0 / max(chi, EPS)),
        "chi": chi,
        "tail_sum": float(values[1:].sum()),
    }
    return result, vector


def spectral_autopsy(formal_root: Path) -> list[dict[str, object]]:
    data = np.load(formal_root / "unseen_response_moments_and_grams.npz", allow_pickle=False)
    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for client_id, architecture in CLIENTS:
            context = f"{system}_ab_client{client_id}"
            for half in ("u1", "u2"):
                frozen, frozen_vector = spectrum_metrics(data[f"{context}_frozen_gram_{half}"])
                for arm in ARMS:
                    current, current_vector = spectrum_metrics(data[f"{context}_{arm}_gram_{half}"])
                    rows.append(
                        {
                            "system": system,
                            "client": client_id,
                            "architecture": architecture,
                            "half": half,
                            "arm": arm,
                            **current,
                            "principal_vector_abs_cosine_vs_frozen": 1.0
                            if arm == "frozen"
                            else abs(cosine(frozen_vector, current_vector)),
                            "trace_relative_change_vs_frozen": (current["trace"] - frozen["trace"])
                            / max(abs(frozen["trace"]), EPS),
                            "lambda1_relative_change_vs_frozen": (current["lambda1"] - frozen["lambda1"])
                            / max(abs(frozen["lambda1"]), EPS),
                            "tail_relative_change_vs_frozen": (current["tail_sum"] - frozen["tail_sum"])
                            / max(abs(frozen["tail_sum"]), EPS),
                            "top1_share_change_vs_frozen": current["lambda1_over_trace"]
                            - frozen["lambda1_over_trace"],
                        }
                    )
    return rows


def architecture_audit(
    formal_root: Path,
    dsa: dict[tuple[str, str], object],
) -> list[dict[str, object]]:
    metrics = read_csv(formal_root / "taxonomy_free_metrics.csv")
    lookup = {(row["system"], int(row["client"]), row["arm"]): row for row in metrics}
    payloads, _ = load_predictions(formal_root)
    task: dict[tuple[str, str, int], dict[str, float]] = {}
    for system in SYSTEMS:
        for arm in ARMS:
            payload = payloads[(system, arm)]
            for local_client, (client_id, _architecture) in enumerate(CLIENTS):
                result = secondary_metrics(
                    payload["probabilities"][None, local_client],
                    payload["labels"],
                    payload["binding"][None, local_client],
                    payload["operator_family_ids"],
                )
                task[(system, arm, client_id)] = {name: float(result[name]) for name in ("cfg", "wcca")}

    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for local_client, (client_id, architecture) in enumerate(CLIENTS):
            chi_f = float(lookup[(system, client_id, "frozen")]["chi_unseen"])
            dsa_f = float(dsa[(system, "frozen")].client[local_client])
            chi_reductions = {
                arm: (chi_f - float(lookup[(system, client_id, arm)]["chi_unseen"])) / max(abs(chi_f), EPS)
                for arm in INTERVENTIONS
            }
            dsa_reductions = {
                arm: dsa_f - float(dsa[(system, arm)].client[local_client]) for arm in INTERVENTIONS
            }
            for arm in INTERVENTIONS:
                rows.append(
                    {
                        "system": system,
                        "client": client_id,
                        "architecture": architecture,
                        "intervention": arm,
                        "relative_delta_chi": chi_reductions[arm],
                        "absolute_delta_dsa": dsa_reductions[arm],
                        "relative_delta_dsa": dsa_reductions[arm] / max(abs(dsa_f), EPS),
                        "delta_cfg": task[(system, "frozen", client_id)]["cfg"] - task[(system, arm, client_id)]["cfg"],
                        "delta_wcca": task[(system, arm, client_id)]["wcca"] - task[(system, "frozen", client_id)]["wcca"],
                        "crsf_minus_rawspec_chi_advantage": chi_reductions["crsf"] - chi_reductions["rawspec"]
                        if arm == "crsf"
                        else "",
                        "crsf_minus_rawspec_dsa_advantage": dsa_reductions["crsf"] - dsa_reductions["rawspec"]
                        if arm == "crsf"
                        else "",
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root.resolve()
    output_dir = args.output_dir.resolve()
    required = (
        formal_root / "oracle_predictions",
        formal_root / "taxonomy_free_metrics.csv",
        formal_root / "unseen_response_moments_and_grams.npz",
    )
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing K1-C-Minimal artifacts: {missing}")

    payloads, dsa = load_predictions(formal_root)
    class_rows, routing_summaries, _class_values = class_routing_audit(payloads, dsa)
    coupling_rows, correlation_rows = coupling_audit(formal_root, dsa)
    spectral_rows = spectral_autopsy(formal_root)
    architecture_rows = architecture_audit(formal_root, dsa)

    write_csv(output_dir / "class_routing_retention.csv", class_rows)
    write_csv(output_dir / "class_routing_summary.csv", routing_summaries)
    write_csv(output_dir / "chi_dsa_coupling.csv", coupling_rows)
    write_csv(output_dir / "chi_dsa_correlations.csv", correlation_rows)
    write_csv(output_dir / "spectral_autopsy.csv", spectral_rows)
    write_csv(output_dir / "architecture_leverage.csv", architecture_rows)
    print(
        json.dumps(
            {
                "class_rows": len(class_rows),
                "routing_summaries": routing_summaries,
                "correlations": correlation_rows,
                "spectral_rows": len(spectral_rows),
                "architecture_rows": len(architecture_rows),
                "gpu_used": False,
                "model_forward_backward": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
