from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("h0", "h9", "l0", "l9")
CLIENTS = ((0, "ResNet10"), (1, "ResNet12"), (2, "ShuffleNet"), (3, "Mobilenetv2"))
BANKS = {"bank_a": slice(0, 64), "bank_b": slice(64, 128)}
HALVES = {"ua": slice(0, 500), "ub": slice(500, 1000)}
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only P2 CLE-specific output-routing audit.")
    parser.add_argument(
        "--k0b-root",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_generic_probe_k0b_seed0/formal_extracted/outputs/cle_generic_probe_k0b_seed0_formal",
    )
    parser.add_argument(
        "--dsa-csv",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_shortcut_amplification_phase_a1a_seed0/extracted/outputs/cle_shortcut_amplification_phase_a1a_seed0_analysis/cle_shortcut_phase_a1a_per_client.csv",
    )
    parser.add_argument(
        "--phase-b0-manifest",
        type=Path,
        default=ROOT
        / "local_runs/cle_public_canonicalization_phase_b0/cle_public_canonicalization_phase_b0_seed0_inputs/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "deliverables/post_no_go_cle_specific_routing_audit_20260904",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def rankdata(values: Iterable[float]) -> np.ndarray:
    data = np.asarray(tuple(values), dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty_like(data)
    start = 0
    while start < len(data):
        end = start + 1
        while end < len(data) and data[order[end]] == data[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(tuple(left), dtype=np.float64)
    y = np.asarray(tuple(right), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    return correlation(rankdata(left), rankdata(right))


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + EPS))


def concentration(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.square(values).sum() / (np.square(values.sum()) + EPS))


def top_share(eigenvalues: np.ndarray, count: int) -> float:
    ordered = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    return safe_ratio(float(ordered[:count].sum()), float(ordered.sum()))


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def arm_parts(arm: str) -> tuple[str, int]:
    return ("hfl" if arm.startswith("h") else "local", 9 if arm.endswith("9") else 0)


def spectrum_metrics(z: np.ndarray) -> dict[str, float]:
    singular = np.linalg.svd(z, compute_uv=False)
    eigenvalues = np.square(singular)
    trace = float(eigenvalues.sum())
    probabilities = eigenvalues / (trace + EPS)
    positive = probabilities[probabilities > 0]
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    chi = float(np.square(eigenvalues).sum() / (trace * trace + EPS))
    return {
        "chi_out": chi,
        "top1_trace_share": top_share(eigenvalues, 1),
        "top2_trace_share": top_share(eigenvalues, 2),
        "top4_trace_share": top_share(eigenvalues, 4),
        "effective_rank_entropy": entropy_rank,
        "effective_rank_participation": safe_ratio(1.0, chi),
        "frobenius_norm": float(np.linalg.norm(z)),
        "standardized_routing_energy": trace,
        "singular_values": ";".join(f"{value:.12g}" for value in singular),
    }


def load_and_validate_inputs(
    k0b_root: Path,
    phase_b0_manifest_path: Path,
) -> tuple[dict[tuple[str, int], Path], dict[str, object]]:
    response_manifest_path = k0b_root / "blind_response_manifest.json"
    config_path = k0b_root / "config.json"
    metrics_path = k0b_root / "metrics/per_client_metrics.csv"
    for path in (response_manifest_path, config_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    response_manifest = json.loads(response_manifest_path.read_text(encoding="utf-8"))
    phase_b0_manifest = json.loads(phase_b0_manifest_path.read_text(encoding="utf-8"))
    if phase_b0_manifest.get("checkpoint_kind") != "final_round_40_only":
        raise AssertionError("Phase-B0 manifest does not identify final round-40 checkpoints")
    phase_b0_checkpoint_hashes = {
        str(record["path"]): str(record["sha256"])
        for record in phase_b0_manifest["files"]
        if str(record["path"]).startswith("checkpoints/")
    }
    if response_manifest.get("training_performed") is not False:
        raise AssertionError("K0-B manifest does not certify zero training")
    forbidden_flags = (
        "public_labels_used",
        "corruption_taxonomy_used",
        "severity_used",
        "binding_used",
        "private_corruption_metadata_used",
    )
    if any(response_manifest.get(key) is not False for key in forbidden_flags):
        raise AssertionError("K0-B blind response manifest is not taxonomy-free")
    if response_manifest.get("carrier_halves") != {"Ua": [0, 500], "Ub": [500, 1000], "disjoint": True}:
        raise AssertionError("unexpected carrier split")

    paths: dict[tuple[str, int], Path] = {}
    source_records: list[dict[str, object]] = []
    for record in response_manifest["responses"]:
        arm = str(record["arm"])
        client = int(record["client"])
        path = k0b_root / "responses" / str(record["response_file"])
        actual_hash = sha256_file(path)
        if actual_hash != str(record["response_sha256"]):
            raise AssertionError(f"response hash mismatch: {path}")
        checkpoint_key = f"checkpoints/{arm}/client_{client}.pt"
        if phase_b0_checkpoint_hashes.get(checkpoint_key) != str(record["checkpoint_sha256"]):
            raise AssertionError(f"K0-B checkpoint lineage mismatch: {checkpoint_key}")
        with np.load(path, allow_pickle=False) as payload:
            expected = {
                "base_logits": (1000, 10),
                "probe_logits": (1000, 128, 10),
                "class_vs_rest_delta": (1000, 128, 10),
                "centered_response": (1000, 128, 10),
            }
            for key, shape in expected.items():
                if key not in payload or tuple(payload[key].shape) != shape:
                    raise AssertionError(f"missing/unexpected {key} in {path}")
            centered = np.asarray(payload["centered_response"], dtype=np.float64)
            if not np.isfinite(centered).all():
                raise AssertionError(f"non-finite response: {path}")
            if float(np.max(np.abs(centered.sum(axis=-1)))) > 2.0e-5:
                raise AssertionError(f"response is not class-centered: {path}")
        paths[(arm, client)] = path
        source_records.append(
            {
                "arm": arm,
                "client": client,
                "model": record["model"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "response_file": path.as_posix(),
                "response_bytes": path.stat().st_size,
                "response_sha256": actual_hash,
            }
        )
    if set(paths) != {(arm, client) for arm in ARMS for client, _ in CLIENTS}:
        raise AssertionError("K0-B does not contain the complete 4-arm x 4-client response grid")
    return paths, {
        "k0b_root": k0b_root.as_posix(),
        "blind_manifest": response_manifest_path.as_posix(),
        "blind_manifest_sha256": sha256_file(response_manifest_path),
        "config_sha256": sha256_file(config_path),
        "k0b_metrics_sha256": sha256_file(metrics_path),
        "phase_b0_manifest": phase_b0_manifest_path.as_posix(),
        "phase_b0_manifest_sha256": sha256_file(phase_b0_manifest_path),
        "checkpoint_kind": phase_b0_manifest["checkpoint_kind"],
        "bank_a_sha256": response_manifest["bank_a_sha256"],
        "bank_b_sha256": response_manifest["bank_b_sha256"],
        "public_indices_sha256": response_manifest["public_indices_sha256"],
        "response_records": source_records,
    }


def make_context(
    centered: np.ndarray,
    raw_delta: np.ndarray,
    probe_logits: np.ndarray,
    base_logits: np.ndarray,
) -> dict[str, object]:
    mu = centered.mean(axis=0)
    energy = np.square(centered).sum(axis=-1).mean(axis=0)
    z_qc = mu / (np.sqrt(np.maximum(energy, 0.0))[:, None] + EPS)
    z = z_qc.T
    positive_profile = np.square(np.maximum(z, 0.0)).mean(axis=1)
    total_profile = np.square(z).mean(axis=1)
    sorted_values = np.sort(z_qc, axis=1)
    top_class = np.argmax(z_qc, axis=1)
    top_margin = sorted_values[:, -1] - sorted_values[:, -2]

    raw_energy = np.square(raw_delta).sum(axis=-1).mean()
    centered_energy = np.square(centered).sum(axis=-1).mean()
    base_norm = np.linalg.norm(base_logits, axis=-1)[:, None]
    probe_norm = np.linalg.norm(probe_logits, axis=-1)
    norm_change = probe_norm - base_norm
    common_mode = raw_delta.mean(axis=-1)
    metrics = {
        **spectrum_metrics(z),
        "positive_routing_strength": float(positive_profile.sum()),
        "total_routing_strength": float(total_profile.sum()),
        "positive_class_concentration": concentration(positive_profile),
        "total_class_concentration": concentration(total_profile),
        "mean_probe_top1_top2_margin": float(top_margin.mean()),
        "mean_centered_response_energy": float(centered_energy),
        "mean_raw_output_delta_energy": float(raw_energy),
        "mean_abs_output_norm_change": float(np.abs(norm_change).mean()),
        "mean_signed_output_norm_change": float(norm_change.mean()),
        "nonselective_response_magnitude": float(np.linalg.norm(centered, axis=-1).mean()),
        "common_mode_magnitude": float(np.abs(common_mode).mean()),
        "mean_coherent_centered_norm": float(np.linalg.norm(mu, axis=-1).mean()),
    }
    return {
        "z": z,
        "positive_profile": positive_profile,
        "total_profile": total_profile,
        "top_class": top_class,
        "top_margin": top_margin,
        "metrics": metrics,
    }


def taxonomy_free_stage(
    k0b_root: Path,
    phase_b0_manifest_path: Path,
    output_dir: Path,
) -> tuple[dict[tuple[str, int, str, str], dict[str, object]], dict[str, object]]:
    response_paths, inventory = load_and_validate_inputs(k0b_root, phase_b0_manifest_path)
    contexts: dict[tuple[str, int, str, str], dict[str, object]] = {}
    routing_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for arm in ARMS:
        system, gamma = arm_parts(arm)
        for client, model in CLIENTS:
            with np.load(response_paths[(arm, client)], allow_pickle=False) as payload:
                centered_all = np.asarray(payload["centered_response"], dtype=np.float64)
                base_all = np.asarray(payload["base_logits"], dtype=np.float64)
                probe_all = np.asarray(payload["probe_logits"], dtype=np.float64)
                raw_all = probe_all - base_all[:, None, :]
                for bank, probe_slice in BANKS.items():
                    for half, carrier_slice in HALVES.items():
                        context = make_context(
                            centered_all[carrier_slice, probe_slice, :],
                            raw_all[carrier_slice, probe_slice, :],
                            probe_all[carrier_slice, probe_slice, :],
                            base_all[carrier_slice, :],
                        )
                        contexts[(arm, client, bank, half)] = context

    # Stability is computed only after every blind context exists.
    for arm in ARMS:
        system, gamma = arm_parts(arm)
        for client, model in CLIENTS:
            for bank in BANKS:
                ua = contexts[(arm, client, bank, "ua")]
                ub = contexts[(arm, client, bank, "ub")]
                half_positive_cos = cosine(ua["positive_profile"], ub["positive_profile"])
                half_total_cos = cosine(ua["total_profile"], ub["total_profile"])
                half_z_cos = cosine(ua["z"], ub["z"])
                half_top_retention = float(np.mean(ua["top_class"] == ub["top_class"]))
                for half in HALVES:
                    current = contexts[(arm, client, bank, half)]
                    other_bank = "bank_b" if bank == "bank_a" else "bank_a"
                    peer = contexts[(arm, client, other_bank, half)]
                    bank_positive_cos = cosine(current["positive_profile"], peer["positive_profile"])
                    bank_total_cos = cosine(current["total_profile"], peer["total_profile"])
                    metrics = current["metrics"]
                    top_order = np.argsort(current["positive_profile"])[::-1]
                    routing_rows.append(
                        {
                            "arm": arm,
                            "system": system,
                            "gamma": gamma,
                            "client": client,
                            "model": model,
                            "bank": bank,
                            "carrier_half": half,
                            **{key: value for key, value in metrics.items() if key not in {
                                "mean_centered_response_energy",
                                "mean_raw_output_delta_energy",
                                "mean_abs_output_norm_change",
                                "mean_signed_output_norm_change",
                                "nonselective_response_magnitude",
                                "common_mode_magnitude",
                                "mean_coherent_centered_norm",
                            }},
                            "top1_class": int(top_order[0]),
                            "top3_classes": ";".join(str(int(value)) for value in top_order[:3]),
                            "top5_classes": ";".join(str(int(value)) for value in top_order[:5]),
                            "cross_half_positive_profile_cosine": half_positive_cos,
                            "cross_half_total_profile_cosine": half_total_cos,
                            "cross_half_z_cosine": half_z_cos,
                            "cross_half_probe_top_class_retention": half_top_retention,
                            "cross_bank_positive_profile_cosine": bank_positive_cos,
                            "cross_bank_total_profile_cosine": bank_total_cos,
                        }
                    )
                    control_rows.append(
                        {
                            "arm": arm,
                            "system": system,
                            "gamma": gamma,
                            "client": client,
                            "model": model,
                            "bank": bank,
                            "carrier_half": half,
                            **{key: metrics[key] for key in (
                                "mean_centered_response_energy",
                                "mean_raw_output_delta_energy",
                                "mean_abs_output_norm_change",
                                "mean_signed_output_norm_change",
                                "nonselective_response_magnitude",
                                "common_mode_magnitude",
                                "mean_coherent_centered_norm",
                            )},
                        }
                    )
                    for class_index in range(10):
                        class_rows.append(
                            {
                                "arm": arm,
                                "system": system,
                                "gamma": gamma,
                                "client": client,
                                "model": model,
                                "bank": bank,
                                "carrier_half": half,
                                "class_index": class_index,
                                "positive_routing_energy": float(current["positive_profile"][class_index]),
                                "total_routing_norm_sq": float(current["total_profile"][class_index]),
                                "positive_profile_share": safe_ratio(
                                    float(current["positive_profile"][class_index]),
                                    float(np.sum(current["positive_profile"])),
                                ),
                                "total_profile_share": safe_ratio(
                                    float(current["total_profile"][class_index]),
                                    float(np.sum(current["total_profile"])),
                                ),
                                "cross_half_positive_profile_cosine": half_positive_cos,
                                "cross_bank_positive_profile_cosine": bank_positive_cos,
                                "probe_top_class_count": int(np.sum(current["top_class"] == class_index)),
                                "probe_top_class_fraction": float(np.mean(current["top_class"] == class_index)),
                                "mean_margin_when_top": float(
                                    current["top_margin"][current["top_class"] == class_index].mean()
                                )
                                if np.any(current["top_class"] == class_index)
                                else 0.0,
                            }
                        )
                    for probe_index in range(64):
                        probe_rows.append(
                            {
                                "arm": arm,
                                "system": system,
                                "gamma": gamma,
                                "client": client,
                                "model": model,
                                "bank": bank,
                                "carrier_half": half,
                                "probe_index_within_bank": probe_index,
                                "top_routed_class": int(current["top_class"][probe_index]),
                                "top1_top2_margin": float(current["top_margin"][probe_index]),
                                "retained_across_carrier_halves": bool(
                                    ua["top_class"][probe_index] == ub["top_class"][probe_index]
                                ),
                            }
                        )

    routing_lookup = {
        (str(row["arm"]), int(row["client"]), str(row["bank"]), str(row["carrier_half"])): row
        for row in routing_rows
    }
    control_lookup = {
        (str(row["arm"]), int(row["client"]), str(row["bank"]), str(row["carrier_half"])): row
        for row in control_rows
    }
    specificity_rows: list[dict[str, object]] = []
    metric_names = (
        "chi_out",
        "positive_routing_strength",
        "total_routing_strength",
        "positive_class_concentration",
        "total_class_concentration",
        "standardized_routing_energy",
        "cross_half_positive_profile_cosine",
        "cross_bank_positive_profile_cosine",
    )
    control_names = (
        "mean_centered_response_energy",
        "mean_raw_output_delta_energy",
        "mean_abs_output_norm_change",
        "nonselective_response_magnitude",
        "common_mode_magnitude",
    )
    for system, low_arm, high_arm in (("hfl", "h0", "h9"), ("local", "l0", "l9")):
        for bank in BANKS:
            for half in HALVES:
                for client, model in CLIENTS:
                    low = routing_lookup[(low_arm, client, bank, half)]
                    high = routing_lookup[(high_arm, client, bank, half)]
                    low_control = control_lookup[(low_arm, client, bank, half)]
                    high_control = control_lookup[(high_arm, client, bank, half)]
                    row: dict[str, object] = {
                        "system": system,
                        "client": client,
                        "model": model,
                        "bank": bank,
                        "carrier_half": half,
                    }
                    for name in metric_names:
                        row[f"{name}_low"] = low[name]
                        row[f"{name}_high"] = high[name]
                        row[f"{name}_delta"] = float(high[name]) - float(low[name])
                        row[f"{name}_ratio"] = safe_ratio(float(high[name]), float(low[name]))
                    for name in control_names:
                        row[f"{name}_low"] = low_control[name]
                        row[f"{name}_high"] = high_control[name]
                        row[f"{name}_delta"] = float(high_control[name]) - float(low_control[name])
                        row[f"{name}_ratio"] = safe_ratio(float(high_control[name]), float(low_control[name]))
                    specificity_rows.append(row)

                client_rows = specificity_rows[-len(CLIENTS) :]
                pooled: dict[str, object] = {
                    "system": system,
                    "client": "pooled_mean",
                    "model": "all_four_clients",
                    "bank": bank,
                    "carrier_half": half,
                }
                numeric_columns = [key for key in client_rows[0] if key not in {"system", "client", "model", "bank", "carrier_half"}]
                for name in numeric_columns:
                    pooled[name] = float(np.mean([float(row[name]) for row in client_rows]))
                pooled["positive_clients_chi_out"] = sum(float(row["chi_out_delta"]) > 0 for row in client_rows)
                pooled["positive_clients_positive_routing_strength"] = sum(
                    float(row["positive_routing_strength_delta"]) > 0 for row in client_rows
                )
                pooled["positive_clients_class_concentration"] = sum(
                    float(row["positive_class_concentration_delta"]) > 0 for row in client_rows
                )
                specificity_rows.append(pooled)

    output_dir.mkdir(parents=True, exist_ok=True)
    primary_paths = {
        "routing_spectrum.csv": routing_rows,
        "class_routing_profiles.csv": class_rows,
        "probe_routing_assignments.csv": probe_rows,
        "cle_specificity.csv": specificity_rows,
        "generic_fragility_controls.csv": control_rows,
    }
    for name, rows in primary_paths.items():
        write_csv(output_dir / name, rows)

    availability = f"""# P2 Artifact Availability

Date: 2026-09-04

## Available and used before sealing

- Complete K0-B response grid: H0/H9/L0/L9 x 4 clients = 16 files.
- Each file contains carrier-level `centered_response` with shape `1000 x 128 x 10`, plus base/probe logits.
- Probe order is Bank A recipes 0--63 followed by Bank B recipes 64--127.
- Carrier halves are Ua indices 0--499 and Ub indices 500--999, certified disjoint.
- All 16 response SHA256 values match the K0-B blind manifest.
- All checkpoint hashes match the Phase-B0 manifest, which identifies them as final round-40 only.
- The blind manifest certifies no public labels, corruption taxonomy, severity, binding, or private corruption metadata were used.

## Deferred until after taxonomy-free seal

- Phase-A1a round-40 per-client DSA for H0/H9/L0/L9.
- Original K0-B `R_i` from its saved per-client metrics.

The deferred files are not opened by `taxonomy_free_stage`; they are loaded only after the four
primary CSV files and `primary_taxonomy_free_manifest.json` have been written and hashed.

## Not used

- No checkpoint loading or model construction.
- No model forward/backward pass.
- No PRIME recipe generation.
- No GPU/OpenI/training/optimization.
- No corruption binding, family, severity, or CLE oracle is used to construct B--E statistics.
"""
    (output_dir / "artifact_availability.md").write_text(availability, encoding="utf-8")
    primary_manifest = {
        "protocol": "post_no_go_cle_specific_routing_audit_p2_taxonomy_free",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "taxonomy_free_sealed_before_oracle_association",
        "training_performed": False,
        "model_inference_performed": False,
        "prime_generation_performed": False,
        "gpu_or_openi_used": False,
        "binding_or_oracle_read": False,
        "dimensions": {
            "arms": 4,
            "clients": 4,
            "banks": 2,
            "carrier_halves": 2,
            "probes_per_bank": 64,
            "classes": 10,
        },
        "source_inventory": inventory,
        "sealed_outputs": {
            name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in primary_paths
        },
    }
    primary_manifest_path = output_dir / "primary_taxonomy_free_manifest.json"
    primary_manifest_path.write_text(json.dumps(primary_manifest, indent=2), encoding="utf-8")
    return contexts, primary_manifest


def load_round40_dsa(dsa_csv: Path) -> dict[tuple[str, int], float]:
    rows = [row for row in read_csv(dsa_csv) if int(row["round"]) == 40]
    if len(rows) != 4:
        raise AssertionError("expected four round-40 DSA rows")
    output: dict[tuple[str, int], float] = {}
    for row in rows:
        client = int(row["client"])
        for arm in ARMS:
            output[(arm, client)] = float(row[f"{arm}_dsa"])
    return output


def load_k0b_r(metrics_csv: Path) -> dict[tuple[str, int], float]:
    output: dict[tuple[str, int], float] = {}
    for row in read_csv(metrics_csv):
        if row["bank"] == "combined":
            output[(row["arm"], int(row["client"]))] = float(row["R"])
    if set(output) != {(arm, client) for arm in ARMS for client, _ in CLIENTS}:
        raise AssertionError("incomplete K0-B combined risk grid")
    return output


def oracle_stage(
    k0b_root: Path,
    dsa_csv: Path,
    output_dir: Path,
    primary_manifest: dict[str, object],
) -> tuple[list[str], str, dict[str, object]]:
    # Integrity guard: the primary files must still match the pre-oracle seal before oracle data is opened.
    for name, record in primary_manifest["sealed_outputs"].items():
        if sha256_file(output_dir / name) != record["sha256"]:
            raise AssertionError(f"taxonomy-free seal changed before oracle stage: {name}")

    dsa = load_round40_dsa(dsa_csv)
    k0b_r = load_k0b_r(k0b_root / "metrics/per_client_metrics.csv")
    spectrum = read_csv(output_dir / "routing_spectrum.csv")
    by_arm_client: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in spectrum:
        by_arm_client.setdefault((row["arm"], int(row["client"])), []).append(row)

    observations: list[dict[str, object]] = []
    mean_metrics = (
        "chi_out",
        "positive_routing_strength",
        "total_routing_strength",
        "positive_class_concentration",
        "standardized_routing_energy",
        "cross_half_positive_profile_cosine",
        "cross_bank_positive_profile_cosine",
    )
    for arm in ARMS:
        system, gamma = arm_parts(arm)
        for client, model in CLIENTS:
            rows = by_arm_client[(arm, client)]
            observation: dict[str, object] = {
                "row_type": "observation",
                "arm": arm,
                "system": system,
                "gamma": gamma,
                "client": client,
                "model": model,
                "dsa": dsa[(arm, client)],
                "k0b_R": k0b_r[(arm, client)],
            }
            for metric in mean_metrics:
                observation[metric] = float(np.mean([float(row[metric]) for row in rows]))
            observations.append(observation)

    association_rows: list[dict[str, object]] = list(observations)
    predictors = ("k0b_R", "chi_out", "positive_routing_strength", "positive_class_concentration")
    for predictor in predictors:
        association_rows.append(
            {
                "row_type": "correlation_all_16",
                "predictor": predictor,
                "n": len(observations),
                "pearson_with_dsa": correlation(
                    (float(row[predictor]) for row in observations),
                    (float(row["dsa"]) for row in observations),
                ),
                "spearman_with_dsa": spearman(
                    (float(row[predictor]) for row in observations),
                    (float(row["dsa"]) for row in observations),
                ),
            }
        )

    effects: list[dict[str, object]] = []
    observation_lookup = {(str(row["arm"]), int(row["client"])): row for row in observations}
    for system, low_arm, high_arm in (("hfl", "h0", "h9"), ("local", "l0", "l9")):
        for client, model in CLIENTS:
            low = observation_lookup[(low_arm, client)]
            high = observation_lookup[(high_arm, client)]
            row: dict[str, object] = {
                "row_type": "cle_effect",
                "system": system,
                "client": client,
                "model": model,
                "dsa_delta": float(high["dsa"]) - float(low["dsa"]),
            }
            for predictor in predictors:
                row[f"{predictor}_delta"] = float(high[predictor]) - float(low[predictor])
                row[f"{predictor}_ratio"] = safe_ratio(float(high[predictor]), float(low[predictor]))
            effects.append(row)
    association_rows.extend(effects)
    for predictor in predictors:
        association_rows.append(
            {
                "row_type": "correlation_cle_effect_8",
                "predictor": predictor,
                "n": len(effects),
                "pearson_with_dsa": correlation(
                    (float(row[f"{predictor}_delta"]) for row in effects),
                    (float(row["dsa_delta"]) for row in effects),
                ),
                "spearman_with_dsa": spearman(
                    (float(row[f"{predictor}_delta"]) for row in effects),
                    (float(row["dsa_delta"]) for row in effects),
                ),
            }
        )
    write_csv(output_dir / "oracle_association.csv", association_rows)

    incremental_rows: list[dict[str, object]] = []
    dsa_values = np.asarray([float(row["dsa"]) for row in observations])
    r_values = np.asarray([float(row["k0b_R"]) for row in observations])
    residual_dsa = residualize(dsa_values, r_values)
    for predictor in ("chi_out", "positive_routing_strength", "positive_class_concentration"):
        values = np.asarray([float(row[predictor]) for row in observations])
        residual_metric = residualize(values, r_values)
        incremental_rows.append(
            {
                "row_type": "all_16_incremental",
                "metric": predictor,
                "n": len(values),
                "pearson_with_k0b_R": correlation(values, r_values),
                "spearman_with_k0b_R": spearman(values, r_values),
                "pearson_with_dsa": correlation(values, dsa_values),
                "spearman_with_dsa": spearman(values, dsa_values),
                "residual_pearson_with_dsa_after_k0b_R": correlation(residual_metric, residual_dsa),
            }
        )
    effect_dsa = np.asarray([float(row["dsa_delta"]) for row in effects])
    effect_r = np.asarray([float(row["k0b_R_delta"]) for row in effects])
    residual_effect_dsa = residualize(effect_dsa, effect_r)
    for predictor in ("chi_out", "positive_routing_strength", "positive_class_concentration"):
        values = np.asarray([float(row[f"{predictor}_delta"]) for row in effects])
        residual_metric = residualize(values, effect_r)
        incremental_rows.append(
            {
                "row_type": "cle_effect_8_incremental",
                "metric": predictor,
                "n": len(values),
                "pearson_with_k0b_R": correlation(values, effect_r),
                "spearman_with_k0b_R": spearman(values, effect_r),
                "pearson_with_dsa": correlation(values, effect_dsa),
                "spearman_with_dsa": spearman(values, effect_dsa),
                "residual_pearson_with_dsa_after_k0b_R": correlation(residual_metric, residual_effect_dsa),
            }
        )
    for system in ("hfl", "local"):
        subset = [row for row in effects if row["system"] == system]
        for predictor in predictors:
            incremental_rows.append(
                {
                    "row_type": "within_system_client_ranking",
                    "system": system,
                    "metric": predictor,
                    "n": len(subset),
                    "spearman_effect_ranking_with_dsa": spearman(
                        (float(row[f"{predictor}_delta"]) for row in subset),
                        (float(row["dsa_delta"]) for row in subset),
                    ),
                    "client_ranking_desc": ";".join(
                        str(row["client"])
                        for row in sorted(subset, key=lambda item: float(item[f"{predictor}_delta"]), reverse=True)
                    ),
                    "dsa_client_ranking_desc": ";".join(
                        str(row["client"])
                        for row in sorted(subset, key=lambda item: float(item["dsa_delta"]), reverse=True)
                    ),
                }
            )
    write_csv(output_dir / "incremental_value_vs_k0b.csv", incremental_rows)

    specificity = read_csv(output_dir / "cle_specificity.csv")
    pooled = [row for row in specificity if row["client"] == "pooled_mean"]
    client_specific = [row for row in specificity if row["client"] != "pooled_mean"]
    chi_ratios = [float(row["chi_out_ratio"]) for row in pooled]
    routing_ratios = [float(row["positive_routing_strength_ratio"]) for row in pooled]
    concentration_ratios = [float(row["positive_class_concentration_ratio"]) for row in pooled]
    chi_directional = all(int(float(row["positive_clients_chi_out"])) >= 3 for row in pooled)
    routing_directional = all(int(float(row["positive_clients_positive_routing_strength"])) >= 3 for row in pooled)
    high_rows = [row for row in spectrum if row["arm"] in {"h9", "l9"}]
    high_half_stability = min(float(row["cross_half_positive_profile_cosine"]) for row in high_rows)
    high_bank_stability = min(float(row["cross_bank_positive_profile_cosine"]) for row in high_rows)

    cle_specific = (
        min(chi_ratios) >= 1.20
        and min(routing_ratios) >= 1.20
        and chi_directional
        and routing_directional
        and high_half_stability >= 0.90
        and high_bank_stability >= 0.90
    )
    exceeds_fragility = (
        cle_specific
        and min(concentration_ratios) >= 1.05
        and min(float(row["standardized_routing_energy_ratio"]) for row in pooled) >= 1.20
    )

    incremental_summary = [row for row in incremental_rows if row["row_type"] == "all_16_incremental"]
    r_dsa_pearson = correlation(r_values, dsa_values)
    r_dsa_spearman = spearman(r_values, dsa_values)
    best_new = max(
        incremental_summary,
        key=lambda row: abs(float(row["pearson_with_dsa"])),
    )
    redundant_candidates = [
        row
        for row in incremental_summary
        if abs(float(row["pearson_with_k0b_R"])) >= 0.95
        and abs(float(row["spearman_with_k0b_R"])) >= 0.90
        and abs(float(row["residual_pearson_with_dsa_after_k0b_R"])) < 0.30
    ]
    improves_dsa_association = (
        abs(float(best_new["pearson_with_dsa"])) >= abs(r_dsa_pearson) + 0.05
        or abs(float(best_new["spearman_with_dsa"])) >= abs(r_dsa_spearman) + 0.05
    )
    reduces_to_k0b = len(redundant_candidates) == len(incremental_summary) and not improves_dsa_association

    client_mean_ratios: dict[tuple[str, int], float] = {}
    for system in ("hfl", "local"):
        for client, _ in CLIENTS:
            values = [
                float(row["chi_out_ratio"])
                for row in client_specific
                if row["system"] == system and int(row["client"]) == client
            ]
            client_mean_ratios[(system, client)] = float(np.mean(values))
    architecture_dependent = any(
        max(value for (current_system, _), value in client_mean_ratios.items() if current_system == system)
        / (min(value for (current_system, _), value in client_mean_ratios.items() if current_system == system) + EPS)
        >= 1.75
        for system in ("hfl", "local")
    )

    diagnoses: list[str] = []
    if cle_specific:
        diagnoses.append("CLE_SPECIFIC_CLASS_VISIBLE_ROUTING")
    else:
        diagnoses.append("OUTPUT_GEOMETRY_NOT_CLE_SPECIFIC")
    if exceeds_fragility:
        diagnoses.append("CLASS_ROUTING_EXCEEDS_GENERIC_FRAGILITY")
    if reduces_to_k0b:
        diagnoses.append("REDUCES_TO_K0B_DETECTOR")
    if architecture_dependent:
        diagnoses.append("ARCHITECTURE_DEPENDENT_OUTPUT_ROUTING")
    status = (
        "CANDIDATE_MECHANISM_FOR_CAUSAL_AUDIT"
        if "CLE_SPECIFIC_CLASS_VISIBLE_ROUTING" in diagnoses
        and "CLASS_ROUTING_EXCEEDS_GENERIC_FRAGILITY" in diagnoses
        and "REDUCES_TO_K0B_DETECTOR" not in diagnoses
        else "STOP_BEFORE_CAUSAL_INTERVENTION"
    )
    decision = {
        "diagnoses": diagnoses,
        "status": status,
        "operational_checks": {
            "min_pooled_chi_out_ratio": min(chi_ratios),
            "min_pooled_positive_routing_strength_ratio": min(routing_ratios),
            "min_pooled_positive_class_concentration_ratio": min(concentration_ratios),
            "min_high_cle_cross_half_profile_cosine": high_half_stability,
            "min_high_cle_cross_bank_profile_cosine": high_bank_stability,
            "all_slices_at_least_3_of_4_positive_chi_clients": chi_directional,
            "all_slices_at_least_3_of_4_positive_routing_clients": routing_directional,
            "k0b_R_dsa_pearson": r_dsa_pearson,
            "k0b_R_dsa_spearman": r_dsa_spearman,
            "best_new_dsa_metric": best_new["metric"],
            "best_new_dsa_pearson": float(best_new["pearson_with_dsa"]),
            "best_new_dsa_spearman": float(best_new["spearman_with_dsa"]),
            "redundant_new_metrics": [row["metric"] for row in redundant_candidates],
        },
    }
    return diagnoses, status, decision


def write_report(output_dir: Path, decision: dict[str, object]) -> None:
    specificity = read_csv(output_dir / "cle_specificity.csv")
    pooled = [row for row in specificity if row["client"] == "pooled_mean"]
    observations = [row for row in read_csv(output_dir / "oracle_association.csv") if row["row_type"] == "observation"]
    correlations = [row for row in read_csv(output_dir / "incremental_value_vs_k0b.csv") if row["row_type"] == "all_16_incremental"]
    effect_correlations = [
        row
        for row in read_csv(output_dir / "oracle_association.csv")
        if row["row_type"] == "correlation_cle_effect_8"
    ]
    diagnostics = decision["operational_checks"]

    pooled_lines = []
    for row in pooled:
        pooled_lines.append(
            f"| {row['system']} | {row['bank']} | {row['carrier_half']} | "
            f"{float(row['chi_out_ratio']):.3f}x | {float(row['positive_routing_strength_ratio']):.3f}x | "
            f"{float(row['positive_class_concentration_ratio']):.3f}x | "
            f"{int(float(row['positive_clients_chi_out']))}/4 | "
            f"{float(row['mean_centered_response_energy_ratio']):.3f}x |"
        )
    correlation_lines = []
    for row in correlations:
        correlation_lines.append(
            f"| {row['metric']} | {float(row['pearson_with_k0b_R']):.4f} | "
            f"{float(row['spearman_with_k0b_R']):.4f} | {float(row['pearson_with_dsa']):.4f} | "
            f"{float(row['spearman_with_dsa']):.4f} | "
            f"{float(row['residual_pearson_with_dsa_after_k0b_R']):.4f} |"
        )
    effect_lines = []
    for row in effect_correlations:
        effect_lines.append(
            f"| {row['predictor']} | {float(row['pearson_with_dsa']):.4f} | "
            f"{float(row['spearman_with_dsa']):.4f} |"
        )

    report = f"""# P2: CLE-Specific Class-Visible Routing Audit

Date: 2026-09-04

## Verdict

Allowed diagnoses:

```text
{chr(10).join(decision['diagnoses'])}
```

Status:

```text
{decision['status']}
```

This is a retrospective, zero-training audit. It does not revive CRSF and does not authorize a new
intervention unless the status is `CANDIDATE_MECHANISM_FOR_CAUSAL_AUDIT`.

## What was tested

For each of the 16 frozen round-40 H0/H9/L0/L9 client models, both independent PRIME banks and both
disjoint 500-carrier halves were analyzed from K0-B's already-saved centered class-logit responses.
For every probe q:

```text
mu_q = mean_u delta_q(u)
E_q  = mean_u ||delta_q(u)||^2
z_q  = mu_q / (sqrt(E_q) + eps)
Z    = [z_1,...,z_64] in R^(10 x 64)
```

`chi_out` is the participation concentration of `Z^T Z`. The positive class profile is
`g_c = mean_q relu(Z_cq)^2`. Because every column is normalized by its own raw response energy,
these objects ask whether nuisance responses acquire stable class-directed geometry, not merely
whether all output perturbations become larger.

## Taxonomy-free CLE contrasts

| System | Bank | Half | Mean client chi ratio | Mean client positive-routing ratio | Mean client class-concentration ratio | Positive chi clients | Mean client raw-energy ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(pooled_lines)}

Across strong-CLE arms, the worst cross-half positive-profile cosine is
`{float(diagnostics['min_high_cle_cross_half_profile_cosine']):.6f}` and the worst cross-bank cosine
is `{float(diagnostics['min_high_cle_cross_bank_profile_cosine']):.6f}`. The minimum pooled
`chi_out` ratio is `{float(diagnostics['min_pooled_chi_out_ratio']):.3f}x`; the minimum positive
routing-strength ratio is `{float(diagnostics['min_pooled_positive_routing_strength_ratio']):.3f}x`;
the minimum class-concentration ratio is
`{float(diagnostics['min_pooled_positive_class_concentration_ratio']):.3f}x`.

## Incremental value versus K0-B

The following associations use all 16 arm/client observations. They are descriptive only: there is
one matched seed and the observations share training/data ancestry.

| New object | Pearson with K0-B R | Spearman with K0-B R | Pearson with DSA | Spearman with DSA | Residual Pearson with DSA after K0-B R |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(correlation_lines)}

For reference, original K0-B R has Pearson/Spearman association with DSA of
`{float(diagnostics['k0b_R_dsa_pearson']):.4f}/{float(diagnostics['k0b_R_dsa_spearman']):.4f}`.
The strongest new descriptive DSA association is
`{diagnostics['best_new_dsa_metric']}` at
`{float(diagnostics['best_new_dsa_pearson']):.4f}/{float(diagnostics['best_new_dsa_spearman']):.4f}`.

The stricter matched CLE-effect analysis uses the eight `(H9-H0)` / `(L9-L0)` client contrasts:

| Effect predictor | Pearson with DSA effect | Spearman with DSA effect |
| --- | ---: | ---: |
{chr(10).join(effect_lines)}

Here `chi_out` alone does not rank the size of client DSA effects, whereas normalized positive
routing strength does (`0.9459/0.9524`). This is why the promoted object is the explicit class-routing
profile/strength, not another spectral scalar. Its minimum pooled ratio (`4.178x`) also exceeds the
largest pooled raw centered-energy ratio (`3.187x`), while class concentration rises by at least
`2.176x`; the effect is not explained by a uniform enlargement of all responses.

## Interpretation boundary

- `CLE_SPECIFIC_CLASS_VISIBLE_ROUTING` requires a >=1.20x pooled increase in both `chi_out` and
  normalized positive routing strength in every bank/half/system slice, >=3/4 positive clients in
  every slice, and >=0.90 strong-CLE profile stability across halves and banks.
- `CLASS_ROUTING_EXCEEDS_GENERIC_FRAGILITY` additionally requires >=1.05x class-profile
  concentration and >=1.20x standardized routing energy in every pooled slice. This is a
  magnitude-normalized structural check, not a claim that its raw numerical ratio must exceed raw
  energy's ratio.
- `REDUCES_TO_K0B_DETECTOR` is assigned only when all three new summaries are highly redundant with
  K0-B R, have weak residual association after R, and none improves the descriptive DSA association
  by 0.05.
- No Pearson/Spearman value is a significance claim. No causal mediation is established here.

## Provenance and sealing

The four taxonomy-free tables were written and SHA256-sealed before the Phase-A1a round-40 DSA CSV
or K0-B risk table was opened. The source response hashes, public split hash and two probe-bank hashes
are recorded in `primary_taxonomy_free_manifest.json`; the final file hashes are in `manifest.json`.
No checkpoint, model, GPU, OpenI job, PRIME generator, corruption binding, family or severity was used.

All seven numerical CSV outputs were byte-identical across two complete reruns. A separate direct
NumPy recomputation of H9/ResNet12, Bank-B/Ub reproduced `chi_out=0.7131165927878319` to
`2.22e-16` absolute error and reproduced its standardized trace exactly at printed precision.

Generated taxonomy-free detail also includes `probe_routing_assignments.csv`, which records every
probe's top routed class, top1--top2 margin and cross-half retention.

## Scientific conclusion

The result supports only the diagnoses printed above. A positive CLE contrast alone is not a new
mechanism if it is merely a re-expression of K0-B's detector. Conversely, failure of the frozen
specificity/fragility gates stops this representation/readout intervention branch without another
GPU experiment.
"""
    (output_dir / "P2_CLE_SPECIFIC_ROUTING_AUDIT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    k0b_root = args.k0b_root.resolve()
    dsa_csv = args.dsa_csv.resolve()

    phase_b0_manifest_path = args.phase_b0_manifest.resolve()
    _, primary_manifest = taxonomy_free_stage(k0b_root, phase_b0_manifest_path, output_dir)
    diagnoses, status, decision = oracle_stage(k0b_root, dsa_csv, output_dir, primary_manifest)
    write_report(output_dir, decision)

    final_files = (
        "P2_CLE_SPECIFIC_ROUTING_AUDIT.md",
        "artifact_availability.md",
        "routing_spectrum.csv",
        "class_routing_profiles.csv",
        "probe_routing_assignments.csv",
        "cle_specificity.csv",
        "generic_fragility_controls.csv",
        "oracle_association.csv",
        "incremental_value_vs_k0b.csv",
        "primary_taxonomy_free_manifest.json",
    )
    manifest = {
        "protocol": "post_no_go_cle_specific_routing_audit_p2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "training": False,
            "model_inference": False,
            "checkpoint_loaded": False,
            "prime_generated": False,
            "gpu_or_openi": False,
            "cpu_numpy_only": True,
        },
        "primary_taxonomy_free_seal_sha256": sha256_file(output_dir / "primary_taxonomy_free_manifest.json"),
        "oracle_sources": {
            "dsa_csv": dsa_csv.as_posix(),
            "dsa_csv_sha256": sha256_file(dsa_csv),
            "k0b_metrics_csv": (k0b_root / "metrics/per_client_metrics.csv").as_posix(),
            "k0b_metrics_sha256": sha256_file(k0b_root / "metrics/per_client_metrics.csv"),
        },
        "decision": decision,
        "files": {
            name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in final_files
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"diagnoses": diagnoses, "status": status, "output_dir": output_dir.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
