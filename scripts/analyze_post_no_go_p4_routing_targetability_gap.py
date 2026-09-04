from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("h0", "h9", "l0", "l9")
CLIENTS = ((0, "ResNet10"), (1, "ResNet12"), (2, "ShuffleNet"), (3, "Mobilenetv2"))
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4 CPU-only routing targetability gap audit.")
    parser.add_argument(
        "--k0b-root",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_generic_probe_k0b_seed0/formal_extracted/outputs/cle_generic_probe_k0b_seed0_formal",
    )
    parser.add_argument(
        "--a1a-predictions",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_shortcut_amplification_phase_a1a_seed0/extracted/outputs/cle_shortcut_amplification_phase_a1a_seed0_analysis/round_040_predictions.npz",
    )
    parser.add_argument(
        "--clean-base",
        type=Path,
        default=ROOT / "outputs/p3a_clean_base_completion_formal/clean_base_outputs.npz",
    )
    parser.add_argument(
        "--p3a-root",
        type=Path,
        default=ROOT
        / "deliverables/post_no_go_p3a_routing_identity_causal_audit_after_clean_completion_20260904",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "deliverables/post_no_go_p4_routing_targetability_gap_20260904",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rankdata(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty_like(data)
    start = 0
    while start < len(data):
        stop = start + 1
        while stop < len(data) and data[order[stop]] == data[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > EPS else float("nan")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > EPS else float("nan")


def concentration(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    return float(np.square(data).sum() / (np.square(data.sum()) + EPS))


def centered_response(probabilities: np.ndarray, clean_logits: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if np.any(values <= 0.0) or not np.isfinite(values).all():
        raise AssertionError("corruption probabilities must be finite and strictly positive")
    corrupt = np.log(values)
    corrupt -= corrupt.mean(axis=-1, keepdims=True)
    clean = np.asarray(clean_logits, dtype=np.float64).copy()
    clean -= clean.mean(axis=-1, keepdims=True)
    return corrupt - clean[:, None, :]


def generic_profile(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Exactly P2: K0-B Bank-A (q=0:64) and carrier half Ua (u=0:500).
    response = np.asarray(centered[:500, :64], dtype=np.float64)
    mu = response.mean(axis=0)
    energy = np.square(response).sum(axis=-1).mean(axis=0)
    z_qc = mu / (np.sqrt(np.maximum(energy, 0.0))[:, None] + EPS)
    profile = np.square(np.maximum(z_qc, 0.0)).mean(axis=0)
    return profile, z_qc.T


def harmful_objects(
    response: np.ndarray,
    labels: np.ndarray,
    client_binding: np.ndarray,
    family_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return M_harm, positive harmful profile and valid support per bound class.

    Row a uses operators in family binding[a] and the same valid-source exclusion as DSA:
    sources whose true class is not any class bound to that family. Column b is output class b.
    """

    matrix = np.empty((10, 10), dtype=np.float64)
    positive = np.empty(10, dtype=np.float64)
    support = np.empty(10, dtype=np.int64)
    for bound_class in range(10):
        family = int(client_binding[bound_class])
        bound_classes = np.flatnonzero(client_binding == family)
        valid = ~np.isin(labels, bound_classes)
        family_response = response[valid][:, family_ids == family, :]
        matrix[bound_class] = family_response.mean(axis=(0, 1))
        positive[bound_class] = np.maximum(family_response[..., bound_class], 0.0).mean()
        support[bound_class] = int(family_response.shape[0] * family_response.shape[1])
    return matrix, positive, support


def top_overlap(left: np.ndarray, right: np.ndarray, count: int) -> float:
    left_top = set(np.argsort(left, kind="mergesort")[-count:].tolist())
    right_top = set(np.argsort(right, kind="mergesort")[-count:].tolist())
    return len(left_top & right_top) / count


def mean_rank_displacement(left: np.ndarray, right: np.ndarray) -> float:
    left_order = np.argsort(-left, kind="mergesort")
    right_order = np.argsort(-right, kind="mergesort")
    left_rank = np.empty(10, dtype=np.int64)
    right_rank = np.empty(10, dtype=np.int64)
    left_rank[left_order] = np.arange(10)
    right_rank[right_order] = np.arange(10)
    return float(np.abs(left_rank - right_rank).mean())


def destructive_score(matrix: np.ndarray, permutation: np.ndarray) -> dict[str, float]:
    classes = np.arange(10, dtype=np.int64)
    before = matrix[classes, classes]
    after = matrix[classes, permutation]
    before_energy = float(np.square(np.maximum(before, 0.0)).sum())
    after_energy = float(np.square(np.maximum(after, 0.0)).sum())
    return {
        "diagonal_signed_before": float(before.mean()),
        "diagonal_signed_after": float(after.mean()),
        "signed_destructive_score": float((before - after).mean()),
        "positive_diagonal_energy_before": before_energy,
        "positive_diagonal_energy_after": after_energy,
        "positive_energy_destructive_score": before_energy - after_energy,
        "positive_energy_reduction_fraction": (before_energy - after_energy) / (before_energy + EPS),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    k0b_root = args.k0b_root.resolve()
    a1a_path = args.a1a_predictions.resolve()
    clean_path = args.clean_base.resolve()
    p3a_root = args.p3a_root.resolve()
    required = (
        a1a_path,
        clean_path,
        p3a_root / "permutations.npz",
        p3a_root / "permutation_manifest.json",
        p3a_root / "oracle_gate_summary.json",
        p3a_root / "random_derangement_null.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    p3a_summary = json.loads((p3a_root / "oracle_gate_summary.json").read_text(encoding="utf-8"))
    if p3a_summary.get("verdict") != "CLASS_IDENTITY_CAUSAL_BUT_GENERIC_PROFILE_NOT_TARGETING":
        raise AssertionError("P4 requires the frozen P3-A NO_GO_TO_METHOD result")
    permutation_manifest = json.loads((p3a_root / "permutation_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(p3a_root / "permutations.npz") != permutation_manifest["permutations_sha256"]:
        raise AssertionError("P3-A permutation artifact hash mismatch")
    with np.load(p3a_root / "permutations.npz", allow_pickle=False) as payload:
        targeted = np.asarray(payload["targeted_new_to_old"], dtype=np.int64)
        random_permutations = np.asarray(payload["random_new_to_old"], dtype=np.int64)
    if targeted.shape != (4, 4, 10) or random_permutations.shape != (4, 4, 1000, 10):
        raise AssertionError("unexpected P3-A permutation shape")
    with np.load(clean_path, allow_pickle=False) as payload:
        clean_logits = np.asarray(payload["clean_logits"], dtype=np.float64)
        clean_labels = np.asarray(payload["labels"], dtype=np.int64)
    with np.load(a1a_path, allow_pickle=False) as payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        binding = np.asarray(payload["binding"], dtype=np.int64)
        family_ids = np.asarray(payload["operator_family_ids"], dtype=np.int64)
    if not np.array_equal(clean_labels, labels):
        raise AssertionError("clean/corruption sample identity mismatch")

    p3a_null_rows = read_csv(p3a_root / "random_derangement_null.csv")
    p3a_null = {
        arm: np.asarray(
            [float(row["pooled_dsa"]) for row in p3a_null_rows if row["arm"] == arm], dtype=np.float64
        )
        for arm in ARMS
    }
    context_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    context_values: dict[tuple[str, int], dict[str, object]] = {}

    for arm_index, arm in enumerate(ARMS):
        print(f"[p4] arm={arm}", flush=True)
        for client, model in CLIENTS:
            response_path = k0b_root / "responses" / f"{arm}_client{client}.npz"
            with np.load(response_path, allow_pickle=False) as payload:
                g_generic, z_generic = generic_profile(payload["centered_response"])
            response = centered_response(
                probabilities[arm_index, client], clean_logits[arm_index, client]
            )
            matrix, g_harm, support = harmful_objects(response, labels, binding[client], family_ids)
            target_action = destructive_score(matrix, targeted[arm_index, client])
            random_signed = np.empty(1000, dtype=np.float64)
            random_energy = np.empty(1000, dtype=np.float64)
            for permutation_id, permutation in enumerate(random_permutations[arm_index, client]):
                action = destructive_score(matrix, permutation)
                random_signed[permutation_id] = action["signed_destructive_score"]
                random_energy[permutation_id] = action["positive_energy_destructive_score"]
            target_signed_percentile = float(np.mean(random_signed <= target_action["signed_destructive_score"]))
            target_energy_percentile = float(
                np.mean(random_energy <= target_action["positive_energy_destructive_score"])
            )
            generic_rank = np.argsort(-g_generic, kind="mergesort")
            harm_rank = np.argsort(-g_harm, kind="mergesort")
            row = {
                "arm": arm,
                "system": "hfl" if arm.startswith("h") else "local",
                "gamma": 9 if arm.endswith("9") else 0,
                "client": client,
                "model": model,
                "cosine_generic_harm": cosine(g_generic, g_harm),
                "pearson_generic_harm": correlation(g_generic, g_harm),
                "spearman_generic_harm": correlation(rankdata(g_generic), rankdata(g_harm)),
                "top1_overlap": top_overlap(g_generic, g_harm, 1),
                "top3_overlap": top_overlap(g_generic, g_harm, 3),
                "top5_overlap": top_overlap(g_generic, g_harm, 5),
                "mean_rank_displacement": mean_rank_displacement(g_generic, g_harm),
                "generic_concentration": concentration(g_generic),
                "harmful_concentration": concentration(g_harm),
                "generic_top_class": int(generic_rank[0]),
                "harmful_top_class": int(harm_rank[0]),
                "generic_top3": ";".join(str(int(x)) for x in generic_rank[:3]),
                "harmful_top3": ";".join(str(int(x)) for x in harm_rank[:3]),
                "matrix_diagonal_mean": float(np.diag(matrix).mean()),
                "matrix_off_diagonal_mean": float(
                    matrix[~np.eye(10, dtype=bool)].mean()
                ),
                "matrix_diagonal_margin": float(
                    np.mean(np.diag(matrix) - (matrix.sum(axis=1) - np.diag(matrix)) / 9.0)
                ),
                **{f"target_{key}": value for key, value in target_action.items()},
                "target_signed_destructive_percentile": target_signed_percentile,
                "target_energy_destructive_percentile": target_energy_percentile,
                "random_signed_mean": float(random_signed.mean()),
                "random_energy_mean": float(random_energy.mean()),
            }
            context_rows.append(row)
            context_values[(arm, client)] = {
                "row": row,
                "random_signed": random_signed,
                "random_energy": random_energy,
            }
            for class_index in range(10):
                class_rows.append(
                    {
                        "arm": arm,
                        "client": client,
                        "model": model,
                        "class": class_index,
                        "binding_family": int(binding[client, class_index]),
                        "generic_strength": float(g_generic[class_index]),
                        "harmful_positive_strength": float(g_harm[class_index]),
                        "harmful_diagonal_signed": float(matrix[class_index, class_index]),
                        "valid_instance_support": int(support[class_index]),
                        "generic_rank": int(np.flatnonzero(generic_rank == class_index)[0]),
                        "harmful_rank": int(np.flatnonzero(harm_rank == class_index)[0]),
                        "target_maps_new_class_from_old_class": int(targeted[arm_index, client, class_index]),
                    }
                )
            for bound_class in range(10):
                for output_class in range(10):
                    matrix_rows.append(
                        {
                            "arm": arm,
                            "client": client,
                            "model": model,
                            "bound_class": bound_class,
                            "binding_family": int(binding[client, bound_class]),
                            "output_class": output_class,
                            "mean_centered_logit_response": float(matrix[bound_class, output_class]),
                        }
                    )

    arm_rows: list[dict[str, object]] = []
    for arm_index, arm in enumerate(ARMS):
        rows = [context_values[(arm, client)]["row"] for client, _ in CLIENTS]
        target_signed = float(np.mean([row["target_signed_destructive_score"] for row in rows]))
        target_energy = float(np.mean([row["target_positive_energy_destructive_score"] for row in rows]))
        random_signed = np.mean(
            np.stack([context_values[(arm, client)]["random_signed"] for client, _ in CLIENTS]), axis=0
        )
        random_energy = np.mean(
            np.stack([context_values[(arm, client)]["random_energy"] for client, _ in CLIENTS]), axis=0
        )
        dsa_reduction = float(p3a_summary["arms"][arm]["identity_dsa"] - p3a_summary["arms"][arm]["targeted_dsa"])
        random_dsa_reduction = float(p3a_summary["arms"][arm]["identity_dsa"]) - p3a_null[arm]
        arm_row = {
            "arm": arm,
            "mean_cosine": float(np.mean([row["cosine_generic_harm"] for row in rows])),
            "mean_pearson": float(np.mean([row["pearson_generic_harm"] for row in rows])),
            "mean_spearman": float(np.mean([row["spearman_generic_harm"] for row in rows])),
            "mean_top3_overlap": float(np.mean([row["top3_overlap"] for row in rows])),
            "mean_rank_displacement": float(np.mean([row["mean_rank_displacement"] for row in rows])),
            "target_signed_destructive_score": target_signed,
            "target_signed_destructive_percentile": float(np.mean(random_signed <= target_signed)),
            "target_energy_destructive_score": target_energy,
            "target_energy_destructive_percentile": float(np.mean(random_energy <= target_energy)),
            "target_dsa_reduction": dsa_reduction,
            "p3a_target_dsa_percentile": float(p3a_summary["arms"][arm]["targeted_percentile"]),
            "random_signed_vs_dsa_reduction_pearson": correlation(random_signed, random_dsa_reduction),
            "random_energy_vs_dsa_reduction_pearson": correlation(random_energy, random_dsa_reduction),
        }
        arm_rows.append(arm_row)
        for permutation_id in range(1000):
            random_rows.append(
                {
                    "arm": arm,
                    "permutation_id": permutation_id,
                    "mean_signed_destructive_score": float(random_signed[permutation_id]),
                    "mean_energy_destructive_score": float(random_energy[permutation_id]),
                    "p3a_dsa_reduction": float(random_dsa_reduction[permutation_id]),
                }
            )

    arm_lookup = {row["arm"]: row for row in arm_rows}
    h9 = arm_lookup["h9"]
    l9 = arm_lookup["l9"]
    client_cosine_advantages = [
        context_values[("l9", client)]["row"]["cosine_generic_harm"]
        - context_values[("h9", client)]["row"]["cosine_generic_harm"]
        for client, _ in CLIENTS
    ]
    client_spearman_advantages = [
        context_values[("l9", client)]["row"]["spearman_generic_harm"]
        - context_values[("h9", client)]["row"]["spearman_generic_harm"]
        for client, _ in CLIENTS
    ]
    # Frozen interpretive thresholds; P4 is explanation-only and can never emit METHOD_GO.
    h9_strong_alignment = h9["mean_cosine"] >= 0.70 and h9["mean_spearman"] >= 0.50
    l9_strong_alignment = l9["mean_cosine"] >= 0.70 and l9["mean_spearman"] >= 0.50
    matched_gap = (
        l9["mean_cosine"] - h9["mean_cosine"] >= 0.10
        and l9["mean_spearman"] - h9["mean_spearman"] >= 0.20
        and np.count_nonzero(np.asarray(client_cosine_advantages) > 0.0) >= 3
        and np.count_nonzero(np.asarray(client_spearman_advantages) > 0.0) >= 3
    )
    destructive_gap = (
        l9["target_signed_destructive_percentile"]
        - h9["target_signed_destructive_percentile"]
        >= 0.30
    )
    if l9_strong_alignment and not h9_strong_alignment and matched_gap and destructive_gap:
        verdict = "GENERIC_ROUTING_DECOUPLES_FROM_HARMFUL_ROUTING_IN_HFL"
        recommendation = "STOP_GENERIC_PROBE_GUIDED_MITIGATION_ROUTE"
    elif h9_strong_alignment:
        verdict = "GENERIC_ROUTING_ALIGNS_WITH_HARMFUL_ROUTING_BUT_TARGET_RULE_FAILED"
        recommendation = "INFORMATION_MAY_EXIST_IDENTIFIABILITY_AUDIT_REQUIRED"
    elif l9_strong_alignment and (
        np.count_nonzero(np.asarray(client_cosine_advantages) > 0.0) < 3
        or np.count_nonzero(np.asarray(client_spearman_advantages) > 0.0) < 3
    ):
        verdict = "LOCAL_ONLY_ALIGNMENT_ARTIFACT"
        recommendation = "NO_METHOD_PROMOTION"
    else:
        verdict = "NO_CLEAR_TARGETABILITY_GAP_EXPLANATION"
        recommendation = "NO_METHOD_PROMOTION"

    write_csv(output_dir / "per_context_alignment.csv", context_rows)
    write_csv(output_dir / "per_class_profiles.csv", class_rows)
    write_csv(output_dir / "harmful_routing_matrices.csv", matrix_rows)
    write_csv(output_dir / "arm_summary.csv", arm_rows)
    write_csv(output_dir / "random_destructive_score_vs_dsa.csv", random_rows)
    arrays_path = output_dir / "p4_permutation_provenance.npz"
    np.savez_compressed(
        arrays_path,
        targeted_new_to_old=targeted,
        random_new_to_old=random_permutations,
        arm_names=np.asarray(ARMS),
    )
    result = {
        "protocol": "p4_routing_targetability_gap_v1",
        "verdict": verdict,
        "recommendation": recommendation,
        "method_go": False,
        "thresholds_frozen_before_harmful_profile_computation": {
            "strong_alignment_mean_cosine": 0.70,
            "strong_alignment_mean_spearman": 0.50,
            "matched_cosine_gap": 0.10,
            "matched_spearman_gap": 0.20,
            "matched_positive_clients": 3,
            "destructive_percentile_gap": 0.30,
        },
        "h9_strong_alignment": bool(h9_strong_alignment),
        "l9_strong_alignment": bool(l9_strong_alignment),
        "matched_gap": bool(matched_gap),
        "destructive_gap": bool(destructive_gap),
        "client_l9_minus_h9_cosine": [float(x) for x in client_cosine_advantages],
        "client_l9_minus_h9_spearman": [float(x) for x in client_spearman_advantages],
        "arms": {row["arm"]: row for row in arm_rows},
        "constraints": {
            "training": False,
            "model_inference": False,
            "new_prime": False,
            "permutation_search": False,
            "p3a_permutation_modified": False,
            "new_loss": False,
        },
    }
    write_json(output_dir / "result.json", result)
    manifest_files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(
        output_dir / "manifest.json",
        {
            "sources": {
                "a1a_predictions": {"path": a1a_path.as_posix(), "sha256": sha256_file(a1a_path)},
                "clean_base": {"path": clean_path.as_posix(), "sha256": sha256_file(clean_path)},
                "p3a_permutations": {
                    "path": (p3a_root / "permutations.npz").as_posix(),
                    "sha256": sha256_file(p3a_root / "permutations.npz"),
                },
            },
            "files": manifest_files,
            "result": result,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
