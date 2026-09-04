from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("h0", "h9", "l0", "l9")
CLIENTS = ((0, "ResNet10"), (1, "ResNet12"), (2, "ShuffleNet"), (3, "Mobilenetv2"))
EPS = 1.0e-12
FORMAL_RANDOM_COUNT = 1000
FORMAL_SEED = 20260904
INVARIANCE_TOLERANCE = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P3-A matched routing-identity causal audit.")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
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
        "--clean-manifest",
        type=Path,
        default=ROOT / "outputs/p3a_clean_base_completion_formal/manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--confirm-formal", action="store_true")
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


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def centered_logits_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise AssertionError("probabilities must be finite and strictly positive")
    logits = np.log(values)
    return logits - logits.mean(axis=-1, keepdims=True)


def rank_reversal(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.arange(len(profile), dtype=np.int64)
    ranking = np.lexsort((classes, -np.asarray(profile, dtype=np.float64)))
    permutation = np.empty_like(ranking)
    for offset in range(len(ranking) // 2):
        high = int(ranking[offset])
        low = int(ranking[-1 - offset])
        permutation[high] = low
        permutation[low] = high
    if len(ranking) % 2:
        middle = int(ranking[len(ranking) // 2])
        permutation[middle] = middle
        # Not used for C=10; retained as a deterministic cyclic repair.
        fixed = np.flatnonzero(permutation == classes)
        if fixed.size:
            permutation[fixed] = np.roll(permutation[fixed], -1)
    validate_permutation(permutation, require_derangement=True)
    return ranking, permutation


def validate_permutation(permutation: np.ndarray, *, require_derangement: bool) -> None:
    value = np.asarray(permutation, dtype=np.int64)
    expected = np.arange(value.size, dtype=np.int64)
    if not np.array_equal(np.sort(value), expected):
        raise AssertionError("class coordinate map is not a permutation")
    if require_derangement and np.any(value == expected):
        raise AssertionError("class coordinate map is not a derangement")


def generate_unique_derangements(rng: np.random.Generator, count: int, classes: int = 10) -> np.ndarray:
    identity = np.arange(classes, dtype=np.int64)
    rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(rows) < count:
        candidate = rng.permutation(classes).astype(np.int64)
        key = tuple(int(x) for x in candidate)
        if np.any(candidate == identity) or key in seen:
            continue
        seen.add(key)
        rows.append(candidate)
    result = np.stack(rows)
    for row in result:
        validate_permutation(row, require_derangement=True)
    return result


def p2_profile(centered_response: np.ndarray) -> np.ndarray:
    response = np.asarray(centered_response[:500, :64], dtype=np.float64)
    mu = response.mean(axis=0)
    energy = np.square(response).sum(axis=-1).mean(axis=0)
    standardized = mu / (np.sqrt(np.maximum(energy, 0.0))[:, None] + EPS)
    return np.square(np.maximum(standardized, 0.0)).mean(axis=0)


def k0b_risk_from_sufficient(
    mu_a: np.ndarray,
    mu_b: np.ndarray,
    energy_a: np.ndarray,
    energy_b: np.ndarray,
    permutation: np.ndarray,
) -> tuple[float, float, float]:
    left = mu_a[:, permutation]
    right = mu_b[:, permutation]
    cross = np.maximum(np.sum(left * right, axis=-1), 0.0)
    kappa = cross / (np.sqrt(np.maximum(energy_a * energy_b, 0.0)) + EPS)
    mean = 0.5 * (left + right)
    top_two = np.partition(mean, kth=8, axis=-1)[:, -2:]
    top_two.sort(axis=-1)
    selectivity = (top_two[:, 1] - top_two[:, 0]) / (np.linalg.norm(mean, axis=-1) + EPS)
    rho = kappa * np.maximum(selectivity, 0.0)
    energy = 0.5 * (energy_a + energy_b)
    active = energy >= np.median(energy)
    selected = rho[active]
    top_count = max(1, int(np.ceil(0.2 * selected.size)))
    risk = float(np.partition(selected, selected.size - top_count)[-top_count:].mean())
    return float(kappa.mean()), float(selectivity.mean()), risk


def response_invariance(response: np.ndarray, permutation: np.ndarray) -> dict[str, float]:
    original = np.asarray(response, dtype=np.float64).reshape(-1, 10)
    changed = original[:, permutation]
    norm_error = float(np.max(np.abs(np.linalg.norm(original, axis=1) - np.linalg.norm(changed, axis=1))))
    subset_indices = np.linspace(0, len(original) - 1, min(256, len(original)), dtype=np.int64)
    left = original[subset_indices]
    right = changed[subset_indices]
    left_gram = np.sum(left[:, None, :] * left[None, :, :], axis=-1)
    right_gram = np.sum(right[:, None, :] * right[None, :, :], axis=-1)
    gram_error = float(np.max(np.abs(left_gram - right_gram)))
    class_gram_original = np.einsum("ni,nj->ij", original, original, optimize=False)
    class_gram_changed = np.einsum("ni,nj->ij", changed, changed, optimize=False)
    expected_changed_gram = class_gram_original[np.ix_(permutation, permutation)]
    spectrum_similarity_error = float(np.max(np.abs(class_gram_changed - expected_changed_gram)))
    trace_original = float(np.trace(class_gram_original))
    trace_changed = float(np.trace(class_gram_changed))
    chi_original = float(np.square(class_gram_original).sum() / (trace_original * trace_original + EPS))
    chi_changed = float(np.square(class_gram_changed).sum() / (trace_changed * trace_changed + EPS))
    energy_original = float(np.square(original).sum())
    energy_changed = float(np.square(changed).sum())
    return {
        "max_row_norm_abs_error": norm_error,
        "pairwise_gram_subset_max_abs_error": gram_error,
        "singular_spectrum_similarity_max_abs_error": spectrum_similarity_error,
        "chi_out_abs_error": abs(chi_original - chi_changed),
        "raw_energy_abs_error": abs(energy_original - energy_changed),
        "raw_energy_relative_error": abs(energy_original - energy_changed) / (abs(energy_original) + EPS),
    }


def dsa_values(
    probabilities: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
    family_ids: np.ndarray,
) -> tuple[np.ndarray, float]:
    probs = np.asarray(probabilities, dtype=np.float64)
    families = int(family_ids.max()) + 1
    family_means = np.stack(
        [probs[:, :, family_ids == family, :].mean(axis=2) for family in range(families)], axis=1
    )
    contrast = family_means - (family_means.sum(axis=1, keepdims=True) - family_means) / (families - 1)
    clients = np.full(probs.shape[0], np.nan, dtype=np.float64)
    for client in range(probs.shape[0]):
        values: list[float] = []
        for family in range(families):
            bound = np.flatnonzero(binding[client] == family)
            if not bound.size:
                continue
            valid = ~np.isin(labels, bound)
            effect = contrast[client, family][:, bound].sum(axis=1)
            values.append(float(effect[valid].mean()))
        clients[client] = float(np.mean(values))
    return clients, float(np.mean(clients))


def random_dsa_null(
    clean_logits: np.ndarray,
    response: np.ndarray,
    permutations: np.ndarray,
    labels: np.ndarray,
    binding: np.ndarray,
    family_ids: np.ndarray,
    *,
    chunk_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    count = permutations.shape[1]
    client_null = np.empty((count, 4), dtype=np.float64)
    pooled_null = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        for position in range(start, stop):
            counterfactual = np.stack(
                [
                    softmax(
                        clean_logits[client, :, None, :]
                        + np.take(response[client], permutations[client, position], axis=-1)
                    )
                    for client in range(4)
                ]
            )
            client_values, pooled = dsa_values(counterfactual, labels, binding, family_ids)
            client_null[position] = client_values
            pooled_null[position] = pooled
    return client_null, pooled_null


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise SystemExit("formal mode requires --confirm-formal")
    random_count = FORMAL_RANDOM_COUNT if args.mode == "formal" else 8
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "outputs" / f"post_no_go_p3a_routing_identity_{args.mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    k0b_root = args.k0b_root.resolve()
    a1a_path = args.a1a_predictions.resolve()
    clean_path = args.clean_base.resolve()
    clean_manifest_path = args.clean_manifest.resolve()
    for path in (a1a_path, clean_path, clean_manifest_path, k0b_root / "blind_response_manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    clean_manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
    if clean_manifest.get("execution_contract", {}).get("clean_forward_only") is not True:
        raise AssertionError("clean-base artifact is not certified as clean-forward-only")
    with np.load(clean_path, allow_pickle=False) as clean_payload:
        clean_logits = np.asarray(clean_payload["clean_logits"], dtype=np.float64)
        clean_labels = np.asarray(clean_payload["labels"], dtype=np.int64)
        if tuple(clean_payload["arm_names"].tolist()) != ARMS:
            raise AssertionError("clean-base arm order mismatch")
        if not np.array_equal(clean_payload["client_ids"], np.arange(4)):
            raise AssertionError("clean-base client order mismatch")
    if clean_logits.shape != (4, 4, 1000, 10):
        raise AssertionError("unexpected clean-base shape")
    print("[p3a] clean base validated", flush=True)

    # Stage 1: taxonomy-free routing design. No Phase-A1a binding/family/DSA is read here.
    rng = np.random.default_rng(FORMAL_SEED)
    targeted = np.empty((4, 4, 10), dtype=np.int64)
    random_permutations = np.empty((4, 4, random_count, 10), dtype=np.int64)
    routing_rows: list[dict[str, object]] = []
    response_paths: dict[tuple[str, int], Path] = {}
    k0b_sufficient: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for arm_index, arm in enumerate(ARMS):
        for client, model in CLIENTS:
            print(f"[p3a] taxonomy-free routing arm={arm} client={client}", flush=True)
            path = k0b_root / "responses" / f"{arm}_client{client}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            response_paths[(arm, client)] = path
            with np.load(path, allow_pickle=False) as payload:
                centered = np.asarray(payload["centered_response"], dtype=np.float64)
            profile = p2_profile(centered)
            ranking, permutation = rank_reversal(profile)
            targeted[arm_index, client] = permutation
            random_permutations[arm_index, client] = generate_unique_derangements(rng, random_count)
            left, right = centered[:500], centered[500:]
            k0b_sufficient[(arm, client)] = (
                left.mean(axis=0),
                right.mean(axis=0),
                np.square(left).sum(axis=-1).mean(axis=0),
                np.square(right).sum(axis=-1).mean(axis=0),
            )
            routing_rows.append(
                {
                    "arm": arm,
                    "client": client,
                    "model": model,
                    "ranking_descending_g": ";".join(str(int(x)) for x in ranking),
                    "target_new_coordinate_to_old_coordinate": ";".join(str(int(x)) for x in permutation),
                    **{f"g_class_{index}": float(profile[index]) for index in range(10)},
                    "k0b_response_sha256": sha256_file(path),
                }
            )
    permutation_path = output_dir / "permutations.npz"
    np.savez_compressed(
        permutation_path,
        targeted_new_to_old=targeted,
        random_new_to_old=random_permutations,
        arm_names=np.asarray(ARMS),
        client_ids=np.arange(4, dtype=np.int64),
    )
    write_csv(output_dir / "taxonomy_free_routing_profiles.csv", routing_rows)
    permutation_manifest = {
        "protocol": "p3a_matched_routing_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "TAXONOMY_FREE_PERMUTATIONS_SEALED_BEFORE_ORACLE",
        "routing_source": "K0-B Bank-A plus carrier-half Ua only",
        "profile": "g_c = mean_q relu(mu_qc/sqrt(E_q))^2",
        "target_algorithm": "stable descending g_c with class-index tie break, then rank reversal",
        "random_algorithm": "independent unique numpy.default_rng derangements per arm/client",
        "random_seed": FORMAL_SEED,
        "random_count": random_count,
        "binding_used": False,
        "real_corruption_family_used": False,
        "severity_used": False,
        "dsa_used": False,
        "permutations_sha256": sha256_file(permutation_path),
        "targeted_array_sha256": sha256_array(targeted),
        "random_array_sha256": sha256_array(random_permutations),
        "source_hashes": {
            "clean_base": sha256_file(clean_path),
            "clean_manifest": sha256_file(clean_manifest_path),
            "a1a_predictions_container_not_opened": sha256_file(a1a_path),
            **{f"{arm}_client{client}": sha256_file(path) for (arm, client), path in response_paths.items()},
        },
    }
    write_json(output_dir / "permutation_manifest.json", permutation_manifest)
    print("[p3a] permutations sealed", flush=True)

    # Stage 2: construct output-space interventions without reading binding/family/DSA arrays.
    with np.load(a1a_path, allow_pickle=False) as a1a_payload:
        corrupt_probabilities = np.asarray(a1a_payload["probabilities"], dtype=np.float32)
    if corrupt_probabilities.shape != (4, 4, 1000, 16, 10):
        raise AssertionError("unexpected Phase-A1a probability grid")
    clean_centered = clean_logits - clean_logits.mean(axis=-1, keepdims=True)
    reconstruction_error = 0.0
    targeted_probabilities = np.empty(corrupt_probabilities.shape, dtype=np.float32)
    invariance_rows: list[dict[str, object]] = []
    maximum_random_risk_error = 0.0
    for arm_index, arm in enumerate(ARMS):
        print(f"[p3a] intervention arm={arm}", flush=True)
        arm_response = centered_logits_from_probabilities(corrupt_probabilities[arm_index]) - clean_centered[
            arm_index, :, :, None, :
        ]
        reconstructed = softmax(clean_logits[arm_index, :, :, None, :] + arm_response)
        reconstruction_error = max(
            reconstruction_error,
            float(np.max(np.abs(reconstructed - corrupt_probabilities[arm_index]))),
        )
        for client, model in CLIENTS:
            permutation = targeted[arm_index, client]
            current_response = arm_response[client]
            targeted_probabilities[arm_index, client] = softmax(
                clean_logits[arm_index, client, :, None, :] + current_response[..., permutation]
            ).astype(np.float32)
            invariance = response_invariance(current_response, permutation)
            mu_a, mu_b, energy_a, energy_b = k0b_sufficient[(arm, client)]
            identity = np.arange(10, dtype=np.int64)
            original_metrics = k0b_risk_from_sufficient(mu_a, mu_b, energy_a, energy_b, identity)
            target_metrics = k0b_risk_from_sufficient(mu_a, mu_b, energy_a, energy_b, permutation)
            target_errors = np.abs(np.asarray(target_metrics) - np.asarray(original_metrics))
            random_risk_error = 0.0
            for random_permutation in random_permutations[arm_index, client]:
                metrics = k0b_risk_from_sufficient(
                    mu_a, mu_b, energy_a, energy_b, random_permutation
                )
                random_risk_error = max(
                    random_risk_error,
                    float(np.max(np.abs(np.asarray(metrics) - np.asarray(original_metrics)))),
                )
            maximum_random_risk_error = max(maximum_random_risk_error, random_risk_error)
            invariance_rows.append(
                {
                    "arm": arm,
                    "client": client,
                    "model": model,
                    **invariance,
                    "k0b_K_target_abs_error": float(target_errors[0]),
                    "k0b_selectivity_target_abs_error": float(target_errors[1]),
                    "k0b_R_target_abs_error": float(target_errors[2]),
                    "k0b_random_max_metric_abs_error": random_risk_error,
                    "random_coordinate_bijections_certified": True,
                }
            )
        del arm_response, reconstructed
    if reconstruction_error > 1.0e-6:
        raise AssertionError(f"identity reconstruction mismatch: {reconstruction_error}")
    targeted_path = output_dir / "targeted_counterfactual_probabilities.npz"
    np.savez_compressed(
        targeted_path,
        targeted_probabilities=targeted_probabilities.astype(np.float32),
        targeted_new_to_old=targeted,
        arm_names=np.asarray(ARMS),
        client_ids=np.arange(4, dtype=np.int64),
    )
    write_csv(output_dir / "invariance_audit.csv", invariance_rows)
    intervention_manifest = {
        "stage": "TAXONOMY_FREE_INTERVENTION_SEALED_BEFORE_ORACLE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "z_counterfactual = z_clean + P_pi P_C(z_corrupt-z_clean)",
        "identity_reconstruction_max_abs_probability_error": reconstruction_error,
        "random_outputs": "deterministically regenerated in chunks from sealed clean/response/permutations",
        "targeted_output_sha256": sha256_file(targeted_path),
        "invariance_csv_sha256": sha256_file(output_dir / "invariance_audit.csv"),
        "permutation_manifest_sha256": sha256_file(output_dir / "permutation_manifest.json"),
        "binding_read_before_seal": False,
        "dsa_read_before_seal": False,
    }
    write_json(output_dir / "taxonomy_free_intervention_manifest.json", intervention_manifest)
    print("[p3a] interventions sealed; opening oracle", flush=True)

    # Stage 3: only now open Phase-A1a labels, binding and corruption-family oracle arrays.
    with np.load(a1a_path, allow_pickle=False) as a1a_payload:
        labels = np.asarray(a1a_payload["labels"], dtype=np.int64)
        binding = np.asarray(a1a_payload["binding"], dtype=np.int64)
        family_ids = np.asarray(a1a_payload["operator_family_ids"], dtype=np.int64)
    if not np.array_equal(labels, clean_labels):
        raise AssertionError("clean and Phase-A1a labels do not align")

    targeted_rows: list[dict[str, object]] = []
    per_client_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    arm_results: dict[str, dict[str, object]] = {}
    for arm_index, arm in enumerate(ARMS):
        print(f"[p3a] oracle arm={arm} random_count={random_count}", flush=True)
        arm_response = centered_logits_from_probabilities(corrupt_probabilities[arm_index]) - clean_centered[
            arm_index, :, :, None, :
        ]
        identity_client, identity_pooled = dsa_values(
            corrupt_probabilities[arm_index], labels, binding, family_ids
        )
        target_client, target_pooled = dsa_values(
            targeted_probabilities[arm_index], labels, binding, family_ids
        )
        null_client, null_pooled = random_dsa_null(
            clean_logits[arm_index],
            arm_response,
            random_permutations[arm_index],
            labels,
            binding,
            family_ids,
        )
        del arm_response
        quantiles = {name: float(np.quantile(null_pooled, value)) for name, value in (
            ("q05", 0.05), ("q10", 0.10), ("q25", 0.25), ("q50", 0.50),
            ("q75", 0.75), ("q90", 0.90), ("q95", 0.95),
        )}
        percentile = float(np.mean(null_pooled <= target_pooled))
        reduction = identity_pooled - target_pooled
        positive_clients = int(np.count_nonzero(identity_client - target_client > 0.0))
        arm_results[arm] = {
            "identity_dsa": identity_pooled,
            "targeted_dsa": target_pooled,
            "absolute_reduction": reduction,
            "relative_reduction": reduction / (abs(identity_pooled) + EPS),
            "positive_clients": positive_clients,
            "random_mean": float(null_pooled.mean()),
            "random_std": float(null_pooled.std()),
            "targeted_percentile": percentile,
            **quantiles,
        }
        targeted_rows.append({"arm": arm, **arm_results[arm]})
        for client, model in CLIENTS:
            per_client_rows.append(
                {
                    "arm": arm,
                    "client": client,
                    "model": model,
                    "identity_dsa": float(identity_client[client]),
                    "targeted_dsa": float(target_client[client]),
                    "absolute_reduction": float(identity_client[client] - target_client[client]),
                    "random_mean": float(null_client[:, client].mean()),
                    "random_q10": float(np.quantile(null_client[:, client], 0.10)),
                    "targeted_percentile": float(np.mean(null_client[:, client] <= target_client[client])),
                }
            )
        for index in range(random_count):
            null_rows.append(
                {
                    "arm": arm,
                    "permutation_id": index,
                    "pooled_dsa": float(null_pooled[index]),
                    **{f"client_{client}_dsa": float(null_client[index, client]) for client in range(4)},
                }
            )
    write_csv(output_dir / "targeted_results.csv", targeted_rows)
    write_csv(output_dir / "per_client_results.csv", per_client_rows)
    write_csv(output_dir / "random_derangement_null.csv", null_rows)

    invariant_values: list[float] = []
    for row in invariance_rows:
        invariant_values.extend(
            abs(float(row[key]))
            for key in (
                "max_row_norm_abs_error",
                "pairwise_gram_subset_max_abs_error",
                "singular_spectrum_similarity_max_abs_error",
                "chi_out_abs_error",
                "raw_energy_relative_error",
                "k0b_K_target_abs_error",
                "k0b_selectivity_target_abs_error",
                "k0b_R_target_abs_error",
                "k0b_random_max_metric_abs_error",
            )
        )
    maximum_invariance_error = max(invariant_values)
    gates = {
        "A_invariance_le_1e_8": maximum_invariance_error <= INVARIANCE_TOLERANCE,
        "B_h9_targeted_reduction_ge_0_05": arm_results["h9"]["absolute_reduction"] >= 0.05,
        "C_l9_targeted_reduction_ge_0_05": arm_results["l9"]["absolute_reduction"] >= 0.05,
        "D_h9_positive_clients_ge_3": arm_results["h9"]["positive_clients"] >= 3,
        "D_l9_positive_clients_ge_3": arm_results["l9"]["positive_clients"] >= 3,
        "E_h9_targeted_le_random_q10": arm_results["h9"]["targeted_dsa"] <= arm_results["h9"]["q10"],
        "E_l9_targeted_le_random_q10": arm_results["l9"]["targeted_dsa"] <= arm_results["l9"]["q10"],
        "F_h0_no_new_dsa_gt_0_02": arm_results["h0"]["targeted_dsa"] - arm_results["h0"]["identity_dsa"] <= 0.02,
        "F_l0_no_new_dsa_gt_0_02": arm_results["l0"]["targeted_dsa"] - arm_results["l0"]["identity_dsa"] <= 0.02,
    }
    if args.mode == "smoke":
        verdict = "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
        status = "EXECUTION_VALIDATION_ONLY"
    elif all(gates.values()):
        verdict = "CAUSAL_ROUTING_IDENTITY_SUPPORTED"
        status = "GO_TO_P3B_ROUTING_STRENGTH_AUDIT"
    elif (
        arm_results["h9"]["random_mean"] < arm_results["h9"]["identity_dsa"]
        and arm_results["l9"]["random_mean"] < arm_results["l9"]["identity_dsa"]
        and (not gates["E_h9_targeted_le_random_q10"] or not gates["E_l9_targeted_le_random_q10"])
    ):
        verdict = "CLASS_IDENTITY_CAUSAL_BUT_GENERIC_PROFILE_NOT_TARGETING"
        status = "NO_GO_TO_METHOD"
    else:
        verdict = "CLASS_ROUTING_NOT_SUFFICIENT_CAUSAL_TARGET"
        status = "STOP_REPRESENTATION_READOUT_ROUTE"
    summary = {
        "protocol": "p3a_matched_routing_identity_v1",
        "mode": args.mode,
        "verdict": verdict,
        "status": status,
        "random_seed": FORMAL_SEED,
        "random_count": random_count,
        "identity_reconstruction_max_abs_probability_error": reconstruction_error,
        "maximum_invariance_error": maximum_invariance_error,
        "maximum_random_k0b_metric_error": maximum_random_risk_error,
        "arms": arm_results,
        "gates": gates,
        "notes": [
            "H9/L9 ResNet10 checkpoints are bit-identical and are not independent replications.",
            "This is output-space mechanism evidence, not a mitigation method or training result.",
        ],
    }
    write_json(output_dir / "oracle_gate_summary.json", summary)
    manifest_files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(output_dir / "manifest.json", {"files": manifest_files, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
