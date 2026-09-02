from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.augmentations.frozen_prime import (  # noqa: E402
    apply_frozen_prime_recipe,
    load_frozen_prime_bank,
)
from fedprime.data.loaders import cifar100_train_images_from_tar, dataset_stats, normalize_batch  # noqa: E402
from fedprime.engine.cle_shared_nuisance_routing import (  # noqa: E402
    aggregate_transfer,
    bootstrap_cross_bank_transfer,
    bootstrap_index_matrix,
    bootstrap_sharedness,
    match_low_energy_probes,
    paired_random_subspace_bases,
    percentile_interval,
    sharedness_statistics,
    weighted_mean,
    weighted_response_subspace,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402
from scripts.run_cle_k1_sdmn_headonly import load_state, public_split, sha256_array, sha256_file, write_json  # noqa: E402


MODEL_NAMES = ("ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2")
SYSTEMS = {"hfl": ("h9", "h0"), "local": ("l9", "l0")}
BANK_ROOT = ROOT / "fedprime/augmentations/assets/cle_generic_probe_k0b"
SELECTION_MANIFEST = ROOT / "fedprime/augmentations/assets/cle_k1_b0/selection_manifest.json"
EXPECTED_BANK_HASHES = {
    "a": "6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01",
    "b": "4A53497EC5DB6EC05C312E6166109FA4B52A5CC402CCE74E6EDB1253D913BF4E",
}
EXPECTED_D_SELECT_SHA256 = "731B8CFFDCBD241474D33B261E323F9EC11C2EA59BC7705261140A3B8572F6CA"
EXPECTED_D_REP_SHA256 = "321C0910E8AA376B10D04D1319F24917EE91EABD25BCC8C31A0BDE66F8E240EE"
BOOTSTRAP_SEED = 20260910
RANDOM_SUBSPACE_SEED = 20260911


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLE K1-B0 shared representation localization gate.")
    parser.add_argument("--mode", choices=("inspect", "smoke", "formal"), default="inspect")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cle_k1_b0_cdr_snr"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def freeze_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not hasattr(model, "backbone") or not hasattr(model, "linear"):
        raise AttributeError("K1-B0 requires model.backbone and model.linear")
    if not isinstance(model.linear, torch.nn.Linear):
        raise TypeError("K1-B0 requires a linear final classifier")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _tensor_batch(images: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(
        device=device, dtype=torch.float32
    ).div_(255.0)


def verify_penultimate_interface(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, object]:
    stats = dataset_stats("cifar10")
    with torch.inference_mode():
        normalized = normalize_batch(_tensor_batch(images[:8], device), stats)
        direct = forward_logits(model, normalized)
        features = model.backbone(normalized).flatten(1)
        reconstructed = model.linear(features)
    difference = float((direct - reconstructed).abs().max().item())
    identical = bool(torch.equal(direct.argmax(1), reconstructed.argmax(1)))
    if difference > 1.0e-5 or not identical:
        raise RuntimeError(f"penultimate reconstruction failed: max_abs={difference}, argmax={identical}")
    return {
        "feature_dimension": int(features.shape[1]),
        "max_abs_logit_error": difference,
        "argmax_identical": identical,
    }


def extract_base_features(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    stats = dataset_stats("cifar10")
    values = []
    with torch.inference_mode():
        for start in range(0, images.shape[0], int(batch_size)):
            batch = _tensor_batch(images[start : start + int(batch_size)], device)
            features = model.backbone(normalize_batch(batch, stats)).flatten(1)
            values.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(values, axis=0)


def extract_delta_for_recipe(
    model: torch.nn.Module,
    images: np.ndarray,
    base: np.ndarray,
    recipe: dict[str, object],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    stats = dataset_stats("cifar10")
    result = np.empty_like(base)
    with torch.inference_mode():
        for start in range(0, images.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), images.shape[0])
            batch = _tensor_batch(images[start:stop], device)
            transformed = apply_frozen_prime_recipe(batch, recipe)
            features = model.backbone(normalize_batch(transformed, stats)).flatten(1)
            result[start:stop] = features.detach().cpu().numpy().astype(np.float32) - base[start:stop]
    return result


def representation_energy_all(
    model: torch.nn.Module,
    images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
    progress_prefix: str = "",
) -> np.ndarray:
    base = extract_base_features(model, images, device=device, batch_size=batch_size)
    energy = np.empty(len(recipes), dtype=np.float64)
    for probe_id, recipe in enumerate(recipes):
        delta = extract_delta_for_recipe(
            model, images, base, recipe, device=device, batch_size=batch_size
        )
        energy[probe_id] = np.square(delta.astype(np.float64)).sum(axis=1).mean()
        if progress_prefix and ((probe_id + 1) % 8 == 0 or probe_id + 1 == len(recipes)):
            log(f"[heartbeat] {progress_prefix} energy probes {probe_id + 1}/{len(recipes)}")
    return energy


def selected_deltas(
    model: torch.nn.Module,
    images: np.ndarray,
    recipes: list[dict[str, object]],
    probe_ids: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    progress_prefix: str = "",
) -> np.ndarray:
    base = extract_base_features(model, images, device=device, batch_size=batch_size)
    output = np.empty((images.shape[0], probe_ids.size, base.shape[1]), dtype=np.float32)
    for position, probe_id in enumerate(probe_ids.tolist()):
        output[:, position] = extract_delta_for_recipe(
            model,
            images,
            base,
            recipes[int(probe_id)],
            device=device,
            batch_size=batch_size,
        )
        if progress_prefix and ((position + 1) % 4 == 0 or position + 1 == probe_ids.size):
            log(f"[heartbeat] {progress_prefix} selected probes {position + 1}/{probe_ids.size}")
    return output


def load_assets() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    banks = {
        name: load_frozen_prime_bank(
            state_path=BANK_ROOT / f"bank_{name}_states.npz",
            manifest_path=BANK_ROOT / f"bank_{name}_manifest.json",
        )
        for name in ("a", "b")
    }
    for name in ("a", "b"):
        if banks[name]["bank_sha256"] != EXPECTED_BANK_HASHES[name]:
            raise ValueError(f"bank {name} hash mismatch")
    manifest = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    if manifest["d_select"]["sha256"] != EXPECTED_D_SELECT_SHA256:
        raise ValueError("frozen D_select manifest mismatch")
    return banks, manifest


def inspect_assets(
    args: argparse.Namespace,
    banks: dict[str, object],
    images: np.ndarray,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    split = public_split(images.shape[0], discover_count=1000, surgery_count=2000, holdout_count=2000)
    checkpoint_rows = []
    for _, arms in SYSTEMS.items():
        for arm in arms:
            for client_id, model_name in enumerate(MODEL_NAMES):
                path = args.checkpoint_root / arm / f"client_{client_id}.pt"
                checkpoint_rows.append(
                    {
                        "arm": arm,
                        "client": client_id,
                        "model": model_name,
                        "exists": path.is_file(),
                        "bytes": path.stat().st_size if path.is_file() else None,
                        "sha256": sha256_file(path) if path.is_file() else None,
                    }
                )
    interface_rows = []
    if device is not None:
        for _, arms in SYSTEMS.items():
            for arm in arms:
                models = build_models(list(MODEL_NAMES), num_classes=10)
                for client_id, model_name in enumerate(MODEL_NAMES):
                    model = freeze_model(models[client_id], device)
                    model.load_state_dict(load_state(args.checkpoint_root / arm / f"client_{client_id}.pt"))
                    check = verify_penultimate_interface(model, images[:8], device=device)
                    interface_rows.append(
                        {"arm": arm, "client": client_id, "model": model_name, **check}
                    )
                del models
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return {
        "protocol": "cle_k1_b0_cdr_snr_inspect_v1",
        "all_16_checkpoints_present": all(row["exists"] for row in checkpoint_rows),
        "checkpoints": checkpoint_rows,
        "all_penultimate_interfaces_valid": bool(interface_rows) and all(
            row["max_abs_logit_error"] <= 1.0e-5 and row["argmax_identical"]
            for row in interface_rows
        ) if device is not None else None,
        "penultimate_interfaces": interface_rows,
        "public_image_count": int(images.shape[0]),
        "d_select_sha256": sha256_array(split["discover"]),
        "d_rep_sha256": sha256_array(split["holdout"]),
        "d_rep_half_a_sha256": sha256_array(split["holdout"][:1000]),
        "d_rep_half_b_sha256": sha256_array(split["holdout"][1000:]),
        "bank_hashes": {name: banks[name]["bank_sha256"] for name in ("a", "b")},
        "labels_loaded": False,
        "training_performed": False,
        "evaluation_assets_read": False,
    }


def _condition_metrics(
    high: np.ndarray,
    matched: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    half = high.shape[0] // 2
    high_stats = sharedness_statistics(high[:half], high[half:])
    matched_stats = sharedness_statistics(matched[:half], matched[half:])
    return {
        "high_sharedness": weighted_mean(high_stats.sharedness, weights),
        "matched_sharedness": weighted_mean(matched_stats.sharedness, weights),
        "specificity_ratio": weighted_mean(high_stats.sharedness, weights)
        / max(weighted_mean(matched_stats.sharedness, weights), 1.0e-12),
        "high_probe_sharedness": high_stats.sharedness.tolist(),
        "matched_probe_sharedness": matched_stats.sharedness.tolist(),
    }


def _transfer_metrics(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    *,
    random_bases: list[np.ndarray],
) -> dict[str, object]:
    half = delta_a.shape[0] // 2
    source_mean = delta_a[:half].astype(np.float64).mean(axis=0)
    target_mean = delta_b[half:].astype(np.float64).mean(axis=0)
    subspace = weighted_response_subspace(source_mean, weights_a)
    true_score, per_probe = aggregate_transfer(target_mean, weights_b, subspace.basis)
    random_scores = np.asarray(
        [aggregate_transfer(target_mean, weights_b, basis)[0] for basis in random_bases],
        dtype=np.float64,
    )
    return {
        "true_G": true_score,
        "per_probe_G": per_probe.tolist(),
        "source_rank": subspace.rank,
        "singular_values": subspace.singular_values.tolist(),
        "random_scores": random_scores,
        "random_q95": float(np.quantile(random_scores, 0.95)),
        "beats_random_q95": bool(true_score > np.quantile(random_scores, 0.95)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ci_payload(values: np.ndarray) -> dict[str, float]:
    lower, upper = percentile_interval(values)
    return {"estimate": float(values.mean()), "ci95_lower": lower, "ci95_upper": upper}


def decide_verdict(rows: list[dict[str, object]], bootstrap: dict[str, dict[str, np.ndarray]]) -> dict[str, object]:
    gates: dict[str, bool] = {}
    summaries: dict[str, object] = {}
    for system, (strong_arm, weak_arm) in SYSTEMS.items():
        system_rows = [row for row in rows if row["system"] == system]
        strong_rows = [row for row in system_rows if row["arm"] == strong_arm]
        weak_rows = [row for row in system_rows if row["arm"] == weak_arm]
        bank_ratios = {}
        for bank in ("a", "b"):
            strong_value = np.mean([row[f"{bank}_high_sharedness"] for row in strong_rows])
            weak_value = np.mean([row[f"{bank}_high_sharedness"] for row in weak_rows])
            bank_ratios[bank] = float(strong_value / max(weak_value, 1.0e-12))
            gates[f"{system}_sharedness_{bank}_ratio_ge_1p25"] = bank_ratios[bank] >= 1.25
        combined_strong = np.mean(
            [row[f"{bank}_high_sharedness"] for row in strong_rows for bank in ("a", "b")]
        )
        combined_weak = np.mean(
            [row[f"{bank}_high_sharedness"] for row in weak_rows for bank in ("a", "b")]
        )
        combined_ratio = float(combined_strong / max(combined_weak, 1.0e-12))
        gates[f"{system}_sharedness_combined_ratio_ge_1p50"] = combined_ratio >= 1.50
        positive = 0
        for client_id in range(4):
            strong_client = next(row for row in strong_rows if row["client"] == client_id)
            weak_client = next(row for row in weak_rows if row["client"] == client_id)
            strong_mean = np.mean([strong_client[f"{bank}_high_sharedness"] for bank in ("a", "b")])
            weak_mean = np.mean([weak_client[f"{bank}_high_sharedness"] for bank in ("a", "b")])
            positive += int(strong_mean > weak_mean)
        gates[f"{system}_sharedness_positive_clients_ge_3of4"] = positive >= 3
        shared_diff_bootstrap = bootstrap[system]["sharedness_combined_difference"]
        shared_ci = _ci_payload(shared_diff_bootstrap)
        gates[f"{system}_sharedness_combined_ci95_lower_gt_0"] = shared_ci["ci95_lower"] > 0.0

        specificity_ratios = {}
        for bank in ("a", "b"):
            high = np.mean([row[f"{bank}_high_sharedness"] for row in strong_rows])
            matched = np.mean([row[f"{bank}_matched_sharedness"] for row in strong_rows])
            specificity_ratios[bank] = float(high / max(matched, 1.0e-12))
            gates[f"{system}_specificity_{bank}_ratio_ge_1p10"] = specificity_ratios[bank] >= 1.10
        combined_specificity = float(
            np.mean([row[f"{bank}_high_sharedness"] for row in strong_rows for bank in ("a", "b")])
            / max(np.mean([row[f"{bank}_matched_sharedness"] for row in strong_rows for bank in ("a", "b")]), 1.0e-12)
        )
        gates[f"{system}_specificity_combined_ratio_ge_1p20"] = combined_specificity >= 1.20

        random_counts = {
            direction: sum(bool(row[f"{direction}_beats_random_q95"]) for row in strong_rows)
            for direction in ("a_to_b", "b_to_a")
        }
        gates[f"{system}_cross_bank_random_ge_3of4_both_directions"] = all(
            value >= 3 for value in random_counts.values()
        )
        transfer_diff_bootstrap = bootstrap[system]["transfer_combined_difference"]
        transfer_ci = _ci_payload(transfer_diff_bootstrap)
        gates[f"{system}_transfer_combined_ci95_lower_gt_0"] = transfer_ci["ci95_lower"] > 0.0
        summaries[system] = {
            "sharedness_bank_ratios": bank_ratios,
            "sharedness_combined_ratio": combined_ratio,
            "sharedness_positive_clients": positive,
            "sharedness_combined_difference_bootstrap": shared_ci,
            "specificity_bank_ratios": specificity_ratios,
            "specificity_combined_ratio": combined_specificity,
            "cross_bank_random_positive_clients": random_counts,
            "transfer_combined_difference_bootstrap": transfer_ci,
        }
    if len(gates) != 20:
        raise AssertionError(f"frozen K1-B0 contract requires exactly 20 gates, got {len(gates)}")
    passed = sum(gates.values())
    return {
        "verdict": "GO_TO_K1_B_SNR_SURGERY" if passed == 20 else "NO_GO_SHARED_NUISANCE_ROUTING",
        "passed_gates": int(passed),
        "total_gates": 20,
        "gates": gates,
        "summaries": summaries,
    }


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    formal = args.mode == "formal"
    device = resolve_device(args.device)
    banks, selection_manifest = load_assets()
    images = cifar100_train_images_from_tar(args.public_root.resolve())
    inspection = inspect_assets(
        args,
        banks,
        images,
        device=device if args.mode == "inspect" else None,
    )
    if inspection["d_select_sha256"] != EXPECTED_D_SELECT_SHA256:
        raise ValueError("D_select hash mismatch")
    if inspection["d_rep_sha256"] != EXPECTED_D_REP_SHA256:
        raise ValueError("D_rep hash mismatch")
    if not inspection["all_16_checkpoints_present"]:
        raise FileNotFoundError("one or more frozen checkpoints are missing")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "inspection.json", inspection)
    if args.mode == "inspect":
        result = {**inspection, "verdict": "INSPECT_PASS_NO_SCIENTIFIC_DECISION"}
        write_json(output_dir / "result.json", result)
        return result

    split = public_split(images.shape[0], discover_count=1000, surgery_count=2000, holdout_count=2000)
    d_select_count = 16 if not formal else 1000
    d_rep_count = 32 if not formal else 2000
    probe_limit = 8 if not formal else 64
    bootstrap_replicates = 8 if not formal else 2000
    random_draws = 8 if not formal else 100
    clients = (0,) if not formal else tuple(range(4))
    systems = {"hfl": SYSTEMS["hfl"]} if not formal else SYSTEMS
    d_select_images = images[split["discover"][:d_select_count]]
    d_rep_images = images[split["holdout"][:d_rep_count]]
    half = d_rep_count // 2
    if half * 2 != d_rep_count:
        raise ValueError("D_rep must split into equal halves")

    index_a = bootstrap_index_matrix(carriers=half, replicates=bootstrap_replicates, seed=BOOTSTRAP_SEED)
    index_b = bootstrap_index_matrix(carriers=half, replicates=bootstrap_replicates, seed=BOOTSTRAP_SEED + 1)
    rows: list[dict[str, object]] = []
    bootstrap: dict[str, dict[str, np.ndarray]] = {}
    selection_records: list[dict[str, object]] = []
    checkpoint_hashes_before = {
        f"{arm}/client_{client_id}": sha256_file(args.checkpoint_root / arm / f"client_{client_id}.pt")
        for arms in systems.values()
        for arm in arms
        for client_id in clients
    }

    for system, (strong_arm, weak_arm) in systems.items():
        system_shared_difference = []
        system_transfer_difference = []
        for client_id in clients:
            log(f"[heartbeat] {system} client={client_id} matching start")
            models = build_models(list(MODEL_NAMES), num_classes=10)
            strong_model = freeze_model(models[client_id], device)
            strong_model.load_state_dict(load_state(args.checkpoint_root / strong_arm / f"client_{client_id}.pt"))
            strong_model.eval()
            strong_interface = verify_penultimate_interface(strong_model, d_rep_images, device=device)
            selection_arm = strong_arm
            matching_by_bank = {}
            for bank_name in ("a", "b"):
                frozen = selection_manifest["selections"][selection_arm][str(client_id)][bank_name]
                high_ids_full = np.asarray(frozen["selected_probe_ids"], dtype=np.int64)
                high_ids = high_ids_full[high_ids_full < probe_limit]
                active_ids = np.asarray(frozen["active_probe_ids"], dtype=np.int64)
                active_ids = active_ids[active_ids < probe_limit]
                if not formal:
                    if high_ids.size == 0:
                        high_ids = active_ids[:1]
                    active_ids = np.unique(np.concatenate((active_ids, high_ids)))
                rho_lookup = {
                    int(probe_id): float(weight)
                    for probe_id, weight in zip(frozen["selected_probe_ids"], frozen["weights"])
                }
                weights = np.asarray([rho_lookup.get(int(probe_id), 1.0) for probe_id in high_ids], dtype=np.float64)
                recipes = list(banks[bank_name]["recipes"][:probe_limit])
                energy = representation_energy_all(
                    strong_model,
                    d_select_images,
                    recipes,
                    device=device,
                    batch_size=args.batch_size,
                    progress_prefix=f"{system} client={client_id} bank={bank_name} D_select",
                )
                matching = match_low_energy_probes(high_ids, active_ids, energy, weights)
                matching_by_bank[bank_name] = matching
                selection_records.append(
                    {
                        "system": system,
                        "selection_arm": selection_arm,
                        "client": client_id,
                        "bank": bank_name,
                        "high_probe_ids": matching.high_probe_ids.tolist(),
                        "matched_probe_ids": matching.matched_probe_ids.tolist(),
                        "weights": matching.high_weights.tolist(),
                        "representation_energy_all": energy.tolist(),
                    }
                )

            arm_payloads: dict[str, dict[str, object]] = {}
            for arm, model, interface in ((strong_arm, strong_model, strong_interface),):
                log(f"[heartbeat] {system} client={client_id} arm={arm} D_rep start")
                bank_payload = {}
                for bank_name in ("a", "b"):
                    matching = matching_by_bank[bank_name]
                    union_ids = np.unique(np.concatenate((matching.high_probe_ids, matching.matched_probe_ids)))
                    union = selected_deltas(
                        model,
                        d_rep_images,
                        list(banks[bank_name]["recipes"]),
                        union_ids,
                        device=device,
                        batch_size=args.batch_size,
                        progress_prefix=f"{system} client={client_id} arm={arm} bank={bank_name} D_rep",
                    )
                    lookup = {int(probe_id): position for position, probe_id in enumerate(union_ids.tolist())}
                    high = union[:, [lookup[int(value)] for value in matching.high_probe_ids]]
                    matched = union[:, [lookup[int(value)] for value in matching.matched_probe_ids]]
                    bank_payload[bank_name] = {"high": high, "matched": matched, "weights": matching.high_weights}
                arm_payloads[arm] = {"banks": bank_payload, "interface": interface}

            del strong_model
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
            weak_models = build_models(list(MODEL_NAMES), num_classes=10)
            weak_model = freeze_model(weak_models[client_id], device)
            weak_model.load_state_dict(load_state(args.checkpoint_root / weak_arm / f"client_{client_id}.pt"))
            weak_model.eval()
            weak_interface = verify_penultimate_interface(weak_model, d_rep_images, device=device)
            log(f"[heartbeat] {system} client={client_id} arm={weak_arm} D_rep start")
            weak_banks = {}
            for bank_name in ("a", "b"):
                matching = matching_by_bank[bank_name]
                union_ids = np.unique(np.concatenate((matching.high_probe_ids, matching.matched_probe_ids)))
                union = selected_deltas(
                    weak_model,
                    d_rep_images,
                    list(banks[bank_name]["recipes"]),
                    union_ids,
                    device=device,
                    batch_size=args.batch_size,
                    progress_prefix=f"{system} client={client_id} arm={weak_arm} bank={bank_name} D_rep",
                )
                lookup = {int(probe_id): position for position, probe_id in enumerate(union_ids.tolist())}
                weak_banks[bank_name] = {
                    "high": union[:, [lookup[int(value)] for value in matching.high_probe_ids]],
                    "matched": union[:, [lookup[int(value)] for value in matching.matched_probe_ids]],
                    "weights": matching.high_weights,
                }
            arm_payloads[weak_arm] = {"banks": weak_banks, "interface": weak_interface}
            del weak_model
            del weak_models
            if device.type == "cuda":
                torch.cuda.empty_cache()

            paired_random: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
            for direction, source_bank in (("a_to_b", "a"), ("b_to_a", "b")):
                ranks = []
                for arm in (strong_arm, weak_arm):
                    payload = arm_payloads[arm]["banks"][source_bank]
                    source_mean = payload["high"][:half].astype(np.float64).mean(axis=0)
                    ranks.append(weighted_response_subspace(source_mean, payload["weights"]).rank)
                feature_dimension = int(arm_payloads[strong_arm]["interface"]["feature_dimension"])
                paired_random[direction] = paired_random_subspace_bases(
                    feature_dimension,
                    ranks[0],
                    ranks[1],
                    draws=random_draws,
                    seed=RANDOM_SUBSPACE_SEED + 100 * client_id + (0 if direction == "a_to_b" else 1),
                )

            row_by_arm = {}
            shared_boot_by_arm = {}
            transfer_boot_by_arm = {}
            for arm_index, arm in enumerate((strong_arm, weak_arm)):
                row: dict[str, object] = {
                    "system": system,
                    "arm": arm,
                    "client": client_id,
                    "model": MODEL_NAMES[client_id],
                    "feature_dimension": arm_payloads[arm]["interface"]["feature_dimension"],
                    "reconstruction_max_abs": arm_payloads[arm]["interface"]["max_abs_logit_error"],
                }
                bank_shared_boot = []
                for bank_name in ("a", "b"):
                    payload = arm_payloads[arm]["banks"][bank_name]
                    metrics = _condition_metrics(payload["high"], payload["matched"], payload["weights"])
                    row[f"{bank_name}_high_sharedness"] = metrics["high_sharedness"]
                    row[f"{bank_name}_matched_sharedness"] = metrics["matched_sharedness"]
                    row[f"{bank_name}_specificity_ratio"] = metrics["specificity_ratio"]
                    bank_shared_boot.append(
                        bootstrap_sharedness(
                            payload["high"][:half],
                            payload["high"][half:],
                            payload["weights"],
                            index_a,
                            index_b,
                        )
                    )
                transfer_boot = []
                for direction, source_bank, target_bank in (("a_to_b", "a", "b"), ("b_to_a", "b", "a")):
                    source = arm_payloads[arm]["banks"][source_bank]
                    target = arm_payloads[arm]["banks"][target_bank]
                    random_bases = paired_random[direction][arm_index]
                    metrics = _transfer_metrics(
                        source["high"],
                        target["high"],
                        source["weights"],
                        target["weights"],
                        random_bases=random_bases,
                    )
                    row[f"{direction}_G"] = metrics["true_G"]
                    row[f"{direction}_rank"] = metrics["source_rank"]
                    row[f"{direction}_random_q95"] = metrics["random_q95"]
                    row[f"{direction}_beats_random_q95"] = metrics["beats_random_q95"]
                    scores, _ = bootstrap_cross_bank_transfer(
                        source["high"][:half],
                        target["high"][half:],
                        source["weights"],
                        target["weights"],
                        index_a,
                        index_b,
                    )
                    transfer_boot.append(scores)
                row_by_arm[arm] = row
                shared_boot_by_arm[arm] = np.mean(np.stack(bank_shared_boot), axis=0)
                transfer_boot_by_arm[arm] = np.mean(np.stack(transfer_boot), axis=0)
                rows.append(row)
            system_shared_difference.append(shared_boot_by_arm[strong_arm] - shared_boot_by_arm[weak_arm])
            system_transfer_difference.append(transfer_boot_by_arm[strong_arm] - transfer_boot_by_arm[weak_arm])
            log(f"[heartbeat] {system} client={client_id} complete")
        bootstrap[system] = {
            "sharedness_combined_difference": np.mean(np.stack(system_shared_difference), axis=0),
            "transfer_combined_difference": np.mean(np.stack(system_transfer_difference), axis=0),
        }

    checkpoint_hashes_after = {
        key: sha256_file(args.checkpoint_root / key.split("/")[0] / f"{key.split('/')[1]}.pt")
        for key in checkpoint_hashes_before
    }
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("a frozen checkpoint changed during K1-B0")
    write_json(output_dir / "selection_and_matching.json", selection_records)
    _write_csv(output_dir / "per_client.csv", rows)
    np.savez_compressed(
        output_dir / "bootstrap_metrics.npz",
        **{
            f"{system}_{metric}": values
            for system, payload in bootstrap.items()
            for metric, values in payload.items()
        },
    )
    if formal:
        decision = decide_verdict(rows, bootstrap)
    else:
        decision = {
            "verdict": "SMOKE_ONLY_NO_SCIENTIFIC_DECISION",
            "passed_gates": None,
            "total_gates": 20,
            "gates": {},
            "summaries": {},
        }
    result = {
        "protocol": "cle_k1_b0_cdr_snr_v1",
        "mode": args.mode,
        "scientific_decision_allowed": formal,
        **decision,
        "configuration": {
            "d_select_count": d_select_count,
            "d_rep_count": d_rep_count,
            "probe_limit": probe_limit,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "random_subspace_draws": random_draws,
            "random_subspace_seed": RANDOM_SUBSPACE_SEED,
            "svd_relative_tolerance": 1.0e-6,
            "statistics_dtype": "float64",
            "labels_loaded": False,
            "evaluation_assets_read": False,
            "training_performed": False,
            "optimizer_constructed": False,
            "backward_called": False,
            "checkpoint_written": False,
        },
        "split": {
            "d_select_sha256": sha256_array(split["discover"]),
            "d_rep_sha256": sha256_array(split["holdout"]),
            "d_rep_half_a_sha256": sha256_array(split["holdout"][:1000]),
            "d_rep_half_b_sha256": sha256_array(split["holdout"][1000:]),
        },
        "bank_hashes": {name: banks[name]["bank_sha256"] for name in ("a", "b")},
        "selection_manifest_sha256": sha256_file(SELECTION_MANIFEST),
        "checkpoint_hashes_before": checkpoint_hashes_before,
        "checkpoint_hashes_after": checkpoint_hashes_after,
        "rows": rows,
    }
    write_json(output_dir / "result.json", result)
    report = [
        "# CLE K1-B0 CDR-SNR Result",
        "",
        f"- Mode: `{args.mode}`",
        f"- Verdict: `{result['verdict']}`",
        f"- Gates: `{result['passed_gates']}/{result['total_gates']}`",
        "- Training/checkpoint updates: `none`",
        "- Labels/CLE taxonomy/evaluation assets used: `none`",
    ]
    (output_dir / "FINAL_REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    artifact_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifact_rows.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(output_dir / "artifact_manifest.json", {"files": artifact_rows})
    return result


def main() -> None:
    args = parse_args()
    result = run_analysis(args)
    print(json.dumps({"mode": args.mode, "verdict": result["verdict"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
