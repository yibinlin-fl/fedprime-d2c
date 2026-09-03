from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.augmentations.frozen_prime import load_frozen_prime_bank  # noqa: E402
from fedprime.data.loaders import cifar100_train_images_from_tar  # noqa: E402
from fedprime.engine.cle_response_spectrum import (  # noqa: E402
    EPS_ENERGY,
    EPS_SPECTRUM,
    bootstrap_clean_concentration,
    bootstrap_count_matrix,
    bootstrap_response_concentration,
    clean_spectrum,
    response_spectrum,
    spectrum_from_eigenvalues,
    spectrum_from_gram,
)
from fedprime.models.factory import build_models  # noqa: E402
from scripts.analyze_cle_k1_b0_cdr_snr import (  # noqa: E402
    extract_base_features,
    extract_delta_for_recipe,
    freeze_model,
    resolve_device,
    verify_penultimate_interface,
)
from scripts.run_cle_k1_sdmn_headonly import (  # noqa: E402
    load_state,
    public_split,
    sha256_array,
    sha256_file,
    write_json,
)


MODEL_NAMES = ("ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2")
SYSTEMS = {"hfl": ("h9", "h0"), "local": ("l9", "l0")}
BANK_ROOT = ROOT / "fedprime/augmentations/assets/cle_generic_probe_k0b"
EXPECTED_BANK_HASHES = {
    "a": "6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01",
    "b": "4A53497EC5DB6EC05C312E6166109FA4B52A5CC402CCE74E6EDB1253D913BF4E",
}
EXPECTED_D_REP_SHA256 = "321C0910E8AA376B10D04D1319F24917EE91EABD25BCC8C31A0BDE66F8E240EE"
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_SAMPLES = 2000
RESPONSE_RATIO_THRESHOLD = 1.25


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLE K1-C0 response-spectrum mechanism gate.")
    parser.add_argument("--mode", choices=("inspect", "smoke", "formal"), default="inspect")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cle_k1_c0_response_spectrum_seed0"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=16)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest().upper()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def load_banks() -> dict[str, dict[str, object]]:
    banks = {
        name: load_frozen_prime_bank(
            state_path=BANK_ROOT / f"bank_{name}_states.npz",
            manifest_path=BANK_ROOT / f"bank_{name}_manifest.json",
        )
        for name in ("a", "b")
    }
    for name, expected in EXPECTED_BANK_HASHES.items():
        if banks[name]["bank_sha256"] != expected:
            raise ValueError(f"Bank {name} hash mismatch")
        if len(banks[name]["recipes"]) != 64:
            raise ValueError(f"Bank {name} must contain exactly 64 recipes")
    return banks


def extract_all_deltas(
    model: torch.nn.Module,
    images: np.ndarray,
    base: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
    progress_prefix: str,
) -> np.ndarray:
    output = np.empty((images.shape[0], len(recipes), base.shape[1]), dtype=np.float32)
    for probe_id, recipe in enumerate(recipes):
        output[:, probe_id] = extract_delta_for_recipe(
            model,
            images,
            base,
            recipe,
            device=device,
            batch_size=batch_size,
        )
        if (probe_id + 1) % 8 == 0 or probe_id + 1 == len(recipes):
            log(f"[heartbeat] {progress_prefix} probes {probe_id + 1}/{len(recipes)}")
    return output


def inspect_inputs(
    args: argparse.Namespace,
    banks: dict[str, dict[str, object]],
    images: np.ndarray,
    *,
    device: torch.device | None,
) -> dict[str, object]:
    split = public_split(images.shape[0], discover_count=1000, surgery_count=2000, holdout_count=2000)
    checkpoints = []
    interfaces = []
    for arms in SYSTEMS.values():
        for arm in arms:
            for client_id, model_name in enumerate(MODEL_NAMES):
                path = args.checkpoint_root / arm / f"client_{client_id}.pt"
                checkpoints.append(
                    {
                        "arm": arm,
                        "client": client_id,
                        "model": model_name,
                        "exists": path.is_file(),
                        "bytes": path.stat().st_size if path.is_file() else None,
                        "sha256": sha256_file(path) if path.is_file() else None,
                    }
                )
    if device is not None and all(row["exists"] for row in checkpoints):
        sample = images[split["holdout"][:8]]
        for arms in SYSTEMS.values():
            for arm in arms:
                models = build_models(list(MODEL_NAMES), num_classes=10)
                for client_id, model_name in enumerate(MODEL_NAMES):
                    model = freeze_model(models[client_id], device)
                    model.load_state_dict(load_state(args.checkpoint_root / arm / f"client_{client_id}.pt"))
                    interfaces.append(
                        {
                            "arm": arm,
                            "client": client_id,
                            "model": model_name,
                            **verify_penultimate_interface(model, sample, device=device),
                        }
                    )
                del models
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return {
        "protocol": "cle_k1_c0_response_spectrum_inspect_v1",
        "all_16_checkpoints_present": all(row["exists"] for row in checkpoints),
        "checkpoints": checkpoints,
        "all_penultimate_interfaces_valid": (
            all(row["max_abs_logit_error"] <= 1.0e-5 and row["argmax_identical"] for row in interfaces)
            if interfaces
            else None
        ),
        "penultimate_interfaces": interfaces,
        "public_image_count": int(images.shape[0]),
        "d_rep_sha256": sha256_array(split["holdout"]),
        "u1_sha256": sha256_array(split["holdout"][:1000]),
        "u2_sha256": sha256_array(split["holdout"][1000:]),
        "u1_u2_disjoint": bool(np.intersect1d(split["holdout"][:1000], split["holdout"][1000:]).size == 0),
        "bank_hashes": {name: banks[name]["bank_sha256"] for name in ("a", "b")},
        "all_64_probes_enabled": all(len(banks[name]["recipes"]) == 64 for name in ("a", "b")),
        "labels_loaded": False,
        "oracle_assets_loaded": False,
        "training_performed": False,
    }


def _ci(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "estimate": float(values.mean()),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
    }


def decide_gates(
    response_rows: list[dict[str, object]],
    clean_rows: list[dict[str, object]],
    bootstrap: dict[str, np.ndarray],
) -> dict[str, object]:
    response = {
        (str(row["arm"]), int(row["client"]), str(row["bank"])): float(row["chi_resp"])
        for row in response_rows
    }
    clean = {
        (str(row["arm"]), int(row["client"])): float(row["chi_clean"])
        for row in clean_rows
    }
    gates: dict[str, bool] = {}
    summaries: dict[str, object] = {}
    prefix = {"hfl": "H", "local": "L"}
    for system, (strong, weak) in SYSTEMS.items():
        bank_means = {}
        for bank in ("a", "b"):
            bank_means[bank] = {
                "strong": float(np.mean([response[(strong, client, bank)] for client in range(4)])),
                "weak": float(np.mean([response[(weak, client, bank)] for client in range(4)])),
            }
        mean_strong = float(np.mean([response[(strong, client, bank)] for client in range(4) for bank in ("a", "b")]))
        mean_weak = float(np.mean([response[(weak, client, bank)] for client in range(4) for bank in ("a", "b")]))
        r_resp = mean_strong / max(mean_weak, EPS_SPECTRUM)
        positives = sum(
            np.mean([response[(strong, client, bank)] for bank in ("a", "b")])
            > np.mean([response[(weak, client, bank)] for bank in ("a", "b")])
            for client in range(4)
        )
        clean_strong = float(np.mean([clean[(strong, client)] for client in range(4)]))
        clean_weak = float(np.mean([clean[(weak, client)] for client in range(4)]))
        r_clean = clean_strong / max(clean_weak, EPS_SPECTRUM)
        d_spec = float(np.log(r_resp + EPS_SPECTRUM) - np.log(r_clean + EPS_SPECTRUM))

        strong_boot = np.mean(
            np.stack([bootstrap[f"response_{strong}_c{client}_{bank}"] for client in range(4) for bank in ("a", "b")]),
            axis=0,
        )
        weak_boot = np.mean(
            np.stack([bootstrap[f"response_{weak}_c{client}_{bank}"] for client in range(4) for bank in ("a", "b")]),
            axis=0,
        )
        delta_boot = strong_boot - weak_boot
        clean_strong_boot = np.mean(
            np.stack([bootstrap[f"clean_{strong}_c{client}"] for client in range(4)]), axis=0
        )
        clean_weak_boot = np.mean(
            np.stack([bootstrap[f"clean_{weak}_c{client}"] for client in range(4)]), axis=0
        )
        r_resp_boot = strong_boot / np.maximum(weak_boot, EPS_SPECTRUM)
        r_clean_boot = clean_strong_boot / np.maximum(clean_weak_boot, EPS_SPECTRUM)
        d_spec_boot = np.log(r_resp_boot + EPS_SPECTRUM) - np.log(r_clean_boot + EPS_SPECTRUM)
        bootstrap[f"aggregate_{system}_delta_chi"] = delta_boot
        bootstrap[f"aggregate_{system}_r_resp"] = r_resp_boot
        bootstrap[f"aggregate_{system}_r_clean"] = r_clean_boot
        bootstrap[f"aggregate_{system}_d_spec"] = d_spec_boot

        letter = prefix[system]
        gates[f"{letter}1_bank_consistency"] = bool(
            bank_means["a"]["strong"] > bank_means["a"]["weak"]
            and bank_means["b"]["strong"] > bank_means["b"]["weak"]
        )
        gates[f"{letter}2_response_ratio_ge_1p25"] = bool(r_resp >= RESPONSE_RATIO_THRESHOLD)
        gates[f"{letter}3_positive_clients_ge_3of4"] = bool(positives >= 3)
        delta_ci = _ci(delta_boot)
        gates[f"{letter}4_delta_chi_ci95_lower_gt_0"] = bool(delta_ci["ci95_lower"] > 0.0)
        d_spec_ci = _ci(d_spec_boot)
        gates[f"{letter}5_response_specificity"] = bool(d_spec > 0.0 and d_spec_ci["ci95_lower"] > 0.0)
        summaries[system] = {
            "bank_means": bank_means,
            "mean_chi_strong": mean_strong,
            "mean_chi_weak": mean_weak,
            "R_resp": r_resp,
            "positive_clients": int(positives),
            "mean_clean_chi_strong": clean_strong,
            "mean_clean_chi_weak": clean_weak,
            "R_clean": r_clean,
            "D_spec": d_spec,
            "delta_chi_bootstrap": delta_ci,
            "D_spec_bootstrap": d_spec_ci,
        }
    if len(gates) != 10:
        raise AssertionError(f"K1-C0 requires exactly ten gates, got {len(gates)}")
    passed = int(sum(gates.values()))
    return {
        "verdict": "GO_TO_K1_C_CRSF_SURGERY" if passed == 10 else "NO_GO_RESPONSE_SPECTRAL_MECHANISM",
        "passed_gates": passed,
        "total_gates": 10,
        "gates": gates,
        "summaries": summaries,
    }


def saved_artifact_recomputation(
    output_dir: Path,
    expected_response_rows: list[dict[str, object]],
    expected_clean_rows: list[dict[str, object]],
    expected_decision: dict[str, object],
) -> dict[str, object]:
    response_rows = []
    with np.load(output_dir / "response_gram_matrices.npz", allow_pickle=False) as archive:
        arms = ("h9", "h0", "l9", "l0")
        for arm in arms:
            for client in range(4):
                for bank in ("a", "b"):
                    u1 = spectrum_from_gram(archive[f"{arm}_c{client}_{bank}_u1"])
                    u2 = spectrum_from_gram(archive[f"{arm}_c{client}_{bank}_u2"])
                    response_rows.append(
                        {"arm": arm, "client": client, "bank": bank, "chi_resp": (u1.concentration + u2.concentration) / 2.0}
                    )
    clean_rows = []
    with np.load(output_dir / "clean_eigenvalues.npz", allow_pickle=False) as archive:
        for arm in ("h9", "h0", "l9", "l0"):
            for client in range(4):
                u1 = spectrum_from_eigenvalues(archive[f"{arm}_c{client}_u1"])
                u2 = spectrum_from_eigenvalues(archive[f"{arm}_c{client}_u2"])
                clean_rows.append(
                    {"arm": arm, "client": client, "chi_clean": (u1.concentration + u2.concentration) / 2.0}
                )
    with np.load(output_dir / "bootstrap_metrics.npz", allow_pickle=False) as archive:
        bootstrap = {key: archive[key].astype(np.float64) for key in archive.files if not key.startswith("aggregate_")}
    decision = decide_gates(response_rows, clean_rows, bootstrap)
    expected_response = {
        (row["arm"], int(row["client"]), row["bank"]): float(row["chi_resp"])
        for row in expected_response_rows
    }
    expected_clean = {
        (row["arm"], int(row["client"])): float(row["chi_clean"])
        for row in expected_clean_rows
    }
    response_mismatches = sum(
        not np.isclose(float(row["chi_resp"]), expected_response[(row["arm"], int(row["client"]), row["bank"])], rtol=1e-10, atol=1e-12)
        for row in response_rows
    )
    clean_mismatches = sum(
        not np.isclose(float(row["chi_clean"]), expected_clean[(row["arm"], int(row["client"]))], rtol=1e-10, atol=1e-12)
        for row in clean_rows
    )
    gate_mismatches = sum(
        bool(decision["gates"][name]) != bool(expected_decision["gates"][name])
        for name in expected_decision["gates"]
    )
    verdict_match = decision["verdict"] == expected_decision["verdict"]
    return {
        "response_metric_mismatches": int(response_mismatches),
        "clean_metric_mismatches": int(clean_mismatches),
        "gate_inconsistencies": int(gate_mismatches),
        "verdict_match": bool(verdict_match),
        "recomputed_verdict": decision["verdict"],
        "audit_pass": bool(response_mismatches == 0 and clean_mismatches == 0 and gate_mismatches == 0 and verdict_match),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    formal = args.mode == "formal"
    np.random.seed(BOOTSTRAP_SEED)
    torch.manual_seed(BOOTSTRAP_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = resolve_device(args.device)
    banks = load_banks()
    images = cifar100_train_images_from_tar(args.public_root.resolve())
    inspection = inspect_inputs(args, banks, images, device=device if args.mode == "inspect" else None)
    if inspection["d_rep_sha256"] != EXPECTED_D_REP_SHA256:
        raise ValueError("D_rep hash mismatch")
    if not inspection["all_16_checkpoints_present"]:
        raise FileNotFoundError("one or more frozen checkpoints are missing")
    if not inspection["u1_u2_disjoint"]:
        raise ValueError("U1 and U2 overlap")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "inspection.json", inspection)
    if args.mode == "inspect":
        result = {**inspection, "verdict": "INSPECT_PASS_NO_SCIENTIFIC_DECISION"}
        write_json(output_dir / "result.json", result)
        return result

    split = public_split(images.shape[0], discover_count=1000, surgery_count=2000, holdout_count=2000)
    half_count = 1000 if formal else 12
    probe_count = 64 if formal else 4
    bootstrap_samples = BOOTSTRAP_SAMPLES if formal else 8
    clients = tuple(range(4)) if formal else (0,)
    systems = SYSTEMS if formal else {"hfl": SYSTEMS["hfl"]}
    u1_indices = split["holdout"][:half_count]
    u2_indices = split["holdout"][1000 : 1000 + half_count]
    used_indices = np.concatenate((u1_indices, u2_indices))
    used_images = images[used_indices]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_u1_indices = rng.integers(0, half_count, size=(bootstrap_samples, half_count), dtype=np.int64)
    bootstrap_u2_indices = rng.integers(0, half_count, size=(bootstrap_samples, half_count), dtype=np.int64)
    counts_u1 = bootstrap_count_matrix(bootstrap_u1_indices, half_count)
    counts_u2 = bootstrap_count_matrix(bootstrap_u2_indices, half_count)

    checkpoint_hashes_before = {
        f"{arm}/client_{client}": sha256_file(args.checkpoint_root / arm / f"client_{client}.pt")
        for arms in systems.values()
        for arm in arms
        for client in clients
    }
    environment = {
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    configuration = {
        "protocol": "cle_k1_c0_response_spectrum_v1",
        "mode": args.mode,
        "scientific_decision_allowed": formal,
        "d_rep_frozen_count": 2000,
        "used_carriers_per_half": half_count,
        "probe_count_per_bank": probe_count,
        "all_64_probes_equal_status": formal,
        "rho_selection_enabled": False,
        "labels_loaded": False,
        "oracle_assets_loaded": False,
        "evaluation_assets_loaded": False,
        "training_performed": False,
        "optimizer_constructed": False,
        "backward_called": False,
        "checkpoint_written": False,
        "eps_energy": EPS_ENERGY,
        "eps_spectrum": EPS_SPECTRUM,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "response_ratio_threshold": RESPONSE_RATIO_THRESHOLD,
        "statistics_dtype": "float64",
    }
    write_json(output_dir / "config.json", configuration)
    write_json(output_dir / "environment.json", environment)
    write_json(
        output_dir / "input_manifest.json",
        {
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "public_root": str(args.public_root.resolve()),
            "checkpoint_count": len(checkpoint_hashes_before),
            "checkpoint_hashes": checkpoint_hashes_before,
            "d_rep_sha256": sha256_array(split["holdout"]),
            "bank_hashes": {name: banks[name]["bank_sha256"] for name in ("a", "b")},
            "evaluation_assets_read": False,
        },
    )
    write_json(
        output_dir / "public_split_manifest.json",
        {
            "d_rep_sha256": sha256_array(split["holdout"]),
            "frozen_u1_sha256": sha256_array(split["holdout"][:1000]),
            "frozen_u2_sha256": sha256_array(split["holdout"][1000:]),
            "used_u1_sha256": sha256_array(u1_indices),
            "used_u2_sha256": sha256_array(u2_indices),
            "used_u1_u2_disjoint": bool(np.intersect1d(u1_indices, u2_indices).size == 0),
            "bootstrap_u1_indices_sha256": sha256_array(bootstrap_u1_indices),
            "bootstrap_u2_indices_sha256": sha256_array(bootstrap_u2_indices),
        },
    )
    write_json(
        output_dir / "prime_bank_manifest.json",
        {
            name: {
                "sha256": banks[name]["bank_sha256"],
                "recipe_count": len(banks[name]["recipes"]),
                "used_recipe_ids": list(range(probe_count)),
            }
            for name in ("a", "b")
        },
    )
    write_json(
        output_dir / "protocol_freeze_audit.json",
        {
            "input_hashes_printed": True,
            "thresholds_frozen": True,
            "eps_energy": EPS_ENERGY,
            "eps_spectrum": EPS_SPECTRUM,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "all_64_probes_enabled": formal,
            "rho_selection_disabled": True,
            "oracle_assets_loaded": False,
        },
    )

    response_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    gate_input_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    response_grams: dict[str, np.ndarray] = {}
    response_eigenvalues: dict[str, np.ndarray] = {}
    clean_eigenvalues: dict[str, np.ndarray] = {}
    bootstrap: dict[str, np.ndarray] = {}
    smoke_checks: dict[str, bool] = {
        "checkpoint_hash_unchanged": True,
        "representation_reconstruction": True,
        "u1_u2_disjoint": bool(np.intersect1d(u1_indices, u2_indices).size == 0),
        "frozen_prime_recipe_reuse": True,
        "delta_h_finite": True,
        "mean_and_energy_finite": True,
        "gram_symmetric": True,
        "gram_psd": True,
        "chi_finite": True,
        "effective_rank_finite": True,
        "clean_covariance_spectrum_finite": True,
        "paired_bootstrap_indices_identical_for_9_0": True,
        "no_parameter_modification": True,
        "deterministic_rerun": True,
    }

    for system, (strong, weak) in systems.items():
        for arm in (strong, weak):
            for client in clients:
                log(f"[heartbeat] {system} arm={arm} client={client} start")
                models = build_models(list(MODEL_NAMES), num_classes=10)
                model = freeze_model(models[client], device)
                checkpoint = args.checkpoint_root / arm / f"client_{client}.pt"
                model.load_state_dict(load_state(checkpoint))
                model.eval()
                state_before = _state_digest(model)
                interface = verify_penultimate_interface(model, used_images[:8], device=device)
                feature_rows.append({"system": system, "arm": arm, "client": client, "model": MODEL_NAMES[client], "module_path": "model.backbone(...).flatten(1)", "classifier_path": "model.linear", **interface})
                base = extract_base_features(model, used_images, device=device, batch_size=args.batch_size)
                base_u1 = base[:half_count]
                base_u2 = base[half_count:]
                clean_u1 = clean_spectrum(base_u1)
                clean_u2 = clean_spectrum(base_u2)
                clean_eigenvalues[f"{arm}_c{client}_u1"] = clean_u1.eigenvalues
                clean_eigenvalues[f"{arm}_c{client}_u2"] = clean_u2.eigenvalues
                clean_boot_u1 = bootstrap_clean_concentration(base_u1, counts_u1, device=device, chunk_size=max(args.bootstrap_chunk_size, 16))
                clean_boot_u2 = bootstrap_clean_concentration(base_u2, counts_u2, device=device, chunk_size=max(args.bootstrap_chunk_size, 16))
                bootstrap[f"clean_{arm}_c{client}"] = (clean_boot_u1 + clean_boot_u2) / 2.0
                clean_rows.append(
                    {
                        "system": system,
                        "arm": arm,
                        "client": client,
                        "model": MODEL_NAMES[client],
                        "chi_clean_u1": clean_u1.concentration,
                        "chi_clean_u2": clean_u2.concentration,
                        "chi_clean": (clean_u1.concentration + clean_u2.concentration) / 2.0,
                        "effective_rank_u1": clean_u1.effective_rank,
                        "effective_rank_u2": clean_u2.effective_rank,
                        "top1_share_u1": clean_u1.top1_share,
                        "top1_share_u2": clean_u2.top1_share,
                        "top3_share_u1": clean_u1.top3_share,
                        "top3_share_u2": clean_u2.top3_share,
                    }
                )
                smoke_checks["clean_covariance_spectrum_finite"] &= bool(np.isfinite(clean_u1.concentration) and np.isfinite(clean_u2.concentration))

                for bank_name in ("a", "b"):
                    recipes = list(banks[bank_name]["recipes"][:probe_count])
                    delta = extract_all_deltas(
                        model,
                        used_images,
                        base,
                        recipes,
                        device=device,
                        batch_size=args.batch_size,
                        progress_prefix=f"{system} arm={arm} client={client} bank={bank_name}",
                    )
                    delta_u1 = delta[:half_count]
                    delta_u2 = delta[half_count:]
                    spec_u1 = response_spectrum(delta_u1)
                    spec_u2 = response_spectrum(delta_u2)
                    response_grams[f"{arm}_c{client}_{bank_name}_u1"] = spec_u1.gram
                    response_grams[f"{arm}_c{client}_{bank_name}_u2"] = spec_u2.gram
                    response_eigenvalues[f"{arm}_c{client}_{bank_name}_u1"] = spec_u1.statistics.eigenvalues
                    response_eigenvalues[f"{arm}_c{client}_{bank_name}_u2"] = spec_u2.statistics.eigenvalues
                    boot_u1 = bootstrap_response_concentration(
                        delta_u1,
                        counts_u1,
                        device=device,
                        chunk_size=args.bootstrap_chunk_size,
                    )
                    boot_u2 = bootstrap_response_concentration(
                        delta_u2,
                        counts_u2,
                        device=device,
                        chunk_size=args.bootstrap_chunk_size,
                    )
                    bootstrap[f"response_{arm}_c{client}_{bank_name}"] = (boot_u1 + boot_u2) / 2.0
                    row = {
                        "system": system,
                        "arm": arm,
                        "client": client,
                        "model": MODEL_NAMES[client],
                        "bank": bank_name,
                        "chi_u1": spec_u1.statistics.concentration,
                        "chi_u2": spec_u2.statistics.concentration,
                        "chi_resp": (spec_u1.statistics.concentration + spec_u2.statistics.concentration) / 2.0,
                        "effective_rank_u1": spec_u1.statistics.effective_rank,
                        "effective_rank_u2": spec_u2.statistics.effective_rank,
                        "top1_share_u1": spec_u1.statistics.top1_share,
                        "top1_share_u2": spec_u2.statistics.top1_share,
                        "top3_share_u1": spec_u1.statistics.top3_share,
                        "top3_share_u2": spec_u2.statistics.top3_share,
                        "mean_response_energy": (spec_u1.mean_response_energy + spec_u2.mean_response_energy) / 2.0,
                        "trace_k_u1": spec_u1.statistics.trace,
                        "trace_k_u2": spec_u2.statistics.trace,
                        "frob_s_sq_u1": spec_u1.statistics.trace,
                        "frob_s_sq_u2": spec_u2.statistics.trace,
                    }
                    response_rows.append(row)
                    if not formal:
                        delta64 = delta.astype(np.float64)
                        smoke_checks["delta_h_finite"] &= bool(np.isfinite(delta64).all())
                        smoke_checks["mean_and_energy_finite"] &= bool(
                            np.isfinite(delta64.mean(axis=0)).all()
                            and np.isfinite(np.square(delta64).sum(axis=-1).mean(axis=0)).all()
                        )
                        smoke_checks["gram_symmetric"] &= bool(
                            np.allclose(spec_u1.gram, spec_u1.gram.T, atol=1e-8, rtol=0.0)
                            and np.allclose(spec_u2.gram, spec_u2.gram.T, atol=1e-8, rtol=0.0)
                        )
                        raw_eigen_u1 = torch.linalg.eigvalsh(
                            torch.as_tensor(spec_u1.gram, dtype=torch.float64)
                        ).cpu().numpy()
                        raw_eigen_u2 = torch.linalg.eigvalsh(
                            torch.as_tensor(spec_u2.gram, dtype=torch.float64)
                        ).cpu().numpy()
                        smoke_checks["gram_psd"] &= bool(
                            raw_eigen_u1.min() >= -1e-8 * max(float(np.abs(raw_eigen_u1).max()), 1.0)
                            and raw_eigen_u2.min() >= -1e-8 * max(float(np.abs(raw_eigen_u2).max()), 1.0)
                        )
                        smoke_checks["chi_finite"] &= bool(
                            np.isfinite(spec_u1.statistics.concentration)
                            and np.isfinite(spec_u2.statistics.concentration)
                        )
                        smoke_checks["effective_rank_finite"] &= bool(
                            np.isfinite(spec_u1.statistics.effective_rank)
                            and np.isfinite(spec_u2.statistics.effective_rank)
                        )
                    if not formal and arm == strong and bank_name == "a":
                        repeated = extract_delta_for_recipe(
                            model,
                            used_images,
                            base,
                            recipes[0],
                            device=device,
                            batch_size=args.batch_size,
                        )
                        smoke_checks["frozen_prime_recipe_reuse"] &= bool(np.allclose(repeated, delta[:, 0], rtol=0.0, atol=1e-6))
                    del delta, delta_u1, delta_u2
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                smoke_checks["no_parameter_modification"] &= _state_digest(model) == state_before
                del model, models, base
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                log(f"[heartbeat] {system} arm={arm} client={client} complete")

    checkpoint_hashes_after = {
        key: sha256_file(args.checkpoint_root / key.split("/")[0] / f"{key.split('/')[1]}.pt")
        for key in checkpoint_hashes_before
    }
    smoke_checks["checkpoint_hash_unchanged"] = checkpoint_hashes_before == checkpoint_hashes_after
    smoke_checks["representation_reconstruction"] = all(row["max_abs_logit_error"] <= 1e-5 and row["argmax_identical"] for row in feature_rows)
    smoke_checks["deterministic_rerun"] = smoke_checks["frozen_prime_recipe_reuse"]

    write_json(output_dir / "checkpoint_manifest.json", {"before": checkpoint_hashes_before, "after": checkpoint_hashes_after, "identical": checkpoint_hashes_before == checkpoint_hashes_after})
    write_json(output_dir / "feature_hook_manifest.json", feature_rows)
    _write_csv(output_dir / "response_spectrum_metrics.csv", response_rows)
    _write_csv(output_dir / "clean_spectrum_metrics.csv", clean_rows)
    np.savez_compressed(output_dir / "response_gram_matrices.npz", **response_grams)
    np.savez_compressed(output_dir / "response_eigenvalues.npz", **response_eigenvalues)
    np.savez_compressed(output_dir / "clean_eigenvalues.npz", **clean_eigenvalues)

    if formal:
        decision = decide_gates(response_rows, clean_rows, bootstrap)
        for system, summary in decision["summaries"].items():
            strong, weak = SYSTEMS[system]
            for client in range(4):
                strong_chi = float(np.mean([next(row["chi_resp"] for row in response_rows if row["arm"] == strong and row["client"] == client and row["bank"] == bank) for bank in ("a", "b")]))
                weak_chi = float(np.mean([next(row["chi_resp"] for row in response_rows if row["arm"] == weak and row["client"] == client and row["bank"] == bank) for bank in ("a", "b")]))
                gate_input_rows.append({"system": system, "client": client, "strong_arm": strong, "weak_arm": weak, "strong_chi": strong_chi, "weak_chi": weak_chi, "delta_chi": strong_chi - weak_chi, "positive": strong_chi > weak_chi, "system_R_resp": summary["R_resp"], "system_R_clean": summary["R_clean"], "system_D_spec": summary["D_spec"]})
        np.savez_compressed(output_dir / "bootstrap_metrics.npz", **bootstrap)
        write_json(output_dir / "bootstrap_response_effect.json", {system: decision["summaries"][system]["delta_chi_bootstrap"] for system in SYSTEMS})
        write_json(output_dir / "bootstrap_response_specificity.json", {system: decision["summaries"][system]["D_spec_bootstrap"] for system in SYSTEMS})
        _write_csv(output_dir / "per_client_gate_inputs.csv", gate_input_rows)
        write_json(output_dir / "gate_table.json", decision["gates"])
        audit = saved_artifact_recomputation(output_dir, response_rows, clean_rows, decision)
        write_json(output_dir / "saved_artifact_audit.json", audit)
        if not audit["audit_pass"]:
            decision = {**decision, "verdict": "AUDIT_FAIL"}
    else:
        np.savez_compressed(output_dir / "bootstrap_metrics.npz", **bootstrap)
        decision = {
            "verdict": "SMOKE_ONLY_NO_SCIENTIFIC_DECISION",
            "passed_gates": None,
            "total_gates": 10,
            "gates": {},
            "summaries": {},
        }
        write_json(output_dir / "smoke_checks.json", smoke_checks)
        if not all(smoke_checks.values()):
            raise RuntimeError(f"K1-C0 smoke failed: {smoke_checks}")
        audit = {"audit_pass": True, "scope": "smoke_execution_only"}

    result = {
        "protocol": "cle_k1_c0_response_spectrum_v1",
        "mode": args.mode,
        "scientific_decision_allowed": formal,
        **decision,
        "configuration": configuration,
        "split": {
            "d_rep_sha256": sha256_array(split["holdout"]),
            "u1_sha256": sha256_array(split["holdout"][:1000]),
            "u2_sha256": sha256_array(split["holdout"][1000:]),
        },
        "bank_hashes": {name: banks[name]["bank_sha256"] for name in ("a", "b")},
        "checkpoint_hashes_before": checkpoint_hashes_before,
        "checkpoint_hashes_after": checkpoint_hashes_after,
        "saved_artifact_audit": audit,
        "smoke_checks": smoke_checks if not formal else None,
        "response_rows": response_rows,
        "clean_rows": clean_rows,
    }
    write_json(output_dir / "result.json", result)
    report = [
        "# CLE K1-C0 Response Spectrum Result",
        "",
        f"- 模式：`{args.mode}`",
        f"- 结论：`{result['verdict']}`",
        f"- 正式门控：`{result['passed_gates']}/{result['total_gates']}`",
        "- 训练、微调、通信或 checkpoint 写入：`均无`",
        "- 标签、CLE taxonomy、DSA/WCCA/CFG：`均未读取`",
        "- 统计对象：全部等权 PRIME probe 的跨 carrier 表征响应谱",
    ]
    (output_dir / "FINAL_REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    artifact_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifact_rows.append({"path": path.relative_to(output_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output_dir / "artifact_manifest.json", {"files": artifact_rows})
    return result


def main() -> None:
    args = parse_args()
    result = run_analysis(args)
    print(json.dumps({"mode": args.mode, "verdict": result["verdict"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
