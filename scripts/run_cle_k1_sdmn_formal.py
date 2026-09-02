from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.augmentations.frozen_prime import load_frozen_prime_bank  # noqa: E402
from fedprime.data.loaders import cifar100_train_images_from_tar, dataset_stats, normalize_batch  # noqa: E402
from fedprime.models.factory import build_models  # noqa: E402
from scripts.run_cle_k1_sdmn_headonly import (  # noqa: E402
    ARMS,
    BANK_A_SHA256,
    BANK_B_SHA256,
    BANK_ROOT,
    MODEL_NAMES,
    freeze_model_for_head_only,
    load_state,
    manifest_files,
    public_split,
    resolve_device,
    run_one_fold_client,
    sha256_array,
    sha256_file,
    write_json,
)


SYSTEMS = ("h9", "l9")
FOLDS = (("ab", "a", "b"), ("ba", "b", "a"))
FORMAL_PROTOCOL = "cle_k1a_sdmn_headonly_formal_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal CLE K1-A head-only SDMN checkpoint surgery.")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cle_k1_sdmn_headonly_seed0_formal"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def load_frozen_contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "cle_k1a_sdmn_headonly_formal_freeze_v1":
        raise ValueError("unexpected K1-A calibration manifest protocol")
    contract = payload["optimizer_contract"]
    expected = {
        "optimizer": "adam",
        "formal_steps": 10,
        "formal_anchor_kl_limit": 0.02,
        "backtracking_factor": 0.5,
        "maximum_backtracks": 12,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"formal optimizer contract changed at {key}")
    for system in SYSTEMS:
        for fold_name, _train_bank, _unseen_bank in FOLDS:
            values = payload["learning_rates"][system][fold_name]
            if len(values) != 4 or any(float(value) not in (1.0e-4, 3.0e-4) for value in values):
                raise ValueError(f"invalid frozen LR vector for {system}/{fold_name}")
    return payload


def split_manifest(split: dict[str, np.ndarray], contract: dict[str, object]) -> dict[str, object]:
    hashes = {name: sha256_array(indices) for name, indices in split.items()}
    frozen = contract["public_split"]
    for name in ("discover", "surgery", "holdout"):
        expected = str(frozen[f"{name if name != 'holdout' else 'formal_holdout'}_sha256"])
        if hashes[name] != expected:
            raise ValueError(f"formal public {name} split hash mismatch")
    return {
        "discover_seed": 20260901,
        "split_seed": 20260906,
        "counts": {name: int(indices.size) for name, indices in split.items()},
        "hashes": hashes,
        "pairwise_disjoint": bool(
            np.intersect1d(split["discover"], split["surgery"]).size == 0
            and np.intersect1d(split["discover"], split["holdout"]).size == 0
            and np.intersect1d(split["surgery"], split["holdout"]).size == 0
        ),
        "public_labels_loaded_or_used": False,
    }


def primary_file_manifest(output_dir: Path) -> dict[str, object]:
    allowed_roots = (
        "config",
        "data_split_manifest",
        "probe_selection",
        "directions",
        "feature_cache_manifest",
        "optimization_traces",
        "checkpoints",
        "unseen_bank_metrics",
        "bootstrap/primary_unseen_R.json",
        "primary_result.json",
    )
    rows = []
    for path in sorted(path for path in output_dir.rglob("*") if path.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if not any(relative == root or relative.startswith(root + "/") for root in allowed_roots):
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "protocol": "cle_k1a_primary_taxonomy_free_seal_v1",
        "sealed_before_oracle_import": True,
        "files": rows,
    }


def aggregate_primary(results: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for system in SYSTEMS:
        system_rows = [row for row in results if row["checkpoint_arm"] == system]
        fold_payload = {}
        for fold_name, _train, _unseen in FOLDS:
            rows = [row for row in system_rows if row["fold"] == fold_name]
            arm_means = {
                arm: float(np.mean([row["metrics"][arm]["R"] for row in rows])) for arm in ARMS
            }
            baseline = arm_means["frozen"]
            reductions = {
                arm: float((baseline - value) / max(abs(baseline), 1.0e-12))
                for arm, value in arm_means.items()
            }
            fold_payload[fold_name] = {
                "R_mean": arm_means,
                "relative_R_reduction": reductions,
                "client_R": {
                    arm: [float(row["metrics"][arm]["R"]) for row in rows] for arm in ARMS
                },
            }
        arm_means = {
            arm: float(np.mean([row["metrics"][arm]["R"] for row in system_rows])) for arm in ARMS
        }
        baseline = arm_means["frozen"]
        reductions = {
            arm: float((baseline - value) / max(abs(baseline), 1.0e-12))
            for arm, value in arm_means.items()
        }
        client_positive = []
        for client_id in range(4):
            client_rows = [row for row in system_rows if row["client"] == client_id]
            frozen = np.mean([row["metrics"]["frozen"]["R"] for row in client_rows])
            targeted = np.mean([row["metrics"]["targeted"]["R"] for row in client_rows])
            client_positive.append(bool(targeted < frozen))
        gates = {
            "combined_targeted_reduction_ge_25pct": reductions["targeted"] >= 0.25,
            "ab_targeted_reduction_ge_15pct": fold_payload["ab"]["relative_R_reduction"]["targeted"] >= 0.15,
            "ba_targeted_reduction_ge_15pct": fold_payload["ba"]["relative_R_reduction"]["targeted"] >= 0.15,
            "positive_clients_ge_3of4": int(sum(client_positive)) >= 3,
            "targeted_minus_sham_ge_10pp": reductions["targeted"] - reductions["direction_sham"] >= 0.10,
            "targeted_minus_random_ge_10pp": reductions["targeted"] - reductions["random_probe"] >= 0.10,
        }
        summary[system] = {
            "folds": fold_payload,
            "combined_R_mean": arm_means,
            "combined_relative_R_reduction": reductions,
            "positive_clients": int(sum(client_positive)),
            "gates": gates,
            "pass": all(gates.values()),
        }
    return summary


def bootstrap_primary(results: list[dict[str, object]], *, samples: int, seed: int = 20260908) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    output: dict[str, object] = {"samples": int(samples), "seed": int(seed), "used_for_gate": False}
    for system in SYSTEMS:
        rows = [row for row in results if row["checkpoint_arm"] == system]
        matrix = {
            arm: np.asarray([row["metrics"][arm]["R"] for row in rows], dtype=np.float64)
            for arm in ARMS
        }
        draws = {"targeted_reduction": [], "targeted_minus_sham": [], "targeted_minus_random": []}
        for _ in range(int(samples)):
            indices = rng.integers(0, len(rows), size=len(rows))
            baseline = float(matrix["frozen"][indices].mean())
            reduction = {
                arm: (baseline - float(values[indices].mean())) / max(abs(baseline), 1.0e-12)
                for arm, values in matrix.items()
            }
            draws["targeted_reduction"].append(reduction["targeted"])
            draws["targeted_minus_sham"].append(reduction["targeted"] - reduction["direction_sham"])
            draws["targeted_minus_random"].append(reduction["targeted"] - reduction["random_probe"])
        output[system] = {
            name: {
                "mean": float(np.mean(values)),
                "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            }
            for name, values in draws.items()
        }
    return output


def infer_probabilities_from_head(
    features: np.ndarray,
    state: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    weight = state["linear.weight"].to(device=device, dtype=torch.float32)
    bias = state["linear.bias"].to(device=device, dtype=torch.float32)
    output = np.empty((features.shape[0], weight.shape[0]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, features.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), features.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(features[start:stop])).to(device)
            output[start:stop] = torch.softmax(torch.nn.functional.linear(batch, weight, bias), dim=-1).cpu().numpy()
    return output


def backbone_features(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    stats = dataset_stats("cifar10")
    output = []
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, images.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), images.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(images[start:stop]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            output.append(model.backbone(normalize_batch(batch, stats)).flatten(1).cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0)


def clean_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    client = 100.0 * (probabilities.argmax(axis=-1) == labels[None]).mean(axis=1)
    return {"avg_acc": float(client.mean()), "worst_acc": float(client.min()), "client_acc": client.tolist()}


def oracle_evaluation(
    *,
    output_dir: Path,
    checkpoint_root: Path,
    evaluation_root: Path,
    device: torch.device,
    batch_size: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    # The taxonomy-bearing module is deliberately imported only after the primary seal exists.
    if not (output_dir / "primary_artifact_manifest.json").is_file():
        raise RuntimeError("primary taxonomy-free artifacts must be sealed before oracle evaluation")
    oracle = importlib.import_module("fedprime.engine.cle_shortcut_alignment")
    clean_images = np.load(evaluation_root / "test_images.npy", allow_pickle=False)
    labels = np.load(evaluation_root / "test_labels.npy", allow_pickle=False).astype(np.int64)
    grid, severities = oracle.deterministic_corruption_grid(clean_images)
    binding = oracle.historical_family_binding(num_clients=4, num_classes=10)
    flat_grid = grid.reshape(-1, *grid.shape[2:])
    result: dict[str, object] = {}
    prediction_dir = output_dir / "oracle_metrics" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for system in SYSTEMS:
        probabilities: dict[str, dict[str, list[np.ndarray]]] = {
            fold: {arm: [] for arm in ARMS} for fold, _train, _unseen in FOLDS
        }
        clean_probabilities: dict[str, dict[str, list[np.ndarray]]] = {
            fold: {arm: [] for arm in ARMS} for fold, _train, _unseen in FOLDS
        }
        for client_id in range(4):
            models = build_models(list(MODEL_NAMES), num_classes=10)
            model = freeze_model_for_head_only(models[client_id], device)
            original = checkpoint_root / system / f"client_{client_id}.pt"
            model.load_state_dict(load_state(original), strict=True)
            grid_features = backbone_features(model, flat_grid, device=device, batch_size=batch_size)
            clean_features = backbone_features(model, clean_images, device=device, batch_size=batch_size)
            for fold_name, _train_bank, _unseen_bank in FOLDS:
                for arm in ARMS:
                    checkpoint = output_dir / "checkpoints" / f"{system}_{fold_name}_client{client_id}_{arm}.pt"
                    state = load_state(checkpoint)
                    probs = infer_probabilities_from_head(
                        grid_features, state, device=device, batch_size=batch_size
                    ).reshape(clean_images.shape[0], grid.shape[1], 10)
                    clean_probs = infer_probabilities_from_head(
                        clean_features, state, device=device, batch_size=batch_size
                    )
                    probabilities[fold_name][arm].append(probs)
                    clean_probabilities[fold_name][arm].append(clean_probs)
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        system_payload: dict[str, object] = {}
        for fold_name, _train_bank, _unseen_bank in FOLDS:
            fold_payload: dict[str, object] = {}
            for arm in ARMS:
                probs = np.stack(probabilities[fold_name][arm], axis=0)
                clean_probs = np.stack(clean_probabilities[fold_name][arm], axis=0)
                dsa = oracle.compute_dsa(probs, labels, binding, oracle.OPERATOR_FAMILY_IDS)
                secondary = oracle.secondary_metrics(probs, labels, binding, oracle.OPERATOR_FAMILY_IDS)
                clean = clean_metrics(clean_probs, labels)
                prediction_path = prediction_dir / f"{system}_{fold_name}_{arm}.npz"
                np.savez_compressed(
                    prediction_path,
                    probabilities=probs,
                    clean_probabilities=clean_probs,
                    labels=labels,
                    binding=binding,
                    operator_family_ids=oracle.OPERATOR_FAMILY_IDS,
                    severities=severities,
                )
                fold_payload[arm] = {
                    "dsa": float(dsa.pooled),
                    "dsa_client": dsa.client.tolist(),
                    "secondary": secondary,
                    "clean": clean,
                    "prediction_sha256": sha256_file(prediction_path),
                }
            system_payload[fold_name] = fold_payload
        result[system] = system_payload
    result["bootstrap_samples"] = int(bootstrap_samples)
    return result


def aggregate_oracle(oracle_result: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for system in SYSTEMS:
        folds = oracle_result[system]
        combined = {}
        for arm in ARMS:
            combined[arm] = {
                "dsa": float(np.mean([folds[name][arm]["dsa"] for name, _a, _b in FOLDS])),
                "dsa_client": np.mean(
                    [folds[name][arm]["dsa_client"] for name, _a, _b in FOLDS], axis=0
                ).tolist(),
                "avg_acc": float(np.mean([folds[name][arm]["secondary"]["avg_acc"] for name, _a, _b in FOLDS])),
                "worst_acc": float(np.mean([folds[name][arm]["secondary"]["worst_acc"] for name, _a, _b in FOLDS])),
                "wcca": float(np.mean([folds[name][arm]["secondary"]["wcca"] for name, _a, _b in FOLDS])),
                "cfg": float(np.mean([folds[name][arm]["secondary"]["cfg"] for name, _a, _b in FOLDS])),
                "clean_avg": float(np.mean([folds[name][arm]["clean"]["avg_acc"] for name, _a, _b in FOLDS])),
                "clean_worst": float(np.mean([folds[name][arm]["clean"]["worst_acc"] for name, _a, _b in FOLDS])),
            }
        baseline = combined["frozen"]
        targeted = combined["targeted"]
        sham = combined["direction_sham"]
        dsa_decrease = baseline["dsa"] - targeted["dsa"]
        dsa_relative = dsa_decrease / max(abs(baseline["dsa"]), 1.0e-12)
        client_delta = np.asarray(combined["frozen"]["dsa_client"]) - np.asarray(targeted["dsa_client"])
        sham_decrease = baseline["dsa"] - sham["dsa"]
        dsa_gates = {
            "absolute_ge_0.05_or_relative_ge_25pct": dsa_decrease >= 0.05 or dsa_relative >= 0.25,
            "positive_clients_ge_3of4": int((client_delta > 0).sum()) >= 3,
            "targeted_minus_sham_decrease_ge_0.02": dsa_decrease - sham_decrease >= 0.02,
        }
        deltas = {
            metric: targeted[metric] - baseline[metric]
            for metric in ("avg_acc", "worst_acc", "wcca", "cfg", "clean_avg", "clean_worst")
        }
        utility_gates = {
            "wcca_delta_ge_1pp": deltas["wcca"] >= 1.0,
            "cfg_delta_le_minus_1pp": deltas["cfg"] <= -1.0,
            "avg_delta_ge_minus_1pp": deltas["avg_acc"] >= -1.0,
            "worst_delta_ge_minus_1pp": deltas["worst_acc"] >= -1.0,
            "clean_avg_delta_ge_minus_1pp": deltas["clean_avg"] >= -1.0,
            "clean_worst_delta_ge_minus_1pp": deltas["clean_worst"] >= -1.0,
        }
        output[system] = {
            "combined": combined,
            "targeted_dsa_decrease": float(dsa_decrease),
            "targeted_dsa_relative_decrease": float(dsa_relative),
            "targeted_positive_dsa_clients": int((client_delta > 0).sum()),
            "targeted_task_deltas": deltas,
            "dsa_gates": dsa_gates,
            "utility_gates": utility_gates,
            "dsa_pass": all(dsa_gates.values()),
            "utility_pass": all(utility_gates.values()),
        }
    return output


def bootstrap_oracle(
    oracle_result: dict[str, object], *, samples: int, seed: int = 20260909
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    output: dict[str, object] = {"samples": int(samples), "seed": int(seed), "used_for_gate": False}
    for system in SYSTEMS:
        baseline = []
        targeted = []
        sham = []
        for fold_name, _train, _unseen in FOLDS:
            fold = oracle_result[system][fold_name]
            baseline.extend(fold["frozen"]["dsa_client"])
            targeted.extend(fold["targeted"]["dsa_client"])
            sham.extend(fold["direction_sham"]["dsa_client"])
        baseline = np.asarray(baseline, dtype=np.float64)
        targeted = np.asarray(targeted, dtype=np.float64)
        sham = np.asarray(sham, dtype=np.float64)
        target_draws = []
        specificity_draws = []
        for _ in range(int(samples)):
            indices = rng.integers(0, baseline.size, size=baseline.size)
            target_decrease = float((baseline[indices] - targeted[indices]).mean())
            sham_decrease = float((baseline[indices] - sham[indices]).mean())
            target_draws.append(target_decrease)
            specificity_draws.append(target_decrease - sham_decrease)
        output[system] = {
            "targeted_DSA_decrease": {
                "mean": float(np.mean(target_draws)),
                "ci95": np.quantile(target_draws, [0.025, 0.975]).tolist(),
            },
            "targeted_minus_sham_DSA_decrease": {
                "mean": float(np.mean(specificity_draws)),
                "ci95": np.quantile(specificity_draws, [0.025, 0.975]).tolist(),
            },
        }
    return output


def generic_invariance_dominates(
    primary: dict[str, object], oracle_summary: dict[str, object]
) -> bool:
    for system in SYSTEMS:
        r = primary[system]["combined_relative_R_reduction"]
        metrics = oracle_summary[system]["combined"]
        baseline = metrics["frozen"]
        targeted = metrics["targeted"]
        generic = metrics["generic_invariance"]
        if r["generic_invariance"] < r["targeted"]:
            return False
        if baseline["dsa"] - generic["dsa"] < baseline["dsa"] - targeted["dsa"]:
            return False
        if generic["wcca"] - baseline["wcca"] < targeted["wcca"] - baseline["wcca"]:
            return False
        if generic["cfg"] - baseline["cfg"] > targeted["cfg"] - baseline["cfg"]:
            return False
        for metric in ("avg_acc", "worst_acc", "clean_avg", "clean_worst"):
            if generic[metric] < targeted[metric]:
                return False
    return True


def decide(primary: dict[str, object], oracle_summary: dict[str, object]) -> dict[str, object]:
    primary_pass = all(bool(primary[system]["pass"]) for system in SYSTEMS)
    dsa_pass = all(bool(oracle_summary[system]["dsa_pass"]) for system in SYSTEMS)
    utility_pass = all(bool(oracle_summary[system]["utility_pass"]) for system in SYSTEMS)
    gi_dominance = generic_invariance_dominates(primary, oracle_summary)
    mechanism_pass = primary_pass and dsa_pass
    if not mechanism_pass:
        verdict = "NO_GO_DIRECTIONAL_SURGERY"
    elif utility_pass and not gi_dominance:
        verdict = "GO_TO_TRAINING_INTEGRATION"
    else:
        verdict = "MECHANISM_PASS_INTEGRATION_NEEDS_REDESIGN"
    return {
        "verdict": verdict,
        "primary_unseen_R_and_specificity_pass": primary_pass,
        "oracle_DSA_pass": dsa_pass,
        "utility_pass": utility_pass,
        "generic_invariance_dominance": gi_dominance,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_frozen_contract(args.calibration_manifest.resolve())
    banks = {
        "a": load_frozen_prime_bank(
            state_path=BANK_ROOT / "bank_a_states.npz", manifest_path=BANK_ROOT / "bank_a_manifest.json"
        ),
        "b": load_frozen_prime_bank(
            state_path=BANK_ROOT / "bank_b_states.npz", manifest_path=BANK_ROOT / "bank_b_manifest.json"
        ),
    }
    if banks["a"]["bank_sha256"] != BANK_A_SHA256 or banks["b"]["bank_sha256"] != BANK_B_SHA256:
        raise ValueError("frozen K0-B bank hash mismatch")
    images = cifar100_train_images_from_tar(args.public_root.resolve())
    split = public_split(images.shape[0], discover_count=1000, surgery_count=2000, holdout_count=2000)
    split_info = split_manifest(split, contract)
    split_dir = output_dir / "data_split_manifest"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in split.items():
        np.save(split_dir / f"{name}_indices.npy", indices, allow_pickle=False)
    write_json(split_dir / "manifest.json", split_info)
    config = {
        "protocol": FORMAL_PROTOCOL,
        "scientific_decision_allowed": True,
        "calibration_manifest": {
            "path": args.calibration_manifest.resolve().as_posix(),
            "sha256": sha256_file(args.calibration_manifest.resolve()),
        },
        "optimizer_contract": contract["optimizer_contract"],
        "public_split": split_info,
        "bank_hashes": {name: banks[name]["bank_sha256"] for name in ("a", "b")},
        "folds": {"ab": "train A, unseen-evaluate B", "ba": "train B, unseen-evaluate A"},
        "arms": list(ARMS),
        "primary_forbidden": contract["formal_forbidden_until_primary_seal"],
        "full_training_performed": False,
        "communication_modified": False,
    }
    write_json(output_dir / "config" / "config.json", config)
    write_json(output_dir / "config" / "frozen_calibration_manifest.json", contract)
    checkpoint_rows = []
    for system in SYSTEMS:
        for client_id in range(4):
            checkpoint = args.checkpoint_root.resolve() / system / f"client_{client_id}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_rows.append(
                {
                    "system": system,
                    "client": client_id,
                    "bytes": checkpoint.stat().st_size,
                    "sha256": sha256_file(checkpoint),
                }
            )
    evaluation_rows = {}
    for name in ("test_images.npy", "test_labels.npy"):
        path = args.evaluation_root.resolve() / name
        if not path.is_file():
            raise FileNotFoundError(path)
        evaluation_rows[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    preflight = {
        "protocol": FORMAL_PROTOCOL,
        "verdict": "PREFLIGHT_PASS_NO_SCIENTIFIC_DECISION",
        "checkpoints": checkpoint_rows,
        "evaluation_assets": evaluation_rows,
        "split": split_info,
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest.resolve()),
    }
    write_json(output_dir / "PREFLIGHT.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2), flush=True)
        return
    device = resolve_device(args.device)
    results = []
    for system in SYSTEMS:
        for client_id in range(4):
            models = build_models(list(MODEL_NAMES), num_classes=10)
            model = freeze_model_for_head_only(models[client_id], device)
            checkpoint = args.checkpoint_root.resolve() / system / f"client_{client_id}.pt"
            model.load_state_dict(load_state(checkpoint), strict=True)
            for fold_name, train_bank, unseen_bank in FOLDS:
                learning_rate = float(contract["learning_rates"][system][fold_name][client_id])
                row = run_one_fold_client(
                    model=model,
                    checkpoint_arm=system,
                    client_id=client_id,
                    fold_name=fold_name,
                    discover_images=images[split["discover"]],
                    surgery_images=images[split["surgery"]],
                    holdout_images=images[split["holdout"]],
                    discover_recipes=list(banks[train_bank]["recipes"]),
                    holdout_recipes=list(banks[unseen_bank]["recipes"]),
                    device=device,
                    batch_size=args.batch_size,
                    learning_rate=learning_rate,
                    steps=int(contract["optimizer_contract"]["formal_steps"]),
                    output_dir=output_dir,
                )
                row.update(
                    {
                        "checkpoint_arm": system,
                        "model": MODEL_NAMES[client_id],
                        "train_bank": train_bank,
                        "unseen_bank": unseen_bank,
                        "learning_rate": learning_rate,
                    }
                )
                results.append(row)
                write_json(output_dir / "unseen_bank_metrics" / f"{system}_{fold_name}_client{client_id}.json", row)
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_json(
        output_dir / "feature_cache_manifest" / "manifest.json",
        {
            f"{row['checkpoint_arm']}_{row['fold']}_client{row['client']}": row["feature_cache_manifest"]
            for row in results
        },
    )
    primary_summary = aggregate_primary(results)
    write_json(
        output_dir / "bootstrap" / "primary_unseen_R.json",
        bootstrap_primary(results, samples=args.bootstrap_samples),
    )
    primary_result = {
        "protocol": FORMAL_PROTOCOL,
        "phase": "PRIMARY_TAXONOMY_FREE_COMPLETE",
        "oracle_loaded": False,
        "summary": primary_summary,
        "results": results,
    }
    write_json(output_dir / "primary_result.json", primary_result)
    write_json(output_dir / "primary_artifact_manifest.json", primary_file_manifest(output_dir))

    oracle_result = oracle_evaluation(
        output_dir=output_dir,
        checkpoint_root=args.checkpoint_root.resolve(),
        evaluation_root=args.evaluation_root.resolve(),
        device=device,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_json(output_dir / "oracle_metrics" / "metrics.json", oracle_result)
    oracle_summary = aggregate_oracle(oracle_result)
    write_json(output_dir / "task_metrics" / "summary.json", oracle_summary)
    write_json(
        output_dir / "bootstrap" / "oracle_DSA.json",
        bootstrap_oracle(oracle_result, samples=args.bootstrap_samples),
    )
    decision = decide(primary_summary, oracle_summary)
    result = {
        "protocol": FORMAL_PROTOCOL,
        **decision,
        "primary": primary_summary,
        "oracle_and_task": oracle_summary,
        "primary_artifact_manifest_sha256": sha256_file(output_dir / "primary_artifact_manifest.json"),
    }
    write_json(output_dir / "result.json", result)
    report = [
        "# CLE K1-A Formal Result",
        "",
        f"Verdict: `{decision['verdict']}`",
        "",
        "Primary taxonomy-free artifacts were sealed before loading CLE binding or task-oracle metadata.",
        "No full training or communication modification was performed.",
    ]
    (output_dir / "FINAL_REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(output_dir / "artifact_manifest.json", manifest_files(output_dir))
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
