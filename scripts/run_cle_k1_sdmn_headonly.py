from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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
from fedprime.data.loaders import (  # noqa: E402
    cifar100_train_images_from_tar,
    dataset_stats,
    normalize_batch,
)
from fedprime.engine.cle_generic_probe_gate import generic_probe_statistics  # noqa: E402
from fedprime.engine.cle_sdmn_headonly import (  # noqa: E402
    centered_response_from_features,
    make_direction_sham,
    match_random_probes,
    run_head_surgery,
    select_high_risk_probes,
)
from fedprime.models.factory import build_models  # noqa: E402


MODEL_NAMES = ("ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2")
DISCOVER_SEED = 20260901
SPLIT_SEED = 20260906
SHAM_SEED = 20260907
BANK_ROOT = ROOT / "fedprime/augmentations/assets/cle_generic_probe_k0b"
BANK_A_SHA256 = "6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01"
BANK_B_SHA256 = "4A53497EC5DB6EC05C312E6166109FA4B52A5CC402CCE74E6EDB1253D913BF4E"
LR_CANDIDATES = (1.0e-4, 3.0e-4, 1.0e-3)
ARMS = ("frozen", "targeted", "direction_sham", "random_probe", "generic_invariance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLE K1-A head-only SDMN checkpoint surgery.")
    parser.add_argument("--mode", choices=("inspect", "smoke", "calibration", "formal"), default="inspect")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cle_k1_sdmn_headonly"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--formal-calibration-manifest", type=Path, default=None)
    parser.add_argument("--formal-surgery-steps", type=int, default=None)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def public_split(total: int, *, discover_count: int, surgery_count: int, holdout_count: int) -> dict[str, np.ndarray]:
    if discover_count + surgery_count + holdout_count > int(total):
        raise ValueError("public split exceeds dataset size")
    discover_rng = np.random.default_rng(DISCOVER_SEED)
    frozen_discover = discover_rng.choice(total, size=1000, replace=False).astype(np.int64)
    discover = frozen_discover[: int(discover_count)]
    available = np.setdiff1d(np.arange(total, dtype=np.int64), frozen_discover, assume_unique=False)
    split_rng = np.random.default_rng(SPLIT_SEED)
    # Freeze the complete 2,000-image surgery pool before drawing holdout.  This
    # preserves the exact calibration surgery set when formal adds a holdout set;
    # np.random.choice(..., size=4000) is not prefix-stable with size=2000.
    frozen_surgery = split_rng.choice(available, size=2000, replace=False).astype(np.int64)
    if int(surgery_count) > frozen_surgery.size:
        raise ValueError("surgery_count exceeds the frozen K1-A surgery pool")
    surgery = frozen_surgery[: int(surgery_count)]
    holdout_candidates = available[~np.isin(available, frozen_surgery)]
    holdout = split_rng.choice(
        holdout_candidates,
        size=int(holdout_count),
        replace=False,
    ).astype(np.int64)
    if np.intersect1d(discover, surgery).size or np.intersect1d(discover, holdout).size:
        raise AssertionError("discover overlaps surgery/holdout")
    if np.intersect1d(surgery, holdout).size:
        raise AssertionError("surgery overlaps holdout")
    return {"discover": discover, "surgery": surgery, "holdout": holdout}


def extract_features(
    model: torch.nn.Module,
    images: np.ndarray,
    recipes: list[dict[str, object]] | None,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    stats = dataset_stats("cifar10")

    def forward_batch(batch_images: np.ndarray, recipe: dict[str, object] | None) -> np.ndarray:
        result = []
        with torch.inference_mode():
            for start in range(0, batch_images.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), batch_images.shape[0])
                batch = torch.from_numpy(np.ascontiguousarray(batch_images[start:stop]))
                batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
                if recipe is not None:
                    batch = apply_frozen_prime_recipe(batch, recipe)
                features = model.backbone(normalize_batch(batch, stats)).flatten(1)
                result.append(features.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(result, axis=0)

    base = forward_batch(images, None)
    if recipes is None:
        return base, None
    probes = np.empty((images.shape[0], len(recipes), base.shape[1]), dtype=np.float32)
    for recipe_id, recipe in enumerate(recipes):
        probes[:, recipe_id] = forward_batch(images, recipe)
    return base, probes


def response_from_numpy_features(
    head: torch.nn.Linear,
    base: np.ndarray,
    probes: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        response = centered_response_from_features(
            head,
            torch.from_numpy(base).to(device),
            torch.from_numpy(probes).to(device),
        )
    return response.cpu().numpy().astype(np.float32)


def arm_metrics(response: np.ndarray) -> dict[str, float | int]:
    result = generic_probe_statistics(np.asarray(response, dtype=np.float64)[None])
    return {
        "S": float(result.S[0]),
        "Dcf": float(result.Dcf[0]),
        "K": float(result.K[0]),
        "R": float(result.R[0]),
        "active_probes": int(result.active[0].sum()),
    }


def freeze_model_for_head_only(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "backbone") or not hasattr(model, "linear"):
        raise AttributeError("K1-A requires model.backbone and model.linear")
    if not isinstance(model.linear, torch.nn.Linear):
        raise TypeError("K1-A classifier head must be torch.nn.Linear")
    for parameter in model.linear.parameters():
        parameter.requires_grad_(True)
    return model


def selection_payload(selection, *, bank_name: str, client_id: int) -> dict[str, object]:
    return {
        "bank": bank_name,
        "client": int(client_id),
        "selected_probe_ids": selection.selected_probe_ids.tolist(),
        "selected_rho": selection.rho[selection.selected_probe_ids].tolist(),
        "selected_energy": selection.energy[selection.selected_probe_ids].tolist(),
        "weights": selection.weights.tolist(),
        "directions_sha256": sha256_array(selection.directions),
        "active_probe_ids": np.flatnonzero(selection.active).astype(int).tolist(),
    }


def trace_payload(trace) -> dict[str, object]:
    return {
        "objective_before": trace.objective_before,
        "objective_after": trace.objective,
        "anchor_kl": trace.anchor_kl,
        "accepted": trace.accepted,
        "effective_learning_rate_trace": trace.learning_rate,
        "nonincrease_steps": int(
            sum(after <= before + 1.0e-12 for before, after in zip(trace.objective_before, trace.objective))
        ),
        "trust_region_events": trace.trust_region_events,
    }


def save_full_checkpoint(model: torch.nn.Module, head: torch.nn.Linear, path: Path) -> None:
    original = model.linear
    model.linear = head
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    model.linear = original


def inspect_assets(args: argparse.Namespace, banks: dict[str, dict[str, object]]) -> dict[str, object]:
    checkpoint_rows = []
    for arm in ("h0", "h9", "l0", "l9"):
        for client_id, model_name in enumerate(MODEL_NAMES):
            path = args.checkpoint_root / arm / f"client_{client_id}.pt"
            checkpoint_rows.append(
                {
                    "arm": arm,
                    "client": client_id,
                    "model": model_name,
                    "exists": path.is_file(),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
            )
    required_evaluation = {
        name: (args.evaluation_root / name).is_file()
        for name in ("test_images.npy", "test_labels.npy")
    }
    public_tar = args.public_root / "cifar-100-python.tar.gz"
    return {
        "protocol": "cle_k1a_sdmn_headonly_inspect_v1",
        "all_16_checkpoints_present": all(row["exists"] for row in checkpoint_rows),
        "checkpoints": checkpoint_rows,
        "public_tar_present": public_tar.is_file(),
        "evaluation_assets": required_evaluation,
        "bank_a_sha256": banks["a"]["bank_sha256"],
        "bank_b_sha256": banks["b"]["bank_sha256"],
        "head_interface": "model.backbone + model.linear",
        "formal_locked": True,
        "formal_lock_reason": (
            "formal optimizer, maximum surgery steps and accepted calibration manifest "
            "must be frozen after smoke/calibration"
        ),
    }


def run_one_fold_client(
    *,
    model: torch.nn.Module,
    checkpoint_arm: str,
    client_id: int,
    fold_name: str,
    discover_images: np.ndarray,
    surgery_images: np.ndarray,
    holdout_images: np.ndarray,
    discover_recipes: list[dict[str, object]],
    holdout_recipes: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    steps: int,
    output_dir: Path,
) -> dict[str, object]:
    discover_base, discover_probes = extract_features(
        model, discover_images, discover_recipes, device=device, batch_size=batch_size
    )
    discover_response = response_from_numpy_features(
        model.linear, discover_base, discover_probes, device=device
    )
    selection = select_high_risk_probes(discover_response)
    random_ids = match_random_probes(selection)
    used_ids = np.unique(np.concatenate((selection.selected_probe_ids, random_ids)))
    surgery_recipe_subset = [discover_recipes[int(probe_id)] for probe_id in used_ids]
    surgery_base, surgery_union = extract_features(
        model, surgery_images, surgery_recipe_subset, device=device, batch_size=batch_size
    )
    union_lookup = {int(probe_id): position for position, probe_id in enumerate(used_ids)}
    targeted_features = surgery_union[:, [union_lookup[int(value)] for value in selection.selected_probe_ids]]
    random_features = surgery_union[:, [union_lookup[int(value)] for value in random_ids]]
    sham_directions, sham_permutations = make_direction_sham(
        selection.directions,
        seed=SHAM_SEED + 100 * int(client_id) + (0 if fold_name == "ab" else 1),
    )

    base_tensor = torch.from_numpy(surgery_base).to(device)
    targeted_tensor = torch.from_numpy(targeted_features).to(device)
    random_tensor = torch.from_numpy(random_features).to(device)
    heads: dict[str, torch.nn.Linear] = {"frozen": model.linear}
    traces: dict[str, object] = {}
    for arm in ARMS[1:]:
        probe_tensor = random_tensor if arm == "random_probe" else targeted_tensor
        directions = (
            sham_directions
            if arm == "direction_sham"
            else (
                discover_response.mean(axis=0)[random_ids]
                if arm == "random_probe"
                else selection.directions
            )
        )
        if arm == "random_probe":
            directions = directions / (np.linalg.norm(directions, axis=-1, keepdims=True) + 1.0e-12)
        repaired, trace = run_head_surgery(
            model.linear,
            base_tensor,
            probe_tensor,
            arm=arm,
            directions=None if arm == "generic_invariance" else directions,
            weights=None if arm == "generic_invariance" else selection.weights,
            learning_rate=float(learning_rate),
            steps=int(steps),
            anchor_limit=0.02,
            optimizer_name="adam",
        )
        heads[arm] = repaired
        traces[arm] = trace_payload(trace)

    holdout_base, holdout_probes = extract_features(
        model, holdout_images, holdout_recipes, device=device, batch_size=batch_size
    )
    metrics = {}
    responses = {}
    for arm, head in heads.items():
        response = response_from_numpy_features(head, holdout_base, holdout_probes, device=device)
        responses[arm] = response
        metrics[arm] = arm_metrics(response)

    selection_dir = output_dir / "probe_selection"
    direction_dir = output_dir / "directions"
    trace_dir = output_dir / "optimization_traces"
    checkpoint_dir = output_dir / "checkpoints"
    response_dir = output_dir / "unseen_bank_metrics" / "responses"
    for path in (selection_dir, direction_dir, trace_dir, checkpoint_dir, response_dir):
        path.mkdir(parents=True, exist_ok=True)
    key = f"{checkpoint_arm}_{fold_name}_client{client_id}"
    write_json(selection_dir / f"{key}.json", selection_payload(selection, bank_name=fold_name[0], client_id=client_id))
    np.savez_compressed(
        direction_dir / f"{key}.npz",
        targeted=selection.directions,
        direction_sham=sham_directions,
        random_probe_ids=random_ids,
        sham_permutations=np.asarray(sham_permutations, dtype=np.int64),
    )
    for arm, payload in traces.items():
        write_json(trace_dir / f"{key}_{arm}.json", payload)
    checkpoint_rows = {}
    for arm, head in heads.items():
        path = checkpoint_dir / f"{key}_{arm}.pt"
        save_full_checkpoint(model, head, path)
        checkpoint_rows[arm] = {"path": path.relative_to(output_dir).as_posix(), "sha256": sha256_file(path)}
        response_path = response_dir / f"{key}_{arm}.npz"
        np.savez_compressed(response_path, centered_response=responses[arm])
    feature_manifest = {
        "discover_base": {"shape": list(discover_base.shape), "sha256": sha256_array(discover_base)},
        "discover_probes": {"shape": list(discover_probes.shape), "sha256": sha256_array(discover_probes)},
        "surgery_base": {"shape": list(surgery_base.shape), "sha256": sha256_array(surgery_base)},
        "surgery_union": {"shape": list(surgery_union.shape), "sha256": sha256_array(surgery_union)},
        "holdout_base": {"shape": list(holdout_base.shape), "sha256": sha256_array(holdout_base)},
        "holdout_probes": {"shape": list(holdout_probes.shape), "sha256": sha256_array(holdout_probes)},
    }
    return {
        "fold": fold_name,
        "client": int(client_id),
        "selected_probe_ids": selection.selected_probe_ids.tolist(),
        "random_probe_ids": random_ids.tolist(),
        "metrics": metrics,
        "traces": traces,
        "checkpoints": checkpoint_rows,
        "feature_cache_manifest": feature_manifest,
    }


def calibration_for_client_fold(
    *,
    model: torch.nn.Module,
    client_id: int,
    fold_name: str,
    discover_images: np.ndarray,
    surgery_images: np.ndarray,
    recipes: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    discover_base, discover_probes = extract_features(
        model, discover_images, recipes, device=device, batch_size=batch_size
    )
    response = response_from_numpy_features(model.linear, discover_base, discover_probes, device=device)
    selection = select_high_risk_probes(response)
    selected_recipes = [recipes[int(probe_id)] for probe_id in selection.selected_probe_ids]
    surgery_base, surgery_probes = extract_features(
        model, surgery_images, selected_recipes, device=device, batch_size=batch_size
    )
    base_tensor = torch.from_numpy(surgery_base).to(device)
    probe_tensor = torch.from_numpy(surgery_probes).to(device)
    rows = []
    for learning_rate in LR_CANDIDATES:
        try:
            _head, trace = run_head_surgery(
                model.linear,
                base_tensor,
                probe_tensor,
                arm="targeted",
                directions=selection.directions,
                weights=selection.weights,
                learning_rate=learning_rate,
                steps=10,
                anchor_limit=0.005,
                optimizer_name="adam",
            )
            payload = trace_payload(trace)
            finite = bool(
                np.isfinite(payload["objective_before"]).all()
                and np.isfinite(payload["objective_after"]).all()
                and np.isfinite(payload["anchor_kl"]).all()
            )
            passed = bool(
                finite
                and payload["nonincrease_steps"] >= 8
                and max(payload["anchor_kl"]) < 0.005
                and all(payload["accepted"])
                and not payload["trust_region_events"]
            )
            rows.append({"candidate_learning_rate": learning_rate, "pass": passed, **payload})
        except (FloatingPointError, RuntimeError) as exc:
            rows.append(
                {
                    "candidate_learning_rate": learning_rate,
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    passing = [row["candidate_learning_rate"] for row in rows if row["pass"]]
    return {
        "fold": fold_name,
        "client": int(client_id),
        "selected_probe_ids": selection.selected_probe_ids.tolist(),
        "chosen_learning_rate": max(passing) if passing else None,
        "candidates": rows,
    }


def manifest_files(output_dir: Path) -> dict[str, object]:
    rows = []
    for path in sorted(path for path in output_dir.rglob("*") if path.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"protocol": "cle_k1a_sdmn_artifacts_v1", "files": rows}


def main() -> None:
    args = parse_args()
    args.public_root = args.public_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.evaluation_root = args.evaluation_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    banks = {
        "a": load_frozen_prime_bank(
            state_path=BANK_ROOT / "bank_a_states.npz",
            manifest_path=BANK_ROOT / "bank_a_manifest.json",
        ),
        "b": load_frozen_prime_bank(
            state_path=BANK_ROOT / "bank_b_states.npz",
            manifest_path=BANK_ROOT / "bank_b_manifest.json",
        ),
    }
    if banks["a"]["bank_sha256"] != BANK_A_SHA256 or banks["b"]["bank_sha256"] != BANK_B_SHA256:
        raise ValueError("frozen K0-B bank hash mismatch")
    inspection = inspect_assets(args, banks)
    write_json(output_dir / "INSPECT.json", inspection)
    if not inspection["all_16_checkpoints_present"] or not inspection["public_tar_present"]:
        raise FileNotFoundError("K1-A asset inspection failed")
    if not all(inspection["evaluation_assets"].values()):
        raise FileNotFoundError("K1-A evaluation assets are incomplete")
    if args.mode == "inspect":
        write_json(output_dir / "result.json", {"verdict": "INSPECT_PASS", "inspection": inspection})
        return
    if args.mode == "formal":
        if args.formal_calibration_manifest is None or args.formal_surgery_steps is None:
            raise RuntimeError(
                "Formal K1-A is locked until an audited calibration manifest and a preregistered "
                "maximum surgery-step budget are provided. Smoke/calibration do not authorize formal."
            )
        raise NotImplementedError("formal oracle/task evaluation remains locked pending calibration audit")

    images = cifar100_train_images_from_tar(args.public_root)
    if args.mode == "smoke":
        counts = {"discover": 8, "surgery": 16, "holdout": 16}
        evaluated_bank_size = 20
        clients = (("h9", 0),)
        folds = (("ab", "a", "b"),)
    else:
        counts = {"discover": 1000, "surgery": 2000, "holdout": 0}
        evaluated_bank_size = 64
        clients = tuple((arm, client_id) for arm in ("h9", "l9") for client_id in range(4))
        folds = (("ab", "a", "b"), ("ba", "b", "a"))
    split = public_split(
        images.shape[0],
        discover_count=counts["discover"],
        surgery_count=counts["surgery"],
        holdout_count=counts["holdout"],
    )
    split_dir = output_dir / "data_split_manifest"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in split.items():
        np.save(split_dir / f"{name}_indices.npy", indices, allow_pickle=False)
    split_manifest = {
        "discover_seed": DISCOVER_SEED,
        "split_seed": SPLIT_SEED,
        "counts": {name: int(indices.size) for name, indices in split.items()},
        "hashes": {name: sha256_array(indices) for name, indices in split.items()},
        "pairwise_disjoint": True,
        "public_labels_loaded_or_used": False,
    }
    write_json(split_dir / "manifest.json", split_manifest)
    device = resolve_device(args.device)
    results = []
    for checkpoint_arm, client_id in clients:
        models = build_models(list(MODEL_NAMES), num_classes=10)
        model = freeze_model_for_head_only(models[client_id], device)
        checkpoint = args.checkpoint_root / checkpoint_arm / f"client_{client_id}.pt"
        model.load_state_dict(load_state(checkpoint), strict=True)
        for fold_name, discover_bank, holdout_bank in folds:
            recipes = list(banks[discover_bank]["recipes"][:evaluated_bank_size])
            if args.mode == "smoke":
                result = run_one_fold_client(
                    model=model,
                    checkpoint_arm=checkpoint_arm,
                    client_id=client_id,
                    fold_name=fold_name,
                    discover_images=images[split["discover"]],
                    surgery_images=images[split["surgery"]],
                    holdout_images=images[split["holdout"]],
                    discover_recipes=recipes,
                    holdout_recipes=list(banks[holdout_bank]["recipes"][:2]),
                    device=device,
                    batch_size=args.batch_size,
                    learning_rate=1.0e-3,
                    steps=2,
                    output_dir=output_dir,
                )
            else:
                result = calibration_for_client_fold(
                    model=model,
                    client_id=client_id,
                    fold_name=fold_name,
                    discover_images=images[split["discover"]],
                    surgery_images=images[split["surgery"]],
                    recipes=recipes,
                    device=device,
                    batch_size=args.batch_size,
                )
                write_json(
                    output_dir / "calibration" / f"{checkpoint_arm}_{fold_name}_client{client_id}.json",
                    result,
                )
            result["checkpoint_arm"] = checkpoint_arm
            result["model"] = MODEL_NAMES[client_id]
            results.append(result)
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.mode == "smoke":
        verdict = "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
        feature_manifest = {row["fold"] + f"_client{row['client']}": row["feature_cache_manifest"] for row in results}
        write_json(output_dir / "feature_cache_manifest" / "manifest.json", feature_manifest)
    else:
        verdict = (
            "CALIBRATION_PASS_READY_FOR_PROTOCOL_FREEZE"
            if all(row["chosen_learning_rate"] is not None for row in results)
            else "CALIBRATION_FAIL"
        )
    config = {
        "protocol": "cle_k1a_sdmn_headonly_v1",
        "mode": args.mode,
        "scientific_decision_allowed": False,
        "optimizer": "adam",
        "smoke_steps": 2,
        "anchor_limit": 0.02,
        "calibration_anchor_limit": 0.005,
        "lr_candidates": LR_CANDIDATES,
        "split": split_manifest,
        "bank_hashes": {name: banks[name]["bank_sha256"] for name in ("a", "b")},
        "forbidden_during_surgery": [
            "CLE binding",
            "corruption type/family/severity",
            "private task metrics",
            "DSA/WCCA/CFG",
            "holdout bank during calibration",
        ],
    }
    write_json(output_dir / "config" / "config.json", config)
    write_json(output_dir / "result.json", {"verdict": verdict, "results": results, "config": config})
    report = [
        "# CLE K1-A head-only SDMN",
        "",
        f"Verdict: `{verdict}`",
        "",
        "This mode is execution/numerical validation only and cannot support a scientific claim.",
        "No RAHFL training or communication update was performed.",
    ]
    (output_dir / "FINAL_REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(output_dir / "artifact_manifest.json", manifest_files(output_dir))


if __name__ == "__main__":
    main()
