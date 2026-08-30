from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import add_vendor_paths, dataset_stats, normalize_batch  # noqa: E402
from fedprime.engine.cle_directional_withdrawal import (  # noqa: E402
    decide_bridge_only_gate,
    directional_withdrawal,
    family_aggregated_withdrawal,
    family_separability_accuracy,
    peak_signal_to_noise_ratio,
    within_source_variance_contraction,
)
from fedprime.engine.cle_probe_directional_promotion import score_binding_retrieval  # noqa: E402
from fedprime.engine.cle_shortcut_alignment import (  # noqa: E402
    OPERATOR_FAMILY_IDS,
    PHASE_A0_SEED,
    PHASE_A0_SEVERITY,
    deterministic_corruption_grid,
    historical_family_binding,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402
from fedprime.models.public_canonicalizer import (  # noqa: E402
    load_public_canonicalizer_checkpoint,
)


ARMS = ("h0", "h9", "l0", "l9")
MODEL_NAMES = ("ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-B0 public canonicalization bridge audit.")
    parser.add_argument("--phase-a1a-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--canonicalizer-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cle_public_canonicalization_phase_b0_seed0"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-sources", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older OpenI PyTorch.
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def _as_uint8(images: torch.Tensor) -> np.ndarray:
    return (
        images.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def canonicalize_images(
    images: np.ndarray,
    checkpoint: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    model, payload = load_public_canonicalizer_checkpoint(str(checkpoint), map_location="cpu")
    model.to(device).eval()
    flat = np.asarray(images, dtype=np.uint8).reshape(-1, 32, 32, 3)
    output = np.empty_like(flat)
    with torch.inference_mode():
        for start in range(0, flat.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), flat.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(flat[start:stop]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            output[start:stop] = _as_uint8(model(batch))
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output.reshape(np.asarray(images).shape), {
        "protocol": payload.get("protocol"),
        "model_config": payload.get("model_config"),
        "train_config": payload.get("train_config"),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def deterministic_augmix_overlay(images: np.ndarray, *, seed: int) -> np.ndarray:
    """Apply the released RAHFL AugMix implementation as an overlay control."""

    add_vendor_paths()
    from Dataset.dataaug import aug

    preprocess = T.ToTensor()
    array = np.asarray(images, dtype=np.uint8)
    flat = array.reshape(-1, 32, 32, 3)
    output = np.empty_like(flat)
    original_state = np.random.get_state()
    try:
        for index, image in enumerate(flat):
            local_seed = int(
                np.random.default_rng(np.random.SeedSequence([seed, index, 7187])).integers(
                    0, 2**31 - 1
                )
            )
            np.random.seed(local_seed)
            mixed = aug(Image.fromarray(image), preprocess)
            output[index] = _as_uint8(mixed.unsqueeze(0))[0]
    finally:
        np.random.set_state(original_state)
    return output.reshape(array.shape)


def infer_model(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    flat = np.asarray(images, dtype=np.uint8).reshape(-1, 32, 32, 3)
    probabilities = np.empty((flat.shape[0], 10), dtype=np.float32)
    stats = dataset_stats("cifar10")
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, flat.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), flat.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(flat[start:stop]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            batch = normalize_batch(batch, stats)
            probabilities[start:stop] = torch.softmax(forward_logits(model, batch), dim=1).cpu().numpy()
    model.to("cpu")
    return probabilities


def infer_all_arms(
    checkpoint_root: Path,
    bridges: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        models = build_models(list(MODEL_NAMES), num_classes=10)
        arm_results = {
            bridge: np.empty((4, *images.shape[:2], 10), dtype=np.float32)
            for bridge, images in bridges.items()
        }
        for client_id in range(4):
            model = models[client_id]
            checkpoint = checkpoint_root / arm / f"client_{client_id}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            print(f"[checkpoint] arm={arm} client={client_id} path={checkpoint}", flush=True)
            model.load_state_dict(load_state(checkpoint), strict=True)
            for bridge, images in bridges.items():
                inferred = infer_model(model, images, device=device, batch_size=batch_size)
                arm_results[bridge][client_id] = inferred.reshape(*images.shape[:2], 10)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[arm] = arm_results
    return results


def _accuracy(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    predictions = np.asarray(probabilities).argmax(axis=-1)
    return np.mean(predictions == np.asarray(labels)[None, :, None], axis=(1, 2))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    phase_root = args.phase_a1a_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    canonicalizer_checkpoint = args.canonicalizer_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not canonicalizer_checkpoint.is_file():
        raise FileNotFoundError(canonicalizer_checkpoint)

    clean_images = np.load(phase_root / "evaluation/test_images.npy", allow_pickle=False)
    labels = np.load(phase_root / "evaluation/test_labels.npy", allow_pickle=False).astype(np.int64)
    max_sources = min(int(args.max_sources), int(labels.size))
    if args.smoke:
        if max_sources < 20:
            raise ValueError("smoke requires at least 20 sources so every class has two examples")
        per_class = max_sources // 10
        selected = np.concatenate(
            [np.flatnonzero(labels == class_id)[:per_class] for class_id in range(10)]
        )
        clean_images = clean_images[selected]
        labels = labels[selected]
        max_sources = int(labels.size)
    else:
        clean_images = clean_images[:max_sources]
        labels = labels[:max_sources]
    if not args.smoke and max_sources != 1000:
        raise ValueError("formal Phase-B0 requires all 1000 frozen evaluation sources")
    grid, severities = deterministic_corruption_grid(
        clean_images,
        severity=PHASE_A0_SEVERITY,
        seed=PHASE_A0_SEED,
    )
    device = resolve_device(args.device)
    print(f"[device] {device}; sources={max_sources}", flush=True)
    canonical_grid, canonicalizer_manifest = canonicalize_images(
        grid,
        canonicalizer_checkpoint,
        device=device,
        batch_size=int(args.batch_size),
    )
    canonical_clean, _ = canonicalize_images(
        clean_images[:, None],
        canonicalizer_checkpoint,
        device=device,
        batch_size=int(args.batch_size),
    )
    overlay_grid = deterministic_augmix_overlay(grid, seed=PHASE_A0_SEED)

    grid_bridges = {
        "original": grid,
        "overlay": overlay_grid,
        "canonical": canonical_grid,
    }
    arm_predictions = infer_all_arms(
        checkpoint_root,
        grid_bridges,
        device=device,
        batch_size=int(args.batch_size),
    )
    clean_predictions = infer_all_arms(
        checkpoint_root,
        {"clean": clean_images[:, None], "canonical_clean": canonical_clean},
        device=device,
        batch_size=int(args.batch_size),
    )

    binding = historical_family_binding(num_clients=4, num_classes=10)
    family_axis = np.arange(4, dtype=np.int64)
    arm_summaries: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        original = arm_predictions[arm]["original"]
        arm_summary: dict[str, object] = {}
        for bridge in ("overlay", "canonical"):
            result = directional_withdrawal(original, arm_predictions[arm][bridge], labels)
            family_matrix = family_aggregated_withdrawal(result.matrix, OPERATOR_FAMILY_IDS)
            retrieval = score_binding_retrieval(family_matrix, binding, family_axis)
            accuracy = _accuracy(arm_predictions[arm][bridge], labels)
            original_accuracy = _accuracy(original, labels)
            arm_summary[bridge] = {
                "scdw_pooled": result.pooled,
                "scdw_client": result.client.tolist(),
                "accuracy": accuracy.tolist(),
                "accuracy_delta": (accuracy - original_accuracy).tolist(),
                "oracle_family_retrieval": retrieval,
            }
            for client_id, model_name in enumerate(MODEL_NAMES):
                rows.append(
                    {
                        "arm": arm,
                        "bridge": bridge,
                        "client": client_id,
                        "model": model_name,
                        "scdw": float(result.client[client_id]),
                        "accuracy": float(accuracy[client_id]),
                        "accuracy_delta": float(accuracy[client_id] - original_accuracy[client_id]),
                        "oracle_map": float(retrieval["client_mean_average_precision"][client_id]),
                        "oracle_hit": float(
                            retrieval["client_class_to_probe_family_hit_rate"][client_id]
                        ),
                    }
                )
        clean_result = directional_withdrawal(
            clean_predictions[arm]["clean"],
            clean_predictions[arm]["canonical_clean"],
            labels,
        )
        arm_summary["clean_artifact"] = {
            "scdw_pooled": clean_result.pooled,
            "scdw_client": clean_result.client.tolist(),
            "clean_accuracy": _accuracy(clean_predictions[arm]["clean"], labels).tolist(),
            "canonical_clean_accuracy": _accuracy(
                clean_predictions[arm]["canonical_clean"], labels
            ).tolist(),
        }
        arm_summaries[arm] = arm_summary

    base_separability = family_separability_accuracy(grid, clean_images, OPERATOR_FAMILY_IDS)
    canonical_separability = family_separability_accuracy(
        canonical_grid, clean_images, OPERATOR_FAMILY_IDS
    )
    overlay_separability = family_separability_accuracy(
        overlay_grid, clean_images, OPERATOR_FAMILY_IDS
    )
    canonical_contraction = within_source_variance_contraction(grid, canonical_grid)
    overlay_contraction = within_source_variance_contraction(grid, overlay_grid)

    def retrieval(arm: str) -> dict[str, object]:
        return arm_summaries[arm]["canonical"]["oracle_family_retrieval"]

    h0, h9, l0, l9 = (retrieval(arm) for arm in ARMS)
    hfl_client_delta = np.asarray(h9["client_mean_average_precision"]) - np.asarray(
        h0["client_mean_average_precision"]
    )
    local_client_delta = np.asarray(l9["client_mean_average_precision"]) - np.asarray(
        l0["client_mean_average_precision"]
    )
    semantic_delta_min = min(
        min(arm_summaries[arm]["canonical"]["accuracy_delta"]) for arm in ARMS
    )
    clean_scdw_max = max(
        float(arm_summaries[arm]["clean_artifact"]["scdw_pooled"]) for arm in ARMS
    )
    gate_values: dict[str, float | int] = {
        "semantic_accuracy_delta_min": float(semantic_delta_min),
        "variance_contraction": canonical_contraction,
        "separability_relative_reduction": float(
            (base_separability - canonical_separability) / max(base_separability, 1.0e-12)
        ),
        "hfl_gamma9_map": float(h9["mean_average_precision"]),
        "hfl_map_delta": float(h9["mean_average_precision"] - h0["mean_average_precision"]),
        "hfl_hit_rate": float(h9["class_to_probe_family_hit_rate"]),
        "hfl_positive_clients": int(np.count_nonzero(hfl_client_delta > 0.0)),
        "local_gamma9_map": float(l9["mean_average_precision"]),
        "local_map_delta": float(l9["mean_average_precision"] - l0["mean_average_precision"]),
        "local_hit_rate": float(l9["class_to_probe_family_hit_rate"]),
        "local_positive_clients": int(np.count_nonzero(local_client_delta > 0.0)),
        "canonical_vs_overlay_contraction_margin": float(
            canonical_contraction - overlay_contraction
        ),
        "clean_scdw_max": clean_scdw_max,
    }
    decision = (
        {"verdict": "SMOKE_ONLY_NO_SCIENTIFIC_DECISION", "values": gate_values}
        if args.smoke
        else decide_bridge_only_gate(gate_values)
    )
    summary = {
        "protocol": "cle_public_canonicalization_phase_b0_seed0",
        "smoke": bool(args.smoke),
        "sources": int(max_sources),
        "device": str(device),
        "canonicalizer": canonicalizer_manifest,
        "integrity": {
            "private_corruption_labels_used_by_canonicalizer": False,
            "private_binding_used_by_withdrawal_estimator": False,
            "binding_and_family_opened_only_for_oracle_scoring": True,
            "final_test_used_for_training_or_selection": False,
            "checkpoint_count": 16,
            "severity": int(PHASE_A0_SEVERITY),
            "severities_exact": bool(np.all(severities == PHASE_A0_SEVERITY)),
        },
        "bridge_quality": {
            "original_psnr": peak_signal_to_noise_ratio(grid, clean_images),
            "overlay_psnr": peak_signal_to_noise_ratio(overlay_grid, clean_images),
            "canonical_psnr": peak_signal_to_noise_ratio(canonical_grid, clean_images),
            "canonical_variance_contraction": canonical_contraction,
            "overlay_variance_contraction": overlay_contraction,
            "original_family_separability": base_separability,
            "overlay_family_separability": overlay_separability,
            "canonical_family_separability": canonical_separability,
        },
        "arms": arm_summaries,
        "decision": decision,
    }
    summary_path = output_dir / "cle_public_canonicalization_phase_b0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_rows(output_dir / "cle_public_canonicalization_phase_b0_per_client.csv", rows)
    np.savez_compressed(
        output_dir / "cle_public_canonicalization_phase_b0_predictions.npz",
        labels=labels,
        operator_family_ids=OPERATOR_FAMILY_IDS,
        binding=binding,
        **{
            f"{arm}_{bridge}_probabilities": values
            for arm, bridges in arm_predictions.items()
            for bridge, values in bridges.items()
        },
    )
    print(json.dumps(decision, indent=2), flush=True)
    print(f"[complete] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
