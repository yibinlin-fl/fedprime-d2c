from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import dataset_stats, normalize_batch  # noqa: E402
from fedprime.models.factory import build_models, forward_logits  # noqa: E402


ARMS = ("h0", "h9", "l0", "l9")
CLIENTS = ((0, "ResNet10"), (1, "ResNet12"), (2, "ShuffleNet"), (3, "Mobilenetv2"))
FORMAL_SOURCE_COUNT = 1000
SMOKE_SOURCE_COUNT = 32
BENCHMARK_SOURCE_COUNT = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P3-A clean-base completion: clean forward only.")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="smoke")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT
        / "local_runs/cle_public_canonicalization_phase_b0/cle_public_canonicalization_phase_b0_seed0_inputs",
    )
    parser.add_argument(
        "--a1a-predictions",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_shortcut_amplification_phase_a1a_seed0/extracted/outputs/cle_shortcut_amplification_phase_a1a_seed0_analysis/round_040_predictions.npz",
    )
    parser.add_argument(
        "--phase-a0-predictions",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_shortcut_alignment_phase_a0_seed0/extracted/outputs/cle_shortcut_alignment_phase_a0_seed0/cle_shortcut_alignment_phase_a0_seed0_predictions.npz",
    )
    parser.add_argument(
        "--smoke-reference",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_k1_c_minimal_formal_seed0/extracted/outputs/cle_k1_c_minimal_seed0_formal/oracle_predictions/h9_ab_frozen.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def configure_inference_numerics(device: torch.device) -> dict[str, object]:
    """Keep Ampere local inference aligned with the V100 reference run."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)
        if device.type == "cuda"
        else None,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)
        if device.type == "cuda"
        else None,
    }


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint payload: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_inputs(
    input_root: Path,
    a1a_predictions: Path,
    phase_a0_predictions: Path,
) -> dict[str, object]:
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "cle_public_canonicalization_phase_b0_seed0_input":
        raise AssertionError("unexpected Phase-B0 input protocol")
    if manifest.get("checkpoint_kind") != "final_round_40_only":
        raise AssertionError("input checkpoints are not certified final round-40")
    records = {str(record["path"]): record for record in manifest["files"]}
    required = [
        "evaluation/test_images.npy",
        "evaluation/test_labels.npy",
        *(f"checkpoints/{arm}/client_{client}.pt" for arm in ARMS for client, _ in CLIENTS),
    ]
    for relative in required:
        path = input_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(records[relative]["sha256"])
        if sha256_file(path) != expected:
            raise AssertionError(f"manifest hash mismatch: {relative}")

    images = np.load(input_root / "evaluation/test_images.npy", allow_pickle=False)
    labels = np.load(input_root / "evaluation/test_labels.npy", allow_pickle=False).astype(np.int64)
    if images.shape != (1000, 32, 32, 3) or images.dtype != np.uint8 or labels.shape != (1000,):
        raise AssertionError("unexpected clean evaluation tensor")
    with np.load(a1a_predictions, allow_pickle=False) as payload:
        if payload["probabilities"].shape != (4, 4, 1000, 16, 10):
            raise AssertionError("unexpected Phase-A1a corruption prediction shape")
        if not np.array_equal(labels, payload["labels"]):
            raise AssertionError("clean labels do not align with Phase-A1a source order")
    with np.load(phase_a0_predictions, allow_pickle=False) as payload:
        source_ids = np.asarray(payload["source_ids"], dtype=np.int64)
        if not np.array_equal(labels, payload["labels"]):
            raise AssertionError("Phase-A0 labels do not align with Phase-A1a/Phase-B0")
    if not np.array_equal(source_ids, np.arange(1000, dtype=np.int64)):
        raise AssertionError("unexpected Phase-A0 source identity/order")
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "records": records,
        "images": images,
        "labels": labels,
        "source_ids": source_ids,
        "test_images_sha256": sha256_file(input_root / "evaluation/test_images.npy"),
        "test_labels_sha256": sha256_file(input_root / "evaluation/test_labels.npy"),
        "source_ids_array_sha256": sha256_array(source_ids),
        "a1a_predictions_sha256": sha256_file(a1a_predictions),
        "phase_a0_predictions_sha256": sha256_file(phase_a0_predictions),
    }


def clean_forward(
    images: np.ndarray,
    checkpoint: Path,
    model_name: str,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    start_total = time.perf_counter()
    model = build_models([model_name], num_classes=10)[0]
    model.load_state_dict(load_state(checkpoint), strict=True)
    model.to(device).eval()
    synchronize(device)
    ready = time.perf_counter()

    logits_chunks: list[np.ndarray] = []
    stats = dataset_stats("cifar10")
    start_forward = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(images), int(batch_size)):
            stop = min(start + int(batch_size), len(images))
            batch = torch.from_numpy(np.ascontiguousarray(images[start:stop]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            logits = forward_logits(model, normalize_batch(batch, stats))
            logits_chunks.append(logits.detach().cpu().numpy().astype(np.float32))
    synchronize(device)
    end_forward = time.perf_counter()
    logits_array = np.concatenate(logits_chunks, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy().astype(np.float32)
    if logits_array.shape != (len(images), 10) or not np.isfinite(logits_array).all():
        raise AssertionError("invalid clean logits")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6):
        raise AssertionError("invalid clean probabilities")
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits_array, probabilities, {
        "model_build_and_checkpoint_load_seconds": ready - start_total,
        "forward_seconds": end_forward - start_forward,
        "total_seconds": end_forward - start_total,
        "images_per_forward_second": len(images) / max(end_forward - start_forward, 1.0e-12),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise SystemExit("formal mode requires the explicit --confirm-formal flag")
    input_root = args.input_root.resolve()
    a1a_predictions = args.a1a_predictions.resolve()
    phase_a0_predictions = args.phase_a0_predictions.resolve()
    smoke_reference = args.smoke_reference.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "outputs" / f"p3a_clean_base_completion_{args.mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    numerical_settings = configure_inference_numerics(device)
    validated = validate_inputs(input_root, a1a_predictions, phase_a0_predictions)
    images = validated.pop("images")
    labels = validated.pop("labels")
    source_ids = validated.pop("source_ids")
    records = validated.pop("records")
    source_count = {
        "smoke": SMOKE_SOURCE_COUNT,
        "benchmark": BENCHMARK_SOURCE_COUNT,
        "formal": FORMAL_SOURCE_COUNT,
    }[args.mode]
    selected_images = images[:source_count]
    selected_ids = source_ids[:source_count]
    selected_labels = labels[:source_count]

    contexts = (
        (("h9", 0, "ResNet10"),)
        if args.mode == "smoke"
        else tuple(("h0", client, model) for client, model in CLIENTS)
        if args.mode == "benchmark"
        else tuple((arm, client, model) for arm in ARMS for client, model in CLIENTS)
    )
    rows: list[dict[str, object]] = []
    all_logits: dict[tuple[str, int], np.ndarray] = {}
    all_probabilities: dict[tuple[str, int], np.ndarray] = {}
    for arm, client, model_name in contexts:
        relative = f"checkpoints/{arm}/client_{client}.pt"
        checkpoint = input_root / relative
        print(f"[clean-forward] mode={args.mode} arm={arm} client={client} model={model_name} n={source_count}", flush=True)
        logits, probabilities, timing = clean_forward(
            selected_images,
            checkpoint,
            model_name,
            device=device,
            batch_size=args.batch_size,
        )
        all_logits[(arm, client)] = logits
        all_probabilities[(arm, client)] = probabilities
        rows.append(
            {
                "arm": arm,
                "client": client,
                "model": model_name,
                "sources": source_count,
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_manifest_sha256": records[relative]["sha256"],
                "logits_shape": list(logits.shape),
                "logits_sha256": sha256_array(logits),
                "probabilities_sha256": sha256_array(probabilities),
                "probability_min": float(probabilities.min()),
                "probability_max": float(probabilities.max()),
                "finite": bool(np.isfinite(logits).all() and np.isfinite(probabilities).all()),
                **timing,
            }
        )

    output_npz = output_dir / "clean_base_outputs.npz"
    if args.mode == "formal":
        clean_logits = np.stack(
            [np.stack([all_logits[(arm, client)] for client, _ in CLIENTS]) for arm in ARMS]
        )
        clean_probabilities = np.stack(
            [np.stack([all_probabilities[(arm, client)] for client, _ in CLIENTS]) for arm in ARMS]
        )
        expected_shape = (4, 4, 1000, 10)
    else:
        clean_logits = np.stack([all_logits[(arm, client)] for arm, client, _ in contexts])
        clean_probabilities = np.stack([all_probabilities[(arm, client)] for arm, client, _ in contexts])
        expected_shape = (len(contexts), source_count, 10)
    if clean_logits.shape != expected_shape or clean_probabilities.shape != expected_shape:
        raise AssertionError("unexpected completion output shape")

    reference_checks: list[dict[str, object]] = []
    if args.mode == "smoke":
        if not smoke_reference.is_file():
            raise FileNotFoundError(smoke_reference)
        with np.load(smoke_reference, allow_pickle=False) as payload:
            selected_clients = np.asarray(payload["selected_client_ids"], dtype=np.int64)
            positions = np.flatnonzero(selected_clients == 0)
            if positions.size != 1:
                raise AssertionError("smoke reference does not uniquely contain H9 client0")
            reference = np.asarray(payload["clean_probabilities"], dtype=np.float32)[
                int(positions[0]), :source_count
            ]
        max_abs_difference = float(np.max(np.abs(clean_probabilities[0] - reference)))
        if max_abs_difference > 1.0e-5:
            raise AssertionError(
                f"clean preprocessing/output mismatch against sealed K1 reference: {max_abs_difference}"
            )
        reference_checks.append({
            "path": smoke_reference.as_posix(),
            "sha256": sha256_file(smoke_reference),
            "arm": "h9",
            "client": 0,
            "sources": source_count,
            "max_abs_probability_difference": max_abs_difference,
            "argmax_agreement": float(
                np.mean(clean_probabilities[0].argmax(axis=1) == reference.argmax(axis=1))
            ),
            "tolerance": 1.0e-5,
            "pass": True,
        })
    elif args.mode == "formal":
        reference_root = smoke_reference.parent
        for arm in ("h9", "l9"):
            reference_path = reference_root / f"{arm}_ab_frozen.npz"
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            with np.load(reference_path, allow_pickle=False) as payload:
                selected_clients = np.asarray(payload["selected_client_ids"], dtype=np.int64)
                reference_probabilities = np.asarray(payload["clean_probabilities"], dtype=np.float32)
            for client in (0, 3):
                positions = np.flatnonzero(selected_clients == client)
                if positions.size != 1:
                    raise AssertionError(
                        f"sealed reference does not uniquely contain {arm.upper()} client{client}"
                    )
                generated = clean_probabilities[ARMS.index(arm), client]
                reference = reference_probabilities[int(positions[0]), :source_count]
                max_abs_difference = float(np.max(np.abs(generated - reference)))
                if max_abs_difference > 1.0e-5:
                    raise AssertionError(
                        "clean preprocessing/output mismatch against sealed K1 reference: "
                        f"{arm} client{client} max_abs_difference={max_abs_difference}"
                    )
                reference_checks.append(
                    {
                        "path": reference_path.as_posix(),
                        "sha256": sha256_file(reference_path),
                        "arm": arm,
                        "client": client,
                        "sources": source_count,
                        "max_abs_probability_difference": max_abs_difference,
                        "argmax_agreement": float(
                            np.mean(generated.argmax(axis=1) == reference.argmax(axis=1))
                        ),
                        "tolerance": 1.0e-5,
                        "pass": True,
                    }
                )
    np.savez_compressed(
        output_npz,
        clean_logits=clean_logits,
        clean_probabilities=clean_probabilities,
        source_ids=selected_ids,
        labels=selected_labels,
        arm_names=np.asarray(ARMS if args.mode == "formal" else [row[0] for row in contexts]),
        client_ids=np.asarray([client for client, _ in CLIENTS] if args.mode == "formal" else [row[1] for row in contexts]),
        model_names=np.asarray([model for _, model in CLIENTS] if args.mode == "formal" else [row[2] for row in contexts]),
    )

    estimate: dict[str, object] | None = None
    if args.mode == "benchmark":
        measured = sum(float(row["total_seconds"]) for row in rows)
        scaled_forward = 4.0 * sum(
            float(row["forward_seconds"]) * FORMAL_SOURCE_COUNT / source_count for row in rows
        )
        scaled_load = 4.0 * sum(float(row["model_build_and_checkpoint_load_seconds"]) for row in rows)
        estimated = scaled_forward + scaled_load
        estimate = {
            "benchmark_contexts": len(rows),
            "benchmark_forward_items": len(rows) * source_count,
            "measured_seconds": measured,
            "estimated_formal_clean_forward_items": 16000,
            "estimated_formal_seconds": estimated,
            "estimated_formal_minutes": estimated / 60.0,
            "conservative_2x_minutes": 2.0 * estimated / 60.0,
            "conservative_3x_minutes": 3.0 * estimated / 60.0,
            "cost_gate": "PASS" if 3.0 * estimated <= 15.0 * 60.0 else "REVIEW_REQUIRED",
            "cost_gate_rule": "3x conservative local ETA must be <=15 minutes",
        }

    manifest = {
        "protocol": "p3a_clean_base_completion_gate",
        "mode": args.mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_contract": {
            "training": False,
            "backward": False,
            "corruption_generation": False,
            "prime_generation": False,
            "checkpoint_modification": False,
            "clean_forward_only": True,
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "numerical_settings": numerical_settings,
        "batch_size": int(args.batch_size),
        "source_count": source_count,
        "source_ids_sha256": sha256_array(selected_ids),
        "source_images_sha256": sha256_array(selected_images),
        "source_labels_sha256": sha256_array(selected_labels),
        "input_integrity": validated,
        "contexts": rows,
        "sealed_reference_checks": reference_checks,
        "benchmark_estimate": estimate,
        "output": {
            "path": output_npz.as_posix(),
            "bytes": output_npz.stat().st_size,
            "sha256": sha256_file(output_npz),
            "clean_logits_shape": list(clean_logits.shape),
            "clean_probabilities_shape": list(clean_probabilities.shape),
        },
        "scientific_verdict": "EXECUTION_ONLY_NO_P3A_DECISION",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"mode": args.mode, "output": output_npz.as_posix(), "estimate": estimate}, indent=2))


if __name__ == "__main__":
    main()
