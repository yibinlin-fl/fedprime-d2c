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

from fedprime.data.loaders import (  # noqa: E402
    _cifar100_train_from_tar,
    dataset_stats,
    normalize_batch,
)
from fedprime.data.corruptions import apply_corruption  # noqa: E402
from fedprime.engine.cle_probe_directional_promotion import (  # noqa: E402
    score_binding_retrieval,
    shuffled_retrieval_nulls,
)
from fedprime.engine.cle_public_carrier_moment import (  # noqa: E402
    decide_public_carrier_gate,
    directional_moment,
    paired_bootstrap_moment_delta,
    public_carrier_responses,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402


ARMS = ("h0", "h9", "l0", "l9")
MODEL_NAMES = ("ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2")
PUBLIC_SEED = 20260901
PROBE_SEED = 20260830
PROBE_SEVERITY = 3
# Frozen operator identities only. Family membership is deliberately unavailable
# until score_saved_responses opens the oracle truth after response hashing.
ORACLE_OPERATOR_NAMES = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "spatter",
    "contrast",
    "brightness",
    "jpeg_compression",
    "pixelate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K0-A public-carrier transfer oracle.")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cle_public_carrier_k0a_seed0"))
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--public-seed", type=int, default=PUBLIC_SEED)
    parser.add_argument("--probe-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
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


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older OpenI PyTorch.
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def infer_logits(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    flat = np.asarray(images, dtype=np.uint8).reshape(-1, 32, 32, 3)
    result = np.empty((flat.shape[0], 10), dtype=np.float32)
    stats = dataset_stats("cifar10")
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, flat.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), flat.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(flat[start:stop]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            batch = normalize_batch(batch, stats)
            result[start:stop] = forward_logits(model, batch).detach().cpu().numpy()
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def select_public_carriers(
    public_root: Path,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    images, _unused_labels = _cifar100_train_from_tar(public_root)
    if count > images.shape[0]:
        raise ValueError("requested more public carriers than CIFAR-100 contains")
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(images.shape[0], size=int(count), replace=False).astype(np.int64)
    return np.asarray(images[indices], dtype=np.uint8), indices


def deterministic_blind_probe_grid(
    clean_images: np.ndarray,
    *,
    operator_names: tuple[str, ...],
    severity: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(clean_images, dtype=np.uint8)
    grid = np.empty((images.shape[0], len(operator_names), 32, 32, 3), dtype=np.uint8)
    for source_id, image in enumerate(images):
        for operator_id, operator in enumerate(operator_names):
            sequence = np.random.SeedSequence([int(seed), int(source_id), int(operator_id)])
            grid[source_id, operator_id] = apply_corruption(
                image,
                operator,
                int(severity),
                np.random.default_rng(sequence),
            )
    return grid, np.full((images.shape[0], len(operator_names)), int(severity), dtype=np.uint8)


def generate_blind_responses(
    checkpoint_root: Path,
    clean_images: np.ndarray,
    probe_images: np.ndarray,
    *,
    arms: tuple[str, ...],
    client_ids: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    response_dir: Path,
) -> list[dict[str, object]]:
    """Generate response files without importing or reading binding metadata."""

    response_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for arm in arms:
        models = build_models(list(MODEL_NAMES), num_classes=10)
        for client_id in client_ids:
            checkpoint = checkpoint_root / arm / f"client_{client_id}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            print(f"[checkpoint] arm={arm} client={client_id} path={checkpoint}", flush=True)
            model = models[client_id]
            model.load_state_dict(load_state(checkpoint), strict=True)
            base_logits = infer_logits(
                model,
                clean_images,
                device=device,
                batch_size=int(batch_size),
            )
            probe_logits = infer_logits(
                model,
                probe_images,
                device=device,
                batch_size=int(batch_size),
            ).reshape(clean_images.shape[0], probe_images.shape[1], 10)
            response = public_carrier_responses(
                base_logits[None],
                probe_logits[None],
            )
            moment = directional_moment(response.centered_response)
            path = response_dir / f"{arm}_client{client_id}.npz"
            np.savez_compressed(
                path,
                base_logits=base_logits,
                probe_logits=probe_logits,
                class_vs_rest_delta=response.class_vs_rest_delta[0].astype(np.float32),
                centered_response=response.centered_response[0].astype(np.float32),
                centered_raw_logit_response=response.centered_raw_logit_response[0].astype(np.float32),
                probability_response=response.probability_response[0].astype(np.float32),
                mean_response=moment.mean_response[0].astype(np.float32),
            )
            rows.append(
                {
                    "arm": arm,
                    "client": int(client_id),
                    "model": MODEL_NAMES[client_id],
                    "checkpoint": f"{arm}/client_{client_id}.pt",
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "response_file": str(path.name),
                    "response_bytes": int(path.stat().st_size),
                    "response_sha256": sha256_file(path),
                }
            )
    return rows


def load_arm_response(response_dir: Path, arm: str, client_ids: tuple[int, ...]) -> dict[str, np.ndarray]:
    keys = (
        "base_logits",
        "probe_logits",
        "class_vs_rest_delta",
        "centered_response",
        "centered_raw_logit_response",
        "probability_response",
        "mean_response",
    )
    collected: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for client_id in client_ids:
        path = response_dir / f"{arm}_client{client_id}.npz"
        with np.load(path, allow_pickle=False) as archive:
            for key in keys:
                collected[key].append(np.asarray(archive[key]))
    return {key: np.stack(values, axis=0) for key, values in collected.items()}


def score_saved_responses(
    response_dir: Path,
    *,
    arms: tuple[str, ...],
    client_ids: tuple[int, ...],
    operator_count: int,
    permutations: int,
    seed: int,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    """Open family/binding truth only after all blind response files are immutable."""

    from fedprime.engine.cle_shortcut_alignment import (  # local import enforces stage boundary.
        OPERATOR_NAMES,
        OPERATOR_FAMILY_IDS,
        historical_family_binding,
    )

    if tuple(OPERATOR_NAMES[:operator_count]) != tuple(ORACLE_OPERATOR_NAMES[:operator_count]):
        raise ValueError("blind operator identities do not match frozen CLE oracle order")
    family_ids = np.asarray(OPERATOR_FAMILY_IDS[:operator_count], dtype=np.int64)
    full_binding = historical_family_binding(num_clients=4, num_classes=10)
    binding = full_binding[np.asarray(client_ids)]
    arm_results: dict[str, dict[str, object]] = {}
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for arm_id, arm in enumerate(arms):
        arrays = load_arm_response(response_dir, arm, client_ids)
        loaded[arm] = arrays
        moment = directional_moment(arrays["centered_response"])
        retrieval = score_binding_retrieval(moment.mean_response, binding, family_ids)
        nulls = shuffled_retrieval_nulls(
            moment.mean_response,
            binding,
            family_ids,
            permutations=int(permutations),
            seed=int(seed) + 1009 * arm_id,
        )
        raw_moment = directional_moment(arrays["centered_raw_logit_response"])
        probability_moment = directional_moment(arrays["probability_response"])
        arm_results[arm] = {
            "directional_strength": float(moment.directional_strength_client.mean()),
            "directional_strength_client": moment.directional_strength_client.tolist(),
            "coherence": float(moment.coherence_client.mean()),
            "coherence_client": moment.coherence_client.tolist(),
            "split_cosine": float(moment.split_cosine_client.mean()),
            "split_cosine_client": moment.split_cosine_client.tolist(),
            "retrieval": retrieval,
            "nulls": nulls,
            "secondary": {
                "centered_raw_logit_retrieval": score_binding_retrieval(
                    raw_moment.mean_response, binding, family_ids
                ),
                "probability_retrieval": score_binding_retrieval(
                    probability_moment.mean_response, binding, family_ids
                ),
            },
        }
    return arm_results, loaded


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(summary: dict[str, object]) -> str:
    lines = [
        "# CLE K0-A Public-Carrier Transfer Oracle",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "K0-A uses frozen classifiers and task-label-disjoint CIFAR-100 carriers. It performs no training.",
        "CIFAR-100 labels are not used. Binding is opened only after response files are saved and hashed.",
        "",
        "| arm | mAP | AUC | hit | directional strength | coherence | split cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, result in summary["arms"].items():
        retrieval = result["retrieval"]
        lines.append(
            f"| {arm} | {retrieval['mean_average_precision']:.6f} | {retrieval['roc_auc']:.6f} | "
            f"{retrieval['class_to_probe_family_hit_rate']:.6f} | {result['directional_strength']:.6f} | "
            f"{result['coherence']:.6f} | {result['split_cosine']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "```json",
            json.dumps(summary["decision"], indent=2, ensure_ascii=False),
            "```",
            "",
            "A formal pass permits only K0-B taxonomy-free generic-probe design. It does not permit training or K1.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    formal = args.mode == "formal"
    public_count = 1000 if formal else 8
    operator_count = len(ORACLE_OPERATOR_NAMES) if formal else 2
    arms = ARMS if formal else ("h0", "h9")
    client_ids = tuple(range(4)) if formal else (0,)
    permutations = int(args.permutations) if formal else min(int(args.permutations), 20)
    bootstrap_samples = int(args.bootstrap_samples) if formal else max(100, min(int(args.bootstrap_samples), 100))
    if formal and (int(args.permutations) != 1000 or int(args.bootstrap_samples) != 1000):
        raise ValueError("formal K0-A freezes permutations=1000 and bootstrap_samples=1000")

    output_dir = args.output_dir.resolve()
    response_dir = output_dir / "responses"
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    clean_images, public_indices = select_public_carriers(
        args.public_root.resolve(), count=public_count, seed=int(args.public_seed)
    )
    selected_path = output_dir / "selected_public_indices.npy"
    np.save(selected_path, public_indices, allow_pickle=False)
    operators = tuple(ORACLE_OPERATOR_NAMES[:operator_count])
    probe_images, severities = deterministic_blind_probe_grid(
        clean_images,
        severity=PROBE_SEVERITY,
        seed=int(args.probe_seed),
        operator_names=operators,
    )
    print(
        f"[setup] mode={args.mode} device={device} carriers={public_count} "
        f"operators={operator_count} arms={arms} clients={client_ids}",
        flush=True,
    )
    response_rows = generate_blind_responses(
        args.checkpoint_root.resolve(),
        clean_images,
        probe_images,
        arms=arms,
        client_ids=client_ids,
        device=device,
        batch_size=int(args.batch_size),
        response_dir=response_dir,
    )
    blind_manifest = {
        "protocol": "cle_public_carrier_k0a_blind_responses",
        "mode": args.mode,
        "binding_or_family_used_during_response_generation": False,
        "public_labels_used": False,
        "public_seed": int(args.public_seed),
        "public_indices_sha256": sha256_array(public_indices),
        "selected_public_indices_file_sha256": sha256_file(selected_path),
        "probe_seed": int(args.probe_seed),
        "severity": int(PROBE_SEVERITY),
        "operators": list(operators),
        "severities_exact": bool(np.all(severities == PROBE_SEVERITY)),
        "responses": response_rows,
    }
    manifest_path = output_dir / "blind_response_manifest.json"
    manifest_path.write_text(json.dumps(blind_manifest, indent=2), encoding="utf-8")
    manifest_sha256_before_scoring = sha256_file(manifest_path)
    print(f"[blind] response manifest sha256={manifest_sha256_before_scoring}", flush=True)

    arm_results, loaded = score_saved_responses(
        response_dir,
        arms=arms,
        client_ids=client_ids,
        operator_count=operator_count,
        permutations=permutations,
        seed=int(args.public_seed),
    )
    bootstrap: dict[str, dict[str, object]] = {}
    if formal:
        bootstrap["hfl"] = paired_bootstrap_moment_delta(
            loaded["h0"]["centered_response"],
            loaded["h9"]["centered_response"],
            samples=bootstrap_samples,
            seed=int(args.public_seed) + 11,
        )
        bootstrap["local"] = paired_bootstrap_moment_delta(
            loaded["l0"]["centered_response"],
            loaded["l9"]["centered_response"],
            samples=bootstrap_samples,
            seed=int(args.public_seed) + 29,
        )
        decision = decide_public_carrier_gate(arm_results, bootstrap)
    else:
        decision = {
            "verdict": "SMOKE_ONLY_NO_SCIENTIFIC_DECISION",
            "scientific_decision_allowed": False,
        }

    per_client_rows: list[dict[str, object]] = []
    arm_rows: list[dict[str, object]] = []
    for arm, result in arm_results.items():
        retrieval = result["retrieval"]
        arm_rows.append(
            {
                "arm": arm,
                "map": retrieval["mean_average_precision"],
                "auc": retrieval["roc_auc"],
                "hit": retrieval["class_to_probe_family_hit_rate"],
                "directional_strength": result["directional_strength"],
                "coherence": result["coherence"],
                "split_cosine": result["split_cosine"],
            }
        )
        for local_id, client_id in enumerate(client_ids):
            per_client_rows.append(
                {
                    "arm": arm,
                    "client": client_id,
                    "model": MODEL_NAMES[client_id],
                    "map": retrieval["client_mean_average_precision"][local_id],
                    "auc": retrieval["client_roc_auc"][local_id],
                    "hit": retrieval["client_class_to_probe_family_hit_rate"][local_id],
                    "directional_strength": result["directional_strength_client"][local_id],
                    "coherence": result["coherence_client"][local_id],
                    "split_cosine": result["split_cosine_client"][local_id],
                }
            )
    write_csv(metrics_dir / "arm_metrics.csv", arm_rows)
    write_csv(metrics_dir / "per_client_metrics.csv", per_client_rows)
    (metrics_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    permutation = {arm: result["nulls"] for arm, result in arm_results.items()}
    (metrics_dir / "permutation_test.json").write_text(
        json.dumps(permutation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config = {
        "protocol": "cle_public_carrier_k0a_20260901",
        "mode": args.mode,
        "public_dataset": "CIFAR-100 train; task labels unused",
        "public_size": public_count,
        "public_seed": int(args.public_seed),
        "probe_seed": int(args.probe_seed),
        "operators": list(operators),
        "severity": int(PROBE_SEVERITY),
        "arms": list(arms),
        "clients": list(client_ids),
        "models": [MODEL_NAMES[index] for index in client_ids],
        "primary_response": "centered class-vs-rest logit delta",
        "bootstrap_samples": bootstrap_samples,
        "permutations": permutations,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        "protocol": config["protocol"],
        "mode": args.mode,
        "verdict": decision["verdict"],
        "scientific_decision_allowed": formal,
        "integrity": {
            "checkpoint_count": len(response_rows),
            "public_labels_used": False,
            "binding_used_during_response_generation": False,
            "binding_opened_only_after_response_manifest": True,
            "blind_manifest_sha256_before_scoring": manifest_sha256_before_scoring,
            "public_indices_sha256": sha256_array(public_indices),
            "severity": int(PROBE_SEVERITY),
            "severities_exact": bool(np.all(severities == PROBE_SEVERITY)),
        },
        "arms": arm_results,
        "bootstrap": bootstrap,
        "decision": decision,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "final_report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"[complete] {result_path}", flush=True)


if __name__ == "__main__":
    main()
