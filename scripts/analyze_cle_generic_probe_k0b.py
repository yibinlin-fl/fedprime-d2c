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
    _cifar100_train_from_tar,
    dataset_stats,
    normalize_batch,
)
from fedprime.engine.cle_generic_probe_gate import (  # noqa: E402
    decide_generic_probe_gate,
    generic_probe_statistics,
    paired_bootstrap_generic_deltas,
    summarize_generic_probe_arm,
)
from fedprime.engine.cle_public_carrier_moment import public_carrier_responses  # noqa: E402
from fedprime.models.factory import build_models, forward_logits  # noqa: E402


ARMS = ("h0", "h9", "l0", "l9")
MODEL_NAMES = ("ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2")
PUBLIC_SEED = 20260901
BANK_A_SEED = 20260902
BANK_B_SEED = 20260903
BOOTSTRAP_HFL_SEED = 20260904
BOOTSTRAP_LOCAL_SEED = 20260905
FORMAL_PUBLIC_COUNT = 1000
FORMAL_BANK_SIZE = 64
K0A_PUBLIC_INDICES_SHA256 = "731B8CFFDCBD241474D33B261E323F9EC11C2EA59BC7705261140A3B8572F6CA"
BANK_A_SHA256 = "6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01"
BANK_B_SHA256 = "4A53497EC5DB6EC05C312E6166109FA4B52A5CC402CCE74E6EDB1253D913BF4E"
FROZEN_BANK_ROOT = ROOT / "fedprime/augmentations/assets/cle_generic_probe_k0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K0-B v2 taxonomy-free generic probe gate.")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cle_generic_probe_k0b_seed0"))
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--oracle-k0a-root", type=Path, default=None)
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


def select_k0a_public_carriers(public_root: Path, *, count: int) -> tuple[np.ndarray, np.ndarray]:
    images, _labels_not_read = _cifar100_train_from_tar(public_root)
    if int(count) > FORMAL_PUBLIC_COUNT:
        raise ValueError("K0-B cannot select more than the frozen K0-A public carriers")
    rng = np.random.default_rng(PUBLIC_SEED)
    full_indices = rng.choice(
        images.shape[0], size=FORMAL_PUBLIC_COUNT, replace=False
    ).astype(np.int64)
    indices = full_indices[: int(count)]
    return np.asarray(images[indices], dtype=np.uint8), indices


def infer_base_logits(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((images.shape[0], 10), dtype=np.float32)
    stats = dataset_stats("cifar10")
    with torch.inference_mode():
        for start in range(0, images.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), images.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(images[start:stop]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            result[start:stop] = (
                forward_logits(model, normalize_batch(batch, stats)).detach().cpu().numpy()
            )
    return result


def infer_recipe_logits(
    model: torch.nn.Module,
    images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((images.shape[0], len(recipes), 10), dtype=np.float32)
    stats = dataset_stats("cifar10")
    with torch.inference_mode():
        for recipe_id, recipe in enumerate(recipes):
            for start in range(0, images.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), images.shape[0])
                batch = torch.from_numpy(np.ascontiguousarray(images[start:stop]))
                batch = batch.permute(0, 3, 1, 2).to(
                    device=device, dtype=torch.float32
                ).div_(255.0)
                transformed = apply_frozen_prime_recipe(batch, recipe)
                result[start:stop, recipe_id] = (
                    forward_logits(model, normalize_batch(transformed, stats))
                    .detach()
                    .cpu()
                    .numpy()
                )
            if (recipe_id + 1) % 8 == 0 or recipe_id + 1 == len(recipes):
                print(f"[probe] completed recipe {recipe_id + 1}/{len(recipes)}", flush=True)
    return result


def generate_responses(
    checkpoint_root: Path,
    public_images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    arms: tuple[str, ...],
    client_ids: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    response_dir: Path,
) -> list[dict[str, object]]:
    """Generate primary response tensors without corruption taxonomy or binding metadata."""

    response_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for arm in arms:
        models = build_models(list(MODEL_NAMES), num_classes=10)
        for client_id in client_ids:
            checkpoint = checkpoint_root / arm / f"client_{client_id}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            print(f"[checkpoint] arm={arm} client={client_id} path={checkpoint}", flush=True)
            model = models[client_id].to(device).eval()
            model.load_state_dict(load_state(checkpoint), strict=True)
            base_logits = infer_base_logits(
                model, public_images, device=device, batch_size=int(batch_size)
            )
            probe_logits = infer_recipe_logits(
                model,
                public_images,
                recipes,
                device=device,
                batch_size=int(batch_size),
            )
            responses = public_carrier_responses(base_logits[None], probe_logits[None])
            path = response_dir / f"{arm}_client{client_id}.npz"
            np.savez_compressed(
                path,
                base_logits=base_logits,
                probe_logits=probe_logits,
                class_vs_rest_delta=responses.class_vs_rest_delta[0].astype(np.float32),
                centered_response=responses.centered_response[0].astype(np.float32),
            )
            rows.append(
                {
                    "arm": arm,
                    "client": int(client_id),
                    "model": MODEL_NAMES[client_id],
                    "checkpoint": f"{arm}/client_{client_id}.pt",
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "response_file": path.name,
                    "response_bytes": int(path.stat().st_size),
                    "response_sha256": sha256_file(path),
                }
            )
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def load_arm_response(response_dir: Path, arm: str, client_ids: tuple[int, ...]) -> np.ndarray:
    responses: list[np.ndarray] = []
    for client_id in client_ids:
        with np.load(response_dir / f"{arm}_client{client_id}.npz", allow_pickle=False) as archive:
            responses.append(np.asarray(archive["centered_response"], dtype=np.float64))
    return np.stack(responses, axis=0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_primary(
    response_dir: Path,
    *,
    arms: tuple[str, ...],
    client_ids: tuple[int, ...],
    bank_size: int,
) -> tuple[dict[str, dict[str, object]], dict[str, np.ndarray], list[dict[str, object]]]:
    arm_results: dict[str, dict[str, object]] = {}
    loaded: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for arm in arms:
        response = load_arm_response(response_dir, arm, client_ids)
        loaded[arm] = response
        combined = generic_probe_statistics(response)
        bank_a = generic_probe_statistics(response[:, :, :bank_size])
        bank_b = generic_probe_statistics(response[:, :, bank_size:])
        result = summarize_generic_probe_arm(combined, bank_a, bank_b)
        arm_results[arm] = result
        for bank_name in ("combined", "bank_a", "bank_b"):
            bank_result = result[bank_name]
            for local_id, client_id in enumerate(client_ids):
                rows.append(
                    {
                        "arm": arm,
                        "bank": bank_name,
                        "client": int(client_id),
                        "model": MODEL_NAMES[client_id],
                        "S": bank_result["S_client"][local_id],
                        "Dcf": bank_result["Dcf_client"][local_id],
                        "K": bank_result["K_client"][local_id],
                        "R": bank_result["R_client"][local_id],
                        "active_probes": bank_result["active_probe_count_client"][local_id],
                    }
                )
    return arm_results, loaded, rows


def markdown_report(summary: dict[str, object]) -> str:
    lines = [
        "# CLE K0-B v2 Taxonomy-Free Generic Probe Gate",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "Frozen classifiers and unlabeled CIFAR-100 carriers only; no training or checkpoint writes.",
        "Primary inference/scoring does not read corruption taxonomy, severity, family, binding, or private metadata.",
        "",
        "| arm | S | Dcf | K | R |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, result in summary["arms"].items():
        combined = result["combined"]
        lines.append(
            f"| {arm} | {combined['S']:.8f} | {combined['Dcf']:.8f} | "
            f"{combined['K']:.8f} | {combined['R']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            "```json",
            json.dumps(summary["decision"], indent=2, ensure_ascii=False),
            "```",
            "",
            "A formal GO permits only the separately specified K1 checkpoint-surgery stage.",
            "",
        ]
    )
    return "\n".join(lines)


def primary_manifest(output_dir: Path, files: list[Path]) -> dict[str, object]:
    rows = []
    for path in sorted(set(files)):
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "protocol": "cle_generic_probe_k0b_primary_artifacts_v2",
        "binding_or_corruption_taxonomy_read": False,
        "primary_artifacts": rows,
    }
    path = output_dir / "primary_artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_file"] = path.name
    manifest["manifest_sha256"] = sha256_file(path)
    return manifest


def optional_oracle_coverage(
    oracle_root: Path,
    loaded: dict[str, np.ndarray],
    *,
    client_ids: tuple[int, ...],
    output_dir: Path,
) -> dict[str, object]:
    """Post-hoc direction coverage only; it is deliberately absent from primary decision inputs."""

    rows: dict[str, object] = {}
    matrices: dict[str, np.ndarray] = {}
    for arm, response in loaded.items():
        generic = generic_probe_statistics(response)
        generic_mean = 0.5 * (generic.mu_a + generic.mu_b)
        arm_rows: list[dict[str, object]] = []
        for local_id, client_id in enumerate(client_ids):
            oracle_path = oracle_root / "responses" / f"{arm}_client{client_id}.npz"
            if not oracle_path.is_file():
                raise FileNotFoundError(oracle_path)
            with np.load(oracle_path, allow_pickle=False) as archive:
                oracle = np.asarray(archive["mean_response"], dtype=np.float64)
            generic_client = generic_mean[local_id]
            denominator = np.linalg.norm(oracle, axis=1)[:, None] * np.linalg.norm(
                generic_client, axis=1
            )[None, :]
            cosine = np.divide(
                oracle @ generic_client.T,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 1.0e-12,
            )
            coverage = cosine.max(axis=1)
            matrices[f"{arm}_client{client_id}"] = cosine.astype(np.float32)
            arm_rows.append(
                {
                    "client": int(client_id),
                    "mean_max_cosine": float(coverage.mean()),
                    "minimum_max_cosine": float(coverage.min()),
                    "maximum_max_cosine": float(coverage.max()),
                }
            )
        rows[arm] = arm_rows
    np.savez_compressed(output_dir / "secondary_oracle_coverage.npz", **matrices)
    result = {
        "protocol": "k0b_posthoc_oracle_coverage_only",
        "changes_primary_verdict": False,
        "used_for_probe_selection": False,
        "coverage": rows,
    }
    (output_dir / "secondary_oracle_coverage.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    formal = args.mode == "formal"
    carrier_count = FORMAL_PUBLIC_COUNT if formal else 8
    evaluated_bank_size = FORMAL_BANK_SIZE if formal else 2
    arms = ARMS if formal else ("h0", "h9")
    client_ids = tuple(range(4)) if formal else (0,)
    bootstrap_samples = int(args.bootstrap_samples) if formal else 100
    if formal and int(args.bootstrap_samples) != 1000:
        raise ValueError("formal K0-B v2 freezes bootstrap_samples=1000")

    output_dir = args.output_dir.resolve()
    bank_dir = output_dir / "probe_banks"
    response_dir = output_dir / "responses"
    metrics_dir = output_dir / "metrics"
    for path in (bank_dir, response_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    bank_a = load_frozen_prime_bank(
        state_path=FROZEN_BANK_ROOT / "bank_a_states.npz",
        manifest_path=FROZEN_BANK_ROOT / "bank_a_manifest.json",
    )
    bank_b = load_frozen_prime_bank(
        state_path=FROZEN_BANK_ROOT / "bank_b_states.npz",
        manifest_path=FROZEN_BANK_ROOT / "bank_b_manifest.json",
    )
    if bank_a["bank_sha256"] != BANK_A_SHA256 or bank_b["bank_sha256"] != BANK_B_SHA256:
        raise ValueError("frozen PRIME bank hash mismatch")
    for name in (
        "bank_a_states.npz",
        "bank_a_manifest.json",
        "bank_b_states.npz",
        "bank_b_manifest.json",
    ):
        shutil.copy2(FROZEN_BANK_ROOT / name, bank_dir / name)
    recipes = (
        list(bank_a["recipes"][:evaluated_bank_size])
        + list(bank_b["recipes"][:evaluated_bank_size])
    )

    public_images, public_indices = select_k0a_public_carriers(
        args.public_root.resolve(), count=carrier_count
    )
    public_indices_path = output_dir / "selected_public_indices.npy"
    np.save(public_indices_path, public_indices, allow_pickle=False)
    public_hash = sha256_array(public_indices)
    if formal and public_hash != K0A_PUBLIC_INDICES_SHA256:
        raise ValueError("formal public carrier indices do not match K0-A")
    device = resolve_device(args.device)
    print(
        f"[setup] mode={args.mode} device={device} carriers={carrier_count} "
        f"bank_size={evaluated_bank_size} arms={arms} clients={client_ids}",
        flush=True,
    )
    response_rows = generate_responses(
        args.checkpoint_root.resolve(),
        public_images,
        recipes,
        arms=arms,
        client_ids=client_ids,
        device=device,
        batch_size=int(args.batch_size),
        response_dir=response_dir,
    )
    blind_manifest = {
        "protocol": "cle_generic_probe_k0b_blind_responses_v2",
        "mode": args.mode,
        "training_performed": False,
        "public_labels_used": False,
        "corruption_taxonomy_used": False,
        "severity_used": False,
        "binding_used": False,
        "private_corruption_metadata_used": False,
        "public_seed": PUBLIC_SEED,
        "public_indices_sha256": public_hash,
        "carrier_halves": {
            "Ua": [0, carrier_count // 2],
            "Ub": [carrier_count // 2, carrier_count],
            "disjoint": True,
        },
        "bank_a_seed": BANK_A_SEED,
        "bank_b_seed": BANK_B_SEED,
        "bank_a_sha256": bank_a["bank_sha256"],
        "bank_b_sha256": bank_b["bank_sha256"],
        "recipe_state_reused_for_every_carrier": True,
        "responses": response_rows,
    }
    blind_path = output_dir / "blind_response_manifest.json"
    blind_path.write_text(json.dumps(blind_manifest, indent=2), encoding="utf-8")
    blind_hash = sha256_file(blind_path)
    print(f"[blind] response manifest sha256={blind_hash}", flush=True)

    arm_results, loaded, metric_rows = score_primary(
        response_dir,
        arms=arms,
        client_ids=client_ids,
        bank_size=evaluated_bank_size,
    )
    write_csv(metrics_dir / "per_client_metrics.csv", metric_rows)
    bootstrap: dict[str, dict[str, object]] = {}
    if formal:
        bootstrap["hfl"] = paired_bootstrap_generic_deltas(
            loaded["h0"],
            loaded["h9"],
            samples=bootstrap_samples,
            seed=BOOTSTRAP_HFL_SEED,
            device=device,
        )
        bootstrap["local"] = paired_bootstrap_generic_deltas(
            loaded["l0"],
            loaded["l9"],
            samples=bootstrap_samples,
            seed=BOOTSTRAP_LOCAL_SEED,
            device=device,
        )
        decision = decide_generic_probe_gate(arm_results, bootstrap)
    else:
        decision = {
            "verdict": "SMOKE_ONLY_NO_SCIENTIFIC_DECISION",
            "scientific_decision_allowed": False,
        }
    (metrics_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )

    config = {
        "protocol": "cle_generic_probe_k0b_v2_20260901",
        "mode": args.mode,
        "training_performed": False,
        "public_dataset": "CIFAR-100 train; labels unused",
        "public_size": carrier_count,
        "public_seed": PUBLIC_SEED,
        "halves": [carrier_count // 2, carrier_count // 2],
        "bank_a": {"seed": BANK_A_SEED, "formal_size": 64, "evaluated_size": evaluated_bank_size},
        "bank_b": {"seed": BANK_B_SEED, "formal_size": 64, "evaluated_size": evaluated_bank_size},
        "primary_response": "centered class-vs-rest logit delta",
        "active_rule": "per-client energy >= within-bank median energy",
        "risk": "CVaR top 20 percent rho among active probes",
        "bootstrap_samples": bootstrap_samples,
        "primary_forbidden_metadata": [
            "CIFAR-C operator",
            "corruption type",
            "family",
            "severity",
            "CLE binding",
            "private corruption metadata",
        ],
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        "protocol": config["protocol"],
        "mode": args.mode,
        "verdict": decision["verdict"],
        "scientific_decision_allowed": formal,
        "integrity": {
            "checkpoint_count": len(response_rows),
            "training_performed": False,
            "public_labels_used": False,
            "taxonomy_or_binding_used_in_primary": False,
            "blind_manifest_sha256_before_primary_scoring": blind_hash,
            "public_indices_sha256": public_hash,
            "bank_a_sha256": bank_a["bank_sha256"],
            "bank_b_sha256": bank_b["bank_sha256"],
            "recipe_state_reused_for_every_carrier": True,
        },
        "arms": arm_results,
        "bootstrap": bootstrap,
        "decision": decision,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = output_dir / "final_report.md"
    report_path.write_text(markdown_report(summary), encoding="utf-8")

    primary_files = [
        public_indices_path,
        bank_dir / "bank_a_states.npz",
        bank_dir / "bank_a_manifest.json",
        bank_dir / "bank_b_states.npz",
        bank_dir / "bank_b_manifest.json",
        blind_path,
        metrics_dir / "per_client_metrics.csv",
        metrics_dir / "bootstrap.json",
        config_path,
        result_path,
        report_path,
        *[response_dir / str(row["response_file"]) for row in response_rows],
    ]
    sealed = primary_manifest(output_dir, primary_files)
    print(f"[primary] sealed manifest sha256={sealed['manifest_sha256']}", flush=True)

    if args.oracle_k0a_root is not None:
        print("[secondary] primary artifacts sealed; opening optional K0-A oracle directions", flush=True)
        optional_oracle_coverage(
            args.oracle_k0a_root.resolve(),
            loaded,
            client_ids=client_ids,
            output_dir=output_dir,
        )
    print(json.dumps(decision, indent=2), flush=True)
    print(f"[complete] {result_path}", flush=True)


if __name__ == "__main__":
    main()
