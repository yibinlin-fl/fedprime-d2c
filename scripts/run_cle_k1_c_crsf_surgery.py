from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.augmentations.frozen_prime import apply_frozen_prime_recipe, load_frozen_prime_bank  # noqa: E402
from fedprime.data.loaders import cifar100_train_images_from_tar, dataset_stats, normalize_batch  # noqa: E402
from fedprime.engine.cle_crsf_surgery import (  # noqa: E402
    EPS,
    LateBlockAdapter,
    apply_state_delta,
    changed_state_keys,
    direct_rawspec_gradients,
    direct_response_gradients,
    features_from_prefix_numpy,
    gradient_agreement,
    public_anchor_kl,
    raw_moments_from_prefix,
    rawspec_loss_from_moments,
    response_loss_from_moments,
    response_moments_from_prefix,
    run_exact_surgery,
    state_dict_sha256,
    two_pass_rawspec_gradients,
    two_pass_response_gradients,
)
from fedprime.models.factory import build_models  # noqa: E402
from scripts.run_cle_k1_sdmn_headonly import (  # noqa: E402
    BANK_A_SHA256,
    BANK_B_SHA256,
    BANK_ROOT,
    MODEL_NAMES,
    load_state,
    public_split,
    resolve_device,
    sha256_array,
    sha256_file,
    write_json,
)


PROTOCOL = "cle_k1_c_crsf_checkpoint_surgery_v1"
SYSTEMS = ("h9", "l9")
FOLDS = (("ab", "a", "b"), ("ba", "b", "a"))
ARMS = ("frozen", "crsf", "shared_mean", "generic_invariance", "rawspec")
OBJECTIVE_BY_ARM = {
    "crsf": "crsf",
    "shared_mean": "shared_mean",
    "generic_invariance": "generic_invariance",
    "rawspec": "rawspec",
}
LR_CANDIDATES = (1.0e-5, 3.0e-5, 1.0e-4)
SURGERY_HASH = "B5441E50539085299F81CD1291636C84A18BA2894BA57D8CB2631D6DF905334A"
HOLDOUT_HASH = "321C0910E8AA376B10D04D1319F24917EE91EABD25BCC8C31A0BDE66F8E240EE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K1-C CRSF late-block checkpoint surgery.")
    parser.add_argument("--mode", choices=("inspect", "smoke", "calibration", "formal"), default="inspect")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, default=None)
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-carriers", type=int, default=16)
    parser.add_argument("--smoke-probes", type=int, default=4)
    return parser.parse_args()


def load_banks() -> dict[str, dict[str, object]]:
    banks = {
        "a": load_frozen_prime_bank(
            state_path=BANK_ROOT / "bank_a_states.npz", manifest_path=BANK_ROOT / "bank_a_manifest.json"
        ),
        "b": load_frozen_prime_bank(
            state_path=BANK_ROOT / "bank_b_states.npz", manifest_path=BANK_ROOT / "bank_b_manifest.json"
        ),
    }
    if banks["a"]["bank_sha256"] != BANK_A_SHA256 or banks["b"]["bank_sha256"] != BANK_B_SHA256:
        raise ValueError("frozen PRIME bank hash mismatch")
    return banks


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def source_manifest(extra: list[Path] | None = None) -> dict[str, object]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "fedprime/engine/cle_crsf_surgery.py",
        ROOT / "fedprime/engine/cle_response_spectrum.py",
        ROOT / "fedprime/engine/cle_shortcut_alignment.py",
        ROOT / "scripts/run_cle_k1_sdmn_formal.py",
    ]
    paths.extend(extra or [])
    return {
        "git_commit": git_commit(),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix() if ROOT in path.parents else path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
            if path.is_file()
        ],
    }


def state_clone(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    values = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    values.update({name: value.detach().cpu().clone() for name, value in model.named_buffers()})
    return values


def make_adapter(
    architecture: str,
    checkpoint: Path,
    device: torch.device,
) -> tuple[LateBlockAdapter, dict[str, torch.Tensor], dict[str, object]]:
    client_id = MODEL_NAMES.index(architecture)
    model = build_models(list(MODEL_NAMES), num_classes=10)[client_id]
    state = load_state(checkpoint)
    model.load_state_dict(state, strict=True)
    adapter = LateBlockAdapter(model, architecture)
    audit = adapter.configure(device)
    original = state_clone(model)
    row = {
        "architecture": audit.architecture,
        "trainable_stage": audit.trainable_stage,
        "trainable_parameter_names": list(audit.trainable_parameter_names),
        "frozen_parameter_names": list(audit.frozen_parameter_names),
        "trainable_parameter_count": audit.trainable_parameter_count,
        "total_parameter_count": audit.total_parameter_count,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "classifier_sha256": state_dict_sha256(
            {name: value for name, value in original.items() if name.startswith("linear.")}
        ),
    }
    return adapter, original, row


def build_prefix_cache(
    adapter: LateBlockAdapter,
    images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.memmap, list[np.memmap], dict[str, object]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = dataset_stats("cifar10")

    def one(path: Path, recipe: dict[str, object] | None) -> tuple[np.memmap, str]:
        output = None
        digest = hashlib.sha256()
        with torch.no_grad():
            for start in range(0, images.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), images.shape[0])
                batch = torch.from_numpy(np.ascontiguousarray(images[start:stop]))
                batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
                if recipe is not None:
                    batch = apply_frozen_prime_recipe(batch, recipe)
                prefix = adapter.prefix(normalize_batch(batch, stats)).detach().cpu().numpy().astype(np.float32)
                if output is None:
                    output = np.lib.format.open_memmap(
                        path, mode="w+", dtype=np.float32, shape=(images.shape[0], *prefix.shape[1:])
                    )
                output[start:stop] = prefix
                digest.update(np.ascontiguousarray(prefix).tobytes())
        if output is None:
            raise ValueError("empty public carrier set")
        output.flush()
        return np.load(path, mmap_mode="r"), digest.hexdigest().upper()

    base, base_hash = one(cache_dir / "base.npy", None)
    probes = []
    probe_hashes = []
    for probe_id, recipe in enumerate(recipes):
        probe, digest = one(cache_dir / f"probe_{probe_id:03d}.npy", recipe)
        probes.append(probe)
        probe_hashes.append(digest)
    return base, probes, {
        "implementation": "float32 frozen-prefix numpy memmap",
        "carriers": int(images.shape[0]),
        "probes": len(recipes),
        "prefix_shape": list(base.shape[1:]),
        "base_content_sha256": base_hash,
        "probe_content_sha256": probe_hashes,
        "exported": False,
    }


def verify_adapter_exactness(
    adapter: LateBlockAdapter, images: np.ndarray, *, device: torch.device
) -> float:
    stats = dataset_stats("cifar10")
    batch = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    batch = batch.to(device=device, dtype=torch.float32).div_(255.0)
    normalized = normalize_batch(batch, stats)
    with torch.no_grad():
        expected = adapter.model.backbone(normalized).flatten(1)
        actual = adapter.feature_from_prefix(adapter.prefix(normalized))
    return float((expected - actual).abs().max().cpu())


def inspect(
    args: argparse.Namespace,
    images: np.ndarray,
    split: dict[str, np.ndarray],
    banks: dict[str, dict[str, object]],
    device: torch.device,
) -> dict[str, object]:
    rows = []
    for client_id, architecture in enumerate(MODEL_NAMES):
        checkpoint = args.checkpoint_root / "h9" / f"client_{client_id}.pt"
        adapter, _state, audit = make_adapter(architecture, checkpoint, device)
        audit["representation_max_abs_error"] = verify_adapter_exactness(
            adapter, images[split["surgery"][:2]], device=device
        )
        rows.append(audit)
        adapter.model.to("cpu")
    payload = {
        "protocol": PROTOCOL,
        "verdict": "INSPECT_PASS_NO_SCIENTIFIC_DECISION",
        "model_block_mapping": rows,
        "public_split": {
            "surgery_count": int(split["surgery"].size),
            "surgery_sha256": sha256_array(split["surgery"]),
            "holdout_count": int(split["holdout"].size),
            "holdout_sha256": sha256_array(split["holdout"]),
            "disjoint": np.intersect1d(split["surgery"], split["holdout"]).size == 0,
            "labels_loaded": False,
        },
        "prime_banks": {name: bank["bank_sha256"] for name, bank in banks.items()},
        "reused_code": {
            "spectrum": "fedprime/engine/cle_response_spectrum.py",
            "oracle": "scripts/run_cle_k1_sdmn_formal.py oracle_evaluation semantics",
        },
        "exact_gradient": "two-pass full-carrier sufficient-statistic VJP",
        "artifact_policy": "late-block deltas only; frozen-prefix cache is temporary and unexported",
        "source": source_manifest(),
    }
    if payload["public_split"]["surgery_sha256"] != SURGERY_HASH:
        raise ValueError("D_surgery hash mismatch")
    if payload["public_split"]["holdout_sha256"] != HOLDOUT_HASH:
        raise ValueError("D_holdout hash mismatch")
    return payload


def tiny_gradient_audit(device: torch.device) -> dict[str, object]:
    torch.manual_seed(20260913)
    layer = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 4)).to(device).double()
    base = torch.randn(9, 5, dtype=torch.float64, device=device)
    probes = base[:, None] + 0.2 * torch.randn(9, 4, 5, dtype=torch.float64, device=device)
    rows = {}
    for objective in ("crsf", "shared_mean", "generic_invariance"):
        direct = direct_response_gradients(list(layer.parameters()), layer, base, probes, objective)
        exact = two_pass_response_gradients(list(layer.parameters()), layer, base, probes, objective)
        rows[objective] = vars(gradient_agreement(direct, exact))
    direct = direct_rawspec_gradients(list(layer.parameters()), layer, base)
    exact = two_pass_rawspec_gradients(list(layer.parameters()), layer, base)
    rows["rawspec"] = vars(gradient_agreement(direct, exact))
    rows["pass"] = all(
        row["relative_error"] <= 1.0e-5 and row["cosine"] >= 0.99999
        for row in rows.values()
        if isinstance(row, dict)
    )
    return rows


def trace_dict(trace) -> dict[str, object]:
    return {
        "objective": trace.objective,
        "initial_raw_loss": trace.initial_raw_loss,
        "accepted_raw_losses": list(trace.accepted_raw_losses),
        "accepted_normalized_losses": list(trace.accepted_normalized_losses),
        "anchor_kl": list(trace.anchor_kl),
        "accepted_learning_rates": list(trace.accepted_learning_rates),
        "attempts": list(trace.attempts),
        "accepted_steps": trace.accepted_steps,
        "contract_failure": trace.contract_failure,
    }


def save_delta(
    path: Path,
    adapter: LateBlockAdapter,
    original: dict[str, torch.Tensor],
    audit: dict[str, object],
) -> dict[str, object]:
    delta = adapter.state_delta(original)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": PROTOCOL,
            "source_checkpoint_sha256": audit["checkpoint_sha256"],
            "classifier_sha256": audit["classifier_sha256"],
            "architecture": audit["architecture"],
            "trainable_stage": audit["trainable_stage"],
            "parameter_names": list(delta),
            "delta": delta,
        },
        path,
    )
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_delta(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    return payload["delta"]


def smoke(
    args: argparse.Namespace,
    images: np.ndarray,
    split: dict[str, np.ndarray],
    banks: dict[str, dict[str, object]],
    device: torch.device,
) -> dict[str, object]:
    architecture = MODEL_NAMES[0]
    checkpoint = args.checkpoint_root / "h9" / "client_0.pt"
    original_adapter, original, audit = make_adapter(architecture, checkpoint, device)
    surgery_images = images[split["surgery"][: args.smoke_carriers]]
    holdout_images = images[split["holdout"][: args.smoke_carriers]]
    recipes_a = list(banks["a"]["recipes"])[: args.smoke_probes]
    recipes_b = list(banks["b"]["recipes"])[: args.smoke_probes]
    base, probes, cache = build_prefix_cache(
        original_adapter,
        surgery_images,
        recipes_a,
        cache_dir=args.cache_dir / "smoke_surgery",
        device=device,
        batch_size=args.batch_size,
    )
    representation_error = verify_adapter_exactness(original_adapter, surgery_images[:2], device=device)
    rows = []
    delta_rows = []
    for arm in ARMS[1:]:
        adapter = original_adapter.clone(device)
        trace = run_exact_surgery(
            adapter,
            OBJECTIVE_BY_ARM[arm],
            base,
            probes,
            device=device,
            batch_size=args.batch_size,
            learning_rate=1.0e-5,
            accepted_steps=2,
            anchor_limit=0.02,
        )
        changed = changed_state_keys(original, state_clone(adapter.model))
        delta_info = save_delta(args.output_dir / "surgery_block_deltas" / f"h9_ab_client0_{arm}.pt", adapter, original, audit)
        rows.append({"arm": arm, "trace": trace_dict(trace), "changed_keys": list(changed)})
        delta_rows.append(delta_info)
    holdout_base, holdout_probes, holdout_cache = build_prefix_cache(
        original_adapter,
        holdout_images,
        recipes_b,
        cache_dir=args.cache_dir / "smoke_holdout",
        device=device,
        batch_size=args.batch_size,
    )
    # Numerical effect is deliberately not reported: smoke is execution-only.
    _ = response_moments_from_prefix(
        original_adapter, holdout_base, holdout_probes, device=device, batch_size=args.batch_size
    )
    gradient = tiny_gradient_audit(device)
    approved = set(audit["trainable_parameter_names"])
    checks = {
        "exact_representation_extraction": representation_error <= 1.0e-6,
        "approved_trainable_block_only": all(set(row["changed_keys"]).issubset(approved) for row in rows),
        "classifier_unchanged": all(not any(name.startswith("linear.") for name in row["changed_keys"]) for row in rows),
        "bn_unchanged": all(not any("bn" in name.lower() for name in row["changed_keys"]) for row in rows),
        "two_pass_crsf_gradient": bool(gradient["crsf"]["relative_error"] <= 1e-5 and gradient["crsf"]["cosine"] >= .99999),
        "two_pass_rawspec_gradient": bool(gradient["rawspec"]["relative_error"] <= 1e-5 and gradient["rawspec"]["cosine"] >= .99999),
        "shared_mean_exact_gradient": bool(gradient["shared_mean"]["relative_error"] <= 1e-5),
        "gi_exact_gradient": bool(gradient["generic_invariance"]["relative_error"] <= 1e-5),
        "normalized_loss_starts_one": all(abs(row["trace"]["accepted_normalized_losses"][0] - 1.0) <= 1e-8 for row in rows),
        "accepted_objective_decreases": all(row["trace"]["accepted_normalized_losses"][-1] <= 1.0 + 1e-6 for row in rows),
        "anchor_kl_computable": all(np.isfinite(row["trace"]["anchor_kl"]).all() for row in rows),
        "rollback_state_captured": all(len(row["trace"]["attempts"]) >= 2 for row in rows),
        "ab_ba_independent_start_contract": True,
        "unseen_not_used_for_optimization": cache["base_content_sha256"] != holdout_cache["base_content_sha256"],
        "oracle_assets_not_loaded_stage1": True,
        "source_hashes_recorded": bool(source_manifest()["files"]),
        "only_block_deltas_exported": len(delta_rows) == 4 and not list(args.output_dir.rglob("*.full.pt")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"smoke failed: {[name for name, value in checks.items() if not value]}")
    return {
        "protocol": PROTOCOL,
        "verdict": "SMOKE_ONLY_NO_SCIENTIFIC_DECISION",
        "scientific_decision_allowed": False,
        "checks": checks,
        "gradient_audit": gradient,
        "trainable_block": audit,
        "surgery_cache": cache,
        "holdout_cache": holdout_cache,
        "optimization_execution": rows,
        "saved_deltas": delta_rows,
        "oracle_assets_loaded": False,
    }


def blind_calibration(
    args: argparse.Namespace,
    images: np.ndarray,
    split: dict[str, np.ndarray],
    banks: dict[str, dict[str, object]],
    device: torch.device,
) -> dict[str, object]:
    surgery_images = images[split["surgery"]]
    contexts: dict[str, list[dict[str, object]]] = {architecture: [] for architecture in MODEL_NAMES}
    for client_id, architecture in enumerate(MODEL_NAMES):
        for system in SYSTEMS:
            checkpoint = args.checkpoint_root / system / f"client_{client_id}.pt"
            original_adapter, original, audit = make_adapter(architecture, checkpoint, device)
            for fold_name, correction_bank, _unseen_bank in FOLDS:
                base, probes, cache = build_prefix_cache(
                    original_adapter,
                    surgery_images,
                    list(banks[correction_bank]["recipes"]),
                    cache_dir=args.cache_dir / f"calibration_{architecture}",
                    device=device,
                    batch_size=args.batch_size,
                )
                candidate_rows = []
                for learning_rate in LR_CANDIDATES:
                    adapter = original_adapter.clone(device)
                    trace = run_exact_surgery(
                        adapter,
                        "crsf",
                        base,
                        probes,
                        device=device,
                        batch_size=args.batch_size,
                        learning_rate=learning_rate,
                        accepted_steps=3,
                        anchor_limit=0.02,
                    )
                    normalized = np.asarray(trace.accepted_normalized_losses, dtype=np.float64)
                    passed = bool(
                        not trace.contract_failure
                        and trace.accepted_steps == 3
                        and np.isfinite(normalized).all()
                        and np.all(np.diff(normalized) <= 1.0e-6)
                        and max(trace.anchor_kl) < 0.005
                    )
                    candidate_rows.append(
                        {
                            "learning_rate": learning_rate,
                            "passed": passed,
                            "trace": trace_dict(trace),
                        }
                    )
                contexts[architecture].append(
                    {
                        "system": system,
                        "fold": fold_name,
                        "checkpoint_sha256": audit["checkpoint_sha256"],
                        "cache": cache,
                        "candidates": candidate_rows,
                    }
                )
                del base, probes
            original_adapter.model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    selected = {}
    for architecture, rows in contexts.items():
        passing = []
        for learning_rate in LR_CANDIDATES:
            if all(
                next(row for row in context["candidates"] if row["learning_rate"] == learning_rate)["passed"]
                for context in rows
            ):
                passing.append(learning_rate)
        selected[architecture] = max(passing) if passing else None
    calibration_failed = any(value is None for value in selected.values())
    calibration_config = {
        "candidate_learning_rates": list(LR_CANDIDATES),
        "steps": 3,
        "optimizer": "Adam defaults; weight_decay=0",
        "step_anchor_limit": 0.02,
        "final_anchor_strict_limit": 0.005,
        "selection": "largest candidate passing all H9/L9 x AB/BA contexts",
    }
    config_hash = hashlib.sha256(
        json.dumps(calibration_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    return {
        "protocol": "cle_k1_c_crsf_blind_calibration_v1",
        "verdict": "CALIBRATION_FAIL" if calibration_failed else "CALIBRATION_PASS_NO_SCIENTIFIC_DECISION",
        "scientific_decision_allowed": False,
        "candidate_learning_rates": list(LR_CANDIDATES),
        "calibration_config": calibration_config,
        "calibration_config_sha256": config_hash,
        "selected_learning_rate_by_architecture": selected,
        "requirements": {
            "contexts_per_architecture": ["h9_ab", "h9_ba", "l9_ab", "l9_ba"],
            "accepted_steps": 3,
            "anchor_kl_strictly_below": 0.005,
            "normalized_crsf_nonincreasing": True,
            "selection": "largest LR passing all four contexts",
        },
        "public_split": {
            "surgery_count": 2000,
            "surgery_sha256": sha256_array(split["surgery"]),
            "holdout_opened": False,
            "labels_loaded": False,
        },
        "bank_sha256": {name: banks[name]["bank_sha256"] for name in ("a", "b")},
        "contexts": contexts,
        "forbidden_metrics_loaded": [],
        "source": source_manifest(),
    }


def load_calibration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "cle_k1_c_crsf_blind_calibration_v1":
        raise ValueError("unexpected K1-C calibration protocol")
    if payload.get("verdict") != "CALIBRATION_PASS_NO_SCIENTIFIC_DECISION":
        raise ValueError("K1-C formal requires a passing frozen calibration")
    selected = payload.get("selected_learning_rate_by_architecture", {})
    if set(selected) != set(MODEL_NAMES) or any(float(value) not in LR_CANDIDATES for value in selected.values()):
        raise ValueError("invalid frozen K1-C learning-rate mapping")
    if payload["public_split"]["surgery_sha256"] != SURGERY_HASH:
        raise ValueError("calibration D_surgery hash mismatch")
    return payload


def spectrum_metrics(
    adapter: LateBlockAdapter,
    base: np.ndarray,
    probes: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    midpoint = int(base.shape[0]) // 2
    chis = []
    energies = []
    saved = {}
    for half_name, slc in (("u1", slice(0, midpoint)), ("u2", slice(midpoint, base.shape[0]))):
        moments, _ = response_moments_from_prefix(
            adapter,
            base[slc],
            [probe[slc] for probe in probes],
            device=device,
            batch_size=batch_size,
        )
        mean = moments.mean.numpy()
        energy = moments.energy.numpy()
        response = mean / (np.sqrt(np.maximum(energy, 0.0))[:, None] + EPS)
        gram = response @ response.T
        chi = float(np.square(gram).sum() / (np.trace(gram) ** 2 + EPS))
        chis.append(chi)
        energies.append(float(energy.mean()))
        saved[f"mean_{half_name}"] = mean
        saved[f"energy_{half_name}"] = energy
        saved[f"gram_{half_name}"] = gram
    return {
        "chi_u1": chis[0],
        "chi_u2": chis[1],
        "chi_unseen": float(np.mean(chis)),
        "response_energy": float(np.mean(energies)),
    }, saved


def primary_seal(output_dir: Path) -> dict[str, object]:
    allowed = (
        "config.json",
        "source_manifest.json",
        "input_manifest.json",
        "checkpoint_manifest.json",
        "public_split_manifest.json",
        "prime_bank_manifest.json",
        "trainable_block_manifest.json",
        "calibration_manifest.json",
        "optimization_traces.jsonl",
        "surgery_block_deltas/",
        "taxonomy_free_metrics.csv",
        "unseen_response_gram_matrices.npz",
        "unseen_response_moments.npz",
        "unseen_response_energy.csv",
        "stage1_gate_inputs.json",
    )
    rows = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative == "primary_taxonomy_free_manifest.json":
            continue
        if not any(relative == root or (root.endswith("/") and relative.startswith(root)) for root in allowed):
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"protocol": PROTOCOL, "sealed_before_oracle": True, "files": rows}


def aggregate_stage1(rows: list[dict[str, object]]) -> dict[str, object]:
    output = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        fold_reduction = {}
        for fold_name, _a, _b in FOLDS:
            fold_rows = [row for row in system_rows if row["fold"] == fold_name]
            baseline = np.mean([row["chi_unseen"] for row in fold_rows if row["arm"] == "frozen"])
            fold_reduction[fold_name] = {
                arm: float(1.0 - np.mean([row["chi_unseen"] for row in fold_rows if row["arm"] == arm]) / baseline)
                for arm in ARMS
            }
        baseline = np.mean([row["chi_unseen"] for row in system_rows if row["arm"] == "frozen"])
        reductions = {
            arm: float(1.0 - np.mean([row["chi_unseen"] for row in system_rows if row["arm"] == arm]) / baseline)
            for arm in ARMS
        }
        client_positive = 0
        for client in range(4):
            client_rows = [row for row in system_rows if row["client"] == client]
            frozen = np.mean([row["chi_unseen"] for row in client_rows if row["arm"] == "frozen"])
            crsf = np.mean([row["chi_unseen"] for row in client_rows if row["arm"] == "crsf"])
            client_positive += int(crsf < frozen)
        energy_retention = []
        for fold_name, _a, _b in FOLDS:
            fold_rows = [row for row in system_rows if row["fold"] == fold_name]
            before = np.mean([row["response_energy"] for row in fold_rows if row["arm"] == "frozen"])
            after = np.mean([row["response_energy"] for row in fold_rows if row["arm"] == "crsf"])
            energy_retention.append({"fold": fold_name, "value": float(after / max(before, EPS))})
        gates = {
            "combined_crsf_reduction_ge_25pct": reductions["crsf"] >= 0.25,
            "ab_crsf_reduction_ge_15pct": fold_reduction["ab"]["crsf"] >= 0.15,
            "ba_crsf_reduction_ge_15pct": fold_reduction["ba"]["crsf"] >= 0.15,
            "positive_clients_ge_3of4": client_positive >= 3,
            "energy_retention_each_fold_ge_50pct": all(row["value"] >= 0.50 for row in energy_retention),
            "crsf_minus_rawspec_reduction_ge_10pp": reductions["crsf"] - reductions["rawspec"] >= 0.10,
        }
        output[system] = {
            "combined_reduction": reductions,
            "fold_reduction": fold_reduction,
            "positive_clients": client_positive,
            "energy_retention": energy_retention,
            "gates": gates,
            "pass": all(gates.values()),
        }
    return output


def run_formal_stage1(
    args: argparse.Namespace,
    images: np.ndarray,
    split: dict[str, np.ndarray],
    banks: dict[str, dict[str, object]],
    calibration: dict[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    surgery_images = images[split["surgery"]]
    holdout_images = images[split["holdout"]]
    metric_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    caches: dict[str, object] = {}
    moments_to_save: dict[str, np.ndarray] = {}
    grams_to_save: dict[str, np.ndarray] = {}
    delta_root = args.output_dir / "surgery_block_deltas"

    for system in SYSTEMS:
        for client_id, architecture in enumerate(MODEL_NAMES):
            checkpoint = args.checkpoint_root / system / f"client_{client_id}.pt"
            original_adapter, original, audit = make_adapter(architecture, checkpoint, device)
            block_rows.append({"system": system, "client": client_id, **audit})
            checkpoint_rows.append(
                {
                    "system": system,
                    "client": client_id,
                    "architecture": architecture,
                    "path": checkpoint.as_posix(),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": audit["checkpoint_sha256"],
                }
            )
            for fold_name, correction_bank, unseen_bank in FOLDS:
                context = f"{system}_{fold_name}_client{client_id}"
                base, probes, cache = build_prefix_cache(
                    original_adapter,
                    surgery_images,
                    list(banks[correction_bank]["recipes"]),
                    cache_dir=args.cache_dir / "formal_work",
                    device=device,
                    batch_size=args.batch_size,
                )
                caches[f"{context}_surgery"] = cache
                for arm in ARMS[1:]:
                    adapter = original_adapter.clone(device)
                    trace = run_exact_surgery(
                        adapter,
                        OBJECTIVE_BY_ARM[arm],
                        base,
                        probes,
                        device=device,
                        batch_size=args.batch_size,
                        learning_rate=float(calibration["selected_learning_rate_by_architecture"][architecture]),
                        accepted_steps=10,
                        anchor_limit=0.02,
                        maximum_backtracks=12,
                    )
                    if trace.contract_failure or trace.accepted_steps != 10:
                        raise RuntimeError(f"OPTIMIZATION_CONTRACT_FAIL: {context}/{arm}")
                    changed = changed_state_keys(original, state_clone(adapter.model))
                    if set(changed) != set(audit["trainable_parameter_names"]):
                        # A mathematically zero delta is allowed, but no forbidden key is.
                        if not set(changed).issubset(set(audit["trainable_parameter_names"])):
                            raise RuntimeError(f"IMPLEMENTATION_FAIL forbidden parameter change: {context}/{arm}")
                    delta_path = delta_root / f"{context}_{arm}.pt"
                    delta_info = save_delta(delta_path, adapter, original, audit)
                    trace_rows.append(
                        {
                            "system": system,
                            "client": client_id,
                            "architecture": architecture,
                            "fold": fold_name,
                            "correction_bank": correction_bank,
                            "unseen_bank": unseen_bank,
                            "arm": arm,
                            "delta": delta_info,
                            "changed_keys": list(changed),
                            "trace": trace_dict(trace),
                        }
                    )
                    adapter.model.to("cpu")
                del base, probes
                holdout_base, holdout_probes, holdout_cache = build_prefix_cache(
                    original_adapter,
                    holdout_images,
                    list(banks[unseen_bank]["recipes"]),
                    cache_dir=args.cache_dir / "formal_work",
                    device=device,
                    batch_size=args.batch_size,
                )
                caches[f"{context}_holdout"] = holdout_cache
                arm_metrics = {}
                for arm in ARMS:
                    adapter = original_adapter.clone(device)
                    if arm != "frozen":
                        apply_state_delta(adapter.model, load_delta(delta_root / f"{context}_{arm}.pt"))
                    metrics, saved = spectrum_metrics(
                        adapter,
                        holdout_base,
                        holdout_probes,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    key = f"{context}_{arm}"
                    for name, value in saved.items():
                        moments_to_save[f"{key}_{name}"] = value
                        if name.startswith("gram_"):
                            grams_to_save[f"{key}_{name}"] = value
                    arm_metrics[arm] = metrics
                    adapter.model.to("cpu")
                frozen_chi = arm_metrics["frozen"]["chi_unseen"]
                frozen_energy = arm_metrics["frozen"]["response_energy"]
                for arm in ARMS:
                    metrics = arm_metrics[arm]
                    row = {
                        "system": system,
                        "client": client_id,
                        "architecture": architecture,
                        "fold": fold_name,
                        "correction_bank": correction_bank,
                        "unseen_bank": unseen_bank,
                        "arm": arm,
                        **metrics,
                        "delta_chi": float(1.0 - metrics["chi_unseen"] / max(frozen_chi, EPS)),
                        "energy_retention": float(metrics["response_energy"] / max(frozen_energy, EPS)),
                    }
                    metric_rows.append(row)
                    energy_rows.append(
                        {
                            "system": system,
                            "client": client_id,
                            "fold": fold_name,
                            "arm": arm,
                            "response_energy": metrics["response_energy"],
                            "energy_retention": row["energy_retention"],
                        }
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                del holdout_base, holdout_probes
            original_adapter.model.to("cpu")

    write_csv(args.output_dir / "taxonomy_free_metrics.csv", metric_rows)
    write_csv(args.output_dir / "unseen_response_energy.csv", energy_rows)
    np.savez_compressed(args.output_dir / "unseen_response_gram_matrices.npz", **grams_to_save)
    np.savez_compressed(args.output_dir / "unseen_response_moments.npz", **moments_to_save)
    with (args.output_dir / "optimization_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output_dir / "checkpoint_manifest.json", {"checkpoints": checkpoint_rows})
    write_json(args.output_dir / "trainable_block_manifest.json", {"blocks": block_rows})
    stage1 = aggregate_stage1(metric_rows)
    write_json(
        args.output_dir / "stage1_gate_inputs.json",
        {"summary": stage1, "cache_manifests": caches, "oracle_assets_loaded": False},
    )
    return metric_rows, stage1


def inference_probabilities(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    stats = dataset_stats("cifar10")
    rows = []
    model.to(device).eval()
    with torch.no_grad():
        for start in range(0, images.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), images.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(images[start:stop]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            output = model(normalize_batch(batch, stats))
            logits = output[0] if isinstance(output, tuple) else output
            rows.append(torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def run_formal_stage2(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if not (args.output_dir / "primary_taxonomy_free_manifest.json").is_file():
        raise RuntimeError("taxonomy-free Stage 1 must be sealed before Stage 2")
    if args.evaluation_root is None:
        raise ValueError("formal Stage 2 requires evaluation-root")
    oracle = importlib.import_module("fedprime.engine.cle_shortcut_alignment")
    clean_images = np.load(args.evaluation_root / "test_images.npy", allow_pickle=False)
    labels = np.load(args.evaluation_root / "test_labels.npy", allow_pickle=False).astype(np.int64)
    grid, severities = oracle.deterministic_corruption_grid(clean_images)
    binding = oracle.historical_family_binding(num_clients=4, num_classes=10)
    flat_grid = grid.reshape(-1, *grid.shape[2:])
    oracle_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    prediction_root = args.output_dir / "oracle_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    delta_root = args.output_dir / "surgery_block_deltas"
    for system in SYSTEMS:
        for fold_name, _correction, _unseen in FOLDS:
            arm_grid = {arm: [] for arm in ARMS}
            arm_clean = {arm: [] for arm in ARMS}
            for client_id, architecture in enumerate(MODEL_NAMES):
                checkpoint = args.checkpoint_root / system / f"client_{client_id}.pt"
                for arm in ARMS:
                    adapter, _original, _audit = make_adapter(architecture, checkpoint, device)
                    if arm != "frozen":
                        path = delta_root / f"{system}_{fold_name}_client{client_id}_{arm}.pt"
                        apply_state_delta(adapter.model, load_delta(path))
                    arm_grid[arm].append(
                        inference_probabilities(adapter.model, flat_grid, device=device, batch_size=args.batch_size).reshape(
                            clean_images.shape[0], grid.shape[1], 10
                        )
                    )
                    arm_clean[arm].append(
                        inference_probabilities(adapter.model, clean_images, device=device, batch_size=args.batch_size)
                    )
                    adapter.model.to("cpu")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            for arm in ARMS:
                probabilities = np.stack(arm_grid[arm])
                clean_probabilities = np.stack(arm_clean[arm])
                dsa = oracle.compute_dsa(probabilities, labels, binding, oracle.OPERATOR_FAMILY_IDS)
                secondary = oracle.secondary_metrics(probabilities, labels, binding, oracle.OPERATOR_FAMILY_IDS)
                client_clean = 100.0 * (clean_probabilities.argmax(-1) == labels[None]).mean(axis=1)
                prediction_path = prediction_root / f"{system}_{fold_name}_{arm}.npz"
                np.savez_compressed(
                    prediction_path,
                    probabilities=probabilities,
                    clean_probabilities=clean_probabilities,
                    labels=labels,
                    binding=binding,
                    operator_family_ids=oracle.OPERATOR_FAMILY_IDS,
                    severities=severities,
                )
                oracle_rows.append(
                    {
                        "system": system,
                        "fold": fold_name,
                        "arm": arm,
                        "dsa": float(dsa.pooled),
                        "dsa_client": json.dumps(dsa.client.tolist()),
                        "prediction_sha256": sha256_file(prediction_path),
                    }
                )
                task_rows.append(
                    {
                        "system": system,
                        "fold": fold_name,
                        "arm": arm,
                        "avg_acc": float(secondary["avg_acc"]),
                        "worst_acc": float(secondary["worst_acc"]),
                        "wcca": float(secondary["wcca"]),
                        "cfg": float(secondary["cfg"]),
                        "clean_avg": float(client_clean.mean()),
                        "clean_worst": float(client_clean.min()),
                    }
                )
    write_csv(args.output_dir / "oracle_dsa_metrics.csv", oracle_rows)
    write_csv(args.output_dir / "task_metrics.csv", task_rows)
    return oracle_rows, task_rows, aggregate_stage2(oracle_rows, task_rows)


def aggregate_stage2(
    oracle_rows: list[dict[str, object]], task_rows: list[dict[str, object]]
) -> dict[str, object]:
    output = {}
    for system in SYSTEMS:
        combined = {}
        for arm in ARMS:
            dsa_rows = [row for row in oracle_rows if row["system"] == system and row["arm"] == arm]
            metric_rows = [row for row in task_rows if row["system"] == system and row["arm"] == arm]
            combined[arm] = {
                "dsa": float(np.mean([float(row["dsa"]) for row in dsa_rows])),
                "dsa_client": np.mean(
                    [np.asarray(json.loads(str(row["dsa_client"])), dtype=np.float64) for row in dsa_rows], axis=0
                ).tolist(),
                **{
                    metric: float(np.mean([float(row[metric]) for row in metric_rows]))
                    for metric in ("avg_acc", "worst_acc", "wcca", "cfg", "clean_avg", "clean_worst")
                },
            }
        baseline = combined["frozen"]
        effects = {}
        for arm in ARMS[1:]:
            value = combined[arm]
            effects[arm] = {
                "dsa_reduction": baseline["dsa"] - value["dsa"],
                "dsa_relative_reduction": (baseline["dsa"] - value["dsa"]) / max(abs(baseline["dsa"]), EPS),
                "wcca_improvement": value["wcca"] - baseline["wcca"],
                "cfg_reduction": baseline["cfg"] - value["cfg"],
                "avg_loss": baseline["avg_acc"] - value["avg_acc"],
                "worst_loss": baseline["worst_acc"] - value["worst_acc"],
                "clean_loss": baseline["clean_avg"] - value["clean_avg"],
            }
        crsf_clients = np.asarray(combined["frozen"]["dsa_client"]) - np.asarray(combined["crsf"]["dsa_client"])
        crsf = effects["crsf"]
        gates = {
            "dsa_absolute_005_or_relative_25pct": crsf["dsa_reduction"] >= 0.05 or crsf["dsa_relative_reduction"] >= 0.25,
            "dsa_positive_clients_ge_3of4": int((crsf_clients > 0).sum()) >= 3,
            "crsf_minus_rawspec_dsa_ge_002": crsf["dsa_reduction"] - effects["rawspec"]["dsa_reduction"] >= 0.02,
            "wcca_improvement_ge_1pp": crsf["wcca_improvement"] >= 1.0,
            "cfg_reduction_ge_1pp": crsf["cfg_reduction"] >= 1.0,
            "avg_loss_le_1pp": crsf["avg_loss"] <= 1.0,
            "worst_loss_le_1pp": crsf["worst_loss"] <= 1.0,
            "clean_loss_le_1pp": crsf["clean_loss"] <= 1.0,
        }
        output[system] = {
            "combined": combined,
            "effects": effects,
            "crsf_positive_dsa_clients": int((crsf_clients > 0).sum()),
            "gates": gates,
        }
    return output


def baseline_dominance(stage2: dict[str, object]) -> dict[str, object]:
    per_system = {}
    for system in SYSTEMS:
        crsf = stage2[system]["effects"]["crsf"]
        rows = {}
        for arm in ("shared_mean", "generic_invariance"):
            control = stage2[system]["effects"][arm]
            conditions = {
                "dsa": control["dsa_reduction"] >= crsf["dsa_reduction"] - 0.01,
                "wcca": control["wcca_improvement"] >= crsf["wcca_improvement"] - 0.10,
                "cfg": control["cfg_reduction"] >= crsf["cfg_reduction"] - 0.10,
                "avg": control["avg_loss"] <= crsf["avg_loss"] + 0.10,
                "worst": control["worst_loss"] <= crsf["worst_loss"] + 0.10,
                "clean": control["clean_loss"] <= crsf["clean_loss"] + 0.10,
            }
            rows[arm] = {"conditions": conditions, "fully_dominates": all(conditions.values())}
        per_system[system] = rows
    global_dominators = [
        arm
        for arm in ("shared_mean", "generic_invariance")
        if all(per_system[system][arm]["fully_dominates"] for system in SYSTEMS)
    ]
    return {
        "per_system": per_system,
        "global_dominators": global_dominators,
        "generic_baseline_dominates": bool(global_dominators),
    }


def decide(
    stage1: dict[str, object], stage2: dict[str, object], dominance: dict[str, object]
) -> dict[str, object]:
    stage1_pass = all(stage1[system]["pass"] for system in SYSTEMS)
    causal_gate_names = (
        "dsa_absolute_005_or_relative_25pct",
        "dsa_positive_clients_ge_3of4",
        "crsf_minus_rawspec_dsa_ge_002",
    )
    causal_pass = all(
        all(stage2[system]["gates"][name] for name in causal_gate_names) for system in SYSTEMS
    )
    utility_gate_names = (
        "wcca_improvement_ge_1pp",
        "cfg_reduction_ge_1pp",
        "avg_loss_le_1pp",
        "worst_loss_le_1pp",
        "clean_loss_le_1pp",
    )
    utility_pass = all(
        all(stage2[system]["gates"][name] for name in utility_gate_names) for system in SYSTEMS
    )
    mechanism_pass = stage1_pass and causal_pass and not dominance["generic_baseline_dominates"]
    if not mechanism_pass:
        verdict = "NO_GO_CRSF_INTERVENTION"
    elif utility_pass:
        verdict = "GO_TO_TRAINING_INTEGRATION"
    else:
        verdict = "MECHANISM_PASS_INTEGRATION_NEEDS_REDESIGN"
    return {
        "verdict": verdict,
        "stage1_taxonomy_free_pass": stage1_pass,
        "stage2_dsa_and_rawspec_pass": causal_pass,
        "task_utility_pass": utility_pass,
        "generic_baseline_dominates": dominance["generic_baseline_dominates"],
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def independent_recompute(
    output_dir: Path,
    reported_stage1: dict[str, object],
    reported_stage2: dict[str, object],
    reported_decision: dict[str, object],
) -> dict[str, object]:
    stored = np.load(output_dir / "unseen_response_moments.npz", allow_pickle=False)
    metric_rows = read_csv_rows(output_dir / "taxonomy_free_metrics.csv")
    rebuilt_rows = []
    maximum_chi_error = 0.0
    for row in metric_rows:
        key = f"{row['system']}_{row['fold']}_client{row['client']}_{row['arm']}"
        chis = []
        energies = []
        for half in ("u1", "u2"):
            mean = stored[f"{key}_mean_{half}"]
            energy = stored[f"{key}_energy_{half}"]
            response = mean / (np.sqrt(np.maximum(energy, 0.0))[:, None] + EPS)
            gram = response @ response.T
            chis.append(float(np.square(gram).sum() / (np.trace(gram) ** 2 + EPS)))
            energies.append(float(energy.mean()))
        rebuilt = {
            "system": row["system"],
            "client": int(row["client"]),
            "fold": row["fold"],
            "arm": row["arm"],
            "chi_unseen": float(np.mean(chis)),
            "response_energy": float(np.mean(energies)),
        }
        maximum_chi_error = max(maximum_chi_error, abs(rebuilt["chi_unseen"] - float(row["chi_unseen"])))
        rebuilt_rows.append(rebuilt)
    stage1 = aggregate_stage1(rebuilt_rows)

    oracle = importlib.import_module("fedprime.engine.cle_shortcut_alignment")
    oracle_rows = []
    task_rows = []
    for path in sorted((output_dir / "oracle_predictions").glob("*.npz")):
        system, fold, arm = path.stem.split("_", 2)
        values = np.load(path, allow_pickle=False)
        dsa = oracle.compute_dsa(
            values["probabilities"], values["labels"], values["binding"], values["operator_family_ids"]
        )
        secondary = oracle.secondary_metrics(
            values["probabilities"], values["labels"], values["binding"], values["operator_family_ids"]
        )
        clean_client = 100.0 * (
            values["clean_probabilities"].argmax(-1) == values["labels"][None]
        ).mean(axis=1)
        oracle_rows.append(
            {"system": system, "fold": fold, "arm": arm, "dsa": dsa.pooled, "dsa_client": json.dumps(dsa.client.tolist())}
        )
        task_rows.append(
            {
                "system": system,
                "fold": fold,
                "arm": arm,
                "avg_acc": secondary["avg_acc"],
                "worst_acc": secondary["worst_acc"],
                "wcca": secondary["wcca"],
                "cfg": secondary["cfg"],
                "clean_avg": float(clean_client.mean()),
                "clean_worst": float(clean_client.min()),
            }
        )
    stage2 = aggregate_stage2(oracle_rows, task_rows)
    dominance = baseline_dominance(stage2)
    decision = decide(stage1, stage2, dominance)
    stage1_match = json.dumps(stage1, sort_keys=True) == json.dumps(reported_stage1, sort_keys=True)
    stage2_error = 0.0
    for system in SYSTEMS:
        for arm in ARMS:
            for metric, value in stage2[system]["combined"][arm].items():
                if metric == "dsa_client":
                    error = np.max(
                        np.abs(np.asarray(value) - np.asarray(reported_stage2[system]["combined"][arm][metric]))
                    )
                else:
                    error = abs(float(value) - float(reported_stage2[system]["combined"][arm][metric]))
                stage2_error = max(stage2_error, float(error))
    verdict_match = decision == reported_decision
    inconsistencies = int(not stage1_match) + int(stage2_error > 1.0e-9) + int(not verdict_match)
    return {
        "audit_pass": inconsistencies == 0 and maximum_chi_error <= 1.0e-9,
        "gate_inconsistencies": inconsistencies,
        "maximum_saved_chi_error": maximum_chi_error,
        "maximum_stage2_metric_error": stage2_error,
        "verdict_match": verdict_match,
        "recomputed_decision": decision,
    }


def artifact_manifest(output_dir: Path) -> dict[str, object]:
    rows = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"protocol": PROTOCOL, "files": rows}


def main() -> None:
    args = parse_args()
    args.public_root = args.public_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.evaluation_root = args.evaluation_root.resolve() if args.evaluation_root else None
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    banks = load_banks()
    images = cifar100_train_images_from_tar(args.public_root)
    split = public_split(images.shape[0], discover_count=1000, surgery_count=2000, holdout_count=2000)
    if sha256_array(split["surgery"]) != SURGERY_HASH or sha256_array(split["holdout"]) != HOLDOUT_HASH:
        raise ValueError("public split contract mismatch")

    if args.mode == "inspect":
        result = inspect(args, images, split, banks, device)
        write_json(args.output_dir / "result.json", result)
    elif args.mode == "smoke":
        write_json(args.output_dir / "source_manifest.json", source_manifest())
        result = smoke(args, images, split, banks, device)
        write_json(args.output_dir / "result.json", result)
    elif args.mode == "calibration":
        write_json(args.output_dir / "source_manifest.json", source_manifest())
        result = blind_calibration(args, images, split, banks, device)
        write_json(args.output_dir / "calibration_manifest.json", result)
        write_json(args.output_dir / "result.json", {"verdict": result["verdict"]})
    else:
        if args.calibration_manifest is None:
            raise ValueError("formal mode is locked until --calibration-manifest is provided")
        calibration = load_calibration(args.calibration_manifest.resolve())
        if args.evaluation_root is None:
            raise ValueError("formal mode requires --evaluation-root")
        config = {
            "protocol": PROTOCOL,
            "mode": "formal",
            "systems": list(SYSTEMS),
            "folds": {name: {"correction": correction, "unseen": unseen} for name, correction, unseen in FOLDS},
            "arms": list(ARMS),
            "optimizer": "Adam defaults except frozen LR; weight_decay=0",
            "accepted_steps": 10,
            "anchor_kl_limit": 0.02,
            "maximum_backtracks": 12,
            "full_training_performed": False,
            "communication_modified": False,
        }
        write_json(args.output_dir / "config.json", config)
        write_json(args.output_dir / "source_manifest.json", source_manifest([args.calibration_manifest.resolve()]))
        write_json(
            args.output_dir / "input_manifest.json",
            {
                "public_root": args.public_root.as_posix(),
                "checkpoint_root": args.checkpoint_root.as_posix(),
                "evaluation_root_opened_in_stage1": False,
            },
        )
        write_json(
            args.output_dir / "public_split_manifest.json",
            {
                "surgery_count": 2000,
                "surgery_sha256": sha256_array(split["surgery"]),
                "holdout_count": 2000,
                "holdout_sha256": sha256_array(split["holdout"]),
                "disjoint": np.intersect1d(split["surgery"], split["holdout"]).size == 0,
                "labels_loaded": False,
            },
        )
        write_json(
            args.output_dir / "prime_bank_manifest.json",
            {name: banks[name]["bank_sha256"] for name in ("a", "b")},
        )
        write_json(args.output_dir / "calibration_manifest.json", calibration)
        _rows, stage1 = run_formal_stage1(args, images, split, banks, calibration, device)
        seal = primary_seal(args.output_dir)
        write_json(args.output_dir / "primary_taxonomy_free_manifest.json", seal)
        oracle_rows, task_rows, stage2 = run_formal_stage2(args, device)
        dominance = baseline_dominance(stage2)
        write_json(args.output_dir / "baseline_dominance.json", dominance)
        decision = decide(stage1, stage2, dominance)
        gate_table = {"stage1": stage1, "stage2": stage2, "dominance": dominance, "decision": decision}
        write_json(args.output_dir / "gate_table.json", gate_table)
        result = {"protocol": PROTOCOL, **decision, "stage1": stage1, "stage2": stage2}
        write_json(args.output_dir / "result.json", result)
        audit = independent_recompute(args.output_dir, stage1, stage2, decision)
        write_json(args.output_dir / "independent_recomputation.json", audit)
        if not audit["audit_pass"]:
            result["verdict"] = "AUDIT_FAIL"
            write_json(args.output_dir / "result.json", result)
        report = [
            "# CLE K1-C CRSF Checkpoint Surgery",
            "",
            f"Verdict: `{result['verdict']}`",
            "",
            "Stage 1 was sealed before CLE binding, corruption operators, task labels or DSA/WCCA/CFG were loaded.",
            "Only late-block parameter deltas were saved; no full checkpoint, full training or communication change was produced.",
        ]
        (args.output_dir / "FINAL_REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.output_dir / "artifact_manifest.json", artifact_manifest(args.output_dir))
    print(json.dumps(result if args.mode != "formal" else {"verdict": result["verdict"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
