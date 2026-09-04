from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.augmentations.frozen_prime import apply_frozen_prime_recipe  # noqa: E402
from fedprime.data.loaders import cifar100_train_images_from_tar, dataset_stats, normalize_batch  # noqa: E402
from fedprime.engine.cle_crsf_surgery import (  # noqa: E402
    EPS,
    LateBlockAdapter,
    apply_state_delta,
    changed_state_keys,
    features_from_prefix_numpy,
    run_exact_surgery,
)
from scripts.run_cle_k1_c_crsf_surgery import (  # noqa: E402
    artifact_manifest as legacy_artifact_manifest,
    load_banks,
    load_delta,
    make_adapter,
    state_clone,
    trace_dict,
    verify_adapter_exactness,
    write_csv,
)
from scripts.run_cle_k1_sdmn_headonly import (  # noqa: E402
    MODEL_NAMES,
    public_split,
    resolve_device,
    sha256_array,
    sha256_file,
    write_json,
)


PROTOCOL = "cle_k1_c_minimal_causal_gate_v1"
CONFIG_PATH = ROOT / "configs/cle_k1_c_minimal_seed0.json"
SYSTEMS = ("h9", "l9")
CLIENTS = ((0, "ResNet10"), (3, "Mobilenetv2"))
ARMS = ("frozen", "crsf", "rawspec")
OBJECTIVE_BY_ARM = {"crsf": "crsf", "rawspec": "rawspec"}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def heartbeat(event: str, **payload: object) -> None:
    fields = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[heartbeat] event={event}{(' ' + fields) if fields else ''}", flush=True)


def scoped_progress(**scope: object):
    def emit(event: str, payload: dict[str, object]) -> None:
        heartbeat(event, **scope, **payload)

    return emit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K1-C-Minimal CRSF causal intervention gate.")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="smoke")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--confirm-formal",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Formal cost approval; accepts a bare flag or an explicit true value for OpenI forms.",
    )
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()


def load_config() -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL or config.get("status") != "FROZEN_BEFORE_FORMAL":
        raise ValueError("K1-C-Minimal config is not frozen")
    formal = config["formal"]
    if formal["systems"] != list(SYSTEMS) or formal["arms"] != list(ARMS):
        raise ValueError("K1-C-Minimal systems/arms config mismatch")
    if [(row["client"], row["architecture"]) for row in formal["clients"]] != list(CLIENTS):
        raise ValueError("K1-C-Minimal client config mismatch")
    return config


def freeze_selection(total: int, config: dict[str, object]) -> dict[str, object]:
    frozen = public_split(total, discover_count=1000, surgery_count=2000, holdout_count=2000)
    selection = config["selection"]
    if sha256_array(frozen["surgery"]) != selection["full_surgery_pool_sha256"]:
        raise ValueError("full surgery pool hash mismatch")
    if sha256_array(frozen["holdout"]) != selection["holdout_pool_sha256"]:
        raise ValueError("holdout pool hash mismatch")
    carrier_rng = np.random.default_rng(int(selection["carrier_seed"]))
    positions = carrier_rng.choice(2000, size=int(selection["carrier_count"]), replace=False).astype(np.int64)
    correction = frozen["surgery"][positions]
    probe_rng = np.random.default_rng(int(selection["probe_seed"]))
    probe_ids = probe_rng.choice(64, size=len(selection["probe_ids"]), replace=False).astype(np.int64)
    checks = {
        "carrier_position_sha256": sha256_array(positions),
        "carrier_global_index_sha256": sha256_array(correction),
        "probe_ids_sha256": sha256_array(probe_ids),
    }
    for key, value in checks.items():
        if value != selection[key]:
            raise ValueError(f"selection hash mismatch: {key}")
    if probe_ids.tolist() != selection["probe_ids"]:
        raise ValueError("selected probe ids mismatch")
    if np.intersect1d(correction, frozen["holdout"]).size:
        raise ValueError("correction and holdout overlap")
    return {
        "correction": correction,
        "correction_positions_in_full_pool": positions,
        "holdout": frozen["holdout"],
        "probe_ids": probe_ids,
        "manifest": {
            "protocol": PROTOCOL,
            "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "config_canonical_sha256": canonical_json_sha256(config),
            "carrier_seed": int(selection["carrier_seed"]),
            "carrier_positions_in_full_pool": positions.tolist(),
            "carrier_global_indices": correction.tolist(),
            "carrier_position_sha256": checks["carrier_position_sha256"],
            "carrier_global_index_sha256": checks["carrier_global_index_sha256"],
            "probe_seed": int(selection["probe_seed"]),
            "probe_ids": probe_ids.tolist(),
            "probe_ids_sha256": checks["probe_ids_sha256"],
            "holdout_indices": frozen["holdout"].tolist(),
            "holdout_sha256": sha256_array(frozen["holdout"]),
            "disjoint": True,
            "selected_before_formal": True,
        },
    }


def source_manifest() -> dict[str, object]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/openi_cle_k1_c_minimal_entry.py",
        ROOT / "fedprime/engine/cle_crsf_surgery.py",
        ROOT / "scripts/run_cle_k1_c_crsf_surgery.py",
        CONFIG_PATH,
        ROOT / "docs/experiments/current/CLE_K1_C_MINIMAL_CAUSAL_GATE_OPENI_ZH.md",
    )
    return {
        "protocol": PROTOCOL,
        "git_commit": git_commit(),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
            if path.is_file()
        ],
    }


def prefix_array(
    adapter: LateBlockAdapter,
    images: np.ndarray,
    recipe: dict[str, object] | None,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    stats = dataset_stats("cifar10")
    rows = []
    with torch.no_grad():
        for start in range(0, int(images.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(images.shape[0]))
            batch = torch.from_numpy(np.ascontiguousarray(images[start:stop]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            if recipe is not None:
                batch = apply_frozen_prime_recipe(batch, recipe)
            rows.append(adapter.prefix(normalize_batch(batch, stats)).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def build_in_memory_prefixes(
    adapter: LateBlockAdapter,
    images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
    scope: str,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    base_started = time.perf_counter()
    base = prefix_array(adapter, images, None, device=device, batch_size=batch_size)
    base_seconds = time.perf_counter() - base_started
    probes = []
    probe_seconds = []
    for probe_id, recipe in enumerate(recipes):
        probe_started = time.perf_counter()
        probes.append(prefix_array(adapter, images, recipe, device=device, batch_size=batch_size))
        probe_seconds.append(time.perf_counter() - probe_started)
        heartbeat(
            "minimal_prefix_probe_complete",
            scope=scope,
            probe=probe_id + 1,
            probes=len(recipes),
            seconds=probe_seconds[-1],
        )
    arrays = [base, *probes]
    return base, probes, {
        "implementation": "bounded in-memory float32 prefix arrays; no transformed-input memmap",
        "carriers": int(images.shape[0]),
        "probes": len(recipes),
        "prefix_shape": list(base.shape[1:]),
        "base_sha256": sha256_array(base),
        "probe_sha256": [sha256_array(value) for value in probes],
        "base_seconds": base_seconds,
        "probe_seconds": probe_seconds,
        "total_seconds": time.perf_counter() - started,
        "resident_array_bytes": int(sum(value.nbytes for value in arrays)),
        "disk_cache_bytes": 0,
    }


def stream_spectrum_multiarm(
    original: LateBlockAdapter,
    adapters: dict[str, LateBlockAdapter],
    images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
    scope: str,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    arm_halves: dict[str, list[tuple[float, float]]] = {arm: [] for arm in adapters}
    saved: dict[str, np.ndarray] = {}
    peak_array_bytes = 0
    midpoint = int(images.shape[0]) // 2
    for half_name, half_images in (("u1", images[:midpoint]), ("u2", images[midpoint:])):
        heartbeat("minimal_unseen_half_start", scope=scope, half=half_name, carriers=int(half_images.shape[0]))
        base_prefix = prefix_array(original, half_images, None, device=device, batch_size=batch_size)
        heartbeat("minimal_unseen_base_prefix_complete", scope=scope, half=half_name)
        base_features = {
            arm: features_from_prefix_numpy(adapter, base_prefix, device=device, batch_size=batch_size)
            for arm, adapter in adapters.items()
        }
        heartbeat("minimal_unseen_base_features_complete", scope=scope, half=half_name)
        sums = {
            arm: np.zeros((len(recipes), value.shape[1]), dtype=np.float64)
            for arm, value in base_features.items()
        }
        energies = {arm: np.zeros(len(recipes), dtype=np.float64) for arm in adapters}
        peak_array_bytes = max(
            peak_array_bytes,
            int(base_prefix.nbytes + sum(value.nbytes for value in base_features.values())),
        )
        for probe_id, recipe in enumerate(recipes):
            probe_prefix = prefix_array(original, half_images, recipe, device=device, batch_size=batch_size)
            current_bytes = int(base_prefix.nbytes + probe_prefix.nbytes + sum(v.nbytes for v in base_features.values()))
            for arm, adapter in adapters.items():
                feature = features_from_prefix_numpy(
                    adapter, probe_prefix, device=device, batch_size=batch_size
                )
                delta = feature.astype(np.float64) - base_features[arm].astype(np.float64)
                sums[arm][probe_id] = delta.sum(axis=0)
                energies[arm][probe_id] = np.square(delta).sum()
                current_bytes += int(feature.nbytes + delta.nbytes)
            peak_array_bytes = max(peak_array_bytes, current_bytes)
            heartbeat(
                "minimal_unseen_probe_complete",
                scope=scope,
                half=half_name,
                probe=probe_id + 1,
                probes=len(recipes),
            )
        count = max(int(half_images.shape[0]), 1)
        for arm in adapters:
            mean = sums[arm] / count
            energy = energies[arm] / count
            response = mean / (np.sqrt(np.maximum(energy, 0.0))[:, None] + EPS)
            # Avoid a Windows/OpenBLAS loader crash observed for tiny 2xd smoke matrices;
            # this is algebraically identical and remains small even for the formal 64xd bank.
            gram = np.sum(response[:, None, :] * response[None, :, :], axis=2)
            chi = float(np.square(gram).sum() / (np.trace(gram) ** 2 + EPS))
            arm_halves[arm].append((chi, float(energy.mean())))
            saved[f"{arm}_mean_{half_name}"] = mean
            saved[f"{arm}_energy_{half_name}"] = energy
            saved[f"{arm}_gram_{half_name}"] = gram
        heartbeat("minimal_unseen_half_complete", scope=scope, half=half_name)
    metrics = {
        arm: {
            "chi_u1": values[0][0],
            "chi_u2": values[1][0],
            "chi_unseen": float(np.mean([row[0] for row in values])),
            "response_energy": float(np.mean([row[1] for row in values])),
        }
        for arm, values in arm_halves.items()
    }
    return metrics, saved, {
        "implementation": "probe-streamed full unseen evaluation; no persistent prefix cache",
        "carriers": int(images.shape[0]),
        "probes": len(recipes),
        "arms": list(adapters),
        "elapsed_seconds": time.perf_counter() - started,
        "estimated_peak_array_bytes": peak_array_bytes,
        "disk_cache_bytes": 0,
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


def aggregate_stage1(rows: list[dict[str, object]], config: dict[str, object]) -> dict[str, object]:
    gate = config["gates"]
    output: dict[str, object] = {}
    for system in SYSTEMS:
        selected = [row for row in rows if row["system"] == system]
        means = {
            arm: float(np.mean([float(row["chi_unseen"]) for row in selected if row["arm"] == arm]))
            for arm in ARMS
        }
        baseline = means["frozen"]
        reductions = {arm: float(1.0 - value / max(baseline, EPS)) for arm, value in means.items()}
        positives = sum(
            float(next(row["chi_unseen"] for row in selected if row["client"] == client and row["arm"] == "crsf"))
            < float(next(row["chi_unseen"] for row in selected if row["client"] == client and row["arm"] == "frozen"))
            for client, _architecture in CLIENTS
        )
        frozen_energy = float(np.mean([float(row["response_energy"]) for row in selected if row["arm"] == "frozen"]))
        crsf_energy = float(np.mean([float(row["response_energy"]) for row in selected if row["arm"] == "crsf"]))
        retention = crsf_energy / max(frozen_energy, EPS)
        gates = {
            "crsf_unseen_chi_reduction_ge_15pct": reductions["crsf"] >= float(gate["per_system_unseen_chi_reduction_min"]),
            "both_selected_clients_positive": positives >= int(gate["positive_selected_clients_required"]),
            "response_energy_retention_ge_50pct": retention >= float(gate["per_system_response_energy_retention_min"]),
            "crsf_minus_rawspec_chi_advantage_ge_10pp": reductions["crsf"] - reductions["rawspec"] >= float(gate["per_system_crsf_minus_rawspec_chi_advantage_min"]),
        }
        output[system] = {
            "mean_chi": means,
            "reductions": reductions,
            "positive_clients": int(positives),
            "response_energy_retention": float(retention),
            "gates": gates,
            "pass": all(gates.values()),
        }
    return output


def aggregate_stage2(
    oracle_rows: list[dict[str, object]], task_rows: list[dict[str, object]], config: dict[str, object]
) -> dict[str, object]:
    gate = config["gates"]
    output: dict[str, object] = {}
    for system in SYSTEMS:
        combined: dict[str, object] = {}
        for arm in ARMS:
            oracle_row = next(row for row in oracle_rows if row["system"] == system and row["arm"] == arm)
            task_row = next(row for row in task_rows if row["system"] == system and row["arm"] == arm)
            combined[arm] = {
                "dsa": float(oracle_row["dsa"]),
                "dsa_client": json.loads(str(oracle_row["dsa_client"])),
                **{name: float(task_row[name]) for name in config["gates"]["task_metrics_report_only"]},
            }
        frozen = combined["frozen"]
        effects = {}
        for arm in ARMS[1:]:
            value = combined[arm]
            effects[arm] = {
                "dsa_reduction": frozen["dsa"] - value["dsa"],
                "dsa_relative_reduction": (frozen["dsa"] - value["dsa"]) / max(abs(frozen["dsa"]), EPS),
            }
        client_delta = np.asarray(frozen["dsa_client"]) - np.asarray(combined["crsf"]["dsa_client"])
        crsf = effects["crsf"]
        rawspec = effects["rawspec"]
        gates = {
            "dsa_absolute_005_or_relative_25pct": (
                crsf["dsa_reduction"] >= float(gate["per_system_dsa_absolute_reduction_min"])
                or crsf["dsa_relative_reduction"] >= float(gate["per_system_dsa_relative_reduction_min"])
            ),
            "both_selected_clients_dsa_positive": int((client_delta > 0).sum()) >= int(gate["positive_selected_clients_required"]),
            "crsf_minus_rawspec_dsa_advantage_ge_002": crsf["dsa_reduction"] - rawspec["dsa_reduction"] >= float(gate["per_system_crsf_minus_rawspec_dsa_advantage_min"]),
        }
        output[system] = {
            "combined": combined,
            "effects": effects,
            "crsf_dsa_client_delta": client_delta.tolist(),
            "gates": gates,
            "pass": all(gates.values()),
        }
    return output


def decide(stage1: dict[str, object], stage2: dict[str, object]) -> dict[str, object]:
    passed = all(stage1[system]["pass"] and stage2[system]["pass"] for system in SYSTEMS)
    return {
        "verdict": "GO_TO_K1_C_MINIMAL_REPLICATION" if passed else "NO_GO_CRSF_INTERVENTION",
        "stage1_taxonomy_free_pass": all(stage1[system]["pass"] for system in SYSTEMS),
        "stage2_causal_dsa_pass": all(stage2[system]["pass"] for system in SYSTEMS),
        "full_training_authorized": False,
    }


def primary_seal(output_dir: Path) -> dict[str, object]:
    forbidden = ("oracle_", "task_metrics", "final_result", "FINAL_REPORT")
    rows = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative == "primary_taxonomy_free_manifest.json" or relative.startswith(forbidden):
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"protocol": PROTOCOL, "sealed_before_oracle": True, "files": rows}


def run_intervention_context(
    args: argparse.Namespace,
    images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    system: str,
    client_id: int,
    architecture: str,
    accepted_steps: int,
) -> tuple[LateBlockAdapter, dict[str, LateBlockAdapter], dict[str, object]]:
    checkpoint = args.checkpoint_root / system / f"client_{client_id}.pt"
    original_adapter, original, audit = make_adapter(architecture, checkpoint, resolve_device(args.device))
    device = next(original_adapter.model.parameters()).device
    base, probes, prefix_manifest = build_in_memory_prefixes(
        original_adapter,
        images,
        recipes,
        device=device,
        batch_size=args.batch_size,
        scope=f"{system}_{architecture}",
    )
    adapters = {"frozen": original_adapter}
    traces = {}
    deltas = {}
    for arm in ARMS[1:]:
        adapter = original_adapter.clone(device)
        trace = run_exact_surgery(
            adapter,
            OBJECTIVE_BY_ARM[arm],
            base,
            probes if arm == "crsf" else [],
            device=device,
            batch_size=args.batch_size,
            learning_rate=1.0e-4,
            accepted_steps=accepted_steps,
            anchor_limit=0.02,
            maximum_backtracks=12,
            progress=scoped_progress(stage="minimal_intervention", system=system, architecture=architecture, arm=arm),
        )
        if trace.contract_failure or trace.accepted_steps != accepted_steps:
            raise RuntimeError(f"OPTIMIZATION_CONTRACT_FAIL: {system}/{architecture}/{arm}")
        changed = changed_state_keys(original, state_clone(adapter.model))
        if not set(changed).issubset(set(audit["trainable_parameter_names"])):
            raise RuntimeError(f"forbidden parameter changed: {system}/{architecture}/{arm}")
        adapters[arm] = adapter
        traces[arm] = {"trace": trace_dict(trace), "changed_keys": list(changed)}
        if args.mode == "formal":
            delta_path = args.output_dir / "surgery_block_deltas" / f"{system}_ab_client{client_id}_{arm}.pt"
            deltas[arm] = save_delta(delta_path, adapter, original, audit)
    return original_adapter, adapters, {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "audit": audit,
        "prefix": prefix_manifest,
        "traces": traces,
        "deltas": deltas,
    }


def smoke(
    args: argparse.Namespace,
    images: np.ndarray,
    selection: dict[str, object],
    banks: dict[str, dict[str, object]],
) -> dict[str, object]:
    device = resolve_device(args.device)
    correction = images[selection["correction"][:8]]
    probe_ids = selection["probe_ids"][:2]
    recipes = [banks["a"]["recipes"][int(value)] for value in probe_ids]
    original, adapters, run = run_intervention_context(
        args, correction, recipes, system="h9", client_id=0, architecture="ResNet10", accepted_steps=1
    )
    holdout = images[selection["holdout"][:8]]
    unseen_recipes = list(banks["b"]["recipes"])[:2]
    metrics, saved, stream = stream_spectrum_multiarm(
        original,
        adapters,
        holdout,
        unseen_recipes,
        device=device,
        batch_size=args.batch_size,
        scope="smoke_h9_resnet10",
    )
    checks = {
        "selection_hashes_verified": True,
        "exact_representation": verify_adapter_exactness(original, correction[:2], device=device) <= 1e-6,
        "three_arms": set(metrics) == set(ARMS),
        "finite_unseen_metrics": all(np.isfinite(list(value.values())).all() for value in metrics.values()),
        "crsf_one_accepted_step": run["traces"]["crsf"]["trace"]["accepted_steps"] == 1,
        "rawspec_one_accepted_step": run["traces"]["rawspec"]["trace"]["accepted_steps"] == 1,
        "bounded_memory_no_disk_cache": run["prefix"]["disk_cache_bytes"] == 0 and stream["disk_cache_bytes"] == 0,
        "oracle_assets_not_loaded": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"minimal smoke failed: {[key for key, value in checks.items() if not value]}")
    np.savez_compressed(args.output_dir / "smoke_unseen_moments.npz", **saved)
    return {
        "protocol": PROTOCOL,
        "verdict": "SMOKE_ONLY_NO_SCIENTIFIC_DECISION",
        "checks": checks,
        "execution": run,
        "unseen_shapes_only": {key: list(value.shape) for key, value in saved.items()},
        "scientific_metrics_reported": False,
    }


def benchmark(
    args: argparse.Namespace,
    images: np.ndarray,
    selection: dict[str, object],
    banks: dict[str, dict[str, object]],
) -> dict[str, object]:
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    correction = images[selection["correction"]]
    recipes = [banks["a"]["recipes"][int(value)] for value in selection["probe_ids"]]
    original, adapters, run = run_intervention_context(
        args, correction, recipes, system="h9", client_id=0, architecture="ResNet10", accepted_steps=1
    )
    correction_seconds = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    sample_carriers = 128
    sample_probes = 8
    _metrics, _saved, stream = stream_spectrum_multiarm(
        original,
        adapters,
        images[selection["holdout"][:sample_carriers]],
        list(banks["b"]["recipes"])[:sample_probes],
        device=device,
        batch_size=args.batch_size,
        scope="benchmark_h9_resnet10",
    )
    evaluation_seconds = time.perf_counter() - evaluation_started

    def five_step_estimate(arm: str) -> float:
        trace = run["traces"][arm]["trace"]
        prep = float(trace["timings_seconds"]["preparation_total"])
        initial_gradient = float(trace["timings_seconds"]["preparation_exact_gradient_vjp"])
        attempts = [row for row in trace["attempts"] if row["accepted"]]
        post = float(sum(row["post_update_exact_objective_seconds"] + row["post_update_anchor_kl_seconds"] for row in attempts))
        # The preparation contains the first exact gradient. Each later accepted step requires one
        # new exact gradient plus a post-update objective/KL evaluation.
        return prep + 5.0 * post + 4.0 * initial_gradient

    correction_context_estimate = float(run["prefix"]["total_seconds"]) + sum(
        five_step_estimate(arm) for arm in ARMS[1:]
    )
    formal_contexts = len(SYSTEMS) * len(CLIENTS)
    evaluation_scale = (2000.0 * 64.0) / (sample_carriers * sample_probes)
    unseen_estimate = evaluation_seconds * evaluation_scale * formal_contexts
    correction_estimate = correction_context_estimate * formal_contexts
    oracle_estimate = unseen_estimate * (17.0 / 65.0)
    total_estimate = correction_estimate + unseen_estimate + oracle_estimate
    gpu_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "protocol": PROTOCOL,
        "verdict": "BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION",
        "scope": "H9 / ResNet10 / Bank A; one accepted step per intervention arm",
        "measured": {
            "correction_carriers": 512,
            "correction_probes": 16,
            "correction_one_step_two_arms_seconds": correction_seconds,
            "correction_prefix": run["prefix"],
            "traces": run["traces"],
            "sample_unseen_carriers": sample_carriers,
            "sample_unseen_probes": sample_probes,
            "sample_unseen_three_arms_seconds": evaluation_seconds,
            "stream": stream,
            "gpu_peak_allocated_bytes": gpu_peak,
        },
        "projected": {
            "assumption": "linear scale from one ResNet10 context; MobileNetV2 is not directly timed",
            "formal_contexts": formal_contexts,
            "correction_seconds": correction_estimate,
            "full_unseen_seconds": unseen_estimate,
            "oracle_seconds_proxy": oracle_estimate,
            "total_wall_seconds": total_estimate,
            "single_gpu_hours": total_estimate / 3600.0,
        },
        "requires_user_cost_approval": True,
        "scientific_decision_allowed": False,
        "forbidden_assets_loaded": [],
    }


def formal(
    args: argparse.Namespace,
    images: np.ndarray,
    selection: dict[str, object],
    banks: dict[str, dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    if not args.confirm_formal:
        raise ValueError("Minimal formal is locked; pass --confirm-formal only after benchmark cost approval")
    if args.evaluation_root is None:
        raise ValueError("Minimal formal requires evaluation-root")
    device = resolve_device(args.device)
    correction_images = images[selection["correction"]]
    correction_recipes = [banks["a"]["recipes"][int(value)] for value in selection["probe_ids"]]
    holdout_images = images[selection["holdout"]]
    unseen_recipes = list(banks["b"]["recipes"])
    metric_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    saved_moments: dict[str, np.ndarray] = {}
    runtime_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for client_id, architecture in CLIENTS:
            context = f"{system}_ab_client{client_id}"
            heartbeat("minimal_formal_context_start", system=system, architecture=architecture)
            original, adapters, run = run_intervention_context(
                args,
                correction_images,
                correction_recipes,
                system=system,
                client_id=client_id,
                architecture=architecture,
                accepted_steps=5,
            )
            checkpoint_rows.append(
                {
                    "system": system,
                    "client": client_id,
                    "architecture": architecture,
                    "path": run["checkpoint"],
                    "sha256": run["checkpoint_sha256"],
                }
            )
            for arm, value in run["traces"].items():
                trace_rows.append(
                    {
                        "system": system,
                        "client": client_id,
                        "architecture": architecture,
                        "fold": "ab",
                        "arm": arm,
                        "trace": value["trace"],
                        "changed_keys": value["changed_keys"],
                        "delta": run["deltas"][arm],
                    }
                )
            metrics, saved, stream = stream_spectrum_multiarm(
                original,
                adapters,
                holdout_images,
                unseen_recipes,
                device=device,
                batch_size=args.batch_size,
                scope=context,
            )
            runtime_rows.append({"context": context, "correction_prefix": run["prefix"], "unseen": stream})
            frozen_chi = metrics["frozen"]["chi_unseen"]
            frozen_energy = metrics["frozen"]["response_energy"]
            for arm in ARMS:
                row = {
                    "system": system,
                    "client": client_id,
                    "architecture": architecture,
                    "fold": "ab",
                    "correction_bank": "a",
                    "unseen_bank": "b",
                    "arm": arm,
                    **metrics[arm],
                    "delta_chi": float(1.0 - metrics[arm]["chi_unseen"] / max(frozen_chi, EPS)),
                    "energy_retention": float(metrics[arm]["response_energy"] / max(frozen_energy, EPS)),
                }
                metric_rows.append(row)
            for key, value in saved.items():
                saved_moments[f"{context}_{key}"] = value
            for adapter in adapters.values():
                adapter.model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    stage1 = aggregate_stage1(metric_rows, config)
    write_csv(args.output_dir / "taxonomy_free_metrics.csv", metric_rows)
    np.savez_compressed(args.output_dir / "unseen_response_moments_and_grams.npz", **saved_moments)
    write_json(args.output_dir / "optimization_traces.json", trace_rows)
    write_json(args.output_dir / "checkpoint_manifest.json", checkpoint_rows)
    write_json(args.output_dir / "runtime_manifest.json", runtime_rows)
    write_json(args.output_dir / "stage1_gate_inputs.json", stage1)
    write_json(args.output_dir / "primary_taxonomy_free_manifest.json", primary_seal(args.output_dir))

    oracle = importlib.import_module("fedprime.engine.cle_shortcut_alignment")
    clean_images = np.load(args.evaluation_root / "test_images.npy", allow_pickle=False)
    labels = np.load(args.evaluation_root / "test_labels.npy", allow_pickle=False).astype(np.int64)
    grid, severities = oracle.deterministic_corruption_grid(clean_images)
    flat_grid = grid.reshape(-1, *grid.shape[2:])
    full_binding = oracle.historical_family_binding(num_clients=4, num_classes=10)
    client_ids = [client for client, _architecture in CLIENTS]
    binding = full_binding[client_ids]
    oracle_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    prediction_root = args.output_dir / "oracle_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    legacy = importlib.import_module("scripts.run_cle_k1_c_crsf_surgery")
    for system in SYSTEMS:
        arm_grid = {arm: [] for arm in ARMS}
        arm_clean = {arm: [] for arm in ARMS}
        for client_id, architecture in CLIENTS:
            checkpoint = args.checkpoint_root / system / f"client_{client_id}.pt"
            for arm in ARMS:
                adapter, _original, _audit = make_adapter(architecture, checkpoint, device)
                if arm != "frozen":
                    delta_path = args.output_dir / "surgery_block_deltas" / f"{system}_ab_client{client_id}_{arm}.pt"
                    apply_state_delta(adapter.model, load_delta(delta_path))
                arm_grid[arm].append(
                    legacy.inference_probabilities(adapter.model, flat_grid, device=device, batch_size=args.batch_size).reshape(
                        clean_images.shape[0], grid.shape[1], 10
                    )
                )
                arm_clean[arm].append(
                    legacy.inference_probabilities(adapter.model, clean_images, device=device, batch_size=args.batch_size)
                )
                adapter.model.to("cpu")
        for arm in ARMS:
            probabilities = np.stack(arm_grid[arm])
            clean_probabilities = np.stack(arm_clean[arm])
            dsa = oracle.compute_dsa(probabilities, labels, binding, oracle.OPERATOR_FAMILY_IDS)
            task = oracle.secondary_metrics(probabilities, labels, binding, oracle.OPERATOR_FAMILY_IDS)
            client_clean = 100.0 * (clean_probabilities.argmax(-1) == labels[None]).mean(axis=1)
            prediction_path = prediction_root / f"{system}_ab_{arm}.npz"
            np.savez_compressed(
                prediction_path,
                probabilities=probabilities,
                clean_probabilities=clean_probabilities,
                labels=labels,
                selected_client_ids=np.asarray(client_ids, dtype=np.int64),
                binding=binding,
                operator_family_ids=oracle.OPERATOR_FAMILY_IDS,
                severities=severities,
            )
            oracle_rows.append(
                {
                    "system": system,
                    "arm": arm,
                    "dsa": float(dsa.pooled),
                    "dsa_client": json.dumps(dsa.client.tolist()),
                    "prediction_sha256": sha256_file(prediction_path),
                }
            )
            task_rows.append(
                {
                    "system": system,
                    "arm": arm,
                    "avg_acc": float(task["avg_acc"]),
                    "worst_acc": float(task["worst_acc"]),
                    "wcca": float(task["wcca"]),
                    "cfg": float(task["cfg"]),
                    "clean_avg": float(client_clean.mean()),
                    "clean_worst": float(client_clean.min()),
                }
            )
    stage2 = aggregate_stage2(oracle_rows, task_rows, config)
    decision = decide(stage1, stage2)
    write_csv(args.output_dir / "oracle_dsa_metrics.csv", oracle_rows)
    write_csv(args.output_dir / "task_metrics.csv", task_rows)
    write_json(args.output_dir / "stage2_gate_inputs.json", stage2)
    write_json(args.output_dir / "gate_table.json", {"stage1": stage1, "stage2": stage2, "decision": decision})
    return {"protocol": PROTOCOL, **decision, "stage1": stage1, "stage2": stage2}


def main() -> None:
    args = parse_args()
    args.public_root = args.public_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.evaluation_root = args.evaluation_root.resolve() if args.evaluation_root else None
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    banks = load_banks()
    selection_config = config["selection"]
    if banks["a"]["bank_sha256"] != selection_config["bank_a_sha256"]:
        raise ValueError("Bank A hash mismatch")
    if banks["b"]["bank_sha256"] != selection_config["bank_b_sha256"]:
        raise ValueError("Bank B hash mismatch")
    images = cifar100_train_images_from_tar(args.public_root)
    selection = freeze_selection(images.shape[0], config)
    write_json(args.output_dir / "frozen_config.json", config)
    write_json(args.output_dir / "selection_manifest.json", selection["manifest"])
    write_json(args.output_dir / "source_manifest.json", source_manifest())
    write_json(
        args.output_dir / "input_manifest.json",
        {
            "public_root": args.public_root.as_posix(),
            "checkpoint_root": args.checkpoint_root.as_posix(),
            "evaluation_opened_before_primary_seal": False,
            "labels_loaded_during_smoke_or_benchmark": False,
        },
    )
    if args.mode == "smoke":
        result = smoke(args, images, selection, banks)
    elif args.mode == "benchmark":
        result = benchmark(args, images, selection, banks)
    else:
        result = formal(args, images, selection, banks, config)
        report = [
            "# CLE K1-C-Minimal Causal Intervention Gate",
            "",
            f"Verdict: `{result['verdict']}`",
            "",
            "A-to-B minimal correction used 512 frozen carriers and 16 frozen Bank-A probes.",
            "Full 2,000-carrier x 64-probe Bank-B evaluation was sealed before CLE oracle evaluation.",
            "A GO only authorizes B-to-A plus the remaining-architecture replication; it does not authorize full training.",
        ]
        (args.output_dir / "FINAL_REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.output_dir / "result.json", result)
    # Reuse the deterministic recursive file manifest implementation; overwrite protocol locally.
    artifacts = legacy_artifact_manifest(args.output_dir)
    artifacts["protocol"] = PROTOCOL
    write_json(args.output_dir / "artifact_manifest.json", artifacts)
    print(json.dumps({"verdict": result["verdict"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
