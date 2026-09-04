from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
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
from fedprime.data.strict_fit_audit import build_strict_fit_audit_loaders  # noqa: E402
from fedprime.engine.cle_shortcut_alignment import (  # noqa: E402
    OPERATOR_FAMILY_IDS,
    compute_dsa,
    deterministic_corruption_grid,
    historical_family_binding,
)
from fedprime.methods.cvrs import (  # noqa: E402
    ProbeSchedule,
    calibrated_regularizer_weight,
    compute_rahfl_augmix_dcl_loss,
    cvrs_loss,
    cvrs_statistics,
    gradient_l2_norm,
    pairwise_public_jsd_loss,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402


PROTOCOL = "cle_cvrs_m0_cheap_method_gate_v1"
CONFIG_PATH = ROOT / "configs/cle_cvrs_m0_seed0.json"
BANK_ROOT = ROOT / "fedprime/augmentations/assets/cle_generic_probe_k0b"
K0B_PUBLIC_SEED = 20260901
K0B_PUBLIC_HASH = "731B8CFFDCBD241474D33B261E323F9EC11C2EA59BC7705261140A3B8572F6CA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CVRS M0 cheap method kill test")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), default="smoke")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older OpenI PyTorch
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint payload: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def load_banks(config: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    banks = []
    for name in ("a", "b"):
        bank = load_frozen_prime_bank(
            state_path=BANK_ROOT / f"bank_{name}_states.npz",
            manifest_path=BANK_ROOT / f"bank_{name}_manifest.json",
        )
        banks.append(bank)
    expected = config["public_training"]["probe_bank_sha256"]
    if banks[0]["bank_sha256"] != expected:
        raise ValueError("Bank-A hash mismatch")
    return banks[0], banks[1]


def frozen_public_split(public_root: Path, config: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    compact_images = public_root / "k0b_public_images.npy"
    compact_indices = public_root / "k0b_public_indices.npy"
    if compact_images.is_file() and compact_indices.is_file():
        selected_images = np.load(compact_images, allow_pickle=False)
        indices = np.load(compact_indices, allow_pickle=False).astype(np.int64, copy=False)
        if selected_images.shape != (1000, 32, 32, 3) or selected_images.dtype != np.uint8:
            raise ValueError("compact K0-B public images have an unexpected shape or dtype")
    else:
        images = cifar100_train_images_from_tar(public_root)
        indices = np.random.default_rng(K0B_PUBLIC_SEED).choice(
            images.shape[0], size=1000, replace=False
        ).astype(np.int64)
        selected_images = images[indices]
    if sha256_array(indices) != K0B_PUBLIC_HASH:
        raise ValueError("K0-B public carrier identity mismatch")
    train_indices = indices[:500]
    heldout_indices = indices[500:756]
    if sha256_array(train_indices) != config["public_training"]["train_carrier_indices_sha256"]:
        raise ValueError("CVRS training carrier hash mismatch")
    if sha256_array(heldout_indices) != config["heldout_routing"]["carrier_indices_sha256"]:
        raise ValueError("CVRS held-out carrier hash mismatch")
    if np.intersect1d(train_indices, heldout_indices).size:
        raise ValueError("public training and held-out carriers overlap")
    return selected_images[:500], selected_images[500:756], indices


def raw_batch(images: np.ndarray, *, device: torch.device) -> torch.Tensor:
    batch = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    return batch.to(device=device, dtype=torch.float32).div_(255.0)


def public_logits(
    model: torch.nn.Module,
    carriers: torch.Tensor,
    recipes: list[dict[str, object]],
) -> tuple[torch.Tensor, torch.Tensor]:
    views = [carriers]
    for recipe in recipes:
        views.append(apply_frozen_prime_recipe(carriers, recipe))
    normalized = normalize_batch(torch.cat(views, dim=0), dataset_stats("cifar10"))
    logits = forward_logits(model, normalized)
    chunks = torch.split(logits, carriers.shape[0])
    return chunks[0], torch.stack(chunks[1:], dim=0)


def public_loss(
    model: torch.nn.Module,
    carriers: torch.Tensor,
    recipes: list[dict[str, object]],
    arm: str,
) -> torch.Tensor:
    was_training = model.training
    model.eval()  # Keep CIFAR-100 carriers out of private BatchNorm running statistics.
    base_logits, probe_logits = public_logits(model, carriers, recipes)
    loss = cvrs_loss(base_logits, probe_logits) if arm == "cvrs" else pairwise_public_jsd_loss(base_logits, probe_logits)
    model.train(was_training)
    return loss


def snapshot_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def restore_buffers(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(snapshot[name])


def trace_private_batch(digest: "hashlib._Hash", images: object, labels: torch.Tensor) -> None:
    for tensor in list(images) + [labels]:
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())


class PublicBatchCycle:
    def __init__(self, images: np.ndarray, batch_size: int):
        self.images = images
        self.batch_size = int(batch_size)
        self.offset = 0

    def next(self) -> np.ndarray:
        positions = (np.arange(self.batch_size, dtype=np.int64) + self.offset) % self.images.shape[0]
        self.offset = int((self.offset + self.batch_size) % self.images.shape[0])
        return self.images[positions]


def build_model(architecture: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_models([architecture], num_classes=10)[0]
    model.load_state_dict(load_state(checkpoint), strict=True)
    return model.to(device)


def build_private_loader(config: dict[str, object], *, batch_size: int, client_id: int):
    private = config["private_training"]
    loaders, _test, _splits, _counts = build_strict_fit_audit_loaders(
        root=resolve_path(config["paths"]["private_root"]),
        num_clients=4,
        train_batch_size=int(batch_size),
        test_batch_size=512,
        num_workers=int(private["num_workers"]),
        split_path=resolve_path(config["paths"]["split_path"]),
        audit_ratio=0.15,
        min_audit_per_class=5,
        min_fit_per_class=2,
        seed=0,
        num_classes=10,
        augmix_module="jsd",
        loader_seed=20260830,
    )
    return loaders[int(client_id)]


def calibrate_lambda(
    model: torch.nn.Module,
    private_batch: tuple[object, torch.Tensor],
    public_images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    arm: str,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    buffers = snapshot_buffers(model)
    model.train()
    task = compute_rahfl_augmix_dcl_loss(
        model, private_batch[0], private_batch[1], device=device, lambda_jsd=12.0
    )
    task_norm = float(gradient_l2_norm(task, model.parameters()).cpu())
    restore_buffers(model, buffers)

    regularizer = public_loss(model, raw_batch(public_images, device=device), recipes, arm)
    regularizer_norm = float(gradient_l2_norm(regularizer, model.parameters()).cpu())
    restore_buffers(model, buffers)
    model.zero_grad(set_to_none=True)
    value = calibrated_regularizer_weight(task_norm, regularizer_norm, ratio=0.1)
    return value, {"task_gradient_norm": task_norm, "regularizer_gradient_norm": regularizer_norm}


def train_arm(
    config: dict[str, object],
    *,
    architecture: str,
    client_id: int,
    arm: str,
    checkpoint: Path,
    train_public: np.ndarray,
    bank_a: dict[str, object],
    device: torch.device,
    mode: str,
) -> tuple[torch.nn.Module, dict[str, object]]:
    private_cfg = config["private_training"]
    public_cfg = config["public_training"]
    if mode == "smoke":
        batch_size, public_batch_size, probes_per_update = 8, 8, 2
        epochs, max_steps, interval = 1, 2, 1
    elif mode == "benchmark":
        batch_size, public_batch_size, probes_per_update = 64, 64, 4
        epochs, max_steps, interval = 1, 8, 4
    else:
        batch_size = int(private_cfg["batch_size"])
        public_batch_size = int(public_cfg["batch_size"])
        probes_per_update = int(public_cfg["probes_per_update"])
        epochs, max_steps = int(private_cfg["epochs"]), None
        interval = int(public_cfg["regularization_interval_private_steps"])

    run_seed = int(config["seed"]) * 100003 + int(client_id) * 1009 + 20260905
    seed_everything(run_seed)
    loader = build_private_loader(config, batch_size=batch_size, client_id=client_id)
    model = build_model(architecture, checkpoint, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(private_cfg["learning_rate"]),
        weight_decay=float(private_cfg["weight_decay"]),
    )
    schedule = ProbeSchedule(
        bank_size=64,
        probes_per_update=probes_per_update,
        seed=int(public_cfg["probe_schedule_seed"]),
    )
    public_batches = PublicBatchCycle(train_public, public_batch_size)
    iterator = iter(loader)
    # AugMix uses process-global RNGs. Fix each fetch independently so extra
    # public forwards/gradient calibration cannot perturb another arm's views.
    seed_everything(run_seed + 1_000_000)
    first_batch = next(iterator)
    first_probe_ids = schedule.next_ids() if arm != "baseline" else np.empty(0, dtype=np.int64)
    first_public = public_batches.next() if arm != "baseline" else np.empty((0, 32, 32, 3), dtype=np.uint8)
    lambda_reg = 0.0
    calibration: dict[str, float] = {}
    if arm != "baseline":
        lambda_reg, calibration = calibrate_lambda(
            model,
            first_batch,
            first_public,
            [bank_a["recipes"][int(value)] for value in first_probe_ids],
            arm=arm,
            device=device,
        )

    private_trace = hashlib.sha256()
    task_losses: list[float] = []
    reg_losses: list[float] = []
    private_seconds = 0.0
    public_seconds = 0.0
    private_steps = 0
    public_steps = 0
    pending_public = (first_public, first_probe_ids)
    stop = False
    for epoch in range(epochs):
        if epoch == 0:
            epoch_iterator = iterator
            pending_private = first_batch
            batch_index = 0
        else:
            epoch_iterator = iter(loader)
            pending_private = None
            batch_index = 0
        while True:
            if max_steps is not None and private_steps >= max_steps:
                stop = True
                break
            if pending_private is not None:
                images, labels = pending_private
                pending_private = None
            else:
                seed_everything(run_seed + 1_000_000 + epoch * 10_000 + batch_index)
                try:
                    images, labels = next(epoch_iterator)
                except StopIteration:
                    break
            batch_index += 1
            trace_private_batch(private_trace, images, labels)
            started = time.perf_counter()
            model.train()
            seed_everything(run_seed + 2_000_000 + private_steps)
            optimizer.zero_grad(set_to_none=True)
            task = compute_rahfl_augmix_dcl_loss(
                model, images, labels, device=device, lambda_jsd=float(private_cfg["lambda_jsd"])
            )
            if not torch.isfinite(task):
                raise FloatingPointError("non-finite private loss")
            task.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(private_cfg["max_grad_norm"]))
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            private_seconds += time.perf_counter() - started
            private_steps += 1
            task_losses.append(float(task.detach().cpu()))

            if arm != "baseline" and private_steps % interval == 0:
                if public_steps == 0:
                    carrier_values, probe_ids = pending_public
                else:
                    carrier_values, probe_ids = public_batches.next(), schedule.next_ids()
                recipes = [bank_a["recipes"][int(value)] for value in probe_ids]
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                regularizer = public_loss(
                    model, raw_batch(carrier_values, device=device), recipes, arm
                )
                weighted = float(lambda_reg) * regularizer
                if not torch.isfinite(weighted):
                    raise FloatingPointError("non-finite public regularizer")
                weighted.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(private_cfg["max_grad_norm"]))
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                public_seconds += time.perf_counter() - started
                public_steps += 1
                reg_losses.append(float(regularizer.detach().cpu()))
            if private_steps == 1 or private_steps % 25 == 0:
                print(
                    f"[heartbeat] mode={mode} arch={architecture} arm={arm} "
                    f"private_step={private_steps} public_steps={public_steps}",
                    flush=True,
                )
        if stop:
            break

    result = {
        "architecture": architecture,
        "client": int(client_id),
        "arm": arm,
        "mode": mode,
        "private_steps": private_steps,
        "public_steps": public_steps,
        "lambda": lambda_reg,
        "lambda_calibration": calibration,
        "private_batch_trace_sha256": private_trace.hexdigest().upper(),
        "mean_task_loss": float(np.mean(task_losses)),
        "mean_regularizer_loss": float(np.mean(reg_losses)) if reg_losses else None,
        "private_seconds": private_seconds,
        "public_seconds": public_seconds,
    }
    return model, result


def infer_probabilities(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    result = np.empty((images.shape[0], 10), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            stop = min(start + batch_size, images.shape[0])
            batch = normalize_batch(raw_batch(images[start:stop], device=device), dataset_stats("cifar10"))
            result[start:stop] = torch.softmax(forward_logits(model, batch), dim=-1).cpu().numpy()
    return result


def evaluate_routing(
    model: torch.nn.Module,
    images: np.ndarray,
    recipes: list[dict[str, object]],
    *,
    device: torch.device,
) -> float:
    sums = np.zeros((len(recipes), 10), dtype=np.float64)
    energies = np.zeros(len(recipes), dtype=np.float64)
    model.eval()
    for probe_start in range(0, len(recipes), 4):
        probe_stop = min(probe_start + 4, len(recipes))
        count = 0
        with torch.inference_mode():
            for start in range(0, images.shape[0], 64):
                batch = raw_batch(images[start : start + 64], device=device)
                base, probes = public_logits(model, batch, recipes[probe_start:probe_stop])
                mu, energy, _routing = cvrs_statistics(base, probes)
                batch_count = int(batch.shape[0])
                sums[probe_start:probe_stop] += mu.double().cpu().numpy() * batch_count
                energies[probe_start:probe_stop] += energy.double().cpu().numpy() * batch_count
                count += batch_count
        sums[probe_start:probe_stop] /= count
        energies[probe_start:probe_stop] /= count
    means = sums
    energy = energies
    return float(np.mean(np.square(means).sum(axis=1) / (energy + 1.0e-12)))


def evaluate_oracle(
    model: torch.nn.Module,
    grid: np.ndarray,
    labels: np.ndarray,
    *,
    client_id: int,
    device: torch.device,
) -> dict[str, float]:
    flat = grid.reshape(-1, 32, 32, 3)
    probabilities = infer_probabilities(model, flat, device=device).reshape(1, grid.shape[0], grid.shape[1], 10)
    binding = historical_family_binding()[[int(client_id)]]
    dsa = compute_dsa(probabilities, labels, binding).pooled
    predictions = probabilities[0].argmax(axis=-1)
    operator_accuracy = 100.0 * (predictions == labels[:, None]).mean(axis=0)
    return {
        "dsa": float(dsa),
        "avg_acc": float(operator_accuracy.mean()),
        "worst_acc": float(operator_accuracy.min()),
    }


def formal_decision(rows: list[dict[str, object]], config: dict[str, object]) -> dict[str, object]:
    gate = config["gates"]
    by_arch = {}
    for context in config["contexts"]:
        architecture = context["architecture"]
        metrics = {row["arm"]: row["oracle"] for row in rows if row["architecture"] == architecture}
        baseline, jsd, cvrs = metrics["baseline"], metrics["jsd"], metrics["cvrs"]
        relative = (baseline["dsa"] - cvrs["dsa"]) / max(abs(baseline["dsa"]), 1.0e-12)
        checks = {
            "cvrs_dsa_relative_reduction": relative >= gate["per_architecture_cvrs_dsa_relative_reduction_min"],
            "cvrs_dsa_advantage_over_jsd": jsd["dsa"] - cvrs["dsa"] >= gate["per_architecture_jsd_minus_cvrs_dsa_min"],
            "avg_accuracy_retained": baseline["avg_acc"] - cvrs["avg_acc"] <= gate["per_architecture_avg_accuracy_drop_max_pp"],
            "worst_accuracy_retained": baseline["worst_acc"] - cvrs["worst_acc"] <= gate["per_architecture_worst_accuracy_drop_max_pp"],
        }
        by_arch[architecture] = {"relative_dsa_reduction": relative, "checks": checks, "pass": all(checks.values())}
    passed = all(value["pass"] for value in by_arch.values())
    return {
        "by_architecture": by_arch,
        "verdict": gate["pass_verdict"] if passed else gate["failure_verdict"],
        "full_hfl_training_authorized": bool(passed),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected CVRS M0 config protocol")
    if args.mode == "formal" and not args.confirm_formal:
        raise ValueError("Formal is intentionally locked; pass --confirm-formal only after cost approval")
    device = resolve_device(args.device)
    output_dir = args.output_dir or (ROOT / "outputs" / f"cle_cvrs_m0_seed0_{args.mode}")
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_a, bank_b = load_banks(config)
    train_public, heldout_public, all_indices = frozen_public_split(
        resolve_path(config["paths"]["public_root"]), config
    )
    write_json(
        output_dir / "frozen_input_manifest.json",
        {
            "protocol": PROTOCOL,
            "mode": args.mode,
            "config_sha256": sha256_file(resolve_path(args.config)),
            "k0b_public_indices_sha256": sha256_array(all_indices),
            "train_public_sha256": sha256_array(train_public),
            "heldout_public_sha256": sha256_array(heldout_public),
            "bank_a_sha256": bank_a["bank_sha256"],
            "bank_b_sha256": bank_b["bank_sha256"],
            "device": str(device),
        },
    )

    rows = []
    checkpoint_root = resolve_path(config["paths"]["checkpoint_root"])
    for context in config["contexts"]:
        client_id, architecture = int(context["client"]), str(context["architecture"])
        checkpoint = checkpoint_root / f"client_{client_id}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for arm in config["arms"]:
            model, row = train_arm(
                config,
                architecture=architecture,
                client_id=client_id,
                arm=str(arm),
                checkpoint=checkpoint,
                train_public=train_public,
                bank_a=bank_a,
                device=device,
                mode=args.mode,
            )
            row["checkpoint_sha256"] = sha256_file(checkpoint)
            if args.mode == "formal":
                state_path = output_dir / "checkpoints" / architecture / f"{arm}.pt"
                state_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), state_path)
                row["output_checkpoint"] = state_path.relative_to(output_dir).as_posix()
                row["output_checkpoint_sha256"] = sha256_file(state_path)
            rows.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for context in config["contexts"]:
        architecture = context["architecture"]
        traces = {row["private_batch_trace_sha256"] for row in rows if row["architecture"] == architecture}
        if len(traces) != 1:
            raise ValueError(f"private stochastic path mismatch across arms for {architecture}")

    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": args.mode,
        "hardware": {
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "rows": rows,
        "private_stochastic_paths_matched": True,
        "smoke_metrics_are_scientific_evidence": False,
    }
    if args.mode == "benchmark":
        private_per_step = float(np.mean([row["private_seconds"] / row["private_steps"] for row in rows]))
        reg_rows = [row for row in rows if row["public_steps"]]
        public_per_step = float(np.mean([row["public_seconds"] / row["public_steps"] for row in reg_rows]))
        fit_counts = [8498, 8497]
        total_private = sum((count // 64) * 3 * 3 for count in fit_counts)
        total_public = sum(((count // 64) * 3 // 4) * 2 for count in fit_counts)
        estimate = total_private * private_per_step + total_public * public_per_step
        result["benchmark"] = {
            "mean_private_step_seconds": private_per_step,
            "mean_public_step_seconds": public_per_step,
            "projected_formal_seconds_excluding_evaluation": estimate,
            "projected_formal_single_gpu_hours_excluding_evaluation": estimate / 3600.0,
            "cost_approval_required_before_formal": True,
        }
        result["verdict"] = "BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION"
    elif args.mode == "smoke":
        result["verdict"] = "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
    else:
        heldout_ids = config["heldout_routing"]["probe_ids"]
        evaluation_root = resolve_path(config["paths"]["evaluation_root"])
        oracle_clean = np.load(evaluation_root / "test_images.npy", allow_pickle=False)
        oracle_labels = np.load(
            evaluation_root / "test_labels.npy", allow_pickle=False
        ).astype(np.int64)
        oracle_grid, _oracle_severity = deterministic_corruption_grid(oracle_clean)
        for row in rows:
            model = build_model(
                str(row["architecture"]),
                output_dir / str(row["output_checkpoint"]),
                device,
            )
            row["routing_strength"] = evaluate_routing(
                model,
                heldout_public,
                [bank_b["recipes"][int(value)] for value in heldout_ids],
                device=device,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        taxonomy_free = {
            "protocol": PROTOCOL,
            "training_complete": True,
            "oracle_assets_opened": False,
            "rows": rows,
        }
        taxonomy_free_path = output_dir / "taxonomy_free_result.json"
        write_json(taxonomy_free_path, taxonomy_free)
        write_json(
            output_dir / "pre_oracle_seal.json",
            {
                "protocol": PROTOCOL,
                "taxonomy_free_result_sha256": sha256_file(taxonomy_free_path),
                "checkpoint_sha256": {
                    f"{row['architecture']}/{row['arm']}": row["output_checkpoint_sha256"]
                    for row in rows
                },
                "sealed_before_oracle": True,
            },
        )
        for row in rows:
            model = build_model(
                str(row["architecture"]),
                output_dir / str(row["output_checkpoint"]),
                device,
            )
            row["oracle"] = evaluate_oracle(
                model,
                oracle_grid,
                oracle_labels,
                client_id=int(row["client"]),
                device=device,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result["decision"] = formal_decision(rows, config)
        result["verdict"] = result["decision"]["verdict"]
    write_json(output_dir / "result.json", result)
    print(f"[done] {result['verdict']} -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
