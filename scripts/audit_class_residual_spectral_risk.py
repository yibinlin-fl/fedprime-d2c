from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import (  # noqa: E402
    CorruptionSkewClientDataset,
    TwoViewTransform,
    _private_test_transform,
    _rahfl_augmix_view_transforms,
)
from fedprime.methods.class_residual_spectral_risk import (  # noqa: E402
    class_balanced_spearman,
    class_conditional_residual_spectral_risk,
    class_operator_cell_correlation,
    cross_split_spectral_metrics,
    decide_crsr_audit0,
    deterministic_random_directions,
    fit_residual_spectral_statistics,
    score_residual_directions,
)
from fedprime.methods.local_fedease import train_local_fedease_epoch  # noqa: E402
from fedprime.models.factory import build_models, forward_logits  # noqa: E402
from fedprime.utils.env import add_vendor_paths, resolve_device, seed_everything  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "RAHFL-master/Dataset/cifar_10_cle_v2/alpha05_gamma09_seed0_split0"
DEFAULT_SPLIT = (
    ROOT
    / "outputs/strict_pew_asymhfl_val_probe_20260804/outputs/partitions"
    / "strict_cle_v2_alpha05_gamma09_seed0_split0.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs/class_residual_spectral_risk_audit0"


class _DummyEnvironmentDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        views, label = self.dataset[index]
        return views, label, 0


class _ProbeDataset(data.Dataset):
    def __init__(self, root: Path, client_id: int, indices: np.ndarray) -> None:
        self.dataset = CorruptionSkewClientDataset(
            root=root,
            client_id=int(client_id),
            train=True,
            transform=_private_test_transform("cifar10"),
            return_corruption=True,
        )
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        source_index = int(self.indices[item])
        image, label, operator_id = self.dataset[source_index]
        return image, label, operator_id, source_index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fit_indices(split_path: Path, client_id: int) -> np.ndarray:
    with np.load(split_path, allow_pickle=False) as archive:
        expected = {"audit_ratio": 0.15, "min_audit_per_class": 5, "min_fit_per_class": 2, "seed": 0}
        for key, value in expected.items():
            if key not in archive or archive[key][0].item() != value:
                raise ValueError(f"unexpected strict split metadata for {key}")
        return np.asarray(archive[f"client_{int(client_id)}_fit"], dtype=np.int64)


def _split_fit_probe(
    fit_indices: np.ndarray,
    labels: np.ndarray,
    *,
    ratio: float,
    min_class_count: int,
    min_probe: int,
    max_probe: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    probe_parts: list[np.ndarray] = []
    for class_id in np.unique(labels[fit_indices]):
        class_indices = fit_indices[labels[fit_indices] == int(class_id)]
        if class_indices.size < int(min_class_count):
            continue
        count = min(
            int(max_probe),
            max(int(min_probe), int(round(float(ratio) * class_indices.size))),
            int(class_indices.size) - 2,
        )
        probe_parts.append(rng.permutation(class_indices)[:count])
    if not probe_parts:
        raise ValueError("no class has enough fit support for the internal probe")
    probe = np.sort(np.concatenate(probe_parts)).astype(np.int64, copy=False)
    train = np.setdiff1d(fit_indices, probe, assume_unique=False).astype(np.int64, copy=False)
    if np.intersect1d(train, probe).size:
        raise AssertionError("fit-internal train and probe overlap")
    return train, probe


def _split_probe_halves(probe: np.ndarray, labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for class_id in np.unique(labels[probe]):
        selected = rng.permutation(probe[labels[probe] == int(class_id)])
        midpoint = selected.size // 2
        if midpoint < 2:
            continue
        left.append(selected[:midpoint])
        right.append(selected[midpoint:])
    return np.sort(np.concatenate(left)), np.sort(np.concatenate(right))


def _build_augmix_dataset(root: Path, client_id: int, indices: np.ndarray) -> data.Dataset:
    add_vendor_paths()
    from Dataset.dataaug import AugMixDataset

    base, weak, preprocess = _rahfl_augmix_view_transforms("cifar10")
    private = CorruptionSkewClientDataset(
        root=root,
        client_id=int(client_id),
        train=True,
        transform=TwoViewTransform(base, weak),
        return_corruption=False,
    )
    subset = data.Subset(private, np.asarray(indices, dtype=np.int64).tolist())
    return _DummyEnvironmentDataset(AugMixDataset(subset, preprocess, jsd_or_nojsd="jsd"))


def _build_train_loader(
    root: Path,
    client_id: int,
    indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> data.DataLoader:
    return data.DataLoader(
        _build_augmix_dataset(root, client_id, indices),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(int(seed)),
    )


def _balanced_update_indices(
    train_indices: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    parts = []
    for class_id in np.unique(labels[train_indices]):
        selected = train_indices[labels[train_indices] == int(class_id)]
        if selected.size < 2:
            continue
        parts.append(rng.permutation(selected)[: min(int(per_class), selected.size)])
    return np.concatenate(parts).astype(np.int64, copy=False)


def _train_base_model(
    model_name: str,
    loader: data.DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    max_batches: int | None,
    client_id: int,
    seed: int,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    seed_everything(int(seed))
    model = build_models([model_name], num_classes=10)[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    history = []
    for epoch in range(int(epochs)):
        diagnostics: dict[str, float] = {}
        loss = train_local_fedease_epoch(
            model,
            loader,
            optimizer,
            device,
            {"ber": {"enabled": False}, "preserve_dcl": True},
            lambda_jsd=12.0,
            max_batches=max_batches,
            max_grad_norm=5.0,
            context=f"CRSR Audit 0 base client={client_id} epoch={epoch}",
            diagnostics=diagnostics,
        )
        row = {key: float(value) for key, value in diagnostics.items()}
        row.update({"epoch": int(epoch), "loss": float(loss)})
        history.append(row)
        print(
            f"[train] client={client_id} model={model_name} epoch={epoch + 1}/{epochs} "
            f"loss={loss:.4f} clean_ce={row.get('clean_ce', float('nan')):.4f}",
            flush=True,
        )
    return model, history


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    dataset: data.Dataset,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    loader = data.DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    probabilities = []
    labels = []
    operators = []
    indices = []
    for images, batch_labels, operator_ids, source_indices in loader:
        logits = forward_logits(model, images.to(device, non_blocking=True))
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())
        operators.append(operator_ids.numpy())
        indices.append(source_indices.numpy())
    merged = {
        "probabilities": np.concatenate(probabilities).astype(np.float64),
        "labels": np.concatenate(labels).astype(np.int64),
        "operator_ids": np.concatenate(operators).astype(np.int64),
        "sample_indices": np.concatenate(indices).astype(np.int64),
    }
    merged["accuracy"] = float(
        (merged["probabilities"].argmax(axis=1) == merged["labels"]).mean()
    )
    return merged


def _directed_signal_metrics(source: dict, target: dict, *, min_class_support: int, min_cell_support: int, seed: int):
    source_stats = fit_residual_spectral_statistics(
        source["probabilities"], source["labels"], min_class_support=int(min_class_support)
    )
    target_stats = fit_residual_spectral_statistics(
        target["probabilities"], target["labels"], min_class_support=int(min_class_support)
    )
    random_directions = deterministic_random_directions(
        target["probabilities"].shape[1], source_stats.keys(), seed=int(seed)
    )
    spectral = score_residual_directions(target["probabilities"], target["labels"], source_stats)
    random = score_residual_directions(
        target["probabilities"], target["labels"], source_stats, directions=random_directions
    )
    labels = target["labels"]
    residuals = target["probabilities"] - np.eye(target["probabilities"].shape[1])[labels]
    ce = -np.log(np.clip(target["probabilities"][np.arange(labels.size), labels], 1.0e-12, 1.0))
    brier = np.sum(residuals * residuals, axis=1)
    errors = (target["probabilities"].argmax(axis=1) != labels).astype(np.float64)
    transfer = cross_split_spectral_metrics(source_stats, target_stats, random_directions)
    cell_metrics = {}
    for name, signal in {"spectral": spectral, "ce": ce, "brier": brier, "random": random}.items():
        correlation, cells = class_operator_cell_correlation(
            signal,
            errors,
            labels,
            target["operator_ids"],
            min_cell_support=int(min_cell_support),
        )
        cell_metrics[f"{name}_cell_correlation"] = correlation
        cell_metrics[f"{name}_valid_cells"] = int(cells)
    return {
        "base_accuracy": float(target["accuracy"]),
        "valid_classes": transfer["valid_classes"],
        "source_top_share": transfer["median_source_top_share"],
        "direction_cosine": transfer["median_direction_cosine"],
        "transfer_share": transfer["median_transfer_share"],
        "random_share": transfer["median_random_share"],
        "transfer_advantage": transfer["median_transfer_advantage"],
        "spectral_ce_abs_correlation": abs(
            class_balanced_spearman(spectral, ce, labels, min_class_support=int(min_class_support))
        ),
        "spectral_brier_abs_correlation": abs(
            class_balanced_spearman(spectral, brier, labels, min_class_support=int(min_class_support))
        ),
        "valid_cells": int(cell_metrics["spectral_valid_cells"]),
        **cell_metrics,
    }


def _model_backbone(model):
    return model.module.backbone if hasattr(model, "module") else model.backbone


def _isolated_update(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    *,
    learning_rate: float,
    spectral_weight: float,
) -> dict[str, float]:
    add_vendor_paths()
    from loss import DCLLoss

    images, labels, _environment_ids = batch
    images = [view.to(device) for view in images]
    labels = labels.to(device).long()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    size = labels.shape[0]
    logits = forward_logits(model, torch.cat(images[:3], dim=0))
    clean, aug1, aug2 = torch.split(logits, size)
    classification, crsr_stats = class_conditional_residual_spectral_risk(
        clean,
        labels,
        spectral_weight=float(spectral_weight),
        min_class_count=2,
    )
    probabilities = [F.softmax(view, dim=1) for view in (clean, aug1, aug2)]
    mixture = torch.clamp(sum(probabilities) / 3.0, 1.0e-7, 1.0).log()
    jsd = sum(F.kl_div(mixture, value, reduction="batchmean") for value in probabilities) / 3.0
    features = _model_backbone(model)(torch.cat([images[0], images[1], images[3]], dim=0))
    features = F.normalize(features.reshape(features.shape[0], -1), dim=1)
    original, strong, weak = torch.split(features, size)
    dcl = DCLLoss(temperature=0.2, device=device, beta=1.0, ddm_temperature=0.2)(
        original_feature=original.unsqueeze(1),
        weak_feature=weak.unsqueeze(1),
        strong_feature=strong.unsqueeze(1),
        labels=labels,
    )
    loss = classification + 12.0 * jsd + dcl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "classification": float(classification.detach().cpu()),
        "balanced_ce": float(crsr_stats["balanced_ce"].detach().cpu()),
        "spectral_radius": float(crsr_stats["spectral_radius"].detach().cpu()),
        "valid_spectral_classes": float(crsr_stats["valid_spectral_classes"].detach().cpu()),
    }


def _probe_risk_metrics(predictions: dict, *, min_cell_support: int) -> dict[str, float]:
    probabilities = predictions["probabilities"]
    labels = predictions["labels"]
    operators = predictions["operator_ids"]
    ce = -np.log(np.clip(probabilities[np.arange(labels.size), labels], 1.0e-12, 1.0))
    cell_values: dict[int, list[float]] = {}
    all_cells = []
    for class_id, operator_id in np.unique(np.stack([labels, operators], axis=1), axis=0):
        selected = (labels == int(class_id)) & (operators == int(operator_id))
        if int(selected.sum()) < int(min_cell_support):
            continue
        value = float(ce[selected].mean())
        all_cells.append(value)
        cell_values.setdefault(int(class_id), []).append(value)
    gaps = [max(values) - min(values) for values in cell_values.values() if len(values) >= 2]
    return {
        "mean_ce": float(ce.mean()),
        "worst_cell_ce": float(max(all_cells)) if all_cells else float("nan"),
        "cell_gap_ce": float(np.mean(gaps)) if gaps else 0.0,
        "valid_cells": float(len(all_cells)),
        "accuracy": float(predictions["accuracy"]),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated Audit 0 for class residual spectral risk.")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split_path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--models", nargs="+", default=["ResNet12", "Mobilenetv2"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--spectral_weight", type=float, default=2.0)
    parser.add_argument("--probe_ratio", type=float, default=0.20)
    parser.add_argument("--probe_min_class_count", type=int, default=64)
    parser.add_argument("--probe_min", type=int, default=32)
    parser.add_argument("--probe_max", type=int, default=128)
    parser.add_argument("--min_class_support", type=int, default=16)
    parser.add_argument("--min_cell_support", type=int, default=4)
    parser.add_argument("--update_per_class", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.clients) != len(args.models):
        raise ValueError("--clients and --models must have equal length")
    if args.smoke:
        args.clients = [int(args.clients[0])]
        args.models = [str(args.models[0])]
        args.epochs = 1
        args.max_train_batches = 2
        args.probe_min_class_count = 16
        args.probe_min = 8
        args.probe_max = 12
        args.min_class_support = 4
        args.min_cell_support = 2
        args.update_per_class = 4
        args.output_dir = Path(args.output_dir) / "smoke"

    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    print(f"[setup] device={device} data={data_root} split={split_path}", flush=True)

    client_results = {}
    directed_metrics = []
    one_step_metrics = []
    raw_arrays = {}
    for client_id, model_name in zip(args.clients, args.models):
        labels = np.load(data_root / f"client_{client_id}" / "train_labels.npy").astype(np.int64)
        fit_indices = _load_fit_indices(split_path, client_id)
        train_indices, probe_indices = _split_fit_probe(
            fit_indices,
            labels,
            ratio=float(args.probe_ratio),
            min_class_count=int(args.probe_min_class_count),
            min_probe=int(args.probe_min),
            max_probe=int(args.probe_max),
            seed=int(args.train_seed) * 1009 + int(client_id),
        )
        left_indices, right_indices = _split_probe_halves(
            probe_indices, labels, seed=7001 + int(client_id)
        )
        print(
            f"[split] client={client_id} train={train_indices.size} probe={probe_indices.size} "
            f"halves={left_indices.size}/{right_indices.size}",
            flush=True,
        )
        train_loader = _build_train_loader(
            data_root,
            client_id,
            train_indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.train_seed) * 1009 + int(client_id),
        )
        model, training = _train_base_model(
            model_name,
            train_loader,
            device,
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            max_batches=args.max_train_batches,
            client_id=int(client_id),
            seed=int(args.train_seed),
        )
        full_probe = _ProbeDataset(data_root, client_id, probe_indices)
        left = _collect_predictions(
            model, _ProbeDataset(data_root, client_id, left_indices), device, batch_size=args.probe_batch_size
        )
        right = _collect_predictions(
            model, _ProbeDataset(data_root, client_id, right_indices), device, batch_size=args.probe_batch_size
        )
        for name, values in (("left", left), ("right", right)):
            prefix = f"client_{client_id}_{name}"
            for key in ("probabilities", "labels", "operator_ids", "sample_indices"):
                raw_arrays[f"{prefix}_{key}"] = values[key]
        rows = []
        for source_name, source, target_name, target, seed_offset in (
            ("left", left, "right", right, 0),
            ("right", right, "left", left, 1),
        ):
            row = _directed_signal_metrics(
                source,
                target,
                min_class_support=int(args.min_class_support),
                min_cell_support=int(args.min_cell_support),
                seed=9101 + int(client_id) * 11 + seed_offset,
            )
            row.update(
                {"client_id": int(client_id), "source": source_name, "target": target_name}
            )
            rows.append(row)
            directed_metrics.append(row)
            print(
                f"[signal] client={client_id} {source_name}->{target_name} "
                f"cos={row['direction_cosine']:.4f} transfer_adv={row['transfer_advantage']:.4f} "
                f"cell_rho={row['spectral_cell_correlation']:.4f}",
                flush=True,
            )

        update_indices = _balanced_update_indices(
            train_indices,
            labels,
            per_class=int(args.update_per_class),
            seed=8101 + int(client_id),
        )
        seed_everything(8201 + int(client_id))
        update_loader = data.DataLoader(
            _build_augmix_dataset(data_root, client_id, update_indices),
            batch_size=int(update_indices.size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        update_batch = next(iter(update_loader))
        control = copy.deepcopy(model)
        candidate = copy.deepcopy(model)
        control_update = _isolated_update(
            control,
            update_batch,
            device,
            learning_rate=float(args.learning_rate),
            spectral_weight=0.0,
        )
        candidate_update = _isolated_update(
            candidate,
            update_batch,
            device,
            learning_rate=float(args.learning_rate),
            spectral_weight=float(args.spectral_weight),
        )
        control_metrics = _probe_risk_metrics(
            _collect_predictions(control, full_probe, device, batch_size=args.probe_batch_size),
            min_cell_support=int(args.min_cell_support),
        )
        candidate_metrics = _probe_risk_metrics(
            _collect_predictions(candidate, full_probe, device, batch_size=args.probe_batch_size),
            min_cell_support=int(args.min_cell_support),
        )
        delta = {
            "client_id": int(client_id),
            "mean_ce_delta": candidate_metrics["mean_ce"] - control_metrics["mean_ce"],
            "worst_cell_ce_delta": candidate_metrics["worst_cell_ce"] - control_metrics["worst_cell_ce"],
            "cell_gap_ce_delta": candidate_metrics["cell_gap_ce"] - control_metrics["cell_gap_ce"],
        }
        one_step_metrics.append(delta)
        print(
            f"[one-step] client={client_id} mean_ce={delta['mean_ce_delta']:+.6f} "
            f"worst_cell={delta['worst_cell_ce_delta']:+.6f} gap={delta['cell_gap_ce_delta']:+.6f}",
            flush=True,
        )
        client_results[int(client_id)] = {
            "model": str(model_name),
            "train_size": int(train_indices.size),
            "probe_size": int(probe_indices.size),
            "training": training,
            "signals": rows,
            "control_update": control_update,
            "candidate_update": candidate_update,
            "control_probe": control_metrics,
            "candidate_probe": candidate_metrics,
            "one_step_delta": delta,
        }
        del model, control, candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    decision = decide_crsr_audit0(directed_metrics, one_step_metrics)
    if args.smoke:
        decision["verdict"] = "SMOKE_ONLY"
    payload = {
        "protocol": {
            "data_root": str(data_root),
            "split_path": str(split_path),
            "split_sha256": _sha256(split_path),
            "clients": [int(value) for value in args.clients],
            "models": [str(value) for value in args.models],
            "epochs": int(args.epochs),
            "spectral_weight": float(args.spectral_weight),
            "implied_minimum_subgroup_mass": float(1.0 / (1.0 + float(args.spectral_weight) ** 2)),
            "uses_environment_metadata_for_training": False,
            "uses_environment_metadata_for_signal_fitting": False,
            "uses_private_audit": False,
            "uses_final_test": False,
            "smoke": bool(args.smoke),
        },
        "clients": client_results,
        "directed_metrics": directed_metrics,
        "one_step_metrics": one_step_metrics,
        "decision": decision,
    }
    safe_payload = _json_safe(payload)
    np.savez_compressed(output_dir / "signals.npz", **raw_arrays)
    (output_dir / "result.json").write_text(
        json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# CRSR Audit 0 Result",
        "",
        f"Verdict: `{decision['verdict']}`",
        "",
        "## Frozen gates",
        "",
    ]
    for name, gate in decision["gates"].items():
        report.append(f"- `{name}`: **{'PASS' if gate['pass'] else 'FAIL'}** — `{json.dumps(_json_safe(gate), ensure_ascii=False)}`")
    report.extend(
        [
            "",
            "Private audit and final test were not read. Operator IDs were used only after prediction and score fitting for post-hoc cell evaluation.",
            "",
        ]
    )
    (output_dir / "RESULT_SUMMARY_ZH.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[result] verdict={decision['verdict']} output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
