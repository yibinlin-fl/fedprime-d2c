from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import (  # noqa: E402
    CorruptionSkewClientDataset,
    TwoViewTransform,
    _rahfl_augmix_view_transforms,
)
from fedprime.methods.counterfactual_regret import (  # noqa: E402
    SeedSignals,
    correct_class_margin,
    counterfactual_regret,
    decide_audit0,
    evaluate_directed_pair,
    per_sample_jsd,
    split_fit_internal_probe,
)
from fedprime.methods.local_fedease import train_local_fedease_epoch  # noqa: E402
from fedprime.models.factory import build_models, forward_logits  # noqa: E402
from fedprime.utils.env import add_vendor_paths  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "RAHFL-master/Dataset/cifar_10_cle_v2/alpha05_gamma09_seed0_split0"
DEFAULT_SPLIT = (
    ROOT
    / "outputs/strict_pew_asymhfl_val_probe_20260804/outputs/partitions"
    / "strict_cle_v2_alpha05_gamma09_seed0_split0.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs/class_conditional_counterfactual_regret_audit0"


class _DummyEnvironmentDataset(data.Dataset):
    """Adapt taxonomy-free AugMix batches to the existing local trainer API."""

    def __init__(self, dataset: data.Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        views, label = self.dataset[index]
        return views, label, 0


class _CounterfactualProbeDataset(data.Dataset):
    """Return three stochastic views plus evaluation-only corruption metadata."""

    def __init__(self, root: Path, client_id: int, indices: np.ndarray) -> None:
        add_vendor_paths()
        from Dataset.dataaug import aug

        base, weak, preprocess = _rahfl_augmix_view_transforms("cifar10")
        self.dataset = CorruptionSkewClientDataset(
            root=root,
            client_id=int(client_id),
            train=True,
            transform=TwoViewTransform(base, weak),
            return_corruption=False,
        )
        self.indices = np.asarray(indices, dtype=np.int64)
        self.preprocess = preprocess
        self.augment = aug

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        source_index = int(self.indices[item])
        paired_views, label = self.dataset[source_index]
        base_image = paired_views[0]
        views = (
            self.preprocess(base_image),
            self.augment(base_image, self.preprocess),
            self.augment(base_image, self.preprocess),
        )
        corruption_id = int(self.dataset.corruption_ids[source_index])
        return views, int(label), corruption_id, source_index


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fit_indices(split_path: Path, client_id: int) -> np.ndarray:
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    with np.load(split_path, allow_pickle=False) as archive:
        expected = {
            "audit_ratio": 0.15,
            "min_audit_per_class": 5,
            "min_fit_per_class": 2,
            "seed": 0,
        }
        for key, expected_value in expected.items():
            if key not in archive or archive[key][0].item() != expected_value:
                raise ValueError(f"unexpected strict split metadata for {key}")
        key = f"client_{int(client_id)}_fit"
        if key not in archive:
            raise KeyError(key)
        return np.asarray(archive[key], dtype=np.int64)


def _build_train_loader(
    *,
    root: Path,
    client_id: int,
    train_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> data.DataLoader:
    add_vendor_paths()
    from Dataset.dataaug import AugMixDataset

    base, weak, preprocess = _rahfl_augmix_view_transforms("cifar10")
    private_dataset = CorruptionSkewClientDataset(
        root=root,
        client_id=int(client_id),
        train=True,
        transform=TwoViewTransform(base, weak),
        return_corruption=False,
    )
    subset = data.Subset(private_dataset, train_indices.tolist())
    augmented = AugMixDataset(subset, preprocess, jsd_or_nojsd="jsd")
    generator = torch.Generator().manual_seed(int(seed))
    return data.DataLoader(
        _DummyEnvironmentDataset(augmented),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _train_probe_model(
    *,
    model_name: str,
    loader: data.DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    max_batches: int | None,
    seed: int,
    client_id: int,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    _seed_everything(seed)
    model = build_models([model_name], num_classes=10)[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    epoch_diagnostics = []
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
            skip_nonfinite=False,
            log_interval=None,
            context=f"C3R Audit 0 client={client_id} epoch={epoch}",
            diagnostics=diagnostics,
        )
        diagnostics = {key: float(value) for key, value in diagnostics.items()}
        diagnostics["epoch"] = int(epoch)
        diagnostics["loss"] = float(loss)
        epoch_diagnostics.append(diagnostics)
        print(
            f"[train] client={client_id} model={model_name} epoch={epoch + 1}/{epochs} "
            f"loss={loss:.4f} clean_ce={diagnostics.get('clean_ce', float('nan')):.4f}",
            flush=True,
        )
    return model, epoch_diagnostics


@torch.no_grad()
def _collect_seed_signals(
    *,
    model: torch.nn.Module,
    dataset: data.Dataset,
    device: torch.device,
    batch_size: int,
    augmentation_seed: int,
) -> SeedSignals:
    _seed_everything(augmentation_seed)
    loader = data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    storage: dict[str, list[np.ndarray]] = {
        "sample_indices": [],
        "labels": [],
        "corruption_ids": [],
        "regret": [],
        "ce": [],
        "jsd": [],
        "robust_error": [],
        "flip_error": [],
        "base_correct": [],
    }
    for views, labels, corruption_ids, sample_indices in loader:
        labels = labels.to(device, non_blocking=True).long()
        tensors = [view.to(device, non_blocking=True) for view in views]
        size = labels.shape[0]
        logits = forward_logits(model, torch.cat(tensors, dim=0))
        logits_base, logits_aug1, logits_aug2 = torch.split(logits, size)
        margins = [correct_class_margin(view, labels) for view in (logits_base, logits_aug1, logits_aug2)]
        regret = counterfactual_regret(margins[0], torch.stack(margins[1:], dim=1))
        ce = torch.nn.functional.cross_entropy(logits_base, labels, reduction="none")
        jsd = per_sample_jsd([logits_base, logits_aug1, logits_aug2])
        predictions = [view.argmax(dim=1) for view in (logits_base, logits_aug1, logits_aug2)]
        base_correct = predictions[0].eq(labels)
        robust_error = ~(predictions[1].eq(labels) & predictions[2].eq(labels))
        flip_error = base_correct & robust_error

        values = {
            "sample_indices": sample_indices.numpy(),
            "labels": labels.cpu().numpy(),
            "corruption_ids": corruption_ids.numpy(),
            "regret": regret.cpu().numpy(),
            "ce": ce.cpu().numpy(),
            "jsd": jsd.cpu().numpy(),
            "robust_error": robust_error.cpu().numpy().astype(np.int64),
            "flip_error": flip_error.cpu().numpy().astype(np.int64),
            "base_correct": base_correct.cpu().numpy().astype(np.int64),
        }
        for key, value in values.items():
            storage[key].append(np.asarray(value))

    merged = {key: np.concatenate(value) for key, value in storage.items()}
    order = np.argsort(merged["sample_indices"])
    merged = {key: value[order] for key, value in merged.items()}
    return SeedSignals(
        sample_indices=merged["sample_indices"].astype(np.int64),
        labels=merged["labels"].astype(np.int64),
        corruption_ids=merged["corruption_ids"].astype(np.int64),
        regret=merged["regret"].astype(np.float64),
        ce=merged["ce"].astype(np.float64),
        jsd=merged["jsd"].astype(np.float64),
        robust_error=merged["robust_error"].astype(np.int64),
        flip_error=merged["flip_error"].astype(np.int64),
        base_accuracy=float(merged["base_correct"].mean()),
    )


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


def _format_metric(value) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.4f}"


def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# C3R Audit 0 Result",
        "",
        f"Verdict: `{payload['decision']['verdict']}`",
        "",
        "## Client summaries",
        "",
        "| client | model | train | probe | mean base acc | median regret>0 | median regret p90 |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for client_id, client in payload["clients"].items():
        lines.append(
            f"| {client_id} | {client['model']} | {client['train_size']} | {client['probe_size']} | "
            f"{np.mean(client['base_accuracies']):.4f} | "
            f"{np.median(client['regret_positive_fractions']):.4f} | "
            f"{np.median(client['regret_p90']):.4f} |"
        )
    lines.extend([
        "",
        "## Directed seed pairs",
        "",
        "| client | pair | flip | persist | C3R AUC | CE AUC | JSD AUC | C3R enrich | C3R cell rho | cells |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for pair in payload["pairs"]:
        signals = pair["signals"]
        lines.append(
            f"| {pair['client_id']} | {pair['source_seed']}->{pair['target_seed']} | "
            f"{_format_metric(pair['flip_prevalence'])} | {_format_metric(pair['regret_persistence'])} | "
            f"{_format_metric(signals['regret']['flip_auroc'])} | "
            f"{_format_metric(signals['ce']['flip_auroc'])} | "
            f"{_format_metric(signals['jsd']['flip_auroc'])} | "
            f"{_format_metric(signals['regret']['top_fraction_enrichment'])} | "
            f"{_format_metric(signals['regret']['cell_correlation'])} | "
            f"{signals['regret']['valid_cells']} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    for name, gate in payload["decision"]["gates"].items():
        lines.append(f"- `{name}`: **{'PASS' if gate['pass'] else 'FAIL'}** — `{json.dumps(gate, ensure_ascii=False)}`")
    lines.extend([
        "",
        "Private audit and final-test data were not read. Corruption IDs were used only after inference for cell summaries.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Signal-only Audit 0 for class-conditional counterfactual regret.")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split_path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--models", nargs="+", default=["ResNet12", "Mobilenetv2"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--augmentation_seeds", type=int, nargs="+", default=[1701, 1702, 1703])
    parser.add_argument("--probe_ratio", type=float, default=0.15)
    parser.add_argument("--probe_min_class_count", type=int, default=32)
    parser.add_argument("--probe_min", type=int, default=16)
    parser.add_argument("--probe_max", type=int, default=64)
    parser.add_argument("--min_class_support", type=int, default=8)
    parser.add_argument("--min_cell_support", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.clients) != len(args.models):
        raise ValueError("--clients and --models must have equal length")
    if len(args.augmentation_seeds) != 3:
        raise ValueError("the pre-registered audit requires exactly three augmentation seeds")
    if args.smoke:
        args.clients = [int(args.clients[0])]
        args.models = [str(args.models[0])]
        args.epochs = 1
        args.max_train_batches = 2
        args.probe_min_class_count = 16
        args.probe_min = 8
        args.probe_max = 8
        args.min_class_support = 4
        args.min_cell_support = 2
        args.output_dir = Path(args.output_dir) / "smoke"

    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(str(args.device))
    print(f"[setup] device={device} data={data_root} split={split_path}", flush=True)

    client_payloads: dict[int, dict[str, object]] = {}
    pair_metrics: list[dict[str, object]] = []
    raw_arrays: dict[str, np.ndarray] = {}
    directed_seed_pairs = list(zip(args.augmentation_seeds, args.augmentation_seeds[1:] + args.augmentation_seeds[:1]))

    for client_id, model_name in zip(args.clients, args.models):
        labels = np.load(data_root / f"client_{client_id}" / "train_labels.npy").astype(np.int64)
        fit_indices = _load_fit_indices(split_path, client_id)
        train_indices, probe_indices = split_fit_internal_probe(
            fit_indices,
            labels,
            ratio=float(args.probe_ratio),
            min_class_count=int(args.probe_min_class_count),
            min_probe=int(args.probe_min),
            max_probe=int(args.probe_max),
            seed=int(args.train_seed) * 1009 + int(client_id),
        )
        print(
            f"[split] client={client_id} model={model_name} train={train_indices.size} probe={probe_indices.size} "
            f"probe_counts={np.bincount(labels[probe_indices], minlength=10).tolist()}",
            flush=True,
        )
        train_loader = _build_train_loader(
            root=data_root,
            client_id=client_id,
            train_indices=train_indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.train_seed) * 1009 + int(client_id),
        )
        model, training = _train_probe_model(
            model_name=model_name,
            loader=train_loader,
            device=device,
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            max_batches=args.max_train_batches,
            seed=int(args.train_seed),
            client_id=int(client_id),
        )
        probe_dataset = _CounterfactualProbeDataset(data_root, client_id, probe_indices)
        seed_signals: dict[int, SeedSignals] = {}
        for augmentation_seed in args.augmentation_seeds:
            signals = _collect_seed_signals(
                model=model,
                dataset=probe_dataset,
                device=device,
                batch_size=int(args.probe_batch_size),
                augmentation_seed=int(augmentation_seed),
            )
            seed_signals[int(augmentation_seed)] = signals
            prefix = f"client_{client_id}_seed_{augmentation_seed}"
            for name in ("sample_indices", "labels", "corruption_ids", "regret", "ce", "jsd", "robust_error", "flip_error"):
                raw_arrays[f"{prefix}_{name}"] = np.asarray(getattr(signals, name))
            print(
                f"[probe] client={client_id} seed={augmentation_seed} base_acc={signals.base_accuracy:.4f} "
                f"flip={signals.flip_error.mean():.4f} regret_pos={(signals.regret > 0).mean():.4f} "
                f"regret_p90={np.quantile(signals.regret, 0.90):.4f}",
                flush=True,
            )

        client_payloads[int(client_id)] = {
            "model": str(model_name),
            "train_size": int(train_indices.size),
            "probe_size": int(probe_indices.size),
            "probe_class_counts": np.bincount(labels[probe_indices], minlength=10).tolist(),
            "training": training,
            "base_accuracies": [seed_signals[int(seed)].base_accuracy for seed in args.augmentation_seeds],
            "regret_positive_fractions": [
                float((seed_signals[int(seed)].regret > 0).mean()) for seed in args.augmentation_seeds
            ],
            "regret_p90": [
                float(np.quantile(seed_signals[int(seed)].regret, 0.90)) for seed in args.augmentation_seeds
            ],
        }
        for source_seed, target_seed in directed_seed_pairs:
            pair = evaluate_directed_pair(
                seed_signals[int(source_seed)],
                seed_signals[int(target_seed)],
                min_class_support=int(args.min_class_support),
                min_cell_support=int(args.min_cell_support),
                top_fraction=0.25,
            )
            pair.update({
                "client_id": int(client_id),
                "source_seed": int(source_seed),
                "target_seed": int(target_seed),
            })
            pair_metrics.append(pair)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    decision = decide_audit0(client_payloads, pair_metrics)
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
            "augmentation_seeds": [int(value) for value in args.augmentation_seeds],
            "uses_private_audit": False,
            "uses_final_test": False,
            "smoke": bool(args.smoke),
        },
        "clients": client_payloads,
        "pairs": pair_metrics,
        "decision": decision,
    }
    safe_payload = _json_safe(payload)
    np.savez_compressed(output_dir / "sample_signals.npz", **raw_arrays)
    (output_dir / "result.json").write_text(
        json.dumps(safe_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(output_dir / "RESULT_SUMMARY_ZH.md", safe_payload)
    print(f"[result] verdict={decision['verdict']} output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
