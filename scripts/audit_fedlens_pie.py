from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.corruptions import (  # noqa: E402
    CIFAR_C_CORE_CORRUPTIONS,
    DEFAULT_UNSEEN_CORRUPTIONS,
)
from fedprime.data.loaders import (  # noqa: E402
    _cifar10_train_from_tar,
    _cifar100_train_from_tar,
)
from fedprime.methods.latent_environment import (  # noqa: E402
    PairedInterventionDataset,
    PairedInterventionEncoder,
    RepresentationAuditThresholds,
    audit_representation,
    representation_audit_gates,
    train_paired_intervention_encoder,
    verify_operator_partition,
)
from fedprime.utils.env import resolve_device, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and audit the label-free FedLENS paired-intervention encoder."
    )
    parser.add_argument("--public_root", type=Path, default=ROOT / "RAHFL-master" / "Dataset")
    parser.add_argument("--public_dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--public_size", type=int, default=5000)
    parser.add_argument("--audit_size", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_chain_length", type=int, default=2)
    parser.add_argument("--clean_fraction", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_batches", type=int)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "local_runs" / "fedlens_pie_audit")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execution-only check; forces a tiny dataset and one training batch.",
    )
    return parser.parse_args()


def _load_public_images(root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "cifar10":
        nested = root / "cifar_10"
        return _cifar10_train_from_tar(nested if nested.is_dir() else root)
    nested = root / "cifar_100"
    return _cifar100_train_from_tar(nested if nested.is_dir() else root)


def _split_indices(total: int, train_size: int, audit_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if train_size < 2 or audit_size < 2:
        raise ValueError("public_size and audit_size must both be at least two")
    if train_size + audit_size > total:
        raise ValueError(
            f"requested {train_size + audit_size} public samples, but only {total} are available"
        )
    rng = np.random.default_rng(int(seed))
    selected = rng.permutation(total)[: train_size + audit_size]
    return selected[:train_size], selected[train_size:]


def _loader(dataset: PairedInterventionDataset, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(int(args.seed))
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=False,
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.public_size = min(int(args.public_size), 8)
        args.audit_size = min(int(args.audit_size), 8)
        args.epochs = 1
        args.batch_size = min(int(args.batch_size), 8)
        args.max_batches = 1
        args.embedding_dim = min(int(args.embedding_dim), 8)
        args.max_chain_length = 1
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    print(f"[heartbeat] PIE load_public dataset={args.public_dataset}", flush=True)
    images, labels = _load_public_images(args.public_root, args.public_dataset)
    train_indices, audit_indices = _split_indices(
        len(images), int(args.public_size), int(args.audit_size), int(args.seed)
    )

    if args.smoke:
        training_operators = ("brightness",)
        heldout_operators = ("pixelate",)
    else:
        heldout_operators = tuple(DEFAULT_UNSEEN_CORRUPTIONS)
        training_operators = tuple(
            name for name in CIFAR_C_CORE_CORRUPTIONS if name not in heldout_operators
        )
    operator_partition = verify_operator_partition(training_operators, heldout_operators)
    train_dataset = PairedInterventionDataset(
        images,
        train_indices,
        operators=training_operators,
        seed=int(args.seed),
        labels=None,
        max_chain_length=int(args.max_chain_length),
        clean_fraction=float(args.clean_fraction),
    )
    seen_audit_dataset = PairedInterventionDataset(
        images,
        audit_indices,
        operators=training_operators,
        seed=int(args.seed) + 1009,
        labels=labels,
        max_chain_length=int(args.max_chain_length),
        clean_fraction=float(args.clean_fraction),
    )
    heldout_audit_dataset = PairedInterventionDataset(
        images,
        audit_indices,
        operators=heldout_operators,
        seed=int(args.seed) + 2027,
        labels=labels,
        max_chain_length=int(args.max_chain_length),
        clean_fraction=float(args.clean_fraction),
    )
    model = PairedInterventionEncoder(embedding_dim=int(args.embedding_dim))
    print(
        f"[heartbeat] PIE train samples={len(train_indices)} device={device} smoke={args.smoke}",
        flush=True,
    )
    history = train_paired_intervention_encoder(
        model,
        _loader(train_dataset, args, shuffle=True),
        device,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        temperature=float(args.temperature),
        max_batches=args.max_batches,
    )
    print(f"[heartbeat] PIE audit_seen samples={len(audit_indices)}", flush=True)
    seen_metrics = audit_representation(
        model, _loader(seen_audit_dataset, args, shuffle=False), device, seed=int(args.seed)
    )
    print(f"[heartbeat] PIE audit_heldout samples={len(audit_indices)}", flush=True)
    heldout_metrics = audit_representation(
        model,
        _loader(heldout_audit_dataset, args, shuffle=False),
        device,
        seed=int(args.seed),
    )
    thresholds = RepresentationAuditThresholds()
    gates = representation_audit_gates(seen_metrics, heldout_metrics, thresholds)

    report = {
        "method": "FedLENS-PIE",
        "phase": "representation_audit_a",
        "formal_evidence": False,
        "smoke_execution_only": bool(args.smoke),
        "smoke_operator_subset": bool(args.smoke),
        "taxonomy_labels_used": False,
        "content_labels_used_for_training": False,
        "content_labels_used_for_audit_only": True,
        "public_dataset": str(args.public_dataset),
        "public_root": str(args.public_root.resolve()),
        "seed": int(args.seed),
        "device": str(device),
        "train_samples": int(len(train_indices)),
        "audit_samples": int(len(audit_indices)),
        "training_config": {
            "embedding_dim": int(args.embedding_dim),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "temperature": float(args.temperature),
            "max_chain_length": int(args.max_chain_length),
            "clean_fraction": float(args.clean_fraction),
            "max_batches": args.max_batches,
        },
        "operator_partition": operator_partition,
        "thresholds": asdict(thresholds),
        "seen": seen_metrics.as_dict(),
        "heldout": heldout_metrics.as_dict(),
        "gates": gates,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "audit_report.json"
    checkpoint_path = args.output_dir / "pie_encoder.pt"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "embedding_dim": int(args.embedding_dim),
            "training_operators": list(training_operators),
            "heldout_operators": list(heldout_operators),
            "taxonomy_labels_used": False,
            "seed": int(args.seed),
        },
        checkpoint_path,
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"[artifact] report={report_path.resolve()}", flush=True)
    print(f"[artifact] checkpoint={checkpoint_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
