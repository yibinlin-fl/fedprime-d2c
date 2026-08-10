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

from fedprime.data.corruptions import CIFAR_C_CORE_CORRUPTIONS  # noqa: E402
from fedprime.data.loaders import (  # noqa: E402
    _cifar10_train_from_tar,
    _cifar100_train_from_tar,
)
from fedprime.methods.latent_environment import (  # noqa: E402
    PairedInterventionDataset,
    PairedInterventionEncoder,
    RepresentationAuditThresholds,
    audit_representation,
    verify_operator_partition,
)
from fedprime.methods.monotone_latent_environment import (  # noqa: E402
    CONFIRMATORY_HELDOUT_OPERATORS,
    ConfirmatoryAttributionThresholds,
    MonotonePairedInterventionEncoder,
    OrderedPairedInterventionDataset,
    confirmatory_audit_gates,
    train_matched_unordered_encoder,
    train_monotone_encoder,
)
from fedprime.utils.env import resolve_device, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched confirmatory audit for unordered PIE-v1 versus radial MPIE-v2."
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
    parser.add_argument("--ordinal_margin", type=float, default=0.25)
    parser.add_argument("--max_chain_length", type=int, default=2)
    parser.add_argument("--audit_clean_fraction", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_batches", type=int)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "local_runs" / "fedlens_mpie_confirmatory_seed1",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_public_images(root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "cifar10":
        nested = root / "cifar_10"
        return _cifar10_train_from_tar(nested if nested.is_dir() else root)
    nested = root / "cifar_100"
    return _cifar100_train_from_tar(nested if nested.is_dir() else root)


def _split_indices(total: int, train_size: int, audit_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if train_size < 2 or audit_size < 2 or train_size + audit_size > total:
        raise ValueError("invalid train/audit sizes")
    selected = np.random.default_rng(int(seed)).permutation(total)[: train_size + audit_size]
    return selected[:train_size], selected[train_size:]


def _loader(dataset, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=False,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )


def _audit(
    model: torch.nn.Module,
    seen_dataset: PairedInterventionDataset,
    heldout_dataset: PairedInterventionDataset,
    args: argparse.Namespace,
    device: torch.device,
    tag: str,
) -> tuple[object, object]:
    print(f"[heartbeat] {tag} audit_seen samples={len(seen_dataset)}", flush=True)
    seen = audit_representation(
        model,
        _loader(seen_dataset, args, shuffle=False),
        device,
        seed=int(args.seed),
    )
    print(f"[heartbeat] {tag} audit_heldout samples={len(heldout_dataset)}", flush=True)
    heldout = audit_representation(
        model,
        _loader(heldout_dataset, args, shuffle=False),
        device,
        seed=int(args.seed),
    )
    return seen, heldout


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.public_size = min(int(args.public_size), 8)
        args.audit_size = min(int(args.audit_size), 8)
        args.epochs = 1
        args.batch_size = min(int(args.batch_size), 8)
        args.embedding_dim = min(int(args.embedding_dim), 8)
        args.max_chain_length = 1
        args.max_batches = 1
    device = resolve_device(str(args.device))
    seed_everything(int(args.seed))
    print(f"[heartbeat] MPIE load_public dataset={args.public_dataset}", flush=True)
    images, labels = _load_public_images(args.public_root, args.public_dataset)
    train_indices, audit_indices = _split_indices(
        len(images), int(args.public_size), int(args.audit_size), int(args.seed)
    )
    if args.smoke:
        training_operators = ("brightness",)
        heldout_operators = ("pixelate",)
    else:
        heldout_operators = tuple(CONFIRMATORY_HELDOUT_OPERATORS)
        training_operators = tuple(
            name for name in CIFAR_C_CORE_CORRUPTIONS if name not in heldout_operators
        )
    operator_partition = verify_operator_partition(training_operators, heldout_operators)
    ordered_train = OrderedPairedInterventionDataset(
        images,
        train_indices,
        operators=training_operators,
        seed=int(args.seed),
        max_chain_length=int(args.max_chain_length),
    )
    seen_audit = PairedInterventionDataset(
        images,
        audit_indices,
        operators=training_operators,
        seed=int(args.seed) + 1009,
        labels=labels,
        max_chain_length=int(args.max_chain_length),
        clean_fraction=float(args.audit_clean_fraction),
    )
    heldout_audit = PairedInterventionDataset(
        images,
        audit_indices,
        operators=heldout_operators,
        seed=int(args.seed) + 2027,
        labels=labels,
        max_chain_length=int(args.max_chain_length),
        clean_fraction=float(args.audit_clean_fraction),
    )

    seed_everything(int(args.seed))
    control = PairedInterventionEncoder(embedding_dim=int(args.embedding_dim))
    print(
        f"[heartbeat] PIE-matched train samples={len(ordered_train)} device={device}",
        flush=True,
    )
    control_history = train_matched_unordered_encoder(
        control,
        _loader(ordered_train, args, shuffle=True),
        device,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        temperature=float(args.temperature),
        ordinal_margin=float(args.ordinal_margin),
        max_batches=args.max_batches,
    )
    control_seen, control_heldout = _audit(
        control, seen_audit, heldout_audit, args, device, "PIE-matched"
    )

    seed_everything(int(args.seed))
    candidate = MonotonePairedInterventionEncoder(embedding_dim=int(args.embedding_dim))
    print(
        f"[heartbeat] MPIE train samples={len(ordered_train)} device={device}", flush=True
    )
    candidate_history = train_monotone_encoder(
        candidate,
        _loader(ordered_train, args, shuffle=True),
        device,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        temperature=float(args.temperature),
        ordinal_margin=float(args.ordinal_margin),
        max_batches=args.max_batches,
    )
    candidate_seen, candidate_heldout = _audit(
        candidate, seen_audit, heldout_audit, args, device, "MPIE"
    )

    absolute_thresholds = RepresentationAuditThresholds()
    attribution_thresholds = ConfirmatoryAttributionThresholds()
    gates = confirmatory_audit_gates(
        control_seen,
        control_heldout,
        candidate_seen,
        candidate_heldout,
        absolute_thresholds=absolute_thresholds,
        attribution_thresholds=attribution_thresholds,
    )
    training_config = {
        "embedding_dim": int(args.embedding_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "temperature": float(args.temperature),
        "ordinal_margin": float(args.ordinal_margin),
        "max_chain_length": int(args.max_chain_length),
        "audit_clean_fraction": float(args.audit_clean_fraction),
        "max_batches": args.max_batches,
    }
    report = {
        "method": "FedLENS-MPIE",
        "phase": "matched_confirmatory_representation_audit",
        "formal_federated_evidence": False,
        "smoke_execution_only": bool(args.smoke),
        "taxonomy_labels_used": False,
        "content_labels_used_for_training": False,
        "content_labels_used_for_audit_only": True,
        "ordered_intervention_signal_used_by_control": False,
        "ordered_intervention_signal_used_by_candidate": True,
        "public_dataset": str(args.public_dataset),
        "seed": int(args.seed),
        "device": str(device),
        "train_samples": int(len(train_indices)),
        "audit_samples": int(len(audit_indices)),
        "training_config": training_config,
        "operator_partition": operator_partition,
        "absolute_thresholds": asdict(absolute_thresholds),
        "attribution_thresholds": asdict(attribution_thresholds),
        "parameter_counts": {
            "control": sum(parameter.numel() for parameter in control.parameters()),
            "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
        },
        "control": {
            "seen": control_seen.as_dict(),
            "heldout": control_heldout.as_dict(),
            "history": control_history,
        },
        "candidate": {
            "seen": candidate_seen.as_dict(),
            "heldout": candidate_heldout.as_dict(),
            "history": candidate_history,
        },
        "gates": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "confirmatory_report.json"
    control_path = args.output_dir / "pie_v1_matched.pt"
    candidate_path = args.output_dir / "mpie_v2.pt"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata = {
        "embedding_dim": int(args.embedding_dim),
        "seed": int(args.seed),
        "training_operators": list(training_operators),
        "heldout_operators": list(heldout_operators),
        "taxonomy_labels_used": False,
        "training_config": training_config,
    }
    torch.save({"state_dict": _cpu_state_dict(control), **metadata}, control_path)
    torch.save({"state_dict": _cpu_state_dict(candidate), **metadata}, candidate_path)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[artifact] report={report_path.resolve()}", flush=True)
    print(f"[artifact] control={control_path.resolve()}", flush=True)
    print(f"[artifact] candidate={candidate_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
