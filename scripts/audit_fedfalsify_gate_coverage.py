from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.methods.fedfalsify.evidence import (  # noqa: E402
    compute_paired_advantage,
    planned_stratified_audit_counts,
)


GAMMA_SPECS = {
    "00": 0.0,
    "06": 0.6,
    "09": 0.9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project FedFalsify class/source gate coverage under the future "
            "label-skewed fit/audit split."
        )
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=Path("outputs/fedfalsify_audit/foreign_tensor"),
    )
    parser.add_argument(
        "--data-template",
        default=(
            "RAHFL-master/Dataset/cifar_10_cle/"
            "alpha05_gamma{token}_seed0"
        ),
    )
    parser.add_argument("--gammas", nargs="+", choices=sorted(GAMMA_SPECS), default=["00", "06", "09"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fedfalsify_audit/gate_coverage"),
    )
    parser.add_argument("--audit-ratio", type=float, default=0.15)
    parser.add_argument("--min-audit-per-class", type=int, default=10)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--shrinkage-nu", type=float, default=10.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_evidence(
    *,
    source_predictions: np.ndarray,
    receiver_predictions: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    projected_count: int,
    repeats: int,
    kappa: float,
    shrinkage_nu: float,
    min_count: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    indices = np.flatnonzero(labels == int(class_id))
    if projected_count < min_count or indices.size < projected_count:
        return 0.0, 0.0, 0.0

    active = []
    strengths = []
    advantages = []
    for _ in range(int(repeats)):
        chosen = rng.choice(indices, size=int(projected_count), replace=False)
        evidence = compute_paired_advantage(
            source_predictions[chosen],
            receiver_predictions[chosen],
            labels[chosen],
            class_id=class_id,
            kappa=kappa,
            shrinkage_nu=shrinkage_nu,
            min_count=min_count,
        )
        active.append(float(evidence.is_active))
        strengths.append(float(evidence.advantage_strength))
        advantages.append(float(evidence.paired_advantage))
    return float(np.mean(active)), float(np.mean(strengths)), float(np.mean(advantages))


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap-repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summaries = {}

    print(
        "[audit] projecting gate coverage from real client label counts; "
        "independent test_same predictions are only used as a bootstrap pool.",
        flush=True,
    )
    for gamma_index, token in enumerate(args.gammas):
        gamma = GAMMA_SPECS[token]
        tensor_path = args.tensor_dir / f"gamma{token}_foreign_predictions.npz"
        if not tensor_path.is_file():
            raise FileNotFoundError(
                f"Missing {tensor_path}; run audit_fedfalsify_foreign_tensor.py first."
            )
        payload = np.load(tensor_path, allow_pickle=False)
        predictions = payload["predictions"]
        labels_by_receiver = payload["labels"]
        num_sources, num_receivers, _ = predictions.shape
        num_classes = int(payload["accuracy"].shape[-1])
        data_root = Path(args.data_template.format(token=token))
        rng = np.random.default_rng(int(args.seed) + gamma_index * 1009)

        gamma_rows = []
        for receiver_id in range(num_receivers):
            train_labels = np.load(
                data_root / f"client_{receiver_id}" / "train_labels.npy"
            ).astype(np.int64)
            train_counts = np.bincount(train_labels, minlength=num_classes)
            projected_counts = planned_stratified_audit_counts(
                train_labels,
                num_classes=num_classes,
                audit_ratio=args.audit_ratio,
            )
            fit_counts = train_counts - projected_counts
            receiver_predictions = predictions[receiver_id, receiver_id]
            receiver_labels = labels_by_receiver[receiver_id]

            for source_id in range(num_sources):
                if source_id == receiver_id:
                    continue
                source_predictions = predictions[source_id, receiver_id]
                weighted_sample_activation = 0.0
                for class_id in range(num_classes):
                    activation, strength, sampled_advantage = bootstrap_evidence(
                        source_predictions=source_predictions,
                        receiver_predictions=receiver_predictions,
                        labels=receiver_labels,
                        class_id=class_id,
                        projected_count=int(projected_counts[class_id]),
                        repeats=args.bootstrap_repeats,
                        kappa=args.kappa,
                        shrinkage_nu=args.shrinkage_nu,
                        min_count=args.min_audit_per_class,
                        rng=rng,
                    )
                    weighted_sample_activation += activation * int(fit_counts[class_id])
                    row = {
                        "gamma": gamma,
                        "source_client": source_id,
                        "receiver_client": receiver_id,
                        "class_id": class_id,
                        "train_count": int(train_counts[class_id]),
                        "projected_audit_count": int(projected_counts[class_id]),
                        "projected_fit_count": int(fit_counts[class_id]),
                        "is_count_auditable": int(
                            projected_counts[class_id] >= args.min_audit_per_class
                        ),
                        "bootstrap_activation_rate": activation,
                        "bootstrap_advantage_strength": strength,
                        "bootstrap_paired_advantage": sampled_advantage,
                    }
                    rows.append(row)
                    gamma_rows.append(row)

                receiver_source_rows = gamma_rows[-num_classes:]
                denominator = max(int(fit_counts.sum()), 1)
                sample_activation = weighted_sample_activation / denominator
                for row in receiver_source_rows:
                    row["receiver_source_sample_activation_rate"] = sample_activation

        auditable = np.asarray(
            [row["is_count_auditable"] for row in gamma_rows],
            dtype=np.float64,
        )
        activation = np.asarray(
            [row["bootstrap_activation_rate"] for row in gamma_rows],
            dtype=np.float64,
        )
        sample_activation = np.asarray(
            [row["receiver_source_sample_activation_rate"] for row in gamma_rows],
            dtype=np.float64,
        )
        summaries[token] = {
            "gamma": gamma,
            "audit_ratio": float(args.audit_ratio),
            "min_audit_per_class": int(args.min_audit_per_class),
            "count_auditable_fraction": float(auditable.mean()),
            "class_gate_activation_fraction": float(activation.mean()),
            "activation_given_auditable": float(
                activation[auditable > 0].mean() if np.any(auditable > 0) else 0.0
            ),
            "mean_projected_sample_activation_rate": float(sample_activation.mean()),
        }
        print(
            f"[audit] gamma={gamma:.1f} auditable="
            f"{summaries[token]['count_auditable_fraction']:.3f} "
            f"class_active={summaries[token]['class_gate_activation_fraction']:.3f} "
            f"sample_active="
            f"{summaries[token]['mean_projected_sample_activation_rate']:.3f}",
            flush=True,
        )

    write_csv(args.output_dir / "gate_coverage.csv", rows)
    summary = {
        "audit_only": True,
        "method": "FedFalsify projected deployment coverage",
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "kappa": float(args.kappa),
        "shrinkage_nu": float(args.shrinkage_nu),
        "gammas": summaries,
    }
    (args.output_dir / "gate_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[audit] wrote gate-coverage audit to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
