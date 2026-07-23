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

from fedprime.methods.fedfalsify.audit_runtime import (  # noqa: E402
    evaluate_all_models_on_receiver,
    load_models_from_archive,
)
from fedprime.methods.fedfalsify.evidence import (  # noqa: E402
    classwise_accuracy_tensor,
)
from fedprime.utils.config import load_config  # noqa: E402
from fedprime.utils.env import resolve_device, seed_everything  # noqa: E402


GAMMA_SPECS = {
    "00": 0.0,
    "06": 0.6,
    "09": 0.9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the FedFalsify foreign-transfer tensor from independent "
            "client-specific CLE-HFL same-environment evaluation splits."
        )
    )
    parser.add_argument(
        "--checkpoint-archive",
        type=Path,
        default=Path("outputs/cle_rahfl_diagnostic_outputs.tar.gz"),
    )
    parser.add_argument(
        "--config-template",
        default="configs/diagnostic_rahfl_cle_alpha05_gamma{token}.yaml",
    )
    parser.add_argument("--gammas", nargs="+", choices=sorted(GAMMA_SPECS), default=["00", "06", "09"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fedfalsify_audit/foreign_tensor"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    transfer_rows: list[dict] = []
    gap_rows: list[dict] = []
    gamma_summaries = {}

    print(
        "[audit] FedFalsify foreign-transfer tensor uses independent test_same "
        "splits for offline hypothesis auditing only.",
        flush=True,
    )
    print(f"[audit] device={device} checkpoint_archive={args.checkpoint_archive}", flush=True)

    for token in args.gammas:
        gamma = GAMMA_SPECS[token]
        config_path = Path(args.config_template.format(token=token))
        config = load_config(config_path)
        seed_everything(int(config.get("seed", 0)))
        model_names = list(config["models"]["names"])
        num_clients = len(model_names)
        num_classes = int(config["data"].get("num_classes", 10))
        experiment_name = str(config["experiment_name"])
        private_root = Path(config["data"]["private_root"])

        print(
            f"[audit] gamma={gamma:.1f} loading {num_clients} checkpoints "
            f"for {experiment_name}",
            flush=True,
        )
        models = load_models_from_archive(
            checkpoint_archive=args.checkpoint_archive,
            experiment_name=experiment_name,
            model_names=model_names,
            num_classes=num_classes,
            device=device,
        )

        receiver_predictions = []
        receiver_labels = []
        for receiver_id in range(num_clients):
            print(
                f"[heartbeat] gamma={gamma:.1f} receiver={receiver_id} "
                "evaluating all source models",
                flush=True,
            )
            predictions, labels = evaluate_all_models_on_receiver(
                models=models,
                dataset_directory=private_root / "test_same" / f"client_{receiver_id}",
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_batches=args.max_batches,
            )
            receiver_predictions.append(predictions)
            receiver_labels.append(labels)

        predictions = np.stack(receiver_predictions, axis=1)
        labels = np.stack(receiver_labels, axis=0)
        accuracy, counts = classwise_accuracy_tensor(
            predictions,
            labels,
            num_classes=num_classes,
        )
        np.savez_compressed(
            args.output_dir / f"gamma{token}_foreign_predictions.npz",
            predictions=predictions,
            labels=labels,
            accuracy=accuracy,
            counts=counts,
            model_names=np.asarray(model_names),
            gamma=np.asarray(gamma, dtype=np.float64),
        )

        gamma_gap_values = []
        for source_id in range(num_clients):
            for receiver_id in range(num_clients):
                for class_id in range(num_classes):
                    transfer_rows.append({
                        "gamma": gamma,
                        "source_client": source_id,
                        "source_model": model_names[source_id],
                        "receiver_client": receiver_id,
                        "class_id": class_id,
                        "sample_count": int(counts[receiver_id, class_id]),
                        "accuracy": float(accuracy[source_id, receiver_id, class_id]),
                        "is_self_environment": int(source_id == receiver_id),
                    })

            foreign_receivers = [
                receiver_id
                for receiver_id in range(num_clients)
                if receiver_id != source_id
            ]
            for class_id in range(num_classes):
                self_accuracy = float(accuracy[source_id, source_id, class_id])
                foreign_accuracy = float(
                    np.nanmean(accuracy[source_id, foreign_receivers, class_id])
                )
                gap = self_accuracy - foreign_accuracy
                gamma_gap_values.append(gap)
                gap_rows.append({
                    "gamma": gamma,
                    "source_client": source_id,
                    "source_model": model_names[source_id],
                    "class_id": class_id,
                    "self_accuracy": self_accuracy,
                    "foreign_mean_accuracy": foreign_accuracy,
                    "foreign_survival_gap": gap,
                })

        gamma_summaries[token] = {
            "gamma": gamma,
            "mean_foreign_survival_gap": float(np.nanmean(gamma_gap_values)),
            "median_foreign_survival_gap": float(np.nanmedian(gamma_gap_values)),
            "positive_gap_fraction": float(np.mean(np.asarray(gamma_gap_values) > 0.0)),
            "num_samples_per_receiver": int(labels.shape[1]),
        }
        print(
            f"[audit] gamma={gamma:.1f} mean_gap="
            f"{gamma_summaries[token]['mean_foreign_survival_gap']:.4f} "
            f"positive_gap_fraction="
            f"{gamma_summaries[token]['positive_gap_fraction']:.3f}",
            flush=True,
        )
        del models

    write_csv(args.output_dir / "foreign_transfer_tensor.csv", transfer_rows)
    write_csv(args.output_dir / "foreign_survival_gap.csv", gap_rows)
    ordered = [gamma_summaries[token]["mean_foreign_survival_gap"] for token in args.gammas]
    summary = {
        "audit_only": True,
        "split": "independent test_same/client_k",
        "warning": (
            "These labels are used only for offline method auditing and must never "
            "be used by a formal training router."
        ),
        "checkpoint_archive": str(args.checkpoint_archive),
        "gammas": gamma_summaries,
        "mean_gap_monotonic_in_requested_order": bool(
            all(right >= left for left, right in zip(ordered, ordered[1:]))
        ),
    }
    (args.output_dir / "foreign_tensor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[audit] wrote foreign-transfer audit to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
