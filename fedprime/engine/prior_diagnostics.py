from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def compare_priors(
    predicted_prior: torch.Tensor,
    oracle_prior: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return per-client metrics comparing predicted and true private priors."""
    predicted = predicted_prior.detach().float().cpu().clamp_min(0.0)
    oracle = oracle_prior.detach().float().cpu().clamp_min(0.0)
    predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(eps)
    oracle = oracle / oracle.sum(dim=-1, keepdim=True).clamp_min(eps)

    l1 = (predicted - oracle).abs().sum(dim=-1)
    kl = (
        oracle
        * (oracle.clamp_min(eps).log() - predicted.clamp_min(eps).log())
    ).sum(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(predicted, oracle, dim=-1)
    predicted_entropy = -(predicted * predicted.clamp_min(eps).log()).sum(dim=-1)
    oracle_entropy = -(oracle * oracle.clamp_min(eps).log()).sum(dim=-1)
    uniform_entropy = math.log(predicted.shape[-1])

    return {
        "prior_l1": l1,
        "prior_kl": kl,
        "prior_cosine_similarity": cosine,
        "predicted_entropy": predicted_entropy,
        "oracle_entropy": oracle_entropy,
        "predicted_normalized_entropy": predicted_entropy / uniform_entropy,
        "oracle_normalized_entropy": oracle_entropy / uniform_entropy,
        "predicted_top_class": predicted.argmax(dim=-1),
        "oracle_top_class": oracle.argmax(dim=-1),
        "top_class_match": (predicted.argmax(dim=-1) == oracle.argmax(dim=-1)).float(),
    }


class PriorDiagnosticsRecorder:
    """Stream small prior-comparison logs without participating in training."""

    BASE_FIELDNAMES = [
        "round",
        "public_batch",
        "client",
        "prior_source",
        "prior_l1",
        "prior_kl",
        "prior_cosine_similarity",
        "predicted_entropy",
        "oracle_entropy",
        "predicted_normalized_entropy",
        "oracle_normalized_entropy",
        "predicted_top_class",
        "oracle_top_class",
        "top_class_match",
    ]

    def __init__(
        self,
        output_dir: Path,
        oracle_prior: torch.Tensor,
        prior_source: str,
        save_rounds: list[int] | None = None,
    ):
        self.output_dir = output_dir
        self.oracle_prior = oracle_prior.detach().float().cpu()
        self.prior_source = prior_source
        self.save_rounds = set(save_rounds or [])
        self.num_classes = int(self.oracle_prior.shape[-1])
        prior_fields = []
        for prefix in ("predicted_prior", "oracle_prior", "used_prior"):
            prior_fields.extend(f"{prefix}_class_{class_id}" for class_id in range(self.num_classes))
        self.fieldnames = self.BASE_FIELDNAMES + prior_fields
        self.csv_path = output_dir / "prior_diagnostics.csv"
        self.snapshot_dir = output_dir / "priors"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._metric_values: dict[str, list[float]] = defaultdict(list)
        self._round_values: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record(
        self,
        round_idx: int,
        public_batch_idx: int,
        predicted_prior: torch.Tensor,
        used_prior: torch.Tensor,
    ) -> None:
        predicted = predicted_prior.detach().float().cpu()
        used = used_prior.detach().float().cpu()
        metrics = compare_priors(predicted, self.oracle_prior)

        for client_id in range(predicted.shape[0]):
            row = {
                "round": round_idx,
                "public_batch": public_batch_idx,
                "client": client_id,
                "prior_source": self.prior_source,
            }
            for name, values in metrics.items():
                value = values[client_id].item()
                row[name] = int(value) if name.endswith("_class") else float(value)
                if name not in {"predicted_top_class", "oracle_top_class"}:
                    self._metric_values[name].append(float(value))
            for class_id in range(self.num_classes):
                row[f"predicted_prior_class_{class_id}"] = float(predicted[client_id, class_id])
                row[f"oracle_prior_class_{class_id}"] = float(
                    self.oracle_prior[client_id, class_id]
                )
                row[f"used_prior_class_{class_id}"] = float(used[client_id, class_id])
            self._writer.writerow(row)
        self._file.flush()

        if round_idx in self.save_rounds:
            self._round_values[round_idx]["predicted_prior"].append(predicted.numpy())
            self._round_values[round_idx]["used_prior"].append(used.numpy())

    def finalize(self) -> None:
        self._file.close()
        for round_idx, values in self._round_values.items():
            predicted = np.mean(values["predicted_prior"], axis=0)
            used = np.mean(values["used_prior"], axis=0)
            np.savez_compressed(
                self.snapshot_dir / f"round_{round_idx:03d}.npz",
                predicted_prior=predicted,
                oracle_prior=self.oracle_prior.numpy(),
                used_prior=used,
            )

        summary = {
            "prior_source": self.prior_source,
            "num_records": len(self._metric_values.get("prior_l1", [])),
            "metrics": {},
        }
        for name, values in sorted(self._metric_values.items()):
            array = np.asarray(values, dtype=np.float64)
            summary["metrics"][name] = {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
            }
        with (self.output_dir / "prior_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
