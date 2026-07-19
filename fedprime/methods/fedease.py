from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fedprime.methods.environment_witness import (
    PEW_ENVIRONMENT_NAMES,
    PublicEnvironmentWitness,
    build_public_environment_loaders,
    infer_environment_annotations,
    load_environment_witness,
    save_environment_witness,
    train_environment_witness,
)
from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment


class FedEASEExperiment(AsymHFLExperiment):
    """FedEASE v2.1 staged experiment runner."""

    def __init__(self, config: dict):
        data_cfg = config.get("data", {})
        method_cfg = config.get("method", {})
        fedease_cfg = method_cfg.get("fedease", {})
        if str(data_cfg.get("scenario", "")).lower() != "cle_hfl":
            raise ValueError("FedEASE requires data.scenario=cle_hfl.")
        if str(method_cfg.get("cl_module", "")).lower() != "fedease":
            raise ValueError("FedEASE requires method.cl_module=fedease.")
        environment_mode = str(fedease_cfg.get("environment_mode", "oracle")).lower()
        if environment_mode not in {"oracle", "learned"}:
            raise ValueError("FedEASE environment_mode must be oracle or learned.")
        communication = str(method_cfg.get("communication", "none")).lower()
        if communication not in {"none", "local_only", "ebst", "ebst_v2"}:
            raise ValueError("FedEASE communication must be none/local_only, ebst, or ebst_v2.")
        super().__init__(config)

    def run(self) -> None:
        fedease_cfg = self.config["method"]["fedease"]
        self._fedease_environment_annotations = None
        if str(fedease_cfg.get("environment_mode", "oracle")).lower() == "learned":
            self._prepare_learned_environment_annotations()
        super().run()

    def _prepare_learned_environment_annotations(self) -> None:
        data_cfg = self.config["data"]
        train_cfg = self.config["train"]
        fedease_cfg = self.config["method"]["fedease"]
        pew_cfg = fedease_cfg.get("pew", fedease_cfg.get("environment_witness", {}))
        if int(fedease_cfg.get("num_environments", len(PEW_ENVIRONMENT_NAMES))) != len(PEW_ENVIRONMENT_NAMES):
            raise ValueError(
                f"Learned PEW requires num_environments={len(PEW_ENVIRONMENT_NAMES)} "
                "(clean + four groups + unknown)."
            )

        checkpoint = Path(pew_cfg.get("checkpoint", self.output_dir / "pew.pt"))
        if checkpoint.is_file() and bool(pew_cfg.get("reuse_checkpoint", True)):
            print(f"[setup] loading PEW checkpoint: {checkpoint}", flush=True)
            witness = load_environment_witness(checkpoint, self.device)
            history = []
        else:
            print("[setup] training PEW from unlabeled CIFAR-100 corruptions", flush=True)
            train_loader, validation_loader = build_public_environment_loaders(
                data_cfg["public_root"],
                public_size=int(pew_cfg.get("public_size", data_cfg.get("public_size", 5000))),
                batch_size=int(pew_cfg.get("batch_size", train_cfg.get("public_batch_size", 128))),
                num_workers=int(self.config.get("num_workers", 2)),
                seed=int(self.config.get("seed", 0)),
                validation_fraction=float(pew_cfg.get("validation_fraction", 0.2)),
            )
            witness = PublicEnvironmentWitness(
                embedding_dim=int(pew_cfg.get("embedding_dim", 32)),
                num_environments=len(PEW_ENVIRONMENT_NAMES),
                severity_levels=int(pew_cfg.get("severity_levels", 5)),
            ).to(self.device)
            history = train_environment_witness(
                witness,
                train_loader,
                validation_loader,
                self.device,
                epochs=int(pew_cfg.get("epochs", pew_cfg.get("pretrain_epochs", 10))),
                learning_rate=float(pew_cfg.get("learning_rate", 1.0e-3)),
                severity_weight=float(pew_cfg.get("severity_weight", 0.25)),
                max_batches=pew_cfg.get("max_batches"),
            )
            save_environment_witness(witness, checkpoint)
            print(f"[setup] saved PEW checkpoint: {checkpoint}", flush=True)

        if history:
            with (self.output_dir / "pew_training.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)

        root = Path(data_cfg["private_root"])
        annotations = {}
        private_correct = 0
        private_total = 0
        prediction_root = self.output_dir / "pew_predictions"
        prediction_root.mkdir(parents=True, exist_ok=True)
        for client_id in range(len(self.config["models"]["names"])):
            client_root = root / f"client_{client_id}"
            images = np.load(client_root / "train_images.npy")
            annotation = infer_environment_annotations(
                witness,
                images,
                self.device,
                batch_size=int(pew_cfg.get("inference_batch_size", 512)),
                confidence_threshold=float(pew_cfg.get("unknown_threshold", 0.55)),
            )
            oracle = np.load(client_root / "train_corruption_ids.npy").astype(np.int64) + 1
            private_correct += int((annotation["environment_ids"] == oracle).sum())
            private_total += int(oracle.size)
            annotations[client_id] = annotation
            np.savez_compressed(prediction_root / f"client_{client_id}.npz", **annotation)
            print(
                f"[heartbeat] PEW inferred client={client_id} "
                f"group_acc={100.0 * (annotation['environment_ids'] == oracle).mean():.2f} "
                f"unknown_rate={(annotation['environment_ids'] == len(PEW_ENVIRONMENT_NAMES) - 1).mean():.3f}",
                flush=True,
            )
        self._fedease_environment_annotations = annotations
        report = {
            "environment_names": PEW_ENVIRONMENT_NAMES,
            "private_group_accuracy": 100.0 * private_correct / max(private_total, 1),
            "private_samples": private_total,
            "checkpoint": str(checkpoint),
        }
        (self.output_dir / "pew_private_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
