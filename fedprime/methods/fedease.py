from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from fedprime.methods.environment_witness import (
    PEW_ENVIRONMENT_NAMES,
    PublicEnvironmentWitness,
    build_public_environment_loaders,
    calibrate_unknown_threshold,
    evaluate_environment_witness,
    infer_environment_annotations,
    load_environment_witness,
    resolve_public_corruption_groups,
    save_environment_witness,
    train_environment_witness,
)
from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment
from fedprime.utils.env import seed_everything


class FedEASEExperiment(AsymHFLExperiment):
    """FedEASE v2.1 staged experiment runner."""

    def __init__(self, config: dict):
        data_cfg = config.get("data", {})
        method_cfg = config.get("method", {})
        fedease_cfg = method_cfg.get("fedease", {})
        if str(data_cfg.get("scenario", "")).lower() not in {"cle_hfl", "cle_hfl_v2"}:
            raise ValueError("FedEASE requires data.scenario=cle_hfl or cle_hfl_v2.")
        if str(method_cfg.get("cl_module", "")).lower() != "fedease":
            raise ValueError("FedEASE requires method.cl_module=fedease.")
        environment_mode = str(fedease_cfg.get("environment_mode", "oracle")).lower()
        if environment_mode not in {"oracle", "oracle_family", "learned", "learned_shuffled"}:
            raise ValueError(
                "FedEASE environment_mode must be oracle, oracle_family, learned, "
                "or learned_shuffled."
            )
        communication = str(method_cfg.get("communication", "none")).lower()
        if communication not in {
            "none",
            "local_only",
            "asymhfl",
            "asymhfl_val",
            "hfl",
            "symmetric_hfl",
        }:
            raise ValueError(
                "FedEASE communication must be none/local_only, hfl/symmetric_hfl, "
                "or asymhfl/asymhfl_val."
            )
        super().__init__(config)

    def run(self) -> None:
        fedease_cfg = self.config["method"]["fedease"]
        self._fedease_environment_annotations = None
        environment_mode = str(fedease_cfg.get("environment_mode", "oracle")).lower()
        if environment_mode in {"learned", "learned_shuffled"}:
            self._prepare_learned_environment_annotations()
            if environment_mode == "learned_shuffled":
                self._shuffle_environment_annotations()
            seed_everything(int(self.config.get("seed", 0)))
            print(
                "[setup] reset experiment RNG after PEW preparation for matched model initialization",
                flush=True,
            )
        elif environment_mode == "oracle_family":
            self._prepare_oracle_family_annotations()
            seed_everything(int(self.config.get("seed", 0)))
            print(
                "[setup] reset experiment RNG after oracle-family preparation for matched initialization",
                flush=True,
            )
        super().run()

    def _shuffle_environment_annotations(self) -> None:
        """Break sample/environment association while preserving PEW marginals."""

        assert self._fedease_environment_annotations is not None
        base_seed = int(self.config.get("seed", 0)) + 91_337
        shuffled = {}
        for client_id, annotation in sorted(self._fedease_environment_annotations.items()):
            rng = np.random.default_rng(base_seed + int(client_id))
            permutation = rng.permutation(len(annotation["environment_ids"]))
            shuffled[client_id] = {
                key: np.asarray(value)[permutation].copy()
                for key, value in annotation.items()
            }
        self._fedease_environment_annotations = shuffled
        print("[setup] shuffled PEW annotations as a frozen negative control", flush=True)

    def _prepare_oracle_family_annotations(self) -> None:
        """Build a reporting-only oracle upper bound from private operator metadata."""

        root = Path(self.config["data"]["private_root"])
        annotations = {}
        for client_id in range(len(self.config["models"]["names"])):
            operator_ids = np.load(
                root / f"client_{client_id}" / "train_corruption_ids.npy"
            ).astype(np.int64)
            environment_ids = self._diagnostic_environment_ids(operator_ids, root)
            annotations[client_id] = {"environment_ids": environment_ids}
        self._fedease_environment_annotations = annotations
        print(
            "[setup] WARNING: oracle-family annotations use private operator metadata; "
            "upper-bound analysis only",
            flush=True,
        )

    def _prepare_learned_environment_annotations(self) -> None:
        prepare_started = time.perf_counter()
        data_cfg = self.config["data"]
        train_cfg = self.config["train"]
        fedease_cfg = self.config["method"]["fedease"]
        pew_cfg = fedease_cfg.get("pew", fedease_cfg.get("environment_witness", {}))
        excluded_operators = tuple(
            sorted({str(operator) for operator in pew_cfg.get("exclude_operators", ())})
        )
        public_corruption_groups = resolve_public_corruption_groups(excluded_operators)
        if int(fedease_cfg.get("num_environments", len(PEW_ENVIRONMENT_NAMES))) != len(PEW_ENVIRONMENT_NAMES):
            raise ValueError(
                f"Learned PEW requires num_environments={len(PEW_ENVIRONMENT_NAMES)} "
                "(clean + four groups + unknown)."
            )

        checkpoint = Path(pew_cfg.get("checkpoint", self.output_dir / "pew.pt"))
        threshold_setting = pew_cfg.get("unknown_threshold", 0.55)
        calibration_requested = str(threshold_setting).lower() == "auto"
        train_loader = None
        validation_loader = None
        if calibration_requested or not (
            checkpoint.is_file() and bool(pew_cfg.get("reuse_checkpoint", True))
        ):
            train_loader, validation_loader = build_public_environment_loaders(
                data_cfg["public_root"],
                public_size=int(pew_cfg.get("public_size", data_cfg.get("public_size", 5000))),
                batch_size=int(pew_cfg.get("batch_size", train_cfg.get("public_batch_size", 128))),
                num_workers=int(self.config.get("num_workers", 2)),
                seed=int(self.config.get("seed", 0)),
                validation_fraction=float(pew_cfg.get("validation_fraction", 0.2)),
                public_dataset=str(data_cfg.get("public_dataset", "cifar100")),
                excluded_operators=excluded_operators,
            )
        if checkpoint.is_file() and bool(pew_cfg.get("reuse_checkpoint", True)):
            print(f"[setup] loading PEW checkpoint: {checkpoint}", flush=True)
            witness = load_environment_witness(checkpoint, self.device)
            checkpoint_exclusions = tuple(
                sorted(getattr(witness, "training_excluded_operators", ()))
            )
            if checkpoint_exclusions != excluded_operators:
                raise ValueError(
                    "PEW checkpoint operator exclusion mismatch: "
                    f"checkpoint={list(checkpoint_exclusions)} "
                    f"config={list(excluded_operators)}"
                )
            history = []
        else:
            print(
                "[setup] training PEW from unlabeled public corruptions; "
                f"excluded_operators={list(excluded_operators)}",
                flush=True,
            )
            assert train_loader is not None and validation_loader is not None
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
            save_environment_witness(
                witness,
                checkpoint,
                excluded_operators=excluded_operators,
            )
            print(f"[setup] saved PEW checkpoint: {checkpoint}", flush=True)

        calibration = None
        if calibration_requested:
            assert validation_loader is not None
            calibration = calibrate_unknown_threshold(witness, validation_loader, self.device)
            unknown_threshold = float(calibration["threshold"])
        else:
            unknown_threshold = float(threshold_setting)

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
                confidence_threshold=unknown_threshold,
            )
            oracle = self._diagnostic_environment_ids(
                np.load(client_root / "train_corruption_ids.npy").astype(np.int64),
                root,
            )
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
            "excluded_public_operators": list(excluded_operators),
            "public_operator_pools": {
                group: list(operators)
                for group, operators in public_corruption_groups.items()
            },
            "private_group_accuracy": 100.0 * private_correct / max(private_total, 1),
            "private_samples": private_total,
            "checkpoint": str(checkpoint),
            "unknown_threshold": unknown_threshold,
            "calibration": calibration,
            "prepare_seconds": time.perf_counter() - prepare_started,
            "validation": (
                evaluate_environment_witness(witness, validation_loader, self.device).as_dict()
                if validation_loader is not None
                else None
            ),
        }
        (self.output_dir / "pew_private_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    def _diagnostic_environment_ids(
        self,
        corruption_ids: np.ndarray,
        root: Path,
    ) -> np.ndarray:
        """Map v2 operator IDs to PEW families for reporting, never for training."""

        if str(self.config["data"].get("scenario", "")).lower() != "cle_hfl_v2":
            return np.asarray(corruption_ids, dtype=np.int64) + 1

        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        operator_to_id = metadata["operator_to_id"]
        operator_families = metadata["operator_families"]
        family_to_pew = {"noise": 1, "blur": 2, "weather": 3, "digital": 4}
        id_to_family = {
            int(operator_id): family_to_pew[operator_families[operator_name]]
            for operator_name, operator_id in operator_to_id.items()
        }
        mapped = np.asarray(
            [id_to_family[int(operator_id)] for operator_id in corruption_ids],
            dtype=np.int64,
        )
        return mapped
