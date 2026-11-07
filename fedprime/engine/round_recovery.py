"""Opt-in round-boundary recovery for the two frozen robust HFL arms."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


RECOVERY_SCHEMA = 2
IDENTITY_KEYS = ("shard", "input_sha256", "code_sha256")


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Recovery CUDA device count differs from saved run")
        torch.cuda.set_rng_state_all(state["cuda"])


def runtime_signature():
    return {
        "python": list(sys.version_info[:2]), "torch": str(torch.__version__),
        "numpy": np.__version__, "cuda": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


class ReplayablePublicIterator:
    """Record the original lazy sampler RNG without changing its execution."""

    def __init__(self, loader):
        if loader.num_workers != 0 or loader.generator is not None:
            raise ValueError("Public iterator recovery requires num_workers=0 and global RNG")
        self.loader = loader
        self.creation_rng = rng_state()
        self.iterator = iter(loader)
        self.first_next_rng = None
        self.yielded = 0
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.first_next_rng is None:
            self.first_next_rng = rng_state()
        try:
            result = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            raise
        self.yielded += 1
        return result

    def state_dict(self):
        return {key: getattr(self, key) for key in (
            "creation_rng", "first_next_rng", "yielded", "exhausted"
        )}

    @classmethod
    def from_state(cls, loader, state):
        current = rng_state()
        try:
            restore_rng(state["creation_rng"])
            result = cls(loader)
            if state["first_next_rng"] is not None:
                restore_rng(state["first_next_rng"])
                for _ in range(state["yielded"]):
                    next(result)
                if state["exhausted"]:
                    try:
                        next(result)
                    except StopIteration:
                        pass
                    else:
                        raise ValueError("Saved public iterator exhaustion is inconsistent")
            return result
        finally:
            restore_rng(current)


def scientific_config(config):
    result = copy.deepcopy(config)
    result.pop("output_root", None)
    result.pop("device", None)
    # Cloud jobs can mount the identical package at different absolute paths.
    for section, keys in (("data", ("private_root", "public_root")),
                          ("checkpoints", ("load_dir",))):
        for key in keys:
            value = result.get(section, {}).get(key)
            if value:
                result[section][key] = "/".join(Path(value).parts[-2:])
    strict = result.get("method", {}).get("strict_fit_audit", {})
    if strict.get("split_path"):
        strict["split_path"] = Path(strict["split_path"]).name
    return result


class RoundRecovery:
    def __init__(self, config, options):
        communication = config["method"].get("communication")
        arm = options["shard"]
        expected = {"aughfl_fidelity": "aughfl_fidelity", "rahfl_fidelity": "asymhfl_val"}
        if expected.get(arm) != communication or int(config.get("num_workers", 0)) != 0:
            raise ValueError("Round recovery only supports the audited AugHFL/RAHFL single arms")
        if int(config["train"].get("pretrain_epochs", 0)) != 0:
            raise ValueError("Round recovery requires the frozen zero-pretraining protocol")
        self.config = scientific_config(config)
        self.options = options
        self.save_rounds = tuple(int(value) for value in options["save_rounds"])
        if not self.save_rounds or any(value < 1 for value in self.save_rounds):
            raise ValueError("Recovery rounds must be positive")
        self.resume_root = Path(options["resume_root"]) if options.get("resume_root") else None

    def restore(self, experiment, models, optimizers, public_loader, public_iter, generators):
        if self.resume_root is None:
            return 0, public_iter
        state_path = self.resume_root / "round_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["schema"] != RECOVERY_SCHEMA or state["config"] != self.config:
            raise ValueError("Resume scientific configuration mismatch")
        if state["runtime"] != runtime_signature():
            raise ValueError("Resume PyTorch/CUDA/device environment mismatch")
        for key in IDENTITY_KEYS:
            if state["identity"][key] != self.options[key]:
                raise ValueError(f"Resume identity mismatch: {key}")
        completed = int(state["completed_rounds"])
        if not 0 < completed <= int(experiment.config["train"]["rounds"]):
            raise ValueError("Invalid recovery round")
        with (self.resume_root / "history/metrics.csv").open(encoding="utf-8", newline="") as handle:
            rounds = [int(row["round"]) for row in csv.DictReader(handle)]
        if rounds != list(range(completed)):
            raise ValueError("Recovery metric history does not match completed rounds")
        if set(models) != set(state["models"]) or len(generators) != len(state["loader_rng"]):
            raise ValueError("Recovery model/loader inventory differs")
        for client, model in models.items():
            model.load_state_dict(state["models"][client], strict=True)
            model.train(state["model_training"][client])
            optimizers[client].load_state_dict(state["optimizers"][client])
        for (generator, _), saved in zip(generators, state["loader_rng"]):
            generator.set_state(saved)
        experiment._communication_strategy.__dict__.update(state["communication"])
        for name, values in state["metrics"].items():
            setattr(experiment, name, values)
        public_iter = ReplayablePublicIterator.from_state(public_loader, state["public_iterator"])
        for path in (self.resume_root / "history").iterdir():
            if path.suffix in {".csv", ".jsonl"}:
                shutil.copy2(path, experiment.output_dir / path.name)
        restore_rng(state["rng"])
        print(f"[round-recovery] restored {completed} completed rounds including optimizer/RNG", flush=True)
        return completed, public_iter

    def save(self, experiment, models, optimizers, public_iter, generators, completed):
        if completed not in self.save_rounds:
            return
        current = rng_state()
        try:
            state = {
                "schema": RECOVERY_SCHEMA, "completed_rounds": completed, "config": self.config,
                "runtime": runtime_signature(),
                "identity": {key: self.options[key] for key in IDENTITY_KEYS},
                "models": {key: model.state_dict() for key, model in models.items()},
                "model_training": {key: model.training for key, model in models.items()},
                "optimizers": {key: opt.state_dict() for key, opt in optimizers.items()},
                "loader_rng": [generator.get_state() for generator, _ in generators],
                "public_iterator": public_iter.state_dict(),
                "communication": experiment._communication_strategy.__dict__,
                "metrics": {name: getattr(experiment, name) for name in (
                    "_last_fedease_metrics", "_last_spurious_metrics", "_last_baseline_metrics"
                )},
                "rng": current,
            }
            directory = experiment.output_dir / "checkpoints" / "round_recovery"
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "round_state.pt.tmp"
            torch.save(state, temporary)
            os.replace(temporary, directory / "round_state.pt")
            status = {
                "status": "training_in_progress", "scientific_evidence": False,
                "completed_rounds": completed, "target_rounds": experiment.config["train"]["rounds"],
                "shard": self.options["shard"],
                "state_sha256": hashlib.sha256((directory / "round_state.pt").read_bytes()).hexdigest(),
            }
            (directory / "PROGRESS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            subprocess.run([
                sys.executable, "-u", str(Path(__file__).resolve().parents[2] / "scripts/publish_cle_round_recovery.py"),
                "--experiment-root", str(experiment.output_dir),
                "--configs-root", self.options["configs_root"],
                "--archive", self.options["archive"],
            ], check=True, timeout=600)
            print(f"[round-recovery] uploaded completed_rounds={completed}", flush=True)
        finally:
            restore_rng(current)
