"""Train a fixed Gate 1-Pilot arm using the original AsymHFL loop."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

from fedprime.data.loaders import _private_test_transform, _private_train_transform
from fedprime.data.strict_fit_audit import StrictClientSplit, _load_split_file, _validate_split, build_client_audit_loaders
from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment
from scripts.run_cle_v2_moderate import moderate_arm_config

from .gate0 import FIT_FILES, SPLIT, read_json, sha256, write_json
from .pilot_data import HERE, MANIFEST, load_aligned_cache, validate
from .ssa import ssa_loss


PILOT = HERE / "gate1_pilot.json"
PROTOCOL = HERE / "gate1_protocol.json"


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def tensor_hash(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PrivateView(Dataset):
    def __init__(self, images, labels, sources, transform, indexed):
        self.images = images
        self.labels = labels
        self.sources = sources
        self.transform = transform
        self.indexed = indexed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, row):
        image = self.transform(self.images[row])
        label = int(self.labels[row])
        if self.indexed:
            return image, label, (row, int(self.sources[row]))
        return image, label


class TracedIterator:
    def __init__(self, iterator, sink):
        self.iterator = iterator
        self.sink = sink

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iterator)
        self.sink(tensor_hash(*batch))
        return batch


class TracedLoader:
    def __init__(self, loader, sink):
        self.loader = loader
        self.sink = sink

    def __iter__(self):
        return TracedIterator(iter(self.loader), self.sink)


class PilotData:
    def __init__(self, root, output, strength, eta, smoke):
        self.root = root
        self.output = output
        self.strength = strength
        self.eta = eta
        self.smoke = smoke
        self.cache = {}

    def loaders(self, cfg):
        split = cfg["method"]["strict_fit_audit"]
        settings = {key: split[key] for key in ("audit_ratio", "min_audit_per_class", "min_fit_per_class", "seed")}
        num_clients = len(cfg["models"]["names"])
        rows = _load_split_file(self.root / SPLIT, num_clients=num_clients, **settings)
        local, splits, counts = [], {}, {}
        for client in range(num_clients):
            directory = self.root / "private" / f"client_{client}"
            images, labels, sources = (np.load(directory / name, allow_pickle=False) for name in FIT_FILES)
            fit, audit = rows[client]
            _validate_split(labels=labels, fit_indices=fit, audit_indices=audit, client_id=client)
            ids, bias = load_aligned_cache(self.root / "cache" / f"client_{client}.npz", fit, sources, labels)
            self.cache[client] = ids, bias, sources, labels
            fit_view = PrivateView(images, labels, sources, _private_train_transform(False), True)
            audit_view = PrivateView(images, labels, sources, _private_test_transform("cifar10"), False)
            generator = torch.Generator().manual_seed(int(split["loader_seed"]) * 1009 + client)
            loader = DataLoader(
                Subset(fit_view, fit.tolist()), batch_size=cfg["train"]["batch_size"], shuffle=True,
                drop_last=True, num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator,
            )
            local.append(loader)
            splits[client] = StrictClientSplit(client, fit, audit, labels, loader, audit_view)
            counts[client] = torch.bincount(torch.as_tensor(labels[fit], dtype=torch.long), minlength=10).float()
        routing = build_client_audit_loaders(splits, batch_size=int(split["audit_batch_size"]), num_workers=0)
        if self.smoke:
            for client, loader in routing.items():
                route_file = self.output / f"audit_trace_c{client}.jsonl"
                routing[client] = TracedLoader(loader, lambda digest, f=route_file: append(f, digest))
        return local, routing, counts

    def task_loss(self, client, round_idx):
        ids, bias, sources, targets = self.cache[client]

        def loss(logits, labels, identity, batch, images):
            rows, source = (value.numpy() for value in identity)
            indices = np.searchsorted(ids, rows)
            if (
                np.any(indices >= len(ids)) or not np.array_equal(ids[indices], rows)
                or not np.array_equal(sources[rows], source)
                or not np.array_equal(targets[rows], labels.detach().cpu().numpy())
            ):
                raise ValueError(f"Wrong SSA fit row/source/label in client {client}, batch {batch}")
            offset = torch.as_tensor(bias[indices], device=logits.device, dtype=logits.dtype)
            value = ssa_loss(logits, labels, offset, strength=self.strength, poe_weight=self.eta)
            if self.smoke:
                append(self.output / "fit_identity_trace.jsonl", {
                    "client": client, "round": round_idx + 1, "batch": batch,
                    "sha256": tensor_hash(images, *identity),
                })
            if batch == 0:
                with torch.no_grad():
                    ce = torch.nn.functional.cross_entropy(logits, labels)
                    adjusted = logits + self.strength * offset
                    poe = torch.nn.functional.cross_entropy(adjusted, labels)
                    truth = torch.nn.functional.one_hot(labels, logits.shape[1])
                    ce_g = (1 - self.eta) * (logits.softmax(1) - truth) / len(labels)
                    poe_g = self.eta * (adjusted.softmax(1) - truth) / len(labels)
                    append(self.output / "ssa_loss_trace.jsonl", {
                        "round": round_idx + 1, "client": client, "ce_weight": 1 - self.eta,
                        "poe_weight": self.eta, "ce": ce.item(), "poe": poe.item(), "total": value.item(),
                        "ce_logit_grad_l2": ce_g.norm().item(), "poe_logit_grad_l2": poe_g.norm().item(),
                    })
            return value

        return loss


class PilotExperiment(AsymHFLExperiment):
    def __init__(self, cfg, data, milestone=None):
        super().__init__(cfg, isolated_training=data)
        self.milestone = milestone
        self._public_trace_iterator = None

    def _collaborative_phase(self, **kwargs):
        if self.isolated_training.smoke:
            def trace(digest):
                append(self.output_dir / "public_trace.jsonl", digest)

            if self._public_trace_iterator is None:
                self._public_trace_iterator = TracedIterator(kwargs["public_iter"], trace)
            kwargs["public_iter"] = self._public_trace_iterator
            kwargs["public_loader"] = TracedLoader(kwargs["public_loader"], trace)
        return super()._collaborative_phase(**kwargs)

    def _save_models(self, models, subdirectory=None):
        super()._save_models(models, subdirectory)
        if self.milestone is not None and subdirectory is not None:
            self.milestone(self.output_dir, subdirectory)


def train(root, output, arm, mode, device, milestone=None):
    pilot, protocol = read_json(PILOT), read_json(PROTOCOL)
    if arm not in pilot["arms"] or mode not in {"smoke", "pilot"}:
        raise ValueError("Unknown training mode or arm")
    if mode == "pilot" and (device != "cuda" or not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name()):
        raise RuntimeError("The frozen 40-round Pilot runs only on a remote NVIDIA V100")
    validate(root)
    if output.exists():
        raise FileExistsError(output)
    eta = pilot["arms"][arm]
    strength = pilot["lambda_abs"]
    cfg = moderate_arm_config("asymhfl_erm", package_root=root, mode="formal", output_root=output.parent, device=device)
    cfg["experiment_name"] = output.name
    cfg["data"]["private_root"] = str(root / "private")
    cfg["method"]["strict_fit_audit"]["split_path"] = str(root / SPLIT)
    cfg["train"]["max_test_batches"] = None
    cfg["checkpoints"]["save_rounds"] = pilot["progress_checkpoints"] if mode == "pilot" else []
    if mode == "smoke":
        cfg["train"].update(pilot["smoke"])
        # Exercise the public iterator reset in round 2 at the original batch size.
        cfg["data"]["public_size"] = 5 * cfg["train"]["public_batch_size"]
    frozen = protocol["backbone_protocol"]
    if mode == "pilot":
        for key, expected in {
            **{key: frozen[key] for key in ("rounds", "local_epochs", "pretrain_epochs",
                                           "public_batch_size", "public_batches_per_round")},
            "batch_size": frozen["private_batch_size"],
            "max_local_batches": frozen["max_local_batches_per_client_per_round"],
        }.items():
            if cfg["train"][key] != expected:
                raise ValueError(f"Pilot drifted from the frozen comparison: {key}")
    data = PilotData(root, output, strength, eta, mode == "smoke")
    experiment = PilotExperiment(cfg, data, milestone)
    started = perf_counter()
    checkpoints = output / "checkpoints"
    contract = {
        "arm": arm, "mode": mode, "probe": pilot["probe"], "lambda": strength, "eta": eta,
        "input_sha256": sha256(root / MANIFEST), "pilot_sha256": sha256(PILOT),
        "protocol_sha256": sha256(PROTOCOL), "reference_setup": frozen,
        "config_sha256": sha256(output / "config.resolved.json"),
        "pilot_source_hashes": {
            p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for p in (
                HERE / "pilot_data.py", HERE / "pilot_train.py", HERE / "ssa.py",
                HERE.parents[1] / "fedprime/methods/rahfl_asymhfl.py",
                HERE.parents[1] / "fedprime/methods/local_rahfl.py",
            )
        },
        "torch": str(torch.__version__), "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "scientific_evidence": False,
    }
    write_json(output / "RUN_STARTED.json", contract)
    experiment.run()
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (
        [int(row["round"]) for row in rows] != list(range(cfg["train"]["rounds"]))
        or any(row["avg_acc"] or row["wcca"] or row["cfg"] for row in rows)
        or not all(math.isfinite(float(row[key])) for row in rows for key in ("local_loss", "col_loss"))
    ):
        raise RuntimeError("Training trace has missing rounds, nonfinite losses, or unsealed evaluation readings")
    done = {
        **contract, "completed_rounds": len(rows), "status": "training_complete_joint_evaluation_pending",
        "training_seconds": perf_counter() - started,
        "peak_cuda_memory_mb": max(float(row["peak_cuda_memory_mb"] or 0) for row in rows),
        "checkpoint_sha256": {f"client_{i}.pt": sha256(checkpoints / f"client_{i}.pt") for i in range(4)},
        "paired_trace_sha256": sha256(output / "local_batch_trace.jsonl"),
    }
    write_json(output / "TRAINING_COMPLETE.json", done)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=read_json(PILOT)["arms"], required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    train(args.input.resolve(), args.output.resolve(), args.arm, args.mode, args.device)
