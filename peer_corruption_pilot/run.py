"""Run the ordinary-corruption RAHFL pilot on one device."""

import argparse
import csv
import hashlib
from importlib import import_module
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T

from fedprime.data.loaders import (
    CIFAR100_MEAN, CIFAR100_STD, NumpyImageClassificationDataset, TwoViewTransform,
    _private_test_transform, _private_train_transform, _rahfl_augmix_view_transforms,
)
from fedprime.methods.local_rahfl import train_local_augmix_dcl_epoch
from fedprime.models.factory import build_models, forward_logits
from fedprime.utils.env import add_vendor_paths, seed_everything
from peer_corruption_pilot.data import LAYOUTS, audit, digest, render


MODELS = ("ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2")
ARMS = ("local", *LAYOUTS)
LEARNING_RATE = 0.001


class DeadlineReached(Exception):
    pass


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def check_deadline(deadline):
    if monotonic() >= deadline:
        raise DeadlineReached


def save_weights(path, models):
    temp = path.with_suffix(".tmp")
    torch.save({
        i: {name: value.detach().cpu() for name, value in model.state_dict().items()}
        for i, model in models.items()
    }, temp)
    temp.replace(path)


def loader(dataset, batch_size, *, shuffle=False, seed=0):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def weights_hash(model):
    return digest(*(value.detach().cpu().numpy() for value in model.state_dict().values()))


def fit_pretrain(model, batches, epochs, device, deadline):
    optim = optimizer(model)
    model.train()
    for _ in range(epochs):
        check_deadline(deadline)
        for images, labels in batches:
            optim.zero_grad(set_to_none=True)
            loss = F.cross_entropy(forward_logits(model, images.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Pretraining produced non-finite loss")
            loss.backward()
            optim.step()


def predict(model, batches, device):
    model.eval()
    with torch.inference_mode():
        scores = [forward_logits(model, images.to(device)).cpu() for images, _ in batches]
        scores = torch.cat(scores)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Evaluation produced non-finite logits")
        return scores.argmax(dim=1).numpy()


def accuracy(predictions, labels):
    return float(100.0 * np.mean(predictions == labels))


def collaborate(models, batches, route_scores, device):
    # Released RAHFL computes every client's public logits before updating any
    # client on that batch, and recreates Adam for each public update.
    teachers = [[j for j in models if j != i and route_scores[i] <= route_scores[j]] for i in models]
    for model in models.values():
        model.train()
    for images, _ in batches:
        images = images.to(device)
        logits = {i: forward_logits(model, images) for i, model in models.items()}
        targets = {i: logits[i].detach().softmax(dim=1) for i in logits}
        for i, model in models.items():
            if not teachers[i]:
                continue
            prediction = logits[i].log_softmax(dim=1)
            loss = torch.stack([F.kl_div(prediction, targets[j], reduction="batchmean") for j in teachers[i]]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Communication produced non-finite loss")
            optim = optimizer(model)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    return teachers


def evaluation_loaders(arrays, role):
    images, ids, labels = (arrays[f"{role}_{suffix}"] for suffix in ("images", "ids", "labels"))
    views = [images] + [render(images, ids, np.full(len(ids), op, dtype=np.uint8)) for op in (1, 2)]
    return [loader(NumpyImageClassificationDataset(view, labels, _private_test_transform()), 128) for view in views]


def fit_arm(arm, seed, arrays, route_loaders, eval_loaders, pretrained, args, deadline, out, previous=None):
    from Dataset.dataaug import AugMixDataset

    check_deadline(deadline)
    started = monotonic()
    clients = (0,) if arm == "local" else tuple(range(4))
    layout = "uniform" if arm == "local" else arm
    models, private = {}, {}
    batch_size = 16 if args.smoke else 64
    public_batch_size = 16 if args.smoke else 128
    base, weak, preprocess = _rahfl_augmix_view_transforms()
    if previous is None:
        write_json(out / "status.json", {"state": "pretraining", "arm": arm, "seed": seed})
        previous_weights = {}
    else:
        previous_weights = torch.load(previous / "weights.pt", map_location="cpu", weights_only=True)
        previous_status = json.loads((previous / "status.json").read_text(encoding="utf-8"))
        if previous_status["state"] != "complete" or set(previous_weights) != set(clients):
            raise ValueError(f"Invalid completed first-stage arm: {previous}")
    for i in clients:
        labels = arrays["fit_labels"][i]
        ids = arrays["fit_ids"][i]
        images = render(arrays["fit_images"][i], ids, arrays[layout][i])
        key = (i, digest(images, labels))
        seed_everything(10_000 * (seed + 1) + i)
        model = build_models([MODELS[i]], 10)[0].to(args.device)
        if previous is not None:
            model.load_state_dict(previous_weights[i])
        elif key not in pretrained:
            print(f"[pretrain] seed={seed} arm={arm} client={i}", flush=True)
            ds = NumpyImageClassificationDataset(images, labels, _private_train_transform(False))
            fit_pretrain(model, loader(ds, batch_size, shuffle=True, seed=seed + i),
                         args.pretrain_epochs, args.device, deadline)
            pretrained[key] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if previous is None:
            model.load_state_dict(pretrained[key])
        models[i] = model
        raw = NumpyImageClassificationDataset(images, labels, TwoViewTransform(base, weak))
        private[i] = AugMixDataset(raw, preprocess, jsd_or_nojsd="jsd")

    public = NumpyImageClassificationDataset(
        arrays["public_images"], np.zeros(len(arrays["public_ids"]), dtype=np.int64),
        T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.RandomRotation(15),
                   T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)]),
    )
    if previous is None:
        initial_hash, trace, rounds = weights_hash(models[0]), [], []
    else:
        initial_hash = previous_status["initial_c0"]
        trace = previous_status["batch_trace_c0"]
        rounds = json.loads((previous / "rounds.json").read_text(encoding="utf-8"))
        if len(trace) != len(rounds) or not 0 < len(rounds) < args.rounds:
            raise ValueError(f"Invalid first-stage round count: {previous}")
    write_json(out / "status.json", {"state": "training", "rounds_completed": len(rounds)})
    for round_no in range(len(rounds) + 1, args.rounds + 1):
        check_deadline(deadline)
        entry = {"round": round_no}
        if arm != "local":
            route_scores = {i: accuracy(predict(model, route_loaders[0], args.device), arrays["route_labels"])
                            for i, model in models.items()}
            measure = round_no in {1, 4, args.rounds}
            if measure:
                entry["c0_corrupt_before"] = accuracy(
                    np.stack([predict(models[0], ds, args.device) for ds in route_loaders[1:]]),
                    arrays["route_labels"],
                )
            public_seed = 100_000 * (seed + 1) + round_no
            seed_everything(public_seed)
            entry["teachers"] = collaborate(
                models, loader(public, public_batch_size, shuffle=True, seed=public_seed), route_scores, args.device,
            )
            entry["route_scores"] = route_scores
            if measure:
                entry["c0_corrupt_after"] = accuracy(
                    np.stack([predict(models[0], ds, args.device) for ds in route_loaders[1:]]),
                    arrays["route_labels"],
                )
        for i, model in models.items():
            local_seed = 1_000_000 * (seed + 1) + 100 * round_no + i
            seed_everything(local_seed)
            batches = loader(private[i], batch_size, shuffle=True, seed=local_seed)
            hashes = []

            def record(*, batch_idx, images, labels):
                hashes.append(digest(labels.numpy(), *(view.numpy() for view in images)))

            entry[f"local_loss_c{i}"] = train_local_augmix_dcl_epoch(
                model, batches, optimizer(model), args.device, max_batches=1 if args.smoke else None,
                context=f"{arm} seed {seed} round {round_no} client {i}",
                batch_trace_fn=record if i == 0 else None,
            )
            if i == 0:
                if len(hashes) != (1 if args.smoke else len(batches)):
                    raise ValueError("Target local update count changed")
                trace.append(hashes)
        rounds.append(entry)
        write_json(out / "rounds.json", rounds)
        if not args.smoke and previous is None and round_no == args.rounds // 2:
            save_weights(out / "midpoint_weights.pt", models)
        write_json(out / "status.json", {"state": "training", "rounds_completed": round_no})
        print(f"[round] seed={seed} arm={arm} {round_no}/{args.rounds}", flush=True)

    save_weights(out / "weights.pt", models)
    predictions = {f"client_{i}": np.stack([predict(model, ds, args.device) for ds in eval_loaders])
                   for i, model in models.items()}
    np.savez_compressed(out / "predictions.npz", source_ids=arrays["eval_ids"],
                        labels=arrays["eval_labels"], **predictions)
    write_json(out / "status.json", {
        "state": "complete", "arm": arm, "seed": seed, "initial_c0": initial_hash,
        "batch_trace_c0": trace, "seconds": monotonic() - started,
    })


def summarize(output, arrays, seeds, smoke):
    rows, decisions = [], []
    for seed in seeds:
        baseline = None
        scores = {}
        for arm in ARMS:
            folder = output / f"seed_{seed}" / arm
            if not (folder / "status.json").exists():
                continue
            status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
            if status["state"] != "complete":
                continue
            pairing = (status["initial_c0"], status["batch_trace_c0"])
            if arm == "local":
                baseline = pairing
            if pairing != baseline:
                raise ValueError(f"Target checkpoint or local batches differ for seed {seed}, arm {arm}")
            with np.load(folder / "predictions.npz", allow_pickle=False) as result:
                if not np.array_equal(result["source_ids"], arrays["eval_ids"]) or not np.array_equal(
                    result["labels"], arrays["eval_labels"]
                ):
                    raise ValueError(f"Evaluation pairing failed for seed {seed}, arm {arm}")
                for key in result.files:
                    if key.startswith("client_"):
                        pred = result[key]
                        labels = arrays["eval_labels"]
                        scores_by_view = [accuracy(view, labels) for view in pred]
                        corrupt = (scores_by_view[1] + scores_by_view[2]) / 2
                        if key == "client_0":
                            scores[arm] = corrupt
                        rows.append({
                            "seed": seed, "arm": arm, "client": key, "clean": scores_by_view[0],
                            "gaussian_noise": scores_by_view[1], "defocus_blur": scores_by_view[2],
                            "corrupt": corrupt, "seconds": status["seconds"],
                        })
        if len(scores) == len(ARMS):
            gains = {arm: scores[arm] - scores["local"] for arm in LAYOUTS}
            decisions.append({"seed": seed, "gains_pp": gains, "flip_at_one_pp": {
                arm: gains["uniform"] >= 1.0 and gains[arm] <= -1.0
                for arm in ("concentrated_a", "concentrated_b")
            }})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "seed", "arm", "client", "clean", "gaussian_noise", "defocus_blur", "corrupt", "seconds",
        ])
        writer.writeheader()
        writer.writerows(rows)
    complete = len(decisions) == len(seeds)
    write_json(output / "collaboration_gains.json", {
        "mode": "smoke" if smoke else "pilot", "complete": complete,
        "seed_level": decisions, "formal_evidence": False,
    })
    return complete


def run_matrix(output, arrays, seeds, args, deadline, route_loaders, eval_loaders, previous=None):
    for seed in seeds:
        pretrained = {}
        for arm in ARMS:
            check_deadline(deadline)
            relative = Path(f"seed_{seed}") / arm
            out = output / relative
            out.mkdir(parents=True)
            fit_arm(
                arm, seed, arrays, route_loaders, eval_loaders, pretrained, args, deadline, out,
                previous / relative if previous is not None else None,
            )
            summarize(output, arrays, seeds, args.smoke)
    if not summarize(output, arrays, seeds, args.smoke):
        raise RuntimeError(f"The paired matrix is incomplete: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-hours", type=float, default=8)
    args = parser.parse_args()
    if not 0 < args.limit_hours <= 10:
        parser.error("--limit-hours must be positive and at most 10")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for --device cuda")
    args.rounds, args.pretrain_epochs = (1, 1) if args.smoke else (8, 5)
    final_rounds = 2 if args.smoke else 12
    seeds = (0, 1)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.input, allow_pickle=False) as handle:
        arrays = dict(handle)
    input_audit = audit(arrays, args.smoke)
    file_hash = hashlib.sha256()
    with args.input.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            file_hash.update(block)
    input_audit["file_sha256"] = file_hash.hexdigest()
    write_json(args.output / "input_audit.json", input_audit)
    write_json(args.output / "run_config.json", {
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "models": MODELS,
        "seeds": seeds, "arms": ARMS, "round_milestones": [args.rounds, final_rounds],
        "pretrain_epochs": args.pretrain_epochs,
        "private_batch_size": 16 if args.smoke else 64, "public_batch_size": 16 if args.smoke else 128,
        "local_epochs_per_round": 1, "learning_rate": LEARNING_RATE, "limit_hours": args.limit_hours,
    })
    deadline = monotonic() + 3600 * args.limit_hours
    add_vendor_paths()
    import_module("Dataset.utils")  # Load the vendored module before applying per-arm seeds.

    route_loaders, eval_loaders = (evaluation_loaders(arrays, role) for role in ("route", "eval"))
    first = args.output / f"round_{args.rounds:02}"
    second = args.output / f"round_{final_rounds:02}"
    first.mkdir()
    stage = first
    completed = []
    write_json(args.output / "status.json", {"state": "running", "stage": stage.name})
    try:
        run_matrix(first, arrays, seeds, args, deadline, route_loaders, eval_loaders)
        completed.append(args.rounds)
        stage = second
        args.rounds = final_rounds
        second.mkdir()
        write_json(args.output / "status.json", {
            "state": "running", "stage": stage.name, "completed_milestones": completed,
        })
        run_matrix(second, arrays, seeds, args, deadline, route_loaders, eval_loaders, previous=first)
        completed.append(final_rounds)
    except DeadlineReached:
        summarize(stage, arrays, seeds, args.smoke)
        write_json(args.output / "status.json", {
            "state": "budget_stopped", "stage": stage.name, "completed_milestones": completed,
            "formal_evidence": False,
        })
        return
    except Exception as exc:
        write_json(args.output / "status.json", {
            "state": "failed", "stage": stage.name, "completed_milestones": completed,
            "error": str(exc), "formal_evidence": False,
        })
        raise
    write_json(args.output / "status.json", {
        "state": "complete", "completed_milestones": completed, "formal_evidence": False,
    })


if __name__ == "__main__":
    main()
