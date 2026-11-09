"""Generate the fixed ordinary-corruption input from CIFAR training splits."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

from fedprime.data.corruptions import apply_corruption


LAYOUTS = {"uniform": (5, 5, 5, 5), "concentrated_a": (5, 5, 2, 8), "concentrated_b": (5, 5, 8, 2)}
OPERATORS = ("gaussian_noise", "defocus_blur")
SEVERITY = 3
DATA_SEED = 20261110


def digest(*arrays):
    value = sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        value.update(f"{array.dtype}:{array.shape}\n".encode())
        value.update(array.tobytes())
    return value.hexdigest()


def render(images, ids, operators):
    result = images.copy()
    for index in np.flatnonzero(operators):
        source = int(ids[index])
        operator = int(operators[index])
        rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, source, operator]))
        result[index] = apply_corruption(images[index], OPERATORS[operator - 1], SEVERITY, rng)
    return result


def make_input(cifar_root, smoke):
    private = CIFAR10(str(cifar_root / "cifar_10"), train=True)
    public = CIFAR100(str(cifar_root / "cifar_100"), train=True)
    per_class, held_per_class, public_count = (20, 2, 32) if smoke else (200, 100, 1000)
    rng = np.random.default_rng(DATA_SEED)
    labels = np.asarray(private.targets, dtype=np.int64)
    class_ids = np.stack([rng.permutation(np.flatnonzero(labels == label)) for label in range(10)])
    fit_ids = class_ids[:, :4 * per_class].reshape(10, 4, per_class).transpose(1, 0, 2)
    route_ids = class_ids[:, 4 * per_class:4 * per_class + held_per_class].flatten()
    eval_ids = class_ids[:, 4 * per_class + held_per_class:4 * per_class + 2 * held_per_class].flatten()
    public_ids = rng.choice(len(public), size=public_count, replace=False)
    rank = np.argsort(rng.random(fit_ids.shape), axis=-1)
    arrays = {
        "fit_ids": fit_ids.reshape(4, -1),
        "fit_images": private.data[fit_ids.reshape(4, -1)],
        "fit_labels": labels[fit_ids.reshape(4, -1)],
        "route_ids": route_ids, "route_images": private.data[route_ids], "route_labels": labels[route_ids],
        "eval_ids": eval_ids, "eval_images": private.data[eval_ids], "eval_labels": labels[eval_ids],
        "public_ids": public_ids, "public_images": public.data[public_ids],
    }
    for layout, quotas in LAYOUTS.items():
        ops = np.zeros(fit_ids.shape, dtype=np.uint8)
        for client, tenths in enumerate(quotas):
            count = per_class * tenths // 10
            ops[client] = np.where(rank[client] < count, rank[client] % 2 + 1, 0)
        arrays[layout] = ops.reshape(4, -1)
    audit(arrays, smoke)
    return arrays


def audit(arrays, smoke):
    per_client, held_count, public_count = (200, 20, 32) if smoke else (2000, 1000, 1000)
    fit_ids, fit_labels = arrays["fit_ids"], arrays["fit_labels"]
    ids = np.concatenate([fit_ids.flatten(), arrays["route_ids"], arrays["eval_ids"]])
    if (fit_ids.shape != (4, per_client) or fit_labels.shape != fit_ids.shape
            or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= 50000
            or any(len(arrays[f"{role}_ids"]) != held_count for role in ("route", "eval"))
            or len(np.unique(arrays["public_ids"])) != public_count):
        raise ValueError("Input size or disjointness audit failed")
    if not np.all([np.bincount(labels, minlength=10) == per_client // 10 for labels in fit_labels]):
        raise ValueError("Fit labels must be balanced within each client")
    for layout, quotas in LAYOUTS.items():
        ops = arrays[layout]
        for client, tenths in enumerate(quotas):
            per_class_bad = per_client // 10 * tenths // 10
            for label in range(10):
                counts = np.bincount(ops[client, fit_labels[client] == label], minlength=3)
                if not np.array_equal(counts, [per_client // 10 - per_class_bad, per_class_bad // 2, per_class_bad // 2]):
                    raise ValueError(f"Corruption quotas failed for {layout}, client {client}, class {label}")
        if not np.array_equal(ops[:2], arrays["uniform"][:2]):
            raise ValueError(f"Fixed-client input changed for {layout}")
    return {
        "data_seed": DATA_SEED, "severity": SEVERITY, "operators": OPERATORS,
        "fit_images_per_client": per_client, "total_bad_images": 2 * per_client,
        "fixed_client0_source_hash": digest(fit_ids[0], fit_labels[0], arrays["fit_images"][0], arrays["uniform"][0]),
        "input_array_hash": digest(*(arrays[name] for name in sorted(arrays))),
        "bad_by_layout": {layout: np.count_nonzero(arrays[layout], axis=1).tolist() for layout in LAYOUTS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target:
        np.savez_compressed(target, **make_input(args.cifar_root, args.smoke))
    print(args.output)


if __name__ == "__main__":
    main()
