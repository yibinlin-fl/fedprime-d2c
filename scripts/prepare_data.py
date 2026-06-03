from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
import torchvision.datasets as datasets
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.utils.config import load_config


def format_rate(rate: float) -> str:
    value = float(rate)
    if value.is_integer():
        return str(int(value))
    return str(value)


def gaussian_noise(image: Image.Image, severity: int) -> Image.Image:
    scales = [0.04, 0.06, 0.08, 0.09, 0.10]
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.clip(arr + np.random.normal(scale=scales[severity - 1], size=arr.shape), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def blur(image: Image.Image, severity: int) -> Image.Image:
    radii = [0.4, 0.6, 0.8, 1.0, 1.2]
    return image.filter(ImageFilter.GaussianBlur(radius=radii[severity - 1]))


def brightness(image: Image.Image, severity: int) -> Image.Image:
    factors = [1.05, 1.12, 1.20, 1.30, 1.40]
    return ImageEnhance.Brightness(image).enhance(factors[severity - 1])


def contrast(image: Image.Image, severity: int) -> Image.Image:
    factors = [0.75, 0.60, 0.45, 0.35, 0.25]
    return ImageEnhance.Contrast(image).enhance(factors[severity - 1])


def pixelate(image: Image.Image, severity: int) -> Image.Image:
    factors = [0.95, 0.90, 0.85, 0.75, 0.65]
    size = max(1, int(32 * factors[severity - 1]))
    return image.resize((size, size), Image.Resampling.BOX).resize((32, 32), Image.Resampling.BOX)


def jpeg(image: Image.Image, severity: int) -> Image.Image:
    qualities = [80, 65, 58, 50, 40]
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=qualities[severity - 1])
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


CORRUPTIONS = [gaussian_noise, blur, brightness, contrast, pixelate, jpeg]


def corrupt_image(image: Image.Image) -> Image.Image:
    fn = random.choice(CORRUPTIONS)
    severity = random.randint(1, 5)
    return fn(image, severity)


def build_random_corrupt_split(images: np.ndarray, labels: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    output = images.copy()
    n_corrupt = int(round(float(rate) * len(images)))
    if n_corrupt > 0:
        indices = rng.choice(np.arange(len(images)), size=n_corrupt, replace=False)
        for index in indices:
            image = Image.fromarray(images[index].astype(np.uint8))
            output[index] = np.asarray(corrupt_image(image)).astype(np.uint8)
    return output.astype(np.uint8), labels.astype(np.uint8)


def prepare_cifar100(public_root: Path, download: bool) -> None:
    public_root.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR100(str(public_root), train=True, download=download)
    datasets.CIFAR100(str(public_root), train=False, download=download)


def prepare_random_cifar10c(private_root: Path, clean_root: Path, rates: list[float], download: bool, seed: int) -> None:
    clean_root.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.CIFAR10(str(clean_root), train=True, download=download)
    test_ds = datasets.CIFAR10(str(clean_root), train=False, download=download)

    private_root.mkdir(parents=True, exist_ok=True)
    for split_name, ds in (("train", train_ds), ("test", test_ds)):
        split_dir = private_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        images = np.asarray(ds.data).astype(np.uint8)
        labels = np.asarray(ds.targets).astype(np.uint8)
        np.save(split_dir / "labels.npy", labels)
        for rate in rates:
            corrupt, out_labels = build_random_corrupt_split(
                images=images,
                labels=labels,
                rate=rate,
                seed=seed + (0 if split_name == "train" else 10000) + int(float(rate) * 1000),
            )
            np.save(split_dir / f"random_corrupt_{format_rate(rate)}.npy", corrupt)
            np.save(split_dir / "labels.npy", out_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RAHFL-style CIFAR data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10/CIFAR-100 if missing.")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--clean_cifar10_root", help="Where clean CIFAR-10 should be stored.")
    args = parser.parse_args()

    config = load_config(args.config)
    private_root = Path(config["data"]["private_root"])
    public_root = Path(config["data"]["public_root"])
    clean_root = Path(args.clean_cifar10_root) if args.clean_cifar10_root else private_root.parent / "cifar_10"
    seed = int(config.get("seed", 0))

    print(f"Preparing CIFAR-100 public data at {public_root}")
    prepare_cifar100(public_root, download=args.download)
    print(f"Preparing RAHFL-style CIFAR-10-C private data at {private_root}")
    prepare_random_cifar10c(private_root, clean_root, args.rates, args.download, seed)
    print("Data preparation finished.")


if __name__ == "__main__":
    main()

