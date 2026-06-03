from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_vendor_paths(root: Path | None = None) -> None:
    root = root or project_root()
    for rel in ("RAHFL-master", "PRIME-augmentations-main"):
        path = str(root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)

