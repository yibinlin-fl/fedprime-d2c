from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a prepared CLE-HFL v2 dataset.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    operator_to_id = {
        str(name): int(value)
        for name, value in metadata["operator_to_id"].items()
    }
    seen = set(str(name) for name in metadata["seen_operators"])
    unseen = set(str(name) for name in metadata["unseen_operators"])
    if seen.intersection(unseen):
        raise ValueError("seen and unseen operators overlap")
    unseen_ids = {operator_to_id[name] for name in unseen}

    per_client = []
    total_matches = 0
    total_samples = 0
    class_map = metadata["class_operator_map"]
    for client_id in range(int(metadata["num_clients"])):
        client_root = root / f"client_{client_id}"
        labels = np.load(client_root / "train_labels.npy").astype(np.int64)
        operator_ids = np.load(
            client_root / "train_corruption_ids.npy"
        ).astype(np.int64)
        leaked = int(np.isin(operator_ids, list(unseen_ids)).sum())
        if leaked:
            raise ValueError(
                f"client {client_id} contains {leaked} unseen training operators"
            )
        matches = 0
        for class_id in range(int(metadata["num_classes"])):
            dominant = str(class_map[str(client_id)][str(class_id)])
            dominant_id = operator_to_id[dominant]
            matches += int(
                ((labels == class_id) & (operator_ids == dominant_id)).sum()
            )
        realized = matches / max(int(labels.size), 1)
        per_client.append(
            {
                "client": client_id,
                "samples": int(labels.size),
                "dominant_match_rate": realized,
                "unseen_train_samples": leaked,
            }
        )
        total_matches += matches
        total_samples += int(labels.size)

    split_counts = {}
    for split in ("test_seen", "test_unseen", "test_balanced", "test_clean"):
        split_root = root / split
        labels = np.load(split_root / "test_labels.npy").astype(np.int64)
        operator_ids = np.load(
            split_root / "test_corruption_ids.npy"
        ).astype(np.int64)
        split_counts[split] = {
            "samples": int(labels.size),
            "classes": np.bincount(
                labels,
                minlength=int(metadata["num_classes"]),
            ).astype(int).tolist(),
            "operator_ids": sorted(int(value) for value in np.unique(operator_ids)),
        }

    seen_ids = {operator_to_id[name] for name in seen}
    if set(split_counts["test_seen"]["operator_ids"]) != seen_ids:
        raise ValueError("test_seen operator IDs do not match metadata")
    if set(split_counts["test_unseen"]["operator_ids"]) != unseen_ids:
        raise ValueError("test_unseen operator IDs do not match metadata")

    gamma = float(metadata["gamma"])
    expected_dominant_rate = gamma + (1.0 - gamma) / len(seen)
    report = {
        "root": str(root),
        "protocol_version": int(metadata["protocol_version"]),
        "seen_operators": sorted(seen),
        "unseen_operators": sorted(unseen),
        "expected_dominant_match_rate": expected_dominant_rate,
        "realized_dominant_match_rate": total_matches / max(total_samples, 1),
        "per_client": per_client,
        "test_splits": split_counts,
        "passed": True,
    }
    output = root / "audit" / "protocol_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote protocol audit: {output}")


if __name__ == "__main__":
    main()
