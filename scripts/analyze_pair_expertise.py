from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FedPRIME-PAIR class-pair expertise snapshots.")
    parser.add_argument("--experiment_dir", required=True, help="Path to outputs/<experiment_name>.")
    parser.add_argument("--round", type=int, default=None, help="Round index to analyze. Defaults to latest snapshot.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    pair_dir = experiment_dir / "pair_expertise"
    snapshots = sorted(pair_dir.glob("round_*.npz"))
    if not snapshots:
        raise FileNotFoundError(f"No pair expertise snapshots found under {pair_dir}.")
    if args.round is None:
        snapshot = snapshots[-1]
    else:
        snapshot = pair_dir / f"round_{args.round:03d}.npz"
        if not snapshot.exists():
            raise FileNotFoundError(snapshot)

    with np.load(snapshot) as data:
        expertise = data["expertise"]
        expertise_raw = data["expertise_raw"]
        counts = data["counts"]

    out_dir = experiment_dir / "pair_expertise_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    round_name = snapshot.stem

    csv_path = out_dir / f"{round_name}_pair_expertise.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["client", "source_class", "target_class", "expertise", "raw_expertise", "source_count"],
        )
        writer.writeheader()
        for client_id in range(expertise.shape[0]):
            for source_class in range(expertise.shape[1]):
                for target_class in range(expertise.shape[2]):
                    if source_class == target_class:
                        continue
                    writer.writerow({
                        "client": client_id,
                        "source_class": source_class,
                        "target_class": target_class,
                        "expertise": float(expertise[client_id, source_class, target_class]),
                        "raw_expertise": float(expertise_raw[client_id, source_class, target_class]),
                        "source_count": float(counts[client_id, source_class]),
                    })

    fig, axes = plt.subplots(1, expertise.shape[0], figsize=(4 * expertise.shape[0], 4), constrained_layout=True)
    if expertise.shape[0] == 1:
        axes = [axes]
    vmax = float(max(expertise.max(), 1e-6))
    for client_id, ax in enumerate(axes):
        im = ax.imshow(expertise[client_id], vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_title(f"client {client_id}")
        ax.set_xlabel("target class j")
        ax.set_ylabel("source class c")
        ax.set_xticks(range(expertise.shape[1]))
        ax.set_yticks(range(expertise.shape[1]))
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out_dir / f"{round_name}_pair_expertise.png", dpi=180)
    plt.close(fig)

    print(f"Analyzed {snapshot}")
    print(f"CSV: {csv_path}")
    print(f"Heatmap: {out_dir / f'{round_name}_pair_expertise.png'}")


if __name__ == "__main__":
    main()
