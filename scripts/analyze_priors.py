from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_round_summary(rows: list[dict[str, str]], path: Path) -> None:
    metric_names = [
        "prior_l1",
        "prior_kl",
        "prior_cosine_similarity",
        "predicted_entropy",
        "oracle_entropy",
        "predicted_normalized_entropy",
        "oracle_normalized_entropy",
        "top_class_match",
    ]
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["round"])].append(row)

    output_rows = []
    for round_idx, round_rows in sorted(grouped.items()):
        output = {"round": round_idx, "records": len(round_rows)}
        for name in metric_names:
            values = np.asarray([float(row[name]) for row in round_rows])
            output[f"{name}_mean"] = float(values.mean())
            output[f"{name}_std"] = float(values.std())
        output_rows.append(output)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)


def plot_round_curves(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["round"])].append(row)

    rounds = sorted(grouped)
    l1 = [np.mean([float(row["prior_l1"]) for row in grouped[r]]) for r in rounds]
    kl = [np.mean([float(row["prior_kl"]) for row in grouped[r]]) for r in rounds]
    cosine = [
        np.mean([float(row["prior_cosine_similarity"]) for row in grouped[r]])
        for r in rounds
    ]
    top_match = [
        np.mean([float(row["top_class_match"]) for row in grouped[r]])
        for r in rounds
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, values, title in zip(
        axes.flat,
        [l1, kl, cosine, top_match],
        ["Prior L1 (lower is better)", "Prior KL (lower is better)",
         "Cosine similarity (higher is better)", "Top-class match (higher is better)"],
    ):
        ax.plot(rounds, values, marker="o", markersize=3)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Round")
    axes[1, 1].set_xlabel("Round")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_heatmap(values: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(3, values.shape[0] * 0.7)))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(values.max()), 1e-8),
    )
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze predicted and oracle D2C priors.")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    diagnostics_path = experiment_dir / "prior_diagnostics.csv"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing prior diagnostics: {diagnostics_path}")

    rows = read_rows(diagnostics_path)
    if not rows:
        raise ValueError(f"No prior diagnostic rows found in {diagnostics_path}")

    out_dir = Path(args.out_dir) if args.out_dir else experiment_dir / "prior_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_round_summary(rows, out_dir / "prior_metrics_by_round.csv")
    plot_round_curves(rows, out_dir / "prior_error_by_round.png")

    snapshots = sorted((experiment_dir / "priors").glob("round_*.npz"))
    if snapshots:
        latest = snapshots[-1]
        data = np.load(latest)
        plot_heatmap(
            data["oracle_prior"],
            f"Oracle Prior ({latest.stem})",
            out_dir / "prior_heatmap_oracle.png",
        )
        plot_heatmap(
            data["predicted_prior"],
            f"Predicted Prior ({latest.stem})",
            out_dir / "prior_heatmap_predicted.png",
        )
        plot_heatmap(
            np.abs(data["predicted_prior"] - data["oracle_prior"]),
            f"Absolute Prior Error ({latest.stem})",
            out_dir / "prior_heatmap_absolute_error.png",
        )

    report = {
        "experiment_dir": str(experiment_dir),
        "records": len(rows),
        "rounds": sorted({int(row["round"]) for row in rows}),
        "latest_snapshot": str(snapshots[-1]) if snapshots else None,
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote prior analysis to {out_dir}")


if __name__ == "__main__":
    main()
