from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def summarize_experiment(exp_dir: Path) -> dict[str, object] | None:
    metrics_path = exp_dir / "metrics.csv"
    config_path = exp_dir / "config.resolved.json"
    if not metrics_path.exists():
        return None

    rows = read_metrics(metrics_path)
    if not rows:
        return None

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    final = rows[-1]
    best_avg = max(rows, key=lambda r: as_float(r, "avg_acc"))
    best_worst = max(rows, key=lambda r: as_float(r, "worst_acc"))

    return {
        "experiment": exp_dir.name,
        "method": config.get("method_name", ""),
        "seed": config.get("seed", ""),
        "alpha": config.get("data", {}).get("dirichlet_alpha", ""),
        "corrupt_rate": config.get("data", {}).get("private_corrupt_rate", ""),
        "final_avg_acc": f"{as_float(final, 'avg_acc'):.4f}",
        "final_worst_acc": f"{as_float(final, 'worst_acc'):.4f}",
        "best_avg_acc": f"{as_float(best_avg, 'avg_acc'):.4f}",
        "best_avg_round": best_avg.get("round", ""),
        "best_worst_acc": f"{as_float(best_worst, 'worst_acc'):.4f}",
        "best_worst_round": best_worst.get("round", ""),
    }


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("No experiments found.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FedPRIME experiment metrics.")
    parser.add_argument("--outputs", default="outputs", help="Root output directory.")
    parser.add_argument("--out_csv", default="outputs/summary.csv")
    parser.add_argument("--out_md", default="outputs/summary.md")
    args = parser.parse_args()

    output_root = Path(args.outputs)
    summaries = []
    for exp_dir in sorted(output_root.glob("*")):
        if exp_dir.is_dir():
            summary = summarize_experiment(exp_dir)
            if summary is not None:
                summaries.append(summary)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if summaries:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    else:
        out_csv.write_text("", encoding="utf-8")

    write_markdown(summaries, Path(args.out_md))
    print(f"Wrote {len(summaries)} summaries to {out_csv}")


if __name__ == "__main__":
    main()

