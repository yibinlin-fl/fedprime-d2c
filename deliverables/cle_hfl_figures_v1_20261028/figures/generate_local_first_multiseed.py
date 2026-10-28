"""Figure 3: matched Local/HFL effects and the communication difference.

Adapted from Vivid Figures advanced.dumbbell (registered source in .vivid/).
Retains paired endpoints, connector arrows, and per-row change information.
True training seeds replace demo metrics; fabricated stars are removed.

Native width 7.15 in; intended manuscript width about 6.7 in. At 0.94x,
9-pt tick labels become 8.46 pt. No manuscript file is changed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / "deliverables/cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv"
INK, MUTED, RULE = "#20252B", "#66727D", "#CBD2D8"
TEAL, RUST, PURPLE = "#287C78", "#B85C38", "#6F5A7E"


def read_data() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seeds = [row for row in rows if row["train_seed"] in {"0", "1", "2"}]
    seeds.sort(key=lambda row: int(row["train_seed"]))
    if [row["train_seed"] for row in seeds] != ["0", "1", "2"]:
        raise ValueError("Expected exactly seeds 0, 1, 2")
    hfl = np.array([float(row["delta_hfl"]) for row in seeds])
    local = np.array([float(row["delta_local"]) for row in seeds])
    comm = np.array([float(row["delta_comm"]) for row in seeds])
    np.testing.assert_allclose(hfl - local, comm, atol=2e-10, rtol=0)
    for mode, calculation in [("mean", np.mean), ("sample_sd", lambda x: np.std(x, ddof=1))]:
        reference = next(row for row in rows if row["train_seed"] == mode)
        for field, vector in [("delta_hfl", hfl), ("delta_local", local), ("delta_comm", comm)]:
            np.testing.assert_allclose(calculation(vector), float(reference[field]), atol=2e-10, rtol=0)
    if not np.isfinite(np.concatenate([hfl, local, comm])).all():
        raise ValueError("Nonfinite effect")
    return seeds, hfl, local, comm


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "mathtext.fontset": "dejavusans", "axes.labelsize": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.edgecolor": RULE, "axes.linewidth": 0.7,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="v1")
    args = parser.parse_args()
    if not args.revision.startswith("v") or not args.revision[1:].isdigit():
        parser.error("revision must be v followed by a number")
    stem = HERE / f"fig_local_first_multiseed_{args.revision}"
    targets = [stem.with_suffix(ext) for ext in [".pdf", ".svg", ".png", ".json", ".csv"]]
    if any(path.exists() for path in targets):
        raise FileExistsError("Preserving existing revision; select a fresh --revision vN")

    seeds, hfl, local, comm = read_data()
    setup_style()
    before, after = local, hfl  # Template endpoints; roles are explicitly labelled below.
    metrics = [f"Seed {row['train_seed']}" for row in seeds]
    y = np.arange(len(metrics))[::-1]
    fig = plt.figure(figsize=(7.15, 3.38))
    ax = fig.add_axes([0.100, 0.300, 0.455, 0.470])
    right = fig.add_axes([0.692, 0.300, 0.258, 0.470], sharey=ax)

    # Original dumbbell structure: paired endpoints + connectors + directional
    # arrows. A separate zero-centred panel shows the exact signed differences.
    ax.grid(axis="x", alpha=0.6, linestyle="-", color="#E5E9EC", lw=0.55)
    ax.set_axisbelow(True)
    for i in range(len(metrics)):
        delta = after[i] - before[i]
        ax.plot([before[i], after[i]], [y[i], y[i]], color=RULE, linewidth=1.6, solid_capstyle="round")
        ax.annotate("", xy=(after[i], y[i]), xytext=(before[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, shrinkA=3.5, shrinkB=3.5), zorder=2)
        ax.scatter(before[i], y[i], color=TEAL, marker="s", s=38, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.scatter(after[i], y[i], color=RUST, marker="o", s=40, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(f"{before[i]:.4f}", (before[i], y[i]), xytext=(0, 8),
                    textcoords="offset points", color=TEAL, fontsize=8, ha="center", va="bottom")
        ax.annotate(f"{after[i]:.4f}", (after[i], y[i]), xytext=(0, -8),
                    textcoords="offset points", color=RUST, fontsize=8, ha="center", va="top")

        # Signed differences are measured in DSA probability units, not accuracy.
        right.plot([0, delta], [y[i], y[i]], color=RULE, lw=1.4, zorder=1)
        right.scatter(delta, y[i], color=PURPLE, s=41, marker="D", edgecolors="white", linewidths=0.8, zorder=3)
        offset = 7 if delta >= 0 else -7
        right.annotate(f"{delta:+.4f}", (delta, y[i]), xytext=(offset, 0),
                       textcoords="offset points", ha="left" if delta >= 0 else "right",
                       va="center", fontsize=8.4, color=PURPLE)

    ax.set_yticks(y, metrics)
    ax.set_ylim(-0.65, 2.65)
    ax.set_xlim(0, 0.16)
    ax.set_xticks([0, 0.04, 0.08, 0.12, 0.16])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-9 else f"{value:.2f}"))
    ax.set_xlabel("CLE effect on DSA", labelpad=7)
    right.set_xlim(-0.0205, 0.023)
    right.set_xticks([-0.015, 0, 0.015])
    right.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-9 else f"{value:+.3f}"))
    right.axvline(0, color=MUTED, lw=0.9, linestyle=(0, (3, 3)), zorder=0)
    right.tick_params(axis="y", left=False, labelleft=False)
    right.set_xlabel(r"$\Delta_{\rm Comm}=\Delta_{\rm HFL}-\Delta_{\rm Local}$", labelpad=7)
    for axis in (ax, right):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=7)
        axis.tick_params(axis="x", length=3, labelsize=8)

    fig.text(0.065, 0.938, "a", fontsize=11, weight="bold", color=TEAL)
    fig.text(0.100, 0.938, "Local and HFL effects", fontsize=10, weight="bold")
    fig.text(0.653, 0.938, "b", fontsize=11, weight="bold", color=PURPLE)
    fig.text(0.692, 0.938, "Communication difference", fontsize=10, weight="bold")
    fig.text(0.100, 0.865, r"$\Delta=\mathrm{DSA}_{\gamma=0.9}-\mathrm{DSA}_{\gamma=0}$", color=MUTED, fontsize=8.3)
    fig.text(0.692, 0.865, "Zero-centred; magnified scale", color=MUTED, fontsize=8.0)
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TEAL, markeredgecolor="white", markersize=6, label="Local"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RUST, markeredgecolor="white", markersize=6, label="HFL"),
    ]
    fig.legend(handles=legend, loc="center", bbox_to_anchor=(0.470, 0.865), frameon=False,
               ncol=2, fontsize=8, columnspacing=0.8, handletextpad=0.3, handlelength=0.9)

    # Summary uses sample SD across three training seeds, never a source CI.
    fig.add_artist(Line2D([0.065, 0.965], [0.160, 0.160], transform=fig.transFigure, color=RULE, lw=0.7))
    fig.text(0.065, 0.105, "Across 3 training seeds", color=INK, fontsize=7.8, weight="bold")
    fig.text(0.065, 0.052, "mean ± sample SD", color=MUTED, fontsize=7.6)
    for x, title, values, color in [
        (0.330, "Local", local, TEAL), (0.555, "HFL", hfl, RUST), (0.780, "Comm.", comm, PURPLE)
    ]:
        fig.text(x, 0.105, title, color=color, weight="bold", fontsize=8)
        fig.text(x, 0.052, f"{values.mean():.5f} ± {values.std(ddof=1):.5f}", color=INK, fontsize=7.7)

    fig.savefig(targets[0], metadata={"Title": "Local-first CLE effects across matched training seeds", "CreationDate": None, "ModDate": None})
    fig.savefig(targets[1], metadata={"Date": None})
    fig.savefig(targets[2], dpi=300)
    plt.close(fig)
    with targets[4].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seeds[0]))
        writer.writeheader()
        writer.writerows(seeds)
    provenance = {
        "figure": stem.name, "source": SOURCE.relative_to(REPO).as_posix(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "template": "advanced.dumbbell", "training_seeds": [0, 1, 2],
        "units": "DSA probability scale", "summary": "mean and sample SD across 3 training seeds",
        "uncertainty_bars": "none; source-bootstrap CIs are not used as training-seed CIs",
        "protocol": "12 rounds x 16 local batches; batch 64; fixed seed0_split0",
        "seed_2_original_verdict": seeds[2]["verdict"],
        "claim_boundary": "Matched factorial contrasts; not a sample-level mediation decomposition or communication equivalence test",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": mpl.__version__},
        "outputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in targets if path != targets[3]},
    }
    targets[3].write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for target in targets:
        print(f"{target.name}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
