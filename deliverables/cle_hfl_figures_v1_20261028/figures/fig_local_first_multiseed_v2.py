"""Figure 3 refined: paired Local/HFL effects and signed communication contrast.

Adapted from Vivid Figures advanced.dumbbell (registered source in .vivid/).
Adapted in place from generate_local_first_multiseed.py; v1 is preserved.
Retains paired endpoints and a separate signed-difference panel. The arbitrary
seed-row coordinate separates symbols; x-values are never shifted or jittered.

Native width 7.06 in matches the present manuscript text block. The large
summary strip moves to the caption/report; exact mean/sample SD stay in JSON.
No manuscript file is changed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import os
from pathlib import Path

os.environ['PATH'] = 'D:/texlive/2026/bin/windows' + os.pathsep + os.environ.get('PATH', '')
os.environ['MPLCONFIGDIR'] = str(Path(__file__).resolve().parents[3] / 'local_runs' / 'mplconfig_figure3_refined')

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / "deliverables/cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv"
INK, MUTED, RULE = "#20252B", "#66727D", "#CBD2D8"
ACCENT = "#2F6A7F"


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
        "font.family": "serif", "font.serif": ["Computer Modern Roman"], "font.size": 8,
        "text.usetex": True, "text.latex.preamble": r"\usepackage{lmodern}",
        "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "axes.edgecolor": RULE, "axes.linewidth": 0.7,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="v2")
    args = parser.parse_args()
    if not args.revision.startswith("v") or not args.revision[1:].isdigit():
        parser.error("revision must be v followed by a number")
    stem = HERE / f"fig_local_first_multiseed_{args.revision}"
    targets = [stem.with_suffix(ext) for ext in [".pdf", ".svg", ".png", ".json", ".csv", ".py"]]
    if any(path.exists() for path in targets):
        raise FileExistsError("Preserving existing revision; select a fresh --revision vN")

    seeds, hfl, local, comm = read_data()
    setup_style()
    before, after = local, hfl  # Matched conditions, not a temporal transition.
    metrics = [f"Seed {row['train_seed']}" for row in seeds]
    y = np.arange(len(metrics))[::-1]
    fig = plt.figure(figsize=(7.06, 2.55))
    ax = fig.add_axes([0.095, 0.255, 0.475, 0.525])
    right = fig.add_axes([0.725, 0.255, 0.235, 0.525], sharey=ax)

    # Plain connectors pair the conditions without implying a time evolution.
    # A small within-seed vertical separation prevents coincident markers.
    ax.grid(axis="x", linestyle="-", color="#E8EBED", lw=0.45)
    ax.set_axisbelow(True)
    for i in range(len(metrics)):
        delta = after[i] - before[i]
        local_y, hfl_y = y[i] + 0.09, y[i] - 0.09
        ax.plot([before[i], after[i]], [local_y, hfl_y], color="#8B949C", linewidth=0.8, zorder=2)
        ax.scatter(before[i], local_y, facecolors="white", edgecolors=INK, marker="s", s=24, zorder=4, linewidths=0.8)
        ax.scatter(after[i], hfl_y, color=ACCENT, marker="o", s=25, zorder=4, edgecolors="white", linewidths=0.6)
        ax.annotate(f"{before[i]:.4f}", (before[i], local_y), xytext=(0, 5),
                    textcoords="offset points", color=INK, fontsize=7.4, ha="center", va="bottom")
        ax.annotate(f"{after[i]:.4f}", (after[i], hfl_y), xytext=(0, -5),
                    textcoords="offset points", color=INK, fontsize=7.4, ha="center", va="top")

        # Signed differences are measured in DSA probability units, not accuracy.
        right.plot([0, delta], [y[i], y[i]], color="#8B949C", lw=0.8, zorder=1)
        right.scatter(delta, y[i], color=INK, s=22, marker="D", edgecolors="white", linewidths=0.5, zorder=3)
        offset = 5 if delta >= 0 else -5
        right.annotate(f"{delta:+.4f}", (delta, y[i]), xytext=(offset, 0),
                       textcoords="offset points", ha="left" if delta >= 0 else "right",
                       va="center", fontsize=7.8, color=INK)

    ax.set_yticks(y, metrics)
    ax.set_ylim(-0.56, 2.56)
    ax.set_xlim(0, 0.16)
    ax.set_xticks([0, 0.04, 0.08, 0.12, 0.16])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-9 else f"{value:.2f}"))
    ax.set_xlabel("CLE effect on DSA", labelpad=5)
    right.set_xlim(-0.0205, 0.023)
    right.set_xticks([-0.015, 0, 0.015])
    right.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-9 else f"{value:+.3f}"))
    right.axvline(0, color=MUTED, lw=0.9, linestyle=(0, (3, 3)), zorder=0)
    right.tick_params(axis="y", left=False, labelleft=False)
    right.set_xlabel(r"$\Delta_{\rm Comm}=\Delta_{\rm HFL}-\Delta_{\rm Local}$", labelpad=5)
    for axis in (ax, right):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=7)
        axis.tick_params(axis="x", length=2.5, labelsize=7.5)

    fig.text(0.095, 0.935, r"\textbf{(a) CLE effects}", fontsize=8.5)
    fig.text(0.725, 0.935, r"\textbf{(b) Communication contrast}", fontsize=8.5)
    fig.text(0.095, 0.840, r"$\Delta=\mathrm{DSA}_{\gamma=0.9}-\mathrm{DSA}_{\gamma=0}$", color=INK, fontsize=7.5)
    fig.text(0.725, 0.840, "Magnified x-scale", color=INK, fontsize=7.5)
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=INK, markersize=4.8, label="Local"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ACCENT, markeredgecolor="white", markersize=5, label="HFL"),
    ]
    fig.legend(handles=legend, loc="center", bbox_to_anchor=(0.470, 0.947), frameon=False,
               ncol=2, fontsize=7.8, columnspacing=0.8, handletextpad=0.3, handlelength=0.9)

    # Exact mean/sample SD are retained in provenance and the caption/report,
    # rather than duplicating the results table in a large in-figure strip.

    fig.savefig(targets[0], metadata={"Title": "Local-first CLE effects across matched training seeds", "CreationDate": None, "ModDate": None})
    fig.savefig(targets[1], metadata={"Date": None})
    fig.savefig(targets[2], dpi=300)
    plotted_coordinates = {
        "local_x": before.tolist(), "hfl_x": after.tolist(), "comm_x": (after-before).tolist(),
        "local_y": (y+0.09).tolist(), "hfl_y": (y-0.09).tolist(), "comm_y": y.tolist(),
        "left_xlim": list(ax.get_xlim()), "right_xlim": list(right.get_xlim()),
    }
    plt.close(fig)
    with targets[4].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seeds[0]))
        writer.writeheader()
        writer.writerows(seeds)
    targets[5].write_bytes(Path(__file__).read_bytes())
    provenance = {
        "figure": stem.name, "source": SOURCE.relative_to(REPO).as_posix(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "template": "advanced.dumbbell", "training_seeds": [0, 1, 2],
        "units": "DSA probability scale", "summary": "mean and sample SD across 3 training seeds",
        "uncertainty_bars": "none; source-bootstrap CIs are not used as training-seed CIs",
        "summary_location": "caption/report and JSON; removed decorative in-figure summary strip",
        "summary_statistics": {name: {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
                               for name, values in [("local", local), ("hfl", hfl), ("comm", comm)]},
        "plot_coordinates": plotted_coordinates,
        "within_seed_vertical_separation": "categorical layout only, not an altered measurement or jittered DSA",
        "axis_ranges_unchanged_from_v1": True,
        "font": "TeX Latin Modern; matched to the TikZ diagram candidates",
        "parent_generator_sha256": hashlib.sha256((HERE / "generate_local_first_multiseed.py").read_bytes()).hexdigest(),
        "template_baseline_sha256": hashlib.sha256((HERE.parent / ".vivid/template-sources/c81322a77c40cac07067eb77.py").read_bytes()).hexdigest(),
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
