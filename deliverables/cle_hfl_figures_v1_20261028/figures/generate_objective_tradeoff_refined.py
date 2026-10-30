"""Figure 4 refined: frozen Axis-I shortcut--utility over 3 training seeds.

Adapted from Vivid academic.error_analysis: retains the scatter comparison,
direct labels, white-edged summary markers. No parameter-size
measurement, efficiency model, or continuous Pareto frontier is available;
those unsupported layers are omitted. Actual seed points and sample-SD
crosses replace the demo bubbles. Adapted from the actual previous generator;
only typography, spacing, marker treatment and emphasis change. Native width
7.06 in. All 12 seed points and sample SD are retained. No manuscript edits.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from refined_data_style import ACCENT, NEUTRAL, setup_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_common import REPO, INK, MUTED, RULE, RUST, BLUE, arguments, export, output_stem, source_record, style

ARMS = ["erm", "cvar_dro", "pew_groupdro", "pew_ber"]
NAMES = ["ERM", "CVaR-DRO", "PEW+GroupDRO", "PEW+BER"]
COLORS = [NEUTRAL, NEUTRAL, NEUTRAL, ACCENT]
MARKERS = ["o", "^", "s", "D"]
SOURCES = [
    REPO / "outputs/openi_downloads/cle_v2_spurious_final_map2_formal/extracted/analysis/RESULT_SUMMARY.json",
    REPO / "outputs/openi_downloads/m1_map2_multiseed/extracted_seed1/analysis/RESULT_SUMMARY.json",
    REPO / "outputs/openi_downloads/m1_map2_multiseed/extracted_seed2/analysis/RESULT_SUMMARY.json",
]
AGGREGATE = REPO / "deliverables/cle_hfl_map2_multiseed_20260922/MULTISEED_SUMMARY.json"
REPORT = REPO / "deliverables/cle_hfl_map2_multiseed_20260922/RESULT_SUMMARY_ZH.md"


def read_data(csv_file: Path | None) -> tuple[list[dict], list[dict]]:
    rows = []
    if csv_file is not None:
        with csv_file.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"train_seed": int(row["train_seed"]), "arm": row["arm"],
                             "dsa": float(row["dsa"]), "grid_accuracy": float(row["grid_accuracy"])})
        sources = [source_record(csv_file)]
    else:
        for seed, path in enumerate(SOURCES):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["mode"] != "formal" or data["rounds"] != 40 or not data["scientific_evidence"]:
                raise ValueError(f"Not a 40-round Formal source: {path}")
            if not data["paired_local_traces"]["matched"]:
                raise ValueError(f"Unmatched source: {path}")
            if "train_seed" in data and data["train_seed"] != seed:
                raise ValueError("Source seed mismatch")
            for arm in ARMS:
                rows.append({"train_seed": seed, "arm": arm, "dsa": data["pooled_dsa"][arm],
                             "grid_accuracy": data["operator_grid_accuracy"][arm]["pooled"]})
        sources = [source_record(path) for path in SOURCES]

    if len(rows) != 12 or {(r["train_seed"], r["arm"]) for r in rows} != {(s, a) for s in range(3) for a in ARMS}:
        raise ValueError("Expected all 4 arms x 3 training seeds")
    if not np.isfinite([[r["dsa"], r["grid_accuracy"]] for r in rows]).all():
        raise ValueError("Nonfinite data")
    # Check exact paired differences against the frozen multi-seed aggregate.
    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    for seed in range(3):
        values = {r["arm"]: r for r in rows if r["train_seed"] == seed}
        actual = {
            "erm_minus_pew_ber_dsa": values["erm"]["dsa"] - values["pew_ber"]["dsa"],
            "pew_groupdro_minus_pew_ber_dsa": values["pew_groupdro"]["dsa"] - values["pew_ber"]["dsa"],
            "pew_ber_minus_cvar_dsa": values["pew_ber"]["dsa"] - values["cvar_dro"]["dsa"],
            "pew_ber_minus_cvar_operator_accuracy_pp": values["pew_ber"]["grid_accuracy"] - values["cvar_dro"]["grid_accuracy"],
        }
        for key, value in actual.items():
            np.testing.assert_allclose(value, aggregate["per_seed_key_deltas"][str(seed)][key], atol=1e-12, rtol=0)

    # Also check all plotted means and SDs against the rounded official table.
    table = {}
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [cell.strip().replace("**", "") for cell in line.split("|")[1:-1]]
            if cells and cells[0] in NAMES and len(cells) >= 3 and "+/-" in cells[1]:
                table[cells[0]] = cells
    for arm, name in zip(ARMS, NAMES):
        subset = sorted((r for r in rows if r["arm"] == arm), key=lambda row: row["train_seed"])
        for col, field, tolerance in [(1, "dsa", 5.1e-7), (2, "grid_accuracy", 5.1e-5)]:
            reference = [float(x.strip()) for x in table[name][col].split("+/-")]
            vector = np.array([r[field] for r in subset])
            np.testing.assert_allclose([vector.mean(), vector.std(ddof=1)], reference, atol=tolerance, rtol=0)
    sources += [source_record(AGGREGATE), source_record(REPORT)]
    return rows, sources


def main() -> None:
    args = arguments(__doc__)
    stem = output_stem("fig_objective_tradeoff", args.revision)
    rows, sources = read_data(args.data_csv)
    setup_style()
    fig, ax = plt.subplots(figsize=(7.06, 3.05))
    fig.subplots_adjust(left=0.100, right=0.970, bottom=0.250, top=0.935)
    offsets = [(-16, 20), (-17, -25), (25, 3), (20, 18)]
    statistics = {}
    # Scatter and direct annotations follow the registered source recipe.
    for i, arm in enumerate(ARMS):
        subset = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["train_seed"])
        acc = np.array([r["grid_accuracy"] for r in subset])
        dsa = np.array([r["dsa"] for r in subset])
        mx, my = acc.mean(), dsa.mean()
        sx, sy = acc.std(ddof=1), dsa.std(ddof=1)
        statistics[arm] = {"accuracy_mean": mx, "accuracy_sd": sx, "dsa_mean": my, "dsa_sd": sy}
        color, marker = COLORS[i], MARKERS[i]
        ax.scatter(acc, dsa, s=18, marker=marker, facecolors="white",
                   edgecolors=color, linewidth=0.75, zorder=3)
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="none", color=color,
                    elinewidth=0.8, capsize=2.0, zorder=2)
        ax.scatter(mx, my, s=46, marker=marker, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(NAMES[i], xy=(mx, my), xytext=offsets[i], textcoords="offset points",
                    fontsize=8.5, color=INK,
                    ha="left" if i >= 2 else "center", va="center")
    ax.set_xlabel(r"Operator-grid accuracy (\%)", labelpad=5)
    ax.set_ylabel("DSA", labelpad=6)
    ax.set_xlim(17.2, 29.4)
    ax.set_ylim(0, 0.29)
    ax.set_xticks([18, 20, 22, 24, 26, 28])
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, labelsize=8)
    ax.grid(linestyle="-", lw=0.45, color="#E8EBED")
    ax.set_axisbelow(True)
    # Single-panel data figure: the figure caption carries protocol and title.
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=NEUTRAL,
               markersize=3.5, label="Individual training seed"),
        Line2D([], [], marker="o", linestyle="none", color=INK, markersize=5,
               label=r"Mean $\pm$ sample SD ($n=3$)"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.545, 0.045), ncol=2,
               frameon=False, columnspacing=1.7, handletextpad=0.5, fontsize=7.8)
    export(fig, stem, rows, {
        "title": "Axis-I shortcut--utility across three training seeds", "sources": sources,
        "recipe": "academic.error_analysis", "n_training_seeds": 3,
        "units": {"x": "operator-grid accuracy percent", "y": "DSA probability scale"},
        "uncertainty": "Axis-aligned mean +/- one sample SD (ddof=1) across training seeds, not a joint confidence region",
        "statistics": statistics,
        "protocol": "mechanism protocol; held-out map2; 40 rounds x 16 local batches; same AsymHFL; no AugMix/JSD/DCL/CDep",
        "scope": "Pooled comparison at fixed partition and binding map; no all-client dominance or continuous frontier claim",
        "size_inches": [7.06, 3.05],
        "plot_coordinates": rows,
        "axis_ranges": {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())},
        "parent_generator": source_record(Path(__file__).with_name('generate_objective_tradeoff.py')),
        "refined_style": source_record(Path(__file__).with_name('refined_data_style.py')),
        "design_changes": "Latin Modern; shorter canvas; hollow seed markers; smaller filled means; neutral baselines; no BER ring or oversized title",
    }, Path(__file__))
    plt.close(fig)


if __name__ == "__main__":
    main()
