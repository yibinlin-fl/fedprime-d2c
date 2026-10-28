"""Figure 4: frozen Axis-I shortcut--utility comparison over 3 training seeds.

Adapted from Vivid academic.error_analysis: retains the scatter comparison,
direct labels, white-edged markers, and emphasis ring. No parameter-size
measurement, efficiency model, or continuous Pareto frontier is available;
those unsupported layers are omitted. Actual seed points and sample-SD
crosses replace the demo bubbles. Native 7.15 x 3.75 in; 6.7-in inclusion
scales 8--10pt text to 7.5--9.4pt. No manuscript file is edited.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_common import REPO, INK, MUTED, RULE, RUST, BLUE, arguments, export, output_stem, source_record, style

ARMS = ["erm", "cvar_dro", "pew_groupdro", "pew_ber"]
NAMES = ["ERM", "CVaR-DRO", "PEW+GroupDRO", "PEW+BER"]
COLORS = ["#87939C", BLUE, "#747488", RUST]
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
    style()
    fig, ax = plt.subplots(figsize=(7.15, 3.75))
    fig.subplots_adjust(left=0.115, right=0.970, bottom=0.215, top=0.825)
    offsets = [(-20, 15), (-23, -29), (26, 2), (23, 2)]
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
        ax.scatter(acc, dsa, s=29, marker=marker, color=color, alpha=0.60,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="none", color=color,
                    elinewidth=0.95, capsize=2.4, alpha=0.85, zorder=2)
        ax.scatter(mx, my, s=82, marker=marker, color=color, edgecolor="white", linewidth=1.1, zorder=4)
        ax.annotate(NAMES[i], xy=(mx, my), xytext=offsets[i], textcoords="offset points",
                    fontsize=9, fontweight="bold" if arm == "pew_ber" else "normal", color=color,
                    ha="left" if i >= 2 else "center", va="center")
        if arm == "pew_ber":
            ax.scatter(mx, my, s=160, marker=marker, facecolor="none", edgecolor=color,
                       linewidth=0.8, alpha=0.7, zorder=4)

    ax.set_xlabel("Operator-grid accuracy (%)  ↑", labelpad=7)
    ax.set_ylabel("DSA  ↓", labelpad=8)
    ax.set_xlim(17.2, 29.4)
    ax.set_ylim(0, 0.29)
    ax.set_xticks([18, 20, 22, 24, 26, 28])
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, labelsize=8)
    ax.grid(alpha=0.65, linestyle="-", lw=0.55, color="#E5E9EC")
    ax.set_axisbelow(True)
    fig.text(0.115, 0.941, "Shortcut–utility across training seeds", fontsize=10.2, weight="bold")
    fig.text(0.115, 0.884, "Held-out map2  ·  same AsymHFL communication  ·  40 rounds  ·  3 training seeds",
             color=MUTED, fontsize=7.8)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=MUTED, alpha=0.6, markersize=3.5, label="Individual training seed"),
        Line2D([], [], marker="o", linestyle="none", color=INK, markersize=6, label="Mean; bars = ±1 sample SD"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.545, 0.068), ncol=2,
               frameon=False, columnspacing=2.0, handletextpad=0.5, fontsize=7.8)
    export(fig, stem, rows, {
        "title": "Axis-I shortcut--utility across three training seeds", "sources": sources,
        "recipe": "academic.error_analysis", "n_training_seeds": 3,
        "units": {"x": "operator-grid accuracy percent", "y": "DSA probability scale"},
        "uncertainty": "Axis-aligned mean +/- one sample SD (ddof=1) across training seeds, not a joint confidence region",
        "statistics": statistics,
        "protocol": "mechanism protocol; held-out map2; 40 rounds x 16 local batches; same AsymHFL; no AugMix/JSD/DCL/CDep",
        "scope": "Pooled comparison at fixed partition and binding map; no all-client dominance or continuous frontier claim",
        "size_inches": [7.15, 3.75],
    }, Path(__file__))
    plt.close(fig)


if __name__ == "__main__":
    main()
