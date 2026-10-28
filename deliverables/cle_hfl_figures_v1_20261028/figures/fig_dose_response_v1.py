"""Figure 5: controlled dose response and the observed task-utility cost.

Adapted from Vivid basic.line: retains ordered measured points, white marker
edges and series emphasis. Only ERM has a measured gamma curve. BER has one
measured gamma=.9 point. No synthetic error band or BER interpolation is drawn.
Right-panel bars are exact accuracy differences, computed from the same table.
Native 7.15 x 3.60 in; at 6.7-in inclusion 8--10pt text is 7.5--9.4pt.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

from figure_common import REPO, INK, MUTED, RULE, RUST, arguments, export, output_stem, source_record, style

SOURCE = REPO / "deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/PAPER_TABLE.csv"
REPORT = REPO / "deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/RESULT_SUMMARY_ZH.md"
METRICS = [
    ("clean_accuracy", "Clean"), ("mixed_accuracy", "Mixed"),
    ("grid_accuracy", "Operator grid"), ("corrupted_balanced_accuracy", "Corrupted"),
    ("seen_accuracy", "Seen"), ("unseen_accuracy", "Unseen"),
]


def read_data(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for item in raw:
        rows.append({key: value if key in {"setting", "method"} else float(value) for key, value in item.items()})
    expected = {("gamma00", "ERM"), ("gamma05", "ERM"), ("gamma09", "AsymHFL-ERM"), ("gamma09", "AsymHFL+PEW+BER")}
    if len(rows) != 4 or {(row["setting"], row["method"]) for row in rows} != expected:
        raise ValueError("Expected exactly three ERM doses and one matched BER arm")
    for row in rows:
        values = [value for value in row.values() if isinstance(value, float)]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite metric")
        for metric, _ in METRICS:
            if not 0 <= row[metric] <= 100:
                raise ValueError("Accuracy is not in percent units")
    return rows


def main() -> None:
    args = arguments(__doc__)
    stem = output_stem("fig_dose_response", args.revision)
    input_csv = args.data_csv if args.data_csv is not None else SOURCE
    rows = read_data(input_csv)
    by_setting = {row["setting"]: row for row in rows if row["method"] != "AsymHFL+PEW+BER"}
    control = by_setting["gamma09"]
    candidate = next(row for row in rows if row["method"] == "AsymHFL+PEW+BER")
    x = np.array([0.0, 0.5, 0.9])
    y = np.array([by_setting[key]["dsa"] for key in ["gamma00", "gamma05", "gamma09"]])
    if not (y[0] < y[1] < y[2]):
        raise ValueError("Frozen table no longer has the reported increasing dose response")
    dsa_reduction = float(control["dsa"] - candidate["dsa"])
    reduction_percent = 100 * dsa_reduction / control["dsa"]
    np.testing.assert_allclose(dsa_reduction, 0.0944899864, atol=2e-10, rtol=0)
    utility_changes = np.array([candidate[key] - control[key] for key, _ in METRICS])
    # Cross-check the rounded published deltas, without using them for plotting.
    np.testing.assert_allclose(utility_changes, [-4.350, -1.793, 0.967, 0.765, 0.920, 0.337], atol=0.000501, rtol=0)

    style()
    fig = plt.figure(figsize=(7.15, 3.60))
    ax = fig.add_axes([0.090, 0.220, 0.385, 0.535])
    right = fig.add_axes([0.683, 0.220, 0.270, 0.535])
    # The measured three-dose ERM sequence retains the source line + white-edge
    # markers. No layer implies a continuous empirical fit or a BER dose curve.
    ax.plot(x, y, "o-", color="#667681", linewidth=1.6, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.9, label="ERM", zorder=3)
    ax.plot([0.9], [candidate["dsa"]], "s", color=RUST, markersize=6.8,
            markeredgecolor="white", markeredgewidth=1.0, label="PEW+BER", zorder=4)
    for xx, yy, offset in zip(x, y, [(10, 3), (-5, 10), (-28, 9)]):
        ax.annotate(f"{yy:.4f}".replace("-", "−"), (xx, yy), xytext=offset,
                    textcoords="offset points", ha="left", va="bottom", color=MUTED, fontsize=8)
    ax.annotate(f"{candidate['dsa']:.4f}", (0.9, candidate["dsa"]), xytext=(-27, -17),
                textcoords="offset points", color=RUST, fontsize=8, weight="bold")
    # A bracket reports the actual matched intervention at gamma=.9.
    top, bottom = control["dsa"], candidate["dsa"]
    ax.plot([0.95, 1.01, 1.01, 0.95], [top, top, bottom, bottom], color=RUST, lw=0.9)
    ax.text(1.045, (top + bottom) / 2, f"−{reduction_percent:.1f}%", color=RUST,
            fontsize=8.5, weight="bold", ha="left", va="center")
    ax.axhline(0, color=RULE, lw=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.set_xlim(-0.05, 1.30)
    ax.set_ylim(-0.015, 0.215)
    ax.set_xticks([0, 0.5, 0.9], ["0", "0.5", "0.9"])
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel(r"Binding strength $\gamma_{\rm CLE}$", labelpad=7)
    ax.set_ylabel("DSA", labelpad=7)
    ax.grid(axis="y", lw=0.55, color="#E5E9EC")
    ax.set_axisbelow(True)

    # Show all six frozen utility endpoints, including clean/mixed degradation.
    ypos = np.arange(len(METRICS))[::-1]
    right.barh(ypos, utility_changes, height=0.48, color=RUST, alpha=0.72, edgecolor="none", zorder=2)
    right.axvline(0, color=MUTED, lw=0.8, linestyle=(0, (3, 3)), zorder=1)
    for yy, value in zip(ypos, utility_changes):
        right.annotate(f"{value:+.2f}".replace("-", "−"), (value, yy),
                       xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                       ha="left" if value >= 0 else "right", va="center", color=INK, fontsize=8)
    right.set_yticks(ypos, [label for _, label in METRICS])
    right.tick_params(axis="y", length=0, pad=7, labelsize=8)
    right.set_ylim(-0.65, 5.65)
    right.set_xlim(-6.0, 2.3)
    right.set_xticks([-4, -2, 0, 2])
    right.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:+.0f}".replace("-", "−")))
    right.set_xlabel("Accuracy change (pp)", labelpad=7)
    right.grid(axis="x", lw=0.55, color="#E5E9EC")
    right.set_axisbelow(True)
    for axis in (ax, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=8, length=3)
    right.spines["left"].set_visible(False)

    fig.text(0.045, 0.941, "a", fontsize=11, weight="bold", color="#667681")
    fig.text(0.090, 0.941, "Binding-strength response", fontsize=9.8, weight="bold")
    fig.text(0.615, 0.941, "b", fontsize=11, weight="bold", color=RUST)
    fig.text(0.657, 0.941, r"Task utility at $\gamma=0.9$", fontsize=9.8, weight="bold")
    fig.text(0.090, 0.875, "Main evaluation protocol  ·  training seed 0  ·  40 rounds", color=MUTED, fontsize=7.7)
    fig.text(0.657, 0.875, "PEW+BER minus matched ERM", color=MUTED, fontsize=7.7)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.025, 1.16), ncol=2, frameon=False,
              columnspacing=0.8, handlelength=1.4, handletextpad=0.4, fontsize=7.5)
    fig.add_artist(Line2D([0.045, 0.965], [0.095, 0.095], transform=fig.transFigure, color=RULE, lw=0.7))
    fig.text(0.090, 0.040, r"Fixed partition, corruption mask and initialization; $\alpha=1.0$, $\rho=0.5$.",
             fontsize=7.6, color=MUTED)
    export(fig, stem, rows, {
        "title": "Main evaluation dose response and task-utility trade-off",
        "sources": [source_record(input_csv), source_record(REPORT)],
        "recipe": "basic.line", "training_seed": 0,
        "units": {"left_y": "DSA probability scale", "right_x": "candidate minus control accuracy, percentage points"},
        "uncertainty": "No multi-training-seed intervals. No synthetic band or per-dose source CI.",
        "dsa_reduction": dsa_reduction, "relative_dsa_reduction_percent": reduction_percent,
        "utility_changes_pp": {key: float(value) for (key, _), value in zip(METRICS, utility_changes)},
        "protocol": "main evaluation protocol; alpha=1; rho=.5; fixed map2/partition seed0/training seed0; 40x64 local batches",
        "scope": "Measured dose points only; BER measured only at gamma .9; no monotonicity theorem or utility-preserving claim",
        "size_inches": [7.15, 3.60],
    }, Path(__file__))
    plt.close(fig)


if __name__ == "__main__":
    main()
