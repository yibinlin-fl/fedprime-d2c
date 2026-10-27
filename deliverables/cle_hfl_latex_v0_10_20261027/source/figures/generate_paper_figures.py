from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent

NAVY = "#233B57"
BLUE = "#4C78A8"
ORANGE = "#D07C3E"
GREEN = "#4E8D73"
RED = "#B95B5B"
PURPLE = "#8064A2"
INK = "#20252B"
MID = "#65717E"
LIGHT = "#E9EEF3"
PALE_BLUE = "#EDF3F8"
PALE_ORANGE = "#F8F0E8"
PALE_GREEN = "#EDF5F1"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.2,
        "axes.labelsize": 8.4,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        if ext == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
        elif ext == "svg":
            kwargs["metadata"] = {"Date": None}
        output = OUT / f"{stem}.{ext}"
        fig.savefig(output, **kwargs)
        if ext == "svg":
            normalized = "\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n"
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(normalized)
    plt.close(fig)


def clean(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def arrow(ax: plt.Axes, start, end, color=NAVY, lw=1.15, style="-|>", zorder=1) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def label_box(ax, xy, wh, title, subtitle="", edge=NAVY, face="white", title_color=INK):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    ax.text(x + 0.04 * w, y + 0.64 * h, title, ha="left", va="center", weight="bold", color=title_color)
    if subtitle:
        ax.text(x + 0.04 * w, y + 0.29 * h, subtitle, ha="left", va="center", color=MID, fontsize=7.0)


def figure_1_evidence_chain() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    clean(ax)

    xs = np.linspace(0.08, 0.92, 6)
    names = ["CLE setup", "Paired diagnosis", "Formation", "Mitigation", "Transfer", "Boundary"]
    subs = [
        "client-specific\nclass--operator binding",
        "same source,\ndifferent operators",
        "HFL vs. Local\nmatched factorial",
        "coarse PEW +\nclass-wise BER",
        "new bindings +\ncommunication base",
        "utility, taxonomy,\nexternal validity",
    ]
    colors = [ORANGE, BLUE, PURPLE, GREEN, BLUE, MID]

    ax.plot([xs[0], xs[-1]], [0.57, 0.57], color="#B9C2CB", linewidth=0.9, zorder=0)
    for i, (x, name, sub, color) in enumerate(zip(xs, names, subs, colors), start=1):
        ax.add_patch(Circle((x, 0.57), 0.025, facecolor="white", edgecolor=color, linewidth=1.3, zorder=3))
        ax.text(x, 0.57, str(i), ha="center", va="center", fontsize=7.0, weight="bold", color=color)
        ax.text(x, 0.43, name, ha="center", va="top", weight="bold", color=INK)
        ax.text(x, 0.32, sub, ha="center", va="top", color=MID, fontsize=6.7, linespacing=1.25)

    ax.text(0.08, 0.83, "scientific question", color=ORANGE, weight="bold", fontsize=7.2)
    ax.plot([0.08, 0.25], [0.78, 0.78], color=ORANGE, linewidth=1.8)
    ax.text(0.42, 0.83, "target-aligned evidence", color=BLUE, weight="bold", fontsize=7.2, ha="center")
    ax.plot([0.29, 0.56], [0.78, 0.78], color=BLUE, linewidth=1.8)
    ax.text(0.76, 0.83, "response and scope", color=GREEN, weight="bold", fontsize=7.2, ha="center")
    ax.plot([0.61, 0.92], [0.78, 0.78], color=GREEN, linewidth=1.8)

    ax.text(0.08, 0.08, "Training-visible", color=GREEN, weight="bold", fontsize=6.8)
    ax.text(0.205, 0.08, "fit labels, public taxonomy, PEW groups", color=MID, fontsize=6.8)
    ax.text(0.57, 0.08, "Evaluation-only", color=PURPLE, weight="bold", fontsize=6.8)
    ax.text(0.695, 0.08, "operator IDs, binding map, paired grid", color=MID, fontsize=6.8)
    save(fig, "fig1_cle_hfl_evidence_chain")


def figure_2_dsa_proxy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.18})
    left, right = axes
    for ax in axes:
        clean(ax)

    left.text(0.01, 0.96, "a", weight="bold", fontsize=9.5, va="top")
    left.text(0.07, 0.96, "Source-paired DSA", weight="bold", color=INK, va="top")
    left.add_patch(Circle((0.12, 0.50), 0.055, facecolor="white", edgecolor=NAVY, linewidth=1.0))
    left.text(0.12, 0.50, r"$(x,y)$", ha="center", va="center", color=INK)
    ops_y = [0.73, 0.50, 0.27]
    op_labels = [r"$T_o(x)$", r"$T_{o'}(x)$", r"$T_{o''}(x)$"]
    op_colors = [ORANGE, BLUE, GREEN]
    for yy, label, color in zip(ops_y, op_labels, op_colors):
        arrow(left, (0.18, 0.50), (0.35, yy), color="#89939D", lw=0.8)
        left.add_patch(Rectangle((0.35, yy - 0.065), 0.20, 0.13, facecolor="white", edgecolor=color, linewidth=1.0))
        left.text(0.45, yy, label, ha="center", va="center", color=INK)
        arrow(left, (0.55, yy), (0.66, yy), color=color, lw=0.9)
        base = 0.66
        vals = [0.21, 0.12, 0.08] if yy == 0.73 else ([0.10, 0.15, 0.12] if yy == 0.50 else [0.08, 0.13, 0.15])
        for j, val in enumerate(vals):
            left.add_patch(Rectangle((base, yy + 0.035 - j * 0.035), val, 0.018, facecolor=color, alpha=0.72, linewidth=0))
    left.annotate("", xy=(0.93, 0.73), xytext=(0.93, 0.27), arrowprops={"arrowstyle": "-[", "lw": 1.0, "color": PURPLE})
    left.text(0.95, 0.50, r"$d_{k,o}(z)$", rotation=90, ha="center", va="center", color=PURPLE, weight="bold")
    left.text(0.07, 0.08, "semantic source fixed", color=MID, fontsize=6.8)
    left.text(0.55, 0.08, "operator changes only", color=MID, fontsize=6.8)

    right.text(0.01, 0.96, "b", weight="bold", fontsize=9.5, va="top")
    right.text(0.07, 0.96, "Proxy non-identifiability", weight="bold", color=INK, va="top")
    label_box(right, (0.28, 0.73), (0.44, 0.12), r"same observable $P(Y,Z)$", "task label and proxy witness", edge=BLUE, face=PALE_BLUE)
    label_box(right, (0.05, 0.39), (0.36, 0.16), "latent world A", r"$D(Y,E)=0$", edge=GREEN, face=PALE_GREEN)
    label_box(right, (0.59, 0.39), (0.36, 0.16), "latent world B", r"$D(Y,E)>0$", edge=RED, face="#FAEEEE")
    arrow(right, (0.42, 0.73), (0.24, 0.55), color=GREEN)
    arrow(right, (0.58, 0.73), (0.76, 0.55), color=RED)
    right.plot([0.20, 0.80], [0.26, 0.26], color="#AAB3BC", linewidth=0.8)
    right.text(0.50, 0.19, "proxy evidence cannot choose the latent world", ha="center", color=MID, fontsize=7.0)
    right.text(0.50, 0.09, "paired, target-aligned DSA remains necessary", ha="center", color=PURPLE, weight="bold", fontsize=7.3)
    save(fig, "fig2_dsa_proxy_boundary")


def figure_3_local_first() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.38})
    ax, ax2 = axes
    seeds = np.array([0, 1, 2])
    hfl = np.array([0.119889, 0.1237172825, 0.1246375956])
    local = np.array([0.107960, 0.1272867853, 0.1312102392])
    hfl_ci = np.array([[0.118050, 0.1218985570, 0.1228536494], [0.121562, 0.1256124678, 0.1265343933]])
    local_ci = np.array([[0.106145, 0.1253805069, 0.1296528507], [0.109678, 0.1291983182, 0.1329373746]])
    hfl_err = np.vstack([hfl - hfl_ci[0], hfl_ci[1] - hfl])
    local_err = np.vstack([local - local_ci[0], local_ci[1] - local])

    ax.errorbar(seeds - 0.045, hfl, yerr=hfl_err, fmt="o-", color=BLUE, linewidth=1.25,
                markersize=4.6, capsize=2.2, label=r"$\Delta_{\mathrm{HFL}}$")
    ax.errorbar(seeds + 0.045, local, yerr=local_err, fmt="s-", color=GREEN, linewidth=1.25,
                markersize=4.2, capsize=2.2, label=r"$\Delta_{\mathrm{Local}}$")
    ax.set_xticks(seeds, ["0", "1", "2"])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("CLE effect (DSA scale)")
    ax.set_ylim(0.102, 0.136)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E1E5E9", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.10, 1.04, "a  Matched CLE effects", transform=ax.transAxes, weight="bold", color=INK)

    comm = np.array([0.011929, -0.0035695028, -0.0065726436])
    comm_ci = np.array([[0.011115, -0.0042906777, -0.0077358433], [0.012712, -0.0028560891, -0.0054346403]])
    comm_err = np.vstack([comm - comm_ci[0], comm_ci[1] - comm])
    ax2.axhline(0, color="#89939D", linewidth=0.8, linestyle="--")
    ax2.errorbar(seeds, comm, yerr=comm_err, fmt="o-", color=PURPLE, linewidth=1.25,
                 markersize=4.7, capsize=2.2)
    ax2.set_xticks(seeds, ["0", "1", "2"])
    ax2.set_xlabel("Training seed")
    ax2.set_ylabel(r"$\Delta_{\mathrm{comm}}$")
    ax2.set_ylim(-0.010, 0.015)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="#E1E5E9", linewidth=0.6)
    ax2.text(-0.13, 1.04, "b  Communication difference", transform=ax2.transAxes, weight="bold", color=INK)
    save(fig, "fig3_local_first_attribution")


def figure_4_axis1() -> None:
    methods = ["ERM", "CVaR-DRO", "PEW+GroupDRO", "PEW+BER"]
    dsa = np.array([0.2505, 0.0867, 0.2054, 0.0715])
    dsa_sd = np.array([0.0087, 0.0055, 0.0080, 0.0113])
    acc = np.array([20.4950, 19.2206, 21.4517, 25.1772])
    acc_sd = np.array([0.3718, 1.1055, 0.5783, 1.2522])
    colors = [MID, ORANGE, BLUE, GREEN]
    y = np.arange(len(methods))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), gridspec_kw={"wspace": 0.48})
    for ax, values, errors, xlabel, panel in [
        (axes[0], dsa, dsa_sd, "DSA (lower is better)", "a  Directional shortcut"),
        (axes[1], acc, acc_sd, "Operator-grid accuracy (%)", "b  Task utility"),
    ]:
        for yi, val, err, color in zip(y, values, errors, colors):
            ax.errorbar(val, yi, xerr=err, fmt="o", color=color, ecolor=color, elinewidth=1.0, capsize=2.5, markersize=5)
        ax.set_yticks(y, methods)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#E1E5E9", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(-0.13, 1.05, panel, transform=ax.transAxes, weight="bold", color=INK)
    axes[0].set_xlim(0.04, 0.28)
    axes[1].set_xlim(17.0, 27.0)
    save(fig, "fig4_axis1_objective_comparison")


def figure_5_protocol_map() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    clean(ax)
    ax.text(0.015, 0.70, "Mechanism\nprotocol", color=ORANGE, weight="bold", fontsize=7.6, va="center")
    ax.text(0.015, 0.27, "Main evaluation\nprotocol", color=BLUE, weight="bold", fontsize=7.2, va="center")

    stages = [
        (0.23, "Data regime"),
        (0.44, "Scientific role"),
        (0.65, "Primary comparison"),
        (0.86, "Reported endpoint"),
    ]
    for x, title in stages:
        ax.text(x, 0.95, title, ha="center", va="top", color=MID, fontsize=6.8, weight="bold")

    top = [
        r"$\alpha=.5,\rho=1,\gamma=.9$",
        "mechanism\nstress test",
        "matched local\nobjectives",
        "DSA + utility\nboundary",
    ]
    bottom = [
        "$\\alpha=1,\\rho=.5$\n$\\gamma\\in\\{0,.5,.9\\}$",
        "dose response +\ntask utility",
        "HFL mechanism\nfamilies",
        "DSA + clean/corrupt\naccuracy",
    ]
    for row_y, values, color, face in [(0.70, top, ORANGE, PALE_ORANGE), (0.27, bottom, BLUE, PALE_BLUE)]:
        for idx, ((x, _), value) in enumerate(zip(stages, values)):
            w = 0.19
            ax.add_patch(
                FancyBboxPatch(
                    (x - w / 2, row_y - 0.09),
                    w,
                    0.18,
                    boxstyle="round,pad=0.006,rounding_size=0.010",
                    linewidth=0.9,
                    edgecolor=color,
                    facecolor=face,
                    zorder=2,
                )
            )
            ax.text(x, row_y, value, ha="center", va="center", color=INK, fontsize=6.8, weight="bold", linespacing=1.2)
            if idx < len(stages) - 1:
                arrow(ax, (x + w / 2, row_y), (stages[idx + 1][0] - w / 2, row_y), color=color, lw=0.9)

    ax.text(0.53, 0.055, "Shared: held-out map2, paired operator grid, sealed evaluation, seed-matched initialization", ha="center", color=MID, fontsize=6.5)
    save(fig, "fig5_protocol_map")


if __name__ == "__main__":
    figure_1_evidence_chain()
    figure_2_dsa_proxy()
    figure_3_local_first()
    figure_4_axis1()
    figure_5_protocol_map()
    print("generated 5 figures in", OUT)
