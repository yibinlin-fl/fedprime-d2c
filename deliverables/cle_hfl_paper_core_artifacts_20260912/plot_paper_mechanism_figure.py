"""Generate the paper-ready CLE-HFL evidence-chain mechanism figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent


def box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color,
        edgecolor="#233044",
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=9.3, linespacing=1.32)


def arrow(ax, x1, x2, y):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="#526173",
        )
    )


fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(
    0.5,
    0.955,
    "CLE-HFL: from directional shortcut identification to controlled mitigation",
    ha="center",
    va="center",
    fontsize=21,
    fontweight="bold",
    color="#182230",
)
ax.text(
    0.5,
    0.915,
    "A problem-diagnosis-attribution-intervention-validation evidence chain",
    ha="center",
    va="center",
    fontsize=12,
    color="#526173",
)

y, w, h = 0.61, 0.168, 0.225
xs = [0.025, 0.222, 0.419, 0.616, 0.813]
boxes = [
    (
        "1  CLE-HFL problem",
        "Client-specific\nclass-corruption binding\ncreates a spurious cue",
        "#FDE8E4",
    ),
    (
        "2  Paired diagnosis",
        "Same source across operators\nDSA measures a binding-\ndirected probability shift",
        "#E8F0FE",
    ),
    (
        "3  Mechanism attribution",
        "Matched HFL vs Local\nShortcut is local-first\nLocal/HFL = 90.05%",
        "#E9F6EE",
    ),
    (
        "4  Local intervention",
        "Public PEW: pseudo-envs\nBER: within-class\nenvironment balancing",
        "#FFF2D9",
    ),
    (
        "5  Controlled validation",
        "New binding map\nSecond communication base\nTwo replication axes",
        "#EEE9FA",
    ),
]
for x, (title, body, color) in zip(xs, boxes):
    box(ax, x, y, w, h, title, body, color)
for index in range(4):
    arrow(ax, xs[index] + w + 0.004, xs[index + 1] - 0.004, y + h / 2)

ax.text(0.025, 0.545, "Formal evidence", fontsize=14, fontweight="bold", color="#182230")

cards = [
    (
        0.025,
        0.30,
        "Shortcut formation",
        "No-CLE DSA approx 0\nStrong-CLE DSA = 0.1196\nShuffled null p = 0.000999",
        "#F7FAFC",
    ),
    (
        0.275,
        0.30,
        "Original AsymHFL map",
        "0.1196 -> 0.0413\nDSA reduction: 65.52%\n4/4 clients improve",
        "#EDF7F1",
    ),
    (
        0.525,
        0.30,
        "New binding map1",
        "0.1138 -> 0.0495\nDSA reduction: 56.46%\nI0/L0/C1/C2: all PASS",
        "#EDF7F1",
    ),
    (
        0.775,
        0.30,
        "FedDF-fidelity",
        "0.1370 -> 0.0290\nDSA reduction: 78.82%\nUtility trade-off remains",
        "#FFF6E5",
    ),
]
for x, y0, title, body, color in cards:
    box(ax, x, y0, 0.20, 0.18, title, body, color)

boundary = FancyBboxPatch(
    (0.025, 0.075),
    0.95,
    0.145,
    boxstyle="round,pad=0.012,rounding_size=0.018",
    facecolor="#F7F8FA",
    edgecolor="#8A94A3",
    linewidth=1.2,
)
ax.add_patch(boundary)
ax.text(0.05, 0.174, "Claim boundary", fontsize=13, fontweight="bold", color="#182230")
ax.text(
    0.05,
    0.122,
    "Supported: directional shortcut formation, local-first attribution, and mitigation replicated across binding maps and communication bases.\n"
    "Not supported: taxonomy-free discovery, universally lossless utility, or generalization across partitions, seeds, datasets, and real domains.",
    fontsize=11.2,
    va="center",
    color="#374151",
    linespacing=1.45,
)

fig.savefig(OUT / "PAPER_MECHANISM_EVIDENCE_CHAIN.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "PAPER_MECHANISM_EVIDENCE_CHAIN.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "PAPER_MECHANISM_EVIDENCE_CHAIN.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)
