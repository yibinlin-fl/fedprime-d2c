from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "local_runs" / "cle_v2_moderate_alpha10_rho05" / "cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0"

INK = "#20252B"
MUTED = "#66727D"
RULE = "#C9D0D6"
PALE = "#F6F8F9"
TEAL = "#287C78"
TEAL_LIGHT = "#DCECE9"
RUST = "#B85C38"
RUST_LIGHT = "#F2E4DD"
BLUE = "#4C6F91"
PURPLE = "#6F5A7E"
GREEN = "#668555"
FAMILY_COLORS = {"noise": TEAL, "blur": BLUE, "weather": GREEN, "digital": PURPLE}
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SHORT_OPERATOR = {
    "gaussian_noise": "Gaussian noise",
    "defocus_blur": "Defocus blur",
    "snow": "Snow",
    "pixelate": "Pixelate",
    "contrast": "Contrast",
    "glass_blur": "Glass blur",
    "elastic_transform": "Elastic",
    "brightness": "Brightness",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2, "axes.linewidth": 0.7,
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str = MUTED, lw: float = 1.0) -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def panel_heading(ax, x: float, y: float, letter: str, title: str, color: str) -> None:
    ax.add_patch(Rectangle((x, y - 0.046), 0.027, 0.054, facecolor=color, edgecolor="none"))
    ax.text(x + 0.0135, y - 0.019, letter, color="white", fontsize=9.3, weight="bold", ha="center", va="center")
    ax.text(x + 0.037, y - 0.019, title, color=INK, fontsize=9.0, weight="bold", ha="left", va="center")


def load_protocol() -> tuple[dict, dict]:
    path = PACKAGE / "data" / "gamma09" / "metadata.json"
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata["class_operator_map"], metadata["operator_families"]


def load_paired_source(source_id: int, operator_ids: list[int]) -> tuple[list[np.ndarray], int]:
    root = PACKAGE / "evaluation" / "paired_dsa"
    images = np.load(root / "test_images.npy", mmap_mode="r")
    labels = np.load(root / "test_labels.npy", mmap_mode="r")
    operators = np.load(root / "test_corruption_ids.npy", mmap_mode="r")
    sources = np.load(root / "test_source_ids.npy", mmap_mode="r")
    rows = np.flatnonzero(sources == source_id)
    selected = []
    for operator_id in operator_ids:
        matches = rows[operators[rows] == operator_id]
        if len(matches) != 1:
            raise RuntimeError(f"source {source_id} operator {operator_id} has {len(matches)} rows")
        selected.append(np.asarray(images[int(matches[0])]))
    return selected, int(labels[int(rows[0])])


def client_binding_table(ax, x: float, y: float, width: float, client_id: int, model_name: str,
                         mapping: dict, families: dict) -> None:
    ax.text(x, y + 0.026, f"Client {client_id}  ·  {model_name}", color=INK, fontsize=6.8, weight="bold", ha="left")
    for j, class_id in enumerate([0, 1, 2, 3]):
        yy = y - j * 0.052
        operator = mapping[str(client_id)][str(class_id)]
        color = FAMILY_COLORS[families[operator]]
        ax.text(x, yy, CLASS_NAMES[class_id], color=MUTED, fontsize=5.7, va="center", ha="left")
        ax.plot([x + 0.071, x + 0.094], [yy, yy], color=RULE, lw=0.8)
        ax.add_patch(Rectangle((x + 0.098, yy - 0.017), width - 0.098, 0.034,
                               facecolor="white", edgecolor=color, linewidth=0.85))
        label = SHORT_OPERATOR.get(operator, operator.replace("_", " ").title())
        ax.text(x + 0.105, yy, label, color=color, fontsize=5.55, weight="bold", va="center", ha="left")


def image_tile(ax, image: np.ndarray, x: float, y: float, width: float, height: float, title: str, color: str) -> None:
    ax.imshow(image, extent=(x, x + width, y, y + height), interpolation="nearest", aspect="auto", zorder=2)
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=0.9, zorder=3))
    ax.text(x + width / 2, y - 0.025, title, ha="center", va="top", fontsize=5.35,
            color=color, weight="bold", linespacing=0.9)


def main() -> None:
    style()
    bindings, families = load_protocol()
    operator_ids = [0, 3, 7, 13]
    operator_names = ["gaussian_noise", "defocus_blur", "snow", "pixelate"]
    operator_labels = ["Gaussian\nnoise", "Defocus\nblur", "Snow", "Pixelate"]
    paired_images, label = load_paired_source(source_id=3, operator_ids=operator_ids)

    fig, ax = plt.subplots(figsize=(9.0, 3.75))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([0.314, 0.314], [0.13, 0.94], color=RULE, lw=0.8)
    ax.plot([0.675, 0.675], [0.13, 0.94], color=RULE, lw=0.8)
    panel_heading(ax, 0.018, 0.938, "a", "Client-specific CLE", RUST)
    panel_heading(ax, 0.334, 0.938, "b", "Source-paired operator grid", TEAL)
    panel_heading(ax, 0.695, 0.938, "c", "Directional contrast", PURPLE)

    ax.text(0.028, 0.805, r"pre-registered $b_k:\mathcal{C}\rightarrow\mathcal{O}$", color=MUTED, fontsize=6.6)
    client_binding_table(ax, 0.035, 0.735, 0.238, 0, "ResNet10", bindings, families)
    client_binding_table(ax, 0.035, 0.465, 0.238, 1, "ResNet12", bindings, families)
    ax.add_patch(Rectangle((0.032, 0.162), 0.248, 0.102, facecolor=RUST_LIGHT, edgecolor="none"))
    ax.text(0.156, 0.226, r"$P_k(O=o\mid Y=c)$", color=INK, fontsize=7.0, ha="center", va="center")
    ax.text(0.156, 0.184, r"$=(1-\gamma)/M+\gamma\,\mathbf{1}\{o=b_k(c)\}$",
            color=RUST, fontsize=6.2, ha="center", va="center")

    ax.text(0.350, 0.810, f"source id 3  ·  true label: {CLASS_NAMES[label]}", color=INK, fontsize=6.7, weight="bold")
    tile_w, tile_h = 0.070, 0.174
    xs = [0.342, 0.423, 0.504, 0.585]
    for x, image, operator, label_text in zip(xs, paired_images, operator_names, operator_labels):
        image_tile(ax, image, x, 0.493, tile_w, tile_h, label_text, FAMILY_COLORS[families[operator]])
    ax.plot([xs[0], xs[-1] + tile_w], [0.735, 0.735], color=TEAL, lw=1.0)
    ax.plot([xs[0], xs[0]], [0.720, 0.750], color=TEAL, lw=1.0)
    ax.plot([xs[-1] + tile_w, xs[-1] + tile_w], [0.720, 0.750], color=TEAL, lw=1.0)
    ax.text(0.503, 0.770, "semantic source fixed; only the operator changes",
            ha="center", color=TEAL, fontsize=6.3, weight="bold")
    ax.add_patch(Rectangle((0.350, 0.230), 0.306, 0.142, facecolor=TEAL_LIGHT, edgecolor="none"))
    ax.text(0.503, 0.330, "What ordinary accuracy cannot isolate", ha="center", color=TEAL, fontsize=6.8, weight="bold")
    ax.text(0.503, 0.292, "image difficulty + semantic error", ha="center", color=INK, fontsize=6.1)
    ax.text(0.503, 0.263, "+ binding-directed response", ha="center", color=INK, fontsize=6.1)
    ax.text(0.503, 0.232, "paired views cancel source content before aggregation", ha="center", color=MUTED, fontsize=5.8)

    ax.add_patch(Rectangle((0.704, 0.728), 0.098, 0.072, facecolor="white", edgecolor=PURPLE, linewidth=0.9))
    ax.text(0.753, 0.764, "operator $o$", ha="center", va="center", color=PURPLE, fontsize=6.5, weight="bold")
    arrow(ax, 0.806, 0.764, 0.841, 0.764, color=PURPLE)
    ax.add_patch(Rectangle((0.846, 0.712), 0.132, 0.104, facecolor="#EEEAF1", edgecolor=PURPLE, linewidth=0.9))
    ax.text(0.912, 0.783, "bound classes", ha="center", color=MUTED, fontsize=5.8)
    ax.text(0.912, 0.744, r"$B_{k,o}=\{c:b_k(c)=o\}$", ha="center", color=INK, fontsize=6.5)
    ax.text(0.704, 0.640, "same source $z=(x,y)$", color=INK, fontsize=6.7, weight="bold")

    ax.add_patch(Rectangle((0.704, 0.537), 0.115, 0.064, facecolor=RUST_LIGHT, edgecolor=RUST, linewidth=0.8))
    ax.text(0.7615, 0.569, r"target view $T_o(x)$", ha="center", va="center", color=RUST, fontsize=6.0, weight="bold")
    arrow(ax, 0.824, 0.569, 0.853, 0.569, color=RUST)
    ax.text(0.858, 0.569, r"$m_{k,o}(z,o)$", ha="left", va="center", color=INK, fontsize=6.7)
    ax.add_patch(Rectangle((0.704, 0.441), 0.115, 0.064, facecolor=PALE, edgecolor=MUTED, linewidth=0.8))
    ax.text(0.7615, 0.473, r"other $T_{o'}(x)$", ha="center", va="center", color=MUTED, fontsize=6.0, weight="bold")
    arrow(ax, 0.824, 0.473, 0.853, 0.473, color=MUTED)
    ax.text(0.858, 0.473, r"$\operatorname{avg}_{o'\ne o}m_{k,o}(z,o')$", ha="left", va="center", color=INK, fontsize=6.35)

    ax.add_patch(Rectangle((0.704, 0.238), 0.274, 0.142, facecolor="#EEEAF1", edgecolor=PURPLE, linewidth=1.0))
    ax.text(0.841, 0.340, "source-level directional contrast", ha="center", color=PURPLE, fontsize=6.5, weight="bold")
    ax.text(0.841, 0.292, r"$d_{k,o}(z)=m_{k,o}(z,o)-\operatorname{avg}_{o'\ne o}m_{k,o}(z,o')$",
            ha="center", color=INK, fontsize=6.35)
    ax.text(0.841, 0.252, r"$\mathrm{DSA}_{k,o}=\mathbb{E}_z[d_{k,o}(z)]$  ·  probability scale",
            ha="center", color=MUTED, fontsize=6.15)

    ax.plot([0.025, 0.975], [0.105, 0.105], color=RULE, lw=0.8)
    ax.text(0.155, 0.068, "CONTROLLED CONSTRUCTION", color=RUST, fontsize=6.1, weight="bold", va="center", ha="center")
    ax.text(0.155, 0.034, r"$b_k,\gamma_{\rm CLE}$", color=MUTED, fontsize=5.9, va="center", ha="center")
    ax.text(0.500, 0.068, "MODEL TRAINING", color=TEAL, fontsize=6.1, weight="bold", va="center", ha="center")
    ax.text(0.500, 0.034, "no operator or binding metadata", color=MUTED, fontsize=5.9, va="center", ha="center")
    ax.text(0.835, 0.068, "SEALED EVALUATION", color=PURPLE, fontsize=6.1, weight="bold", va="center", ha="center")
    ax.text(0.835, 0.034, "paired grid + registered binding", color=MUTED, fontsize=5.9, va="center", ha="center")

    output = HERE / "fig_cle_paired_dsa_v3"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
