from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image, ImageEnhance, ImageFilter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA_ROOT = (
    ROOT
    / "local_runs"
    / "cle_v2_moderate_alpha10_rho05"
    / "cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0"
)

INK = "#20252B"
MUTED = "#66727D"
RULE = "#CBD2D8"
PALE = "#F5F7F8"
TEAL = "#287C78"
TEAL_LIGHT = "#DCECE9"
RUST = "#B85C38"
RUST_LIGHT = "#F2E4DD"
BLUE = "#4C6F91"
EVAL = "#6F5A7E"

CLASS_NAMES = [
    "airplane", "auto", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "mathtext.fontset": "stixsans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def public_image() -> np.ndarray:
    archive = DATA_ROOT / "public" / "cifar-100-python.tar.gz"
    with tarfile.open(archive, "r:gz") as tf:
        member = tf.getmember("cifar-100-python/train")
        handle = tf.extractfile(member)
        if handle is None:
            raise RuntimeError("missing CIFAR-100 train member")
        batch = pickle.load(io.BytesIO(handle.read()), encoding="bytes")
    flat = np.asarray(batch[b"data"])[42]
    return flat.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)


def private_images() -> list[tuple[np.ndarray, str]]:
    client = DATA_ROOT / "data" / "gamma09" / "client_0"
    images = np.load(client / "train_images.npy", mmap_mode="r")
    labels = np.load(client / "train_labels.npy", mmap_mode="r")
    corrupted = np.load(client / "train_is_corrupted.npy", mmap_mode="r")
    chosen: list[int] = []
    seen: set[int] = set()
    for idx, (label, is_corrupted) in enumerate(zip(labels, corrupted)):
        label_i = int(label)
        if int(is_corrupted) == 1 and label_i not in seen:
            chosen.append(idx)
            seen.add(label_i)
        if len(chosen) == 4:
            break
    result = []
    for idx in chosen:
        image = np.asarray(images[idx])
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        result.append((image.astype(np.uint8), CLASS_NAMES[int(labels[idx])]))
    return result


def variants(image: np.ndarray) -> list[tuple[np.ndarray, str]]:
    base = Image.fromarray(image)
    rng = np.random.default_rng(7)

    noise = np.asarray(base).astype(np.float32)
    noise += rng.normal(0, 34, noise.shape)
    noise = np.clip(noise, 0, 255).astype(np.uint8)

    blur = np.asarray(base.filter(ImageFilter.GaussianBlur(radius=1.6)))

    weather = np.asarray(base).astype(np.float32)
    fog = np.full_like(weather, 235.0)
    weather = np.clip(0.62 * weather + 0.38 * fog, 0, 255).astype(np.uint8)

    tiny = base.resize((8, 8), Image.Resampling.BILINEAR)
    digital = np.asarray(tiny.resize((32, 32), Image.Resampling.NEAREST))
    digital = np.asarray(ImageEnhance.Contrast(Image.fromarray(digital)).enhance(1.25))

    return [(noise, "noise"), (blur, "blur"), (weather, "weather"), (digital, "digital")]


def arrow(ax, x1: float, y1: float, x2: float, y2: float, *, color=INK, lw=1.25, dashed=False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def image_tile(ax, image: np.ndarray, x: float, y: float, w: float, h: float, label: str | None = None) -> None:
    ax.imshow(image, extent=(x, x + w, y, y + h), zorder=2, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#87919A", linewidth=0.75, zorder=3))
    if label:
        ax.text(x + w / 2, y - 0.018, label, ha="center", va="top", fontsize=6.6, color=MUTED)


def section_label(ax, x: float, y: float, letter: str, title: str, color: str) -> None:
    ax.text(x, y, letter, ha="left", va="center", fontsize=10.5, weight="bold", color="white",
            bbox=dict(boxstyle="square,pad=0.22", facecolor=color, edgecolor=color))
    ax.text(x + 0.028, y, title, ha="left", va="center", fontsize=9.2, weight="bold", color=INK)


def network_icon(ax, x: float, y: float, w: float, h: float) -> None:
    cols = [3, 4, 3]
    xs = np.linspace(x + 0.015, x + w - 0.015, len(cols))
    nodes: list[list[tuple[float, float]]] = []
    for xi, count in zip(xs, cols):
        ys = np.linspace(y + 0.018, y + h - 0.018, count)
        nodes.append([(xi, yi) for yi in ys])
    for left, right in zip(nodes[:-1], nodes[1:]):
        for p in left:
            for q in right:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#9AA6AF", lw=0.45, zorder=1)
    for layer in nodes:
        for px, py in layer:
            ax.add_patch(Circle((px, py), 0.0043, facecolor=TEAL, edgecolor="white", linewidth=0.35, zorder=3))


def support_matrix(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=RULE, linewidth=0.9))
    cols = ["C", "N", "B", "W", "D", "U"]
    rows = ["$c_1$", "$c_2$", "$c_3$"]
    # Schematic support cells, not measurements or a quantitative heatmap.
    intensities = np.array([[0.35, 0.85, 0.20, 0.10, 0.35, 0.05], [0.30, 0.15, 0.78, 0.28, 0.12, 0.08], [0.40, 0.25, 0.12, 0.70, 0.22, 0.04]])
    cell_w = w * 0.105
    cell_h = h * 0.18
    x0 = x + w * 0.25
    y0 = y + h * 0.20
    for j, name in enumerate(cols):
        ax.text(x0 + (j + 0.5) * cell_w, y + h * 0.86, name, ha="center", va="center", fontsize=6.1, color=MUTED)
    for i, name in enumerate(rows):
        yy = y0 + (2 - i) * cell_h
        ax.text(x + w * 0.17, yy + cell_h / 2, name, ha="center", va="center", fontsize=6.5, color=INK)
        for j in range(6):
            alpha = 0.15 + 0.75 * intensities[i, j]
            ax.add_patch(Rectangle((x0 + j * cell_w, yy), cell_w * 0.82, cell_h * 0.72,
                                   facecolor=RUST, alpha=alpha, edgecolor="none"))
    ax.text(x + w / 2, y + 0.012, r"$n_{k,c,e}$ (schematic)", ha="center", va="bottom", fontsize=6.0, color=MUTED)


def main() -> None:
    out = HERE / "fig_pew_ber_execution_v3"
    if any(out.with_suffix(extension).exists() for extension in [".pdf", ".svg", ".png", ".py"]):
        raise FileExistsError("Preserve existing Figure 2 revision; select a new suffix")
    style()
    # Sized close to a two-column conference figure so text is judged at its
    # eventual reading scale rather than designed on an oversized canvas.
    fig, ax = plt.subplots(figsize=(9.0, 3.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Flat lane guides: visual organization without card-wall styling.
    ax.add_patch(Rectangle((0.012, 0.615), 0.976, 0.335, facecolor="#FBFCFC", edgecolor=RULE, linewidth=0.8))
    ax.add_patch(Rectangle((0.012, 0.115), 0.976, 0.445, facecolor="white", edgecolor=RULE, linewidth=0.8))
    section_label(ax, 0.025, 0.912, "a", "Public witness construction", TEAL)
    section_label(ax, 0.025, 0.522, "b", "Client $k$: class-conditional local risk", RUST)

    # Public lane.
    clean = public_image()
    image_tile(ax, clean, 0.055, 0.705, 0.072, 0.145, "public source")
    arrow(ax, 0.132, 0.778, 0.167, 0.778, color=MUTED)
    ax.text(0.148, 0.806, r"$T_f(\cdot)$", ha="center", va="bottom", fontsize=7.3, color=MUTED)
    for j, (img, family) in enumerate(variants(clean)):
        image_tile(ax, img, 0.177 + j * 0.079, 0.705, 0.063, 0.126, family)
    ax.text(0.302, 0.870, "clean + synthetic family supervision", ha="center", va="center", fontsize=6.8, color=MUTED)
    arrow(ax, 0.491, 0.778, 0.535, 0.778, color=TEAL)

    ax.add_patch(Rectangle((0.545, 0.684), 0.155, 0.188, facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=1.15))
    network_icon(ax, 0.565, 0.724, 0.050, 0.102)
    ax.text(0.635, 0.803, "PEW", fontsize=10.2, weight="bold", color=TEAL, ha="center")
    ax.text(0.635, 0.756, r"$q_\phi(\hat e\mid x)$", fontsize=9.2, color=INK, ha="center")
    ax.text(0.622, 0.708, "coarse family witness", fontsize=6.8, color=MUTED, ha="center")
    ax.text(0.719, 0.806, "freeze", fontsize=7.0, weight="bold", color=TEAL, ha="left")
    arrow(ax, 0.700, 0.778, 0.755, 0.778, color=TEAL)
    ax.plot([0.755, 0.755, 0.342], [0.778, 0.465, 0.465], color=TEAL, lw=1.25)
    arrow(ax, 0.342, 0.465, 0.342, 0.444, color=TEAL)

    # Private lane: actual private thumbnails and task labels.
    samples = private_images()
    for j, (img, _label) in enumerate(samples):
        image_tile(ax, img, 0.045 + j * 0.052, 0.265, 0.045, 0.105, "")
    ax.text(0.142, 0.405, "private fit minibatch", ha="center", va="center", fontsize=7.3, weight="bold", color=INK)
    ax.text(0.142, 0.237, r"samples $x_i$ with task labels $y_i$", ha="center", va="center", fontsize=6.35, color=MUTED)
    arrow(ax, 0.247, 0.320, 0.292, 0.320, color=MUTED)

    # PEW proxy tags.
    ax.text(0.342, 0.424, "frozen PEW", ha="center", va="center", fontsize=7.4, weight="bold", color=TEAL)
    families = [("clean", MUTED), ("noise", TEAL), ("blur", BLUE), ("weather", "#6B8E5D"), ("digital", "#7C6A92"), ("unknown", MUTED)]
    for j, (name, color) in enumerate(families):
        yy = 0.367 - j * 0.040
        ax.add_patch(Rectangle((0.300, yy), 0.085, 0.030, facecolor="white", edgecolor=color, linewidth=0.9))
        ax.text(0.3425, yy + 0.015, name, ha="center", va="center", fontsize=6.1, color=color, weight="bold")
    ax.text(0.342, 0.145, r"pseudo-environment $\hat e_i$", ha="center", va="center", fontsize=6.3, color=MUTED)
    arrow(ax, 0.395, 0.320, 0.427, 0.320, color=TEAL)

    # Support structure and BER objective.
    support_matrix(ax, 0.438, 0.205, 0.155, 0.232)
    arrow(ax, 0.603, 0.320, 0.630, 0.320, color=RUST)
    ax.add_patch(Rectangle((0.641, 0.205), 0.190, 0.232, facecolor=RUST_LIGHT, edgecolor=RUST, linewidth=1.1))
    ax.text(0.736, 0.400, "BER", ha="center", va="center", fontsize=9.5, weight="bold", color=RUST)
    ax.text(0.736, 0.338, r"$a_{k,c,e}\propto\mathbf{1}[n_{k,c,e}\geq m]$", ha="center", va="center", fontsize=6.8, color=INK)
    ax.text(0.736, 0.296, r"$\min(n_{k,c,e},K_{\rm cap})^{\gamma_{\rm BER}}$", ha="center", va="center", fontsize=7.6, color=INK)
    ax.plot([0.664, 0.808], [0.266, 0.266], color="#D8B7A8", lw=0.7)
    ax.text(0.736, 0.236, r"$R_k=\operatorname{avg}_{c\in\mathcal{C}^{\rm valid}_k}\sum_e a_{k,c,e}R_{k,c,e}$",
            ha="center", va="center", fontsize=6.3, color=INK)
    arrow(ax, 0.839, 0.320, 0.866, 0.320, color=RUST)

    # Local model and unchanged HFL communication.
    ax.add_patch(Rectangle((0.875, 0.250), 0.055, 0.140, facecolor="white", edgecolor=INK, linewidth=1.0))
    ax.text(0.9025, 0.340, "local", ha="center", va="center", fontsize=6.5, color=MUTED)
    ax.text(0.9025, 0.305, r"$f_k$", ha="center", va="center", fontsize=9.2, weight="bold", color=INK)
    ax.text(0.9025, 0.272, "logits", ha="center", va="center", fontsize=6.0, color=MUTED)
    ax.plot([0.943, 0.943], [0.180, 0.445], color=RULE, lw=0.9, linestyle="--")
    arrow(ax, 0.930, 0.320, 0.958, 0.320, color=MUTED, dashed=True)
    ax.add_patch(Circle((0.970, 0.320), 0.018, facecolor=PALE, edgecolor=MUTED, linewidth=0.9))
    ax.text(0.970, 0.320, "HFL", ha="center", va="center", fontsize=5.7, weight="bold", color=INK)
    ax.text(0.970, 0.366, "existing", ha="center", va="center", fontsize=5.5, color=MUTED)
    ax.text(0.970, 0.274, "unchanged", ha="center", va="center", fontsize=5.5, weight="bold", color=MUTED)

    # Explicit information boundary.
    ax.plot([0.025, 0.975], [0.075, 0.075], color=RULE, lw=0.8)
    ax.text(0.035, 0.042, "TRAIN-TIME", ha="left", va="center", fontsize=6.6, weight="bold", color=TEAL)
    ax.text(0.128, 0.042, r"$x_i,y_i$; public family labels; proxy $\hat e_i$", ha="left", va="center", fontsize=6.0, color=MUTED)
    ax.text(0.585, 0.042, "EVAL-ONLY", ha="left", va="center", fontsize=6.6, weight="bold", color=EVAL)
    ax.text(0.675, 0.042, r"true operator; binding map; paired DSA", ha="left", va="center", fontsize=6.2, color=MUTED)

    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.03)
    out.with_suffix(".py").write_bytes(Path(__file__).read_bytes())
    plt.close(fig)


if __name__ == "__main__":
    main()

