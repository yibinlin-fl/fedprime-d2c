"""Render the two CLE-HFL paper schematics without external assets."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "cle_hfl_latex_v0_1_20260915" / "figures"

NAVY = HexColor("#1F3A5F")
BLUE = HexColor("#4C78A8")
CYAN = HexColor("#3C9D9B")
ORANGE = HexColor("#E67E22")
GREEN = HexColor("#3B8F55")
RED = HexColor("#C94C4C")
PURPLE = HexColor("#8F63A8")
INK = HexColor("#17212B")
MUTED = HexColor("#52606D")
LINE = HexColor("#8A98A8")
TRAIN_BG = HexColor("#FFF7EC")
EVAL_BG = HexColor("#F2F7FC")


def text(drawing, x, y, value, size=9, color=INK, anchor="middle", bold=False):
    drawing.add(
        String(
            x,
            y,
            value,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
        )
    )


def box(drawing, x, y, w, h, title, body_lines, color, tag=None):
    drawing.add(Rect(x, y, w, h, rx=9, ry=9, fillColor=color.clone(alpha=0.09), strokeColor=color, strokeWidth=1.3))
    if tag:
        text(drawing, x + 9, y + h - 13, tag, 6.8, color, "start", True)
    text(drawing, x + w / 2, y + h * 0.60, title, 9.2, INK, "middle", True)
    for idx, line in enumerate(body_lines):
        text(drawing, x + w / 2, y + h * 0.34 - idx * 12, line, 7.6, MUTED)


def arrow(drawing, x1, y1, x2, y2, color=LINE):
    drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.4))
    dx, dy = x2 - x1, y2 - y1
    norm = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    size = 7
    drawing.add(
        Polygon(
            [
                x2,
                y2,
                x2 - ux * size + px * size * 0.55,
                y2 - uy * size + py * size * 0.55,
                x2 - ux * size - px * size * 0.55,
                y2 - uy * size - py * size * 0.55,
            ],
            fillColor=color,
            strokeColor=color,
        )
    )


def write_all(drawing, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(OUT / f"{stem}.pdf"))
    renderSVG.drawToFile(drawing, str(OUT / f"{stem}.svg"))


def figure_one():
    d = Drawing(950, 300)
    text(d, 475, 282, "CLE-HFL evidence chain and information boundary", 13, INK, "middle", True)
    d.add(Rect(15, 157, 920, 105, rx=10, ry=10, fillColor=TRAIN_BG, strokeColor=None))
    d.add(Rect(15, 25, 920, 105, rx=10, ry=10, fillColor=EVAL_BG, strokeColor=None))
    text(d, 28, 244, "TRAIN-TIME PATH", 7.5, ORANGE, "start", True)
    text(d, 28, 112, "SEALED EVALUATION AND ATTRIBUTION", 7.5, BLUE, "start", True)

    w, h = 232, 68
    xs = [50, 359, 668]
    box(d, xs[0], 173, w, h, "1  CLE-HFL private data", ["client-specific class-operator binding"], ORANGE, "CONTROLLED INPUT")
    box(d, xs[1], 173, w, h, "2  Local shortcut formation", ["robust local learning can still exploit binding"], RED, "OBSERVED FAILURE")
    box(d, xs[2], 173, w, h, "5  PEW + BER", ["coarse family witness + within-class support balance"], GREEN, "LOCAL MITIGATION")
    box(d, xs[0], 41, w, h, "3  Paired operator grid", ["same source and label; intervene on operator"], BLUE, "REPORTING ONLY")
    box(d, xs[1], 41, w, h, "4  DSA + local-first attribution", ["binding-directed mass shift; HFL versus Local"], PURPLE, "TARGET-ALIGNED")
    box(d, xs[2], 41, w, h, "6  Matched validation", ["ERM, CVaR, shared-PEW GroupDRO, maps/bases"], NAVY, "BOUNDARY-AWARE")

    arrow(d, 282, 207, 359, 207, RED)
    arrow(d, 166, 173, 166, 109, BLUE)
    arrow(d, 475, 173, 475, 109, PURPLE)
    arrow(d, 282, 75, 359, 75, PURPLE)
    arrow(d, 784, 173, 784, 109, NAVY)
    arrow(d, 590, 109, 690, 173, GREEN)
    text(d, 654, 138, "motivates local target", 7.0, GREEN)
    write_all(d, "fig1_cle_hfl_evidence_chain")


def figure_two():
    d = Drawing(950, 330)
    text(d, 475, 313, "Directional diagnosis and the proxy-identifiability boundary", 13, INK, "middle", True)
    d.add(Line(475, 22, 475, 292, strokeColor=LINE, strokeWidth=0.8))

    text(d, 24, 282, "(a) Source-paired Directional Shortcut Alignment", 10.2, INK, "start", True)
    box(d, 35, 120, 135, 70, "Source (x, y)", ["semantic content fixed"], NAVY)
    operator_y = [220, 125, 30]
    operator_titles = ["Operator o", "Operator o'", "Operator o''"]
    operator_bodies = [["bound classes B(k,o)"], ["alternative corruption"], ["alternative corruption"]]
    operator_colors = [ORANGE, BLUE, CYAN]
    for y, title_value, body_value, color in zip(operator_y, operator_titles, operator_bodies, operator_colors):
        box(d, 220, y, 145, 62, title_value, body_value, color)
        arrow(d, 170, 155, 220, y + 31, color)
        arrow(d, 365, y + 31, 395, 155, color)
    box(d, 395, 111, 70, 88, "DSA", ["mass to B(k,o)", "under o versus", "other operators"], PURPLE)
    text(d, 250, 12, "Operator-invariant source terms cancel; shuffled binding supplies a null.", 7.5, MUTED)

    text(d, 494, 282, "(b) Same proxy evidence, different latent CLE", 10.2, INK, "start", True)
    box(d, 645, 222, 180, 58, "Same observable P(Y,Z)", ["task label Y + PEW proxy Z"], BLUE)
    box(d, 515, 105, 170, 72, "Latent world A", ["E independent of Y", "D(Y,E) = 0"], GREEN)
    box(d, 785, 105, 150, 72, "Latent world B", ["E = g(Y)", "D(Y,E) > 0"], RED)
    arrow(d, 700, 222, 600, 177, GREEN)
    arrow(d, 770, 222, 860, 177, RED)
    text(d, 725, 84, "identical proxy evidence, different true dependence", 8.0, PURPLE, "middle", True)
    arrow(d, 725, 75, 725, 58, PURPLE)
    box(d, 590, 8, 270, 50, "Target-aligned evaluation remains necessary", ["Use paired DSA; report proxy error separately"], PURPLE)
    write_all(d, "fig2_dsa_proxy_boundary")


def main():
    figure_one()
    figure_two()


if __name__ == "__main__":
    main()
