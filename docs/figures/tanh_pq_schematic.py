"""
Schematic of the tanh.pq relaxation (drawn into the hybopt.methods.tanh docstring by ``.. plot::``).

Every curve comes from the package (Piece, tanh_pieces, ...), so the figure follows the code.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from hybopt.methods.tanh import (
    Piece,
    curvature_range,
    evaluate,
    tanh_pieces,
    tanh_second_derivative,
)

BLUE, ORANGE, GRAY = "tab:blue", "tab:orange", "0.55"
ERROR = "tab:red"  # the band width w
ARROW = "-|>,head_length=0.6,head_width=0.16"  # slender head
WIDTHS = [0.22, 0.22, 0.22, 0.22]  # panels: network, 1, 2, 3
GAP = 0.03
BOTTOM, HEIGHT = 0.03, 0.78

fig = plt.figure(figsize=(11, 3.3))
lefts = np.concatenate([[0.015], 0.015 + np.cumsum(np.add(WIDTHS, GAP))[:-1]])


def panel(slot: int, left: float = 0.0, width: float = 1.0, frame: bool = True) -> plt.Axes:
    """Axes covering [left, left + width] (fractions) of a slot, without ticks."""
    axis = fig.add_axes([lefts[slot] + left * WIDTHS[slot], BOTTOM, width * WIDTHS[slot], HEIGHT])
    axis.set_box_aspect(1)  # square panels
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(GRAY)
        spine.set_visible(frame)
    return axis


def title(slot: int, text: str) -> None:
    fig.text(lefts[slot] + WIDTHS[slot] / 2, 0.82, text, ha="center", va="bottom", fontsize=10.5)


def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
    fig.patches.append(
        FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle=ARROW,
                        mutation_scale=14, linewidth=1.2, color="black",
                        joinstyle="miter", capstyle="butt")  # pointed tip, not rounded
    )


a, b = 0.3, 2.6
piece = Piece.make(a, b)
x_piece = np.linspace(a, b, 200)
x_wide = np.linspace(-0.3, 3.0, 300)
t = np.linspace(-3.0, 3.0, 400)

# trained network
title(0, "Trained tanh network")
network = panel(0, frame=False)
layers = [2, 3, 3, 1]
nodes = [[(k, (n - 1) / 2 - i) for i in range(n)] for k, n in enumerate(layers)]
for left_layer, right_layer in zip(nodes[:-1], nodes[1:]):
    for p in left_layer:
        for q in right_layer:
            network.plot([p[0], q[0]], [p[1], q[1]], color="black", linewidth=1, zorder=1)
for layer in nodes:
    for p in layer:
        network.scatter(*p, s=300, color="#e6e6e6", edgecolor="black", linewidth=1.5, zorder=2)
for k, name in ((0, "input"), (1, "tanh"), (2, "tanh"), (3, "linear")):
    network.text(k, -1.32, name, ha="center", va="top", fontsize=9.5)
network.set_xlim(-0.4, 3.4)
network.set_ylim(-2.0, 1.8)

# 1. chord on a piece
title(1, "1. Chord on a piece")
chord_axis = panel(1)
chord = piece.coefficients(0.0)
chord_axis.plot(x_wide, np.tanh(x_wide), color="black", linewidth=1.4)
chord_axis.plot(x_piece, evaluate(chord, x_piece), color=GRAY, linewidth=1.8)
chord_axis.fill_between(x_piece, evaluate(chord, x_piece), np.tanh(x_piece), color=GRAY,
                        alpha=0.2, linewidth=0)
chord_axis.plot([a, b], np.tanh([a, b]), "o", color="black", markersize=4.5)
chord_axis.text(a + 0.08, np.tanh(a) - 0.13, "a", fontsize=10)
chord_axis.text(b - 0.05, np.tanh(b) - 0.18, "b", fontsize=10, ha="right")
chord_axis.text(1.7, evaluate(chord, 1.7) - 0.05, "chord", color="0.35", fontsize=10, va="top")
chord_axis.text(1.0, np.tanh(1.0) + 0.06, "tanh", fontsize=10, ha="right", va="bottom")
chord_axis.set_ylim(-0.4, 1.2)

# 2. curvature range -> enclosing quadratics, with tanh'' as an inset
title(2, "2. Quadratic bounds")
band_axis = panel(2)
band_axis.plot(x_wide, np.tanh(x_wide), color="black", linewidth=1.2)
band_axis.fill_between(x_piece, evaluate(piece.lower, x_piece), evaluate(piece.upper, x_piece),
                       color=BLUE, alpha=0.25, linewidth=0)
band_axis.plot(x_piece, evaluate(piece.upper, x_piece), color=BLUE, linewidth=1.4)
band_axis.plot(x_piece, evaluate(piece.lower, x_piece), color=ORANGE, linewidth=1.4)
band_axis.text(0.68, evaluate(piece.upper, 0.68) + 0.02, r"$q_{\mathrm{up}}$", color=BLUE,
               fontsize=10, ha="right", va="bottom")
band_axis.text(2.1, evaluate(piece.lower, 2.1) - 0.05, r"$q_{\mathrm{lo}}$", color=ORANGE,
               fontsize=10, va="top")
band_axis.set_ylim(-0.35, 1.5)
# the band is the error: tanh lies somewhere inside it, at most w from either quadratic
mid = (a + b) / 2
mid_lower, mid_upper = evaluate(piece.lower, mid), evaluate(piece.upper, mid)
band_axis.annotate("", xy=(mid, mid_lower), xytext=(mid, mid_upper),
                   arrowprops=dict(arrowstyle="<->", color=ERROR, linewidth=1.3,
                                   shrinkA=0, shrinkB=0))
band_axis.text(mid + 0.1, (np.tanh(mid) + mid_upper) / 2, "$w$", va="center", fontsize=11,
               color=ERROR)
inset = band_axis.inset_axes([0.5, 0.05, 0.46, 0.38])
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color(GRAY)
inset.plot(t, tanh_second_derivative(t), color="black", linewidth=1.0)
inset.axvspan(a, b, color=BLUE, alpha=0.15, linewidth=0)
c_min, _ = curvature_range(a, b)
inset.axhline(c_min, color=BLUE, linewidth=1.0)
inset.axhline(0.0, color=ORANGE, linewidth=1.0)
inset.text(-1.15, 0.72, r"$\tanh''$", fontsize=8.5, ha="right", va="center")
inset.set_ylim(-0.95, 0.95)
middle = BOTTOM + HEIGHT / 2

# 3. bisection of the widest band
title(3, "3. Halve the worst piece")
bisection_axis = panel(3, frame=False)
steps = range(2, 6)
scale = 0.85 / max(p.width for p in tanh_pieces(-3.0, 3.0, max_pieces=2))
for row, n_pieces in enumerate(steps):
    pieces = tanh_pieces(-3.0, 3.0, tol=1e-9, max_pieces=n_pieces)
    widest = int(np.argmax([p.width for p in pieces]))
    base = len(steps) - 1 - row
    for s, p in enumerate(pieces):
        xs = np.linspace(p.a, p.b, 100)
        band = evaluate(p.upper, xs) - evaluate(p.lower, xs)
        bisection_axis.fill_between(xs, base, base + scale * band, linewidth=0, alpha=0.6,
                                    color=ORANGE if s == widest else BLUE)
        bisection_axis.plot([p.a] * 2, [base, base + 0.85], color=GRAY, linestyle=":", lw=0.8)
    bisection_axis.plot([3.0] * 2, [base, base + 0.85], color=GRAY, linestyle=":", lw=0.8)
    bisection_axis.plot([-3.0, 3.0], [base, base], color="black", linewidth=0.8)
    # the largest error w of each row (the piece split next), as in panel 2; labeled once
    peak = pieces[widest]
    top = base + scale * peak.width
    centre = (peak.a + peak.b) / 2
    style = "<->" if row < 2 else "-"  # rows 3 and 4 are too short for arrowheads
    bisection_axis.annotate("", xy=(centre, base), xytext=(centre, top),
                            arrowprops=dict(arrowstyle=style, color=ERROR, linewidth=1.1,
                                            mutation_scale=10, shrinkA=0, shrinkB=0))
    if row == 0:
        bisection_axis.text(centre + 0.12, base + scale * peak.width / 2, "$w$",
                            va="center", fontsize=11, color=ERROR)
bisection_axis.set_ylim(-0.1, 3.95)

# flow: network -> 1 -> 2 -> 3
for slot in range(3):
    arrow((lefts[slot] + WIDTHS[slot] + 0.003, middle), (lefts[slot + 1] - 0.003, middle))
