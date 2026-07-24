"""Drag force on electrons (the Chandrasekar figure), as REPL cell (# %%) script.

A standalone schematic of the collisional drag force against parallel momentum,
with the Dreicer and avalanche regions marked. Reads the digitised drag curve
from `Chandrasekar.csv` in the repository root.
"""

# %%
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.stats

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV = REPO_ROOT / "Chandrasekar.csv"
OUTPUT = os.environ.get("CDO_OUTPUT_FILE", "")  # e.g. drag_force.png; empty shows the figure

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "xtick.major.size": 0,
    "xtick.minor.visible": False,
    "xtick.top": False,
    "ytick.direction": "in",
    "ytick.major.size": 0,
    "ytick.minor.visible": False,
    "ytick.right": False,
    "font.size": 20,
    "figure.dpi": 150,
    "lines.linewidth": 3,
    "axes.linewidth": 2,
    "figure.constrained_layout.use": True,
})


# %%
# Interpolate the digitised drag curve and build a schematic thermal bulk.
points = np.genfromtxt(CSV, delimiter=",")
spline = scipy.interpolate.splrep(points[:, 0], points[:, 1])
p = np.linspace(0, 125, 200)
drag = scipy.interpolate.splev(p, spline)

border = 70  # split between the sub-critical curve and the runaway region
min_after_border = np.argmin(drag[border:]) + border
bulk = 1600 * scipy.stats.norm(loc=0, scale=16).pdf(p)


# %%
# Assemble the figure.
fig = plt.figure(figsize=(8, 5))
axes = plt.gca()

axes.plot(p, bulk, zorder=1, c="orange", linewidth=2)
axes.plot(p[:border], drag[:border], c="k", zorder=5)
axes.plot(p[border - 1:], drag[border - 1:], c="red", zorder=6)
axes.fill_between(p, bulk, where=bulk >= 0, interpolate=True, color="orange", alpha=0.3)

axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)
axes.set_ylim(0, 53)
axes.set_xlim(0, 125)
axes.set_xticks([p[border - 1], 120])
axes.set_xticklabels([r"$p_c$", r"$p_{\parallel}$"])
axes.set_yticks([48, drag[border - 1], drag[min_after_border]])
axes.set_yticklabels([r"$F_s$", r"$eE_{\parallel}$", r"$eE_c$"])
for color, tick in zip(["red", "k"], axes.xaxis.get_ticklabels()):
    tick.set_color(color)
for color, tick in zip(["k", "red", "k"], axes.yaxis.get_ticklabels()):
    tick.set_color(color)

# Critical-momentum and critical-field guide lines.
axes.axvline(p[border - 1], ymax=drag[border - 1] / 53, c="red", ls=(0, (5, 5)), linewidth=2, zorder=3)
axes.axhline(drag[border - 1], xmax=p[border - 1] / 125, c="red", ls=(0, (5, 5)), linewidth=2, zorder=3)
axes.axvline(p[min_after_border], ymax=drag[min_after_border] / 53, c="k", ls=(0, (5, 5)), linewidth=2, zorder=2)
axes.axhline(drag[min_after_border], xmax=p[min_after_border] / 125, c="k", ls=(0, (5, 5)), linewidth=2, zorder=2)

# Axis arrowheads.
axes.plot(0.995, 0, ">k", transform=axes.get_yaxis_transform(), clip_on=False, ms=10)
axes.plot(0, 0.996, "^k", transform=axes.get_xaxis_transform(), clip_on=False, ms=10)

arrow = dict(tail_width=2, head_width=10, head_length=10)
axes.add_patch(patches.FancyArrowPatch(
    (35, 1), (52, 1), connectionstyle="arc3,rad=-.3",
    arrowstyle=f"Simple, tail_width={arrow['tail_width']}, "
               f"head_width={arrow['head_width']}, head_length={arrow['head_length']}",
    color="green", zorder=10))
axes.add_patch(patches.FancyArrowPatch(
    (15, 30), (62, 10), connectionstyle="arc3,rad=-.3",
    arrowstyle=f"Simple, tail_width={arrow['tail_width']}, "
               f"head_width={arrow['head_width']}, head_length={arrow['head_length']}",
    color="blue", zorder=10))

axes.annotate(xy=(8, 20), xytext=(6, 15), text=r"$\langle f_p\rangle$", c="#954900")
axes.annotate(xy=(0, 0), xytext=(17, 0.7), text="Dreicer", c="green", fontsize="small", fontweight="bold")
axes.annotate(xy=(0, 0), xytext=(23, 32), text="Avalanche", c="blue", fontsize="small", fontweight="bold")
axes.annotate(xy=(0, 0), xytext=(73, 8), text="Runaway region", c="red", fontsize="small", fontweight="bold")

if OUTPUT:
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)
