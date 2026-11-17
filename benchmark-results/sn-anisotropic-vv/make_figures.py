"""Figure for the anisotropic S_N V&V note: mesh convergence of k on the
LA-13511 problems, anisotropic problems next to their isotropic controls.

Data: docs/qa/sph_audit_evidence/verify_sn_sood.out (GARNET, quadrature (4, 16),
SCB) and the independent 1-D S_N reference (tools/slab_sn_1d.py, S256).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SLOT = ("#2a78d6", "#eb6834", "#1baf7a")          # validated categorical slots 1-3

CYL_CELLS = (16, 32, 64, 128, 256)
CYL = {   # k - 1 in pcm
    "13 isotropic (control)": (-999.82, -387.51, -166.77, -81.35, -47.26),
    "36 P1, $\\Sigma_{s1}/\\Sigma_{s0}$ = 0.17": (-919.29, -357.85, -155.27, -76.60, -45.45),
    "37 P1, 0.86 (negative kernel)": (-546.46, -230.84, -111.46, -60.88, -41.24),
}
SLAB_CELLS = (80, 160, 320, 640, 1280, 2560, 5120)
SLAB = {
    "54 isotropic (control)": (-795.80, -294.05, -113.28, -45.33, -17.80, -5.78, -0.22),
    "71 P1 within-group and down-scatter": (-678.81, -266.19, -118.31, -63.99, -42.66,
                                            -33.60, -29.49),
}
ONE_D_71 = -31.523            # independent 1-D S256, Richardson


def panel(ax, cells, series, title, xlabel, yticks):
    markers = ("o", "s", "^")
    for i, (label, vals) in enumerate(series.items()):
        y = [-v for v in vals]
        ax.plot(cells, y, color=SLOT[i], lw=2, marker=markers[i], ms=6,
                markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(cells)
    ax.set_xticklabels([str(c) for c in cells])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:g}" for t in yticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel("$-(k-1)$  [pcm]", color=MUTED)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
panel(axes[0], CYL_CELLS, CYL, "(a) one-group bare cylinders", "cells across the diameter",
      (50, 100, 200, 500, 1000))
panel(axes[1], SLAB_CELLS, SLAB, "(b) two-group bare slabs", "cells across the thickness",
      (0.3, 1, 3, 10, 30, 100, 300, 1000))
axes[1].axhline(-ONE_D_71, color=MUTED, lw=1.2, ls="--", zorder=2)
axes[1].annotate("71: independent 1-D S$_N$", (SLAB_CELLS[0], -ONE_D_71),
                 xytext=(0, -11), textcoords="offset points", color=INK, fontsize=7.5)
for ax in axes:
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper right")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "figures", f"convergence.{ext}"), dpi=200)
print("wrote figures/convergence.pdf/.png")
