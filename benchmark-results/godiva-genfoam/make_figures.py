"""Figures and LaTeX tables for the Godiva / GeN-Foam report from data/.

data/godiva_1d.json  tools/godiva_1d.py (1D spherical S_N)
data/3d/*.json       tools/godiva_3d.py records (prisms, GH200)
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq, curve_fit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, TAB = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

K_GENFOAM = 0.980106
K_EXP = 1.0
# categorical palette (dataviz reference, light): fixed order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "legend.frameon": False})

d1 = json.load(open(os.path.join(HERE, "data", "godiva_1d.json")))
K1D = d1["k_converged"]
runs = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "data", "3d", "*.json"))]


def series(quad, scheme="scb", accel_pref=("power", "cmfd")):
    """{h: record} for one quadrature and scheme, power iteration preferred."""
    out = {}
    for r in runs:
        if r["quad"] != quad or r["scheme"] != scheme or r.get("treatment") != "blend":
            continue
        h = r["side"]
        acc = r.get("accel", "cmfd")
        if h not in out:
            out[h] = r
            continue
        a_old = accel_pref.index(out[h].get("accel", "cmfd"))
        a_new = accel_pref.index(acc)
        if a_new < a_old or (a_new == a_old and r["solve_s"] < out[h]["solve_s"]):
            out[h] = r                              # fastest equivalent run
    return dict(sorted(out.items(), reverse=True))


def pcm(k, ref):
    return 1e5 * (k - ref)


base = series("8x16")
h = np.array(list(base))
k = np.array([base[x]["k"] for x in h])


def model(hh, kinf, C, p):
    return kinf - C * hh ** p


fits = []
for name, sel in (("all 5 meshes", slice(0, 5)), ("finest 4", slice(1, 5)),
                  ("finest 3 (exact)", slice(2, 5))):
    hh, kk = h[sel], k[sel]
    if len(hh) == 3:
        g = lambda p: ((kk[1] - kk[0]) / (kk[2] - kk[1])
                       - (hh[0] ** p - hh[1] ** p) / (hh[1] ** p - hh[2] ** p))
        p = brentq(g, 0.3, 4.0)
        C = (kk[2] - kk[1]) / (hh[1] ** p - hh[2] ** p)
        kinf = kk[2] + C * hh[2] ** p
    else:
        (kinf, C, p), _ = curve_fit(model, hh, kk, p0=(0.994, 0.02, 1.5))
    res = np.abs(kk - model(hh, kinf, C, p)).max()
    fits.append((name, p, kinf, C, res))
k3d = np.mean([f[2] for f in fits])
spread = max(f[2] for f in fits) - min(f[2] for f in fits)

# ---- tables -----------------------------------------------------------------
with open(os.path.join(TAB, "mesh.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrrl}\n\\hline\n$h$ (cm) & prisms & $k$ & vs 1D (pcm) "
            "& solve (s) & outer acceleration\\\\\n\\hline\n")
    for x in h:
        r = base[x]
        acc = "power" if r.get("accel") == "power" else "CMFD (lagged LU)"
        f.write(f"{x:g} & {r['active']:,} & {r['k']:.6f} & {pcm(r['k'], K1D):+.0f} & "
                f"{r['solve_s']:.0f} & {acc}\\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
with open(os.path.join(TAB, "fits.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\hline\nfit & $p$ & $k_\\infty$ & vs 1D (pcm) "
            "& max residual (pcm)\\\\\n\\hline\n")
    for name, p, kinf, C, res in fits:
        f.write(f"{name} & {p:.2f} & {kinf:.6f} & {pcm(kinf, K1D):+.0f} & {1e5 * res:.1f}\\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

quads = [("8x16", "product $8\\times16$", 128), ("16x32", "product $16\\times32$", 512),
         ("quadratureSet", "GeN-Foam default", 32), ("quadratureSet16", "GeN-Foam 16/octant", 128)]
with open(os.path.join(TAB, "quad.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\hline\nset & directions & P2 Gram error & "
            "$h = 0.7$ & $h = 0.5$ & $h = 0.35$\\\\\n\\hline\n")
    for q, label, ndir in quads:
        s = series(q)
        gram = next((r.get("gram_error") for r in s.values()), None)
        cells = []
        for x in (0.7, 0.5, 0.35):
            if x in s and x in base:
                val = s[x]["k"]
                cells.append(f"{val:.6f}" if q == "8x16" else f"{pcm(val, base[x]['k']):+.1f}")
            else:
                cells.append("--")
        g = "--" if gram is None else (f"{gram:.1e}" if gram > 1e-12 else "exact")
        f.write(f"{label} & {ndir} & {g} & " + " & ".join(cells) + "\\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

step = series("quadratureSet", scheme="step")
with open(os.path.join(TAB, "step.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrr}\n\\hline\n$h$ (cm) & mean prism volume (cm$^3$) & "
            "$k$ & vs 3D reference (pcm) & vs 6-corner SCB (pcm)\\\\\n\\hline\n")
    for x, r in step.items():
        vol = (np.sqrt(3) / 4) * x * x * (x * np.sqrt(3) / 2)
        scb = series("quadratureSet").get(x)
        f.write(f"{x:g} & {vol:.4f} & {r['k']:.6f} & {pcm(r['k'], k3d):+.0f} & "
                + (f"{pcm(r['k'], scb['k']):+.0f}" if scb else "--") + "\\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

with open(os.path.join(TAB, "oned.tex"), "w") as f:
    f.write("\\begin{tabular}{lrr}\n\\hline\ncase (S$_{64}$, 200 shells unless stated) & $k$ & "
            "vs P2 (pcm)\\\\\n\\hline\n")
    f.write(f"P2, mesh and angle extrapolated & {K1D:.6f} & --\\\\\n")
    p2 = d1["scattering"]["P2"]
    for name, val in d1["scattering"].items():
        f.write(f"{name} & {val:.6f} & {pcm(val, p2):+.0f}\\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

with open(os.path.join(TAB, "numbers.tex"), "w") as f:
    f.write(f"\\newcommand{{\\kOneD}}{{{K1D:.6f}}}\n")
    f.write(f"\\newcommand{{\\kThreeD}}{{{k3d:.4f}}}\n")
    f.write(f"\\newcommand{{\\kThreeDspread}}{{{1e5 * spread / 2:.0f}}}\n")
    f.write(f"\\newcommand{{\\kGenFoam}}{{{K_GENFOAM:.6f}}}\n")
    f.write(f"\\newcommand{{\\GenFoamGap}}{{{pcm(K_GENFOAM, k3d):+.0f}}}\n")
    f.write(f"\\newcommand{{\\DataGap}}{{{pcm(K1D, K_EXP):+.0f}}}\n")

# ---- figure 1: k against mesh size ------------------------------------------------
# (the quadrature variants differ from 8 x 16 by < 20 pcm: table quad.tex)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
hf = np.linspace(0.0, 0.95, 200)
name, p, kinf, C, _ = fits[0]
ax.plot(hf, model(hf, kinf, C, p), color=BLUE, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.plot(h, k, "o", color=BLUE, ms=7, mew=0, zorder=3,
        label="6-corner SCB, $8\\times16$; fit $k_\\infty - C h^{%.2f}$" % p)
ax.plot(list(step), [r["k"] for r in step.values()], "^-", color=ORANGE, ms=8, lw=1.6,
        mew=0, zorder=3, label="step (first-order upwind), GeN-Foam 32-direction set")
ax.axhline(K1D, color=INK2, lw=1, zorder=1)
ax.text(0.94, K1D - 0.0004, "1D reference %.6f" % K1D, ha="right", va="top",
        color=INK2, fontsize=8)
v_gf = 0.016                                     # GeN-Foam mean tetrahedron volume
h_gf = (v_gf / (3.0 / 8.0)) ** (1.0 / 3.0)       # prism of equal volume
ax.plot([h_gf], [K_GENFOAM], "*", color=INK, ms=13, mew=0, zorder=4,
        label="GeN-Foam tutorial, at its mean cell volume")
ax.set_xlabel("prism side $h$ (cm)")
ax.set_ylabel("$k_{\\mathrm{eff}}$")
ax.set_xlim(0, 0.95)
ax.set_ylim(0.968, 0.9965)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "k_vs_h.pdf"))
plt.close(fig)

# ---- figure 2: 1D angular and mesh convergence ---------------------------------
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6))
a = d1["angle"]
axs[0].loglog(a["n_mu"][:-1], [abs(pcm(x, a["extrapolated"])) for x in a["k"][:-1]], "o-",
              color=BLUE, ms=6, mew=0, lw=1.6)
axs[0].set_xlabel("Gauss--Legendre order $N$")
axs[0].set_ylabel("$|k - k_\\infty|$ (pcm)")
axs[0].set_title("angle, 200 shells", fontsize=9)
m = d1["mesh"]
axs[1].loglog(m["shells"][:-1], [abs(pcm(x, m["extrapolated"])) for x in m["k"][:-1]], "o-",
              color=BLUE, ms=6, mew=0, lw=1.6)
for ax_, ticks in ((axs[0], a["n_mu"][:-1]), (axs[1], m["shells"][:-1])):
    ax_.xaxis.set_major_locator(FixedLocator(ticks))
    ax_.xaxis.set_minor_locator(NullLocator())
    ax_.xaxis.set_minor_formatter(NullFormatter())
    ax_.set_xticklabels([str(t) for t in ticks])
axs[1].set_xlabel("shells")
axs[1].set_title("mesh, S$_{64}$", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "oned_convergence.pdf"))
plt.close(fig)
print(f"k3d {k3d:.6f} spread {1e5 * spread:.1f} pcm; fits:", [(f[0], round(f[1], 3), round(f[2], 6)) for f in fits])
