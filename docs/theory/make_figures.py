"""Tables, figures and inline numbers for docs/theory/report.tex from data/*.json.

data/*.json are written by run_cases.py; the HP-MR geometry figure is drawn
from garnet.benchmarks.hpmr directly.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
FIG, TAB = HERE / "figures", HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

# categorical palette (dataviz reference, light), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "lines.markersize": 5, "legend.frameon": False,
                     "pdf.fonttype": 42})


def load(name):
    return json.loads((HERE / "data" / f"{name}.json").read_text())


def order(errs, ratio=2.0):
    """Observed orders between successive meshes (|err| ratios)."""
    return [math.log(abs(a) / abs(b)) / math.log(ratio) for a, b in zip(errs, errs[1:])]


def fmt_order(p):
    return "--" if p is None else f"{p:.2f}"


def tex_table(path, head, rows, align):
    with open(TAB / path, "w") as f:
        f.write(f"\\begin{{tabular}}{{{align}}}\n\\hline\n" + " & ".join(head) + "\\\\\n\\hline\n")
        for r in rows:
            f.write(r if isinstance(r, str) else " & ".join(r) + "\\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def cost(r):
    o = r.get("outers")
    return [str(o) if o is not None else "--", f"{r['inners']:,}" if "inners" in r else "--",
            f"{r['seconds']:.2f}" if r["seconds"] < 10 else f"{r['seconds']:.1f}"]


num = {}
env = load("environment")

# ---- verification: eigenvalue convergence ---------------------------------------
box, cyl, prism, skew = load("box"), load("cylinder"), load("prism"), load("skewed")
refl = load("reflective")
vrows, series = [], []


def add_study(title, ref, rows, key, label_fmt):
    errs = [r["err_pcm"] for r in rows]
    ps = [None] + order(errs)
    vrows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}, reference $k = {ref:.7f}$}}\\\\\n")
    for r, p in zip(rows, ps):
        vrows.append([label_fmt(r), f"{r['k']:.7f}", f"{r['err_pcm']:+.3f}", fmt_order(p)] + cost(r))
    return errs, ps


e, p = add_study("Cartesian diffusion, bare cube", box["diffusion"]["k_ref"],
                 box["diffusion"]["rows"], "n", lambda r: f"${r['n']}^3$")
series.append(("Cartesian diffusion", [r["h"] for r in box["diffusion"]["rows"]], e))
num["boxOrder"] = p[-1]
e, p = add_study("Cartesian SP$_3$, bare cube", box["sp3"]["k_ref"], box["sp3"]["rows"], "n",
                 lambda r: f"${r['n']}^3$")
series.append(("Cartesian SP$_3$", [r["h"] for r in box["sp3"]["rows"]], e))
num["spOrder"] = p[-1]
e, p = add_study("r--z diffusion, bare cylinder", cyl["k_ref"], cyl["rows"], "n",
                 lambda r: f"${r['n']}\\times{2 * r['n']}$")
series.append(("r--z cylinder", [r["h"] for r in cyl["rows"]], e))
num["cylOrder"] = p[-1]
e, p = add_study("triangular prisms, slab in $z$", prism["k_ref"], prism["rows"], "nz",
                 lambda r: f"$n_z = {r['nz']}$")
series.append(("triangular prisms ($\\Delta z$)", [r["dz"] for r in prism["rows"]], e))
num["prismOrder"] = p[-1]
e, p = add_study("unstructured triangles (diagonal cut), non-orthogonal correction",
                 skew["k_ref"], skew["corrected"], "n", lambda r: f"{r['cells']:,} cells")
num["skewOrder"] = p[-1]
e2, _ = add_study("same meshes, two-point flux only", skew["k_ref"], skew["two-point only"], "n",
                  lambda r: f"{r['cells']:,} cells")
num["skewUncorrected"] = e2[-1]
tex_table("verification.tex", ["mesh", "$k$", "error (pcm)", "order", "outers", "inners",
                               "time (s)"], vrows, "lrrrrrr")

tex_table("reflective.tex", ["lattice", "$k$", "$k - k_\\infty$ (pcm)", "outers", "inners",
                             "time (s)"],
          [[r["case"], f"{r['k']:.10f}", f"{r['err_pcm']:+.1e}"] + cost(r) for r in refl["rows"]],
          "lrrrrr")
num["kinf"] = refl["k_ref"]
num["reflMax"] = max(abs(r["err_pcm"]) for r in refl["rows"])

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
ax = axs[0]
for (lab, hs, es), c in zip(series, C):
    hs = np.array(hs) / hs[0]
    ax.loglog(hs, np.abs(es), "o-", color=c, label=lab, mew=0)
hh = np.array([1, 1 / 16])
ax.loglog(hh, 3000 * hh ** 2, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(0.09, 3000 * 0.09 ** 2 * 2.2, "$\\propto h^2$", color=INK2, fontsize=8)
ax.set_xlabel("mesh spacing / coarsest")
ax.set_ylabel("$|k - k_\\mathrm{exact}|$ (pcm)")
ax.invert_xaxis()
ax.legend(fontsize=7, loc="lower left")
ax = axs[1]
n = [r["n"] for r in skew["corrected"]]
ax.semilogx(n, [r["err_pcm"] for r in skew["corrected"]], "o-", color=C[0], mew=0,
            label="non-orthogonal correction")
ax.semilogx(n, [r["err_pcm"] for r in skew["two-point only"]], "^-", color=C[1], mew=0,
            label="two-point flux only")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(n)
ax.set_xticklabels([str(x) for x in n])
ax.minorticks_off()
ax.set_xlabel("squares per side (2 triangles each)")
ax.set_ylabel("$k - k_\\mathrm{exact}$ (pcm)")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "convergence.pdf")
plt.close(fig)

# ---- verification: kinetics ------------------------------------------------------
kin = load("kinetics")
num["Pref"] = kin["P_ref"]
krows = []
fig, ax = plt.subplots(figsize=(4.4, 3.0))
nominal = {"backward Euler": 1, "Crank-Nicolson": 2, "BDF2": 2, "BDF3 (graded start)": 3,
           "BDF4 (graded start)": 4}
for (name, rows), c in zip(kin["schemes"].items(), C):
    errs = [r["rel_err"] for r in rows]
    ps = order(errs)
    krows.append([name, str(nominal[name])] + [f"{abs(x):.2e}" for x in errs]
                 + [f"{ps[-1]:.2f}"])
    ax.loglog([r["dt"] * 1e3 for r in rows], np.abs(errs), "o-", color=c, mew=0, label=name)
ax.set_xlabel("time step (ms)")
ax.set_ylabel("$|P(0.2\\,\\mathrm{s})/P_\\mathrm{ref} - 1|$")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "kinetics.pdf")
plt.close(fig)
steps = [r["steps"] for r in next(iter(kin["schemes"].values()))]
tex_table("kinetics.tex", ["scheme", "nominal"] + [f"{s} steps" for s in steps] + ["order"],
          krows, "lr" + "r" * (len(steps) + 1))

# ---- validation ------------------------------------------------------------------
bib, iaea, vver, tw = load("biblis"), load("iaea"), load("vver"), load("twigl")
bff = load("biblis_femffusion")
num["biblisRef"] = bib["k_ref"]
num["biblisRefUnc"] = bff["uncertainty_pcm"]
vrows = []
vser = []
for title, d, lab, key in (
        ("BIBLIS-2D (Cartesian, vacuum), FEMFFUSION FE3 limit", bib, "cells/assembly",
         "cells_per_assembly"),
        ("IAEA-3D (Cartesian, albedo), FEMFFUSION FE3 limit", iaea, "cells/node",
         "cells_per_node"),
        ("VVER-440 2-D (triangles), FEMFFUSION diffusion", vver, "refine", "refine"),
        ("TWIGL static (Cartesian), Hageman \\& Yasinsky", dict(k_ref=tw["k_ref"],
                                                               rows=tw["static"]),
         "cells/8 cm", "cells_per_8cm")):
    vrows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}: $k = {d['k_ref']:.7g}$}}\\\\\n")
    for r in d["rows"]:
        cells = f"{r['cells']:,}" if "cells" in r else "--"
        vrows.append([f"{r[key]}", cells, f"{r['k']:.6f}", f"{r['err_pcm']:+.1f}"] + cost(r))
    vser.append((title.split(" (")[0], [r[key] for r in d["rows"]],
                 [r["err_pcm"] for r in d["rows"]]))
tex_table("validation.tex", ["mesh", "active cells", "$k$", "$\\Delta k$ (pcm)", "outers",
                             "inners", "time (s)"], vrows, "rrrrrrr")
num["biblisFine"] = bib["rows"][-1]["err_pcm"]
num["biblisTwo"] = bib["rows"][1]["k"]
num["iaeaFine"] = iaea["rows"][-1]["err_pcm"]
num["vverFine"] = vver["rows"][-1]["err_pcm"]
num["twiglFine"] = tw["static"][-1]["err_pcm"]

trows = []
for r in tw["transient"]:
    trows.append([r["perturbation"], str(r["cells_per_8cm"]), f"{r['dt'] * 1e3:g}",
                  f"{r['P01']:.3f}", f"{r['P05']:.3f}", f"{r['inners']:,}",
                  f"{r['seconds']:.1f}"])
ref = {p: (r["ref01"], r["ref05"]) for r in tw["transient"] for p in [r["perturbation"]]}
for p, (a, b) in ref.items():
    trows.append([f"{p}, reference", "", "", f"{a:.2f}", f"{b:.2f}", "", ""])
tex_table("twigl.tex", ["perturbation", "cells/8 cm", "$\\Delta t$ (ms)", "$P(0.1\\,$s$)$",
                        "$P(0.5\\,$s$)$", "inners", "time (s)"], trows, "lrrrrrr")

fig, ax = plt.subplots(figsize=(4.4, 3.0))
for (lab, x, e), c in zip(vser, C):
    x = np.array(x, float)
    ax.loglog(x / x[0], np.abs(e), "o-", color=c, mew=0, label=lab)
ax.set_xlabel("refinement (cells per assembly side, relative)")
ax.set_ylabel("$|k - k_\\mathrm{ref}|$ (pcm)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "validation.pdf")
plt.close(fig)

# ---- HP-MR -------------------------------------------------------------------------
hp = load("hpmr")
refines = sorted(hp["rows"], key=int)
fine = hp["rows"][refines[-1]]


def worth(rows):
    """Reactivity (pcm) relative to the withdrawn state (180 deg)."""
    k180 = next(r["k"] for r in rows if r["angle"] == 180)
    return [1e5 * (1 / k180 - 1 / r["k"]) for r in rows]


hrows = []
angles = [r["angle"] for r in fine]
for i, a in enumerate(angles):
    cells = []
    for rf in refines:
        r = hp["rows"][rf][i]
        cells += [f"{r['k']:.5f}", f"{worth(hp['rows'][rf])[i]:+.0f}"]
    hrows.append([f"{a}"] + cells)
tex_table("hpmr.tex", ["angle"] + sum(([f"$k$, $r={rf}$", "$\\rho$ (pcm)"] for rf in refines),
                                      []), hrows, "r" + "rr" * len(refines))
hcost = []
for rf in refines:
    rr = hp["rows"][rf]
    hcost.append([rf, f"{rr[0]['cells']:,}", f"{rr[0]['cells'] * hp['groups']:,}",
                  f"{np.mean([r['outers'] for r in rr]):.0f}",
                  f"{np.mean([r['inners'] for r in rr]):,.0f}",
                  f"{np.mean([r['seconds'] for r in rr]):.1f}"])
hp32, noabs = load("hpmr32"), load("hpmr_noabs")
r32 = {r["angle"]: r for r in hp32["rows"]}
hcost.append(["32 (A100)", f"{r32[0]['cells']:,}", f"{r32[0]['cells'] * hp['groups']:,}",
              f"{np.mean([r['outers'] for r in r32.values()]):.0f}",
              f"{np.mean([r['inners'] for r in r32.values()]):,.0f}",
              f"{np.mean([r['seconds'] for r in r32.values()]):.1f}"])
tex_table("hpmr_cost.tex", ["refine", "cells", "unknowns", "outers", "inners", "time (s)"],
          [[str(c) for c in r] for r in hcost], "rrrrrr")
series_rows, prev = [], None
na = {r["refine"]: r["k"] for r in noabs["rows"]}
for rf in [int(x) for x in refines] + [32]:
    if rf == 32:
        k0, k180 = r32[0]["k"], r32[180]["k"]
    else:
        rows = hp["rows"][str(rf)]
        k0, k180 = rows[0]["k"], rows[-1]["k"]
    wt = 1e5 * (1 / k0 - 1 / k180)
    side = 26.752 / (math.sqrt(3) * rf)
    series_rows.append([str(rf), f"{side:.2f}", f"{k0:.5f}", f"{k180:.5f}", f"{wt:,.0f}",
                        "--" if prev is None else f"{wt - prev:+.0f}", f"{na[rf]:.6f}"])
    prev = wt
tex_table("hpmr_series.tex", ["refine", "side (cm)", "$k$, 0$^\\circ$", "$k$, 180$^\\circ$",
                              "worth (pcm)", "change", "$k$, no B$_4$C"],
          series_rows, "rrrrrrr")
num["hpmrWorthFine"] = 1e5 * (1 / r32[0]["k"] - 1 / r32[180]["k"])
num["noabsSpread"] = 1e5 * (max(na.values()) - min(na.values()))
num["rThirtyTwoTime"] = np.mean([r["seconds"] for r in r32.values()])
w = {rf: worth(hp["rows"][rf]) for rf in refines}
num["hpmrWorth"] = -w[refines[-1]][0]
num["hpmrWorthMid"] = -w[refines[-2]][0]
num["hpmrWorthCoarse"] = -w[refines[0]][0]
num["hpmrKout"] = fine[-1]["k"]
num["hpmrKin"] = fine[0]["k"]

fig, ax = plt.subplots(figsize=(4.4, 3.0))
for rf, c in zip(refines, C):
    ax.plot(angles, w[rf], "o-", color=c, mew=0, label=f"refine {rf}")
ax.set_xticks(angles)
ax.set_xlabel("drum angle (deg; 0 = arcs toward the core)")
ax.set_ylabel("reactivity vs 180$^\\circ$ (pcm)")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "hpmr_worth.pdf")
plt.close(fig)

# geometry: the raster as the solver sees it, with the exact absorber arcs
from garnet.benchmarks import hpmr as H  # noqa: E402

ANGLE = 90.0
ras = H.hpmr_raster(6, np.full(12, ANGLE), paint_absorber=False)
mm = ras.material_map
polys, cols = [], []
colour = {H.FUEL: C[0], H.CENTRAL: C[3], H.BE_REFLECTOR: C[2], H.DRUM_BE: "#b3b2ab"}
for a in range(mm.shape[0]):
    for b in range(mm.shape[1]):
        for t in range(2):
            m = int(mm[a, b, t])
            if m:
                polys.append(ras.cell_vertices(a, b, t))
                cols.append(colour[m])
fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="white", linewidths=0.15))
drum_xy, arc_az = H._drum_geometry(np.full(12, ANGLE))
for (x, y), az in zip(drum_xy, arc_az):
    deg = math.degrees(az)
    ax.add_patch(Wedge((x, y), H.DRUM_RADIUS, deg - H.DRUM_ARC_HALF_DEG,
                       deg + H.DRUM_ARC_HALF_DEG, width=H.DRUM_RADIUS - H.DRUM_ABSORBER_INNER,
                       facecolor="#e34948", edgecolor="none"))
ax.set_aspect("equal")
ax.autoscale_view()
ax.axis("off")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C[0], C[3], C[2], "#b3b2ab", "#e34948")]
ax.legend(handles, ["fuel assembly", "central cell", "Be reflector", "drum body (Be)",
                    "B$_4$C arc (exact area)"], fontsize=7, loc="upper center",
          bbox_to_anchor=(0.5, 0.0), ncol=3)
fig.tight_layout()
fig.savefig(FIG / "hpmr_geometry.pdf")
plt.close(fig)
num["hpmrCellsSix"] = len(polys)

# ---- inline numbers ----------------------------------------------------------------
with open(TAB / "numbers.tex", "w") as f:
    def cmd(name, val):
        f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
    cmd("boxOrder", f"{num['boxOrder']:.2f}")
    cmd("spOrder", f"{num['spOrder']:.2f}")
    cmd("cylOrder", f"{num['cylOrder']:.2f}")
    cmd("prismOrder", f"{num['prismOrder']:.2f}")
    cmd("skewOrder", f"{num['skewOrder']:.2f}")
    cmd("skewUncorrected", f"{num['skewUncorrected']:+.0f}")
    cmd("reflMax", f"{num['reflMax']:.0e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
    cmd("Pref", f"{num['Pref']:.10f}")
    cmd("kinfval", f"{num['kinf']:.7f}")
    cmd("biblisRef", f"{num['biblisRef']:.6f}")
    cmd("biblisRefUnc", f"{num['biblisRefUnc']:.1f}")
    cmd("biblisFine", f"{num['biblisFine']:+.1f}")
    cmd("biblisTwo", f"{num['biblisTwo']:.5f}")
    cmd("iaeaFine", f"{num['iaeaFine']:+.1f}")
    cmd("vverFine", f"{num['vverFine']:+.1f}")
    cmd("twiglFine", f"{num['twiglFine']:+.1f}")
    cmd("hpmrWorth", f"{num['hpmrWorth']:,.0f}")
    cmd("hpmrWorthMid", f"{num['hpmrWorthMid']:,.0f}")
    cmd("hpmrWorthCoarse", f"{num['hpmrWorthCoarse']:,.0f}")
    cmd("hpmrWorthFine", f"{num['hpmrWorthFine']:,.0f}")
    cmd("noabsSpread", f"{num['noabsSpread']:.1f}")
    cmd("rThirtyTwoTime", f"{num['rThirtyTwoTime']:.0f}")
    cmd("hpmrKout", f"{num['hpmrKout']:.5f}")
    cmd("hpmrKin", f"{num['hpmrKin']:.5f}")
    cmd("hpmrCellsSix", f"{num['hpmrCellsSix']:,}")
    cmd("runHost", env["host"].split(".")[0])
    cmd("runDate", env["date"])
    cmd("runNumpy", env["numpy"])
    cmd("runPython", env["python"])
print({k: v for k, v in num.items()})
