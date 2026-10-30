"""Figures, LaTeX tables and summary.json for the Carreno et al. (2024) V&V report.

    MPLCONFIGDIR=/tmp/garnet-matplotlib python make_figures.py

Reads data/runs*/ (records of tools/assess_carreno2024.py), data/timing/
(best-of-3 timings), data/paper-figures.json (extract_paper_figures.py) and
data/nea-c5g7-mcnp-pin-powers.json. Writes figures/*.pdf, tables/*.tex and
data/summary.json. pcm are 1e5 (k - k_ref), signed; the paper's Tables 6 and 7
also list 1e5 |k - k_ref| (not the relative form of its Eq. 49).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG, TAB = HERE / "figures", HERE / "tables"
METHODS = ["SP1", "SP3", "SP5", "SP7", "SDP1", "SDP2", "SDP3"]
BL_MESHES = [40, 80, 160, 320, 640]
K_OPENMOC, K_S16, K_MCNP = 0.80536, 0.80613, 1.18655
PAPER_BL = {"SDP1": 0.80161, "SDP2": 0.80373, "SDP3": 0.80402, "SP1": 0.77680,
            "SP3": 0.79904, "SP5": 0.80280, "SP7": 0.80354}
PAPER_BL_CPU = {"SDP1": 1.5, "SDP2": 2.0, "SDP3": 2.6, "SP1": 1.1, "SP3": 1.5, "SP5": 2.0, "SP7": 2.6}
PAPER_C5G7 = {"SDP1": (1.18242, 0.68, 0.03), "SDP2": (1.18219, 0.57, 0.02),
              "SDP3": (1.18286, 0.64, 0.03), "SP1": (1.18336, 1.43, 0.05),
              "SP3": (1.18271, 0.75, 0.03), "SP5": (1.18221, 0.59, 0.02),
              "SP7": (1.18231, 0.57, 0.02)}
COLORS = dict(zip(METHODS, ["#000000", "#1f77b4", "#2ca02c", "#9467bd",
                            "#ff7f0e", "#d62728", "#8c564b"]))
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def pcm(k, ref):
    return 1e5 * (k - ref)


def load_runs():
    runs = []
    for sub in sorted(HERE.glob("data/runs*")):
        for f in sorted(sub.glob("*.json")):
            j = json.loads(f.read_text())
            for r in j["runs"]:
                r.update(device=j["args"]["device"], file=f"{sub.name}/{f.name}",
                         node=j["environment"].get("node"))
                runs.append(r)
    return runs


def load_timing():
    out = {}
    for f in sorted(HERE.glob("data/timing/*.json")):
        j = json.loads(f.read_text())
        for r in j["runs"]:
            r["device"] = j["args"]["device"]
            out[(r["case"], r["method"], r["vacuum"], r["mesh"], r["device"])] = r
    return out


def fmt(v, spec, missing="--"):
    return missing if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def write_table(name, header, rows, align):
    lines = [r"\begin{tabular}{" + align + "}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    for row in rows:
        lines.append(r"\hline" if row is None else " & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- BL

def richardson3(k1, k2, k3, ratio=2.0):
    """Observed order, extrapolated k and GCI (Roache, Fs = 1.25) from three
    meshes refined by `ratio` (k3 finest). Falls back to p = 2 when the order
    is not defined (non-monotone differences)."""
    d1, d2 = k2 - k1, k3 - k2
    p = np.log(d1 / d2) / np.log(ratio) if d1 * d2 > 0 else np.nan
    pe = p if np.isfinite(p) and 0.5 < p < 4.0 else 2.0
    kinf = k3 + d2 / (ratio ** pe - 1.0)
    gci = 1.25 * abs(d2) / (ratio ** pe - 1.0)
    return p, kinf, gci


def bl_analysis(runs, figs):
    bl = {(r["method"], r["vacuum"], r["mesh"]): r for r in runs if r["case"] == "bl"}
    res = {}
    for m in METHODS + ["SP3-SPN", "SDP1-SDPN"]:
        for v in ("marshak", "robin"):
            vv = "marshak" if m == "SP1" else v
            ks = [bl.get((m, vv, n), {}).get("k") for n in BL_MESHES]
            # The finest three consecutive meshes. The 80/160/320 and
            # 160/320/640 triples agree within 0.21 pcm wherever both exist,
            # so the SDP2/SDP3 runs at n = 640 (GMRES, 8-20 h) were not run.
            avail = [i for i, k in enumerate(ks) if k is not None]
            if len(avail) < 3 or avail[-1] - avail[-3] != 2:
                continue
            triple = [ks[i] for i in avail[-3:]]
            p, kinf, gci = richardson3(*triple)
            res[(m, v)] = dict(k=dict(zip(BL_MESHES, ks)), p=p, kinf=kinf, gci=gci,
                               finest=BL_MESHES[avail[-1]],
                               converged=all(bl[(m, vv, n)]["converged"] for n in BL_MESHES
                                             if (m, vv, n) in bl))
    out = {f"{m}|{v}": dict(p=x["p"], kinf=x["kinf"], gci_pcm=1e5 * x["gci"],
                            k={str(n): k for n, k in x["k"].items()})
           for (m, v), x in res.items()}

    # Table: convergence.
    rows = []
    for m in METHODS:
        first = True
        for v in (("marshak",) if m == "SP1" else ("marshak", "robin")):
            x = res.get((m, v))
            if not x:
                continue
            rows.append([m if first else "", "--" if m == "SP1" else v.capitalize()]
                        + [fmt(x["k"][n], ".6f") for n in (80, 160, 320, 640)]
                        + [fmt(x["p"], ".2f"), fmt(x["kinf"], ".6f"), fmt(1e5 * x["gci"], ".2f")])
            first = False
    write_table("bl_convergence", ["Method", "Vacuum", "$k_{80}$", "$k_{160}$", "$k_{320}$",
                                   "$k_{640}$", "$p$", "$k_\\infty$", "GCI (pcm)"], rows, "llrrrrrrr")

    # Table: against the paper and the transport references.
    rows = []
    for m in METHODS:
        xm, xr = res.get((m, "marshak")), res.get((m, "robin"))
        if not xm:
            continue
        rows.append([m, f"{PAPER_BL[m]:.5f}", f"{pcm(PAPER_BL[m], K_OPENMOC):+.0f}",
                     f"{xm['kinf']:.5f}", f"{pcm(xm['kinf'], PAPER_BL[m]):+.1f}",
                     fmt(None if m == "SP1" or not xr else pcm(xr["kinf"], PAPER_BL[m]), "+.1f"),
                     f"{pcm(xm['kinf'], K_OPENMOC):+.0f}"])
    write_table("bl_vs_paper", ["Method", "Paper $k$", "Paper $-$ OpenMOC", "\\garnet{} $k_\\infty$",
                                "$-$ paper", "Robin $-$ paper", "\\garnet{} $-$ OpenMOC"],
                rows, "lrrrrrr")

    # Table: error budget on the test mesh (n = 80).
    rows = []
    split = {}
    for m in METHODS:
        xm, xr = res.get((m, "marshak")), res.get((m, "robin"))
        if not xm:
            continue
        disc = pcm(xm["k"][80], xm["kinf"])
        bcv = None if m == "SP1" or not xr else pcm(xr["kinf"], xm["kinf"])
        ang = pcm(xm["kinf"], K_OPENMOC)
        km80 = pcm(xm["k"][80], PAPER_BL[m])
        kr80 = None if m == "SP1" or not xr else pcm(xr["k"][80], PAPER_BL[m])
        split[m] = dict(discretization_80=disc, robin_minus_marshak=bcv, angular=ang,
                        marshak_80_minus_paper=km80, robin_80_minus_paper=kr80)
        rows.append([m, f"{disc:+.1f}", fmt(bcv, "+.1f"), f"{ang:+.0f}",
                     f"{km80:+.1f}", fmt(kr80, "+.1f")])
    write_table("bl_split", ["Method", "Mesh error at $n=80$", "Robin $-$ Marshak",
                             "Angular ($k_\\infty - k_{\\rm OpenMOC}$)",
                             "$k_{80} -$ paper, Marshak", "$k_{80} -$ paper, Robin"],
                rows, "lrrrrr")
    out["split"] = split

    # Table: equivalent code paths.
    rows = []
    eq = {}
    # The dedicated SDP1 solver has no Marshak term (its Marshak runs are the
    # SDPN block itself), so SDP1 is compared with Robin only.
    for a, b, vacs in (("SP3", "SP3-SPN", ("marshak", "robin")), ("SDP1", "SDP1-SDPN", ("robin",))):
        for v in vacs:
            d = [abs(bl[(a, v, n)]["k"] - bl[(b, v, n)]["k"]) for n in BL_MESHES
                 if (a, v, n) in bl and (b, v, n) in bl]
            if d:
                eq[f"{a}|{v}"] = max(d)
                rows.append([f"{a} dedicated vs {b.split('-')[1]} block", v.capitalize(),
                             f"{max(d):.1e}", "same condition" if v == "marshak"
                             else "per-moment term on different moments"])
    write_table("bl_equivalence", ["Comparison", "Vacuum", "max $|\\Delta k|$, $n=40$--$640$", "Note"],
                rows, "llrl")
    out["equivalence_max_dk"] = eq

    # Table: SDP3 coefficient-table variants (Sec. A.6 of the report).
    rows = []
    variants = {}
    for m, label in (("SDP3", "re-derived (\\garnet)"), ("SDP3-FFN", "FEMFFUSION code"),
                     ("SDP3-A6", "paper Eq.~(A.6) as printed")):
        ks = {n: bl.get((m, "marshak", n), {}).get("k") for n in BL_MESHES}
        tri = [n for n in (80, 160, 320) if ks[n] is not None]
        if len(tri) < 2:
            continue
        if len(tri) == 3:
            pp, kinf, _ = richardson3(*[ks[n] for n in tri])
        else:
            pp, kinf = 2.0, ks[160] + (ks[160] - ks[80]) / 3.0
        variants[m] = dict(k=ks, p=pp, kinf=kinf, minus_paper=pcm(kinf, PAPER_BL["SDP3"]))
        rows.append([label] + [fmt(ks[n], ".6f") for n in (80, 160, 320)]
                    + [fmt(pp, ".2f"), f"{kinf:.6f}", f"{pcm(kinf, PAPER_BL['SDP3']):+.1f}"])
    write_table("bl_sdp3_tables", ["SDP3 tables", "$k_{80}$", "$k_{160}$", "$k_{320}$", "$p$",
                                   "$k_\\infty$ (80--320)", "$-$ paper"], rows, "lrrrrrr")
    out["sdp3_tables"] = variants

    # Fig.: mesh convergence.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, v in zip(axes, ("marshak", "robin")):
        for m in METHODS:
            x = res.get((m, v)) or (res.get((m, "marshak")) if m == "SP1" else None)
            if not x:
                continue
            ns = [n for n in BL_MESHES if x["k"][n] is not None and n < x["finest"]]
            h = np.array([10.0 / n for n in ns])
            e = np.abs([x["k"][n] - x["kinf"] for n in ns]) * 1e5
            ax.loglog(h, e, "o-", ms=3, color=COLORS[m], label=m)
        hh = np.array([0.03, 0.25])
        ax.loglog(hh, 0.3 * (hh / 0.03) ** 2 * 10, "k--", lw=0.7, label="slope 2")
        ax.set_xlabel("cell size $h$ (cm)")
        ax.set_title(f"{v.capitalize()} vacuum")
    axes[0].set_ylabel(r"$|k(h) - k_\infty|$ (pcm)")
    axes[1].legend(fontsize=6, ncol=2)
    fig.savefig(FIG / "bl_convergence.pdf")
    plt.close(fig)

    # Fig.: extrapolated k against the paper and the transport references.
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    xpos = np.arange(len(METHODS))
    for i, m in enumerate(METHODS):
        xm, xr = res.get((m, "marshak")), res.get((m, "robin"))
        ax.plot(i - 0.15, pcm(PAPER_BL[m], K_OPENMOC), "s", color="0.3", ms=5,
                label="paper (FEM, Marshak)" if i == 0 else None)
        if xm:
            ax.plot(i, pcm(xm["kinf"], K_OPENMOC), "o", color="C0", ms=5,
                    label=r"GARNET $k_\infty$, Marshak" if i == 0 else None)
        if xr and m != "SP1":
            ax.plot(i + 0.15, pcm(xr["kinf"], K_OPENMOC), "^", color="C3", ms=5,
                    label=r"GARNET $k_\infty$, Robin" if i == 1 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(pcm(K_S16, K_OPENMOC), color="k", lw=0.6, ls=":", label="TWODANT S$_{16}$")
    ax.set_xticks(xpos, METHODS)
    ax.set_ylabel("$k - k_{\\rm OpenMOC}$ (pcm)")
    ax.set_ylim(-700, 120)
    ax.text(0.02, 0.04, f"SP1: paper {pcm(PAPER_BL['SP1'], K_OPENMOC):+.0f}, off scale",
            transform=ax.transAxes, fontsize=6)
    ax.legend(fontsize=6, loc="lower right")
    fig.savefig(FIG / "bl_keff.pdf")
    plt.close(fig)

    # Fig. 4: flux along y = 4.5 cm.
    paper = figs["fig4"]
    ref = paper["curves"]["OpenMOC"]
    rx, ry = np.array(ref["x"]), np.array(ref["y"])
    flux = {}
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(rx, ry, color="k", lw=1.6, label="OpenMOC (paper Fig. 4)")
    for name, mk in (("SDP3", "o"), ("SP7", "s")):
        mx, my = np.array(paper["markers"][name]["x"]), np.array(paper["markers"][name]["y"])
        c, dev = _scaled_dev(mx, my, rx, ry)
        flux[f"paper {name}"] = dict(rms_pct_peak=dev[0], max_pct_peak=dev[1], scale=c)
        ax.plot(mx, my, mk, ms=2.5, mfc="none", color="0.4", label=f"paper {name} (markers)")
    finest = max((n for (m, v, n) in bl if m == "SDP3"), default=None)
    for m in ("SP3", "SP7", "SDP1", "SDP3"):
        r = bl.get((m, "marshak", finest))
        if not r:
            continue
        gx, gy = np.array(r["line_x"]), np.array(r["line_phi"])
        c, dev = _scaled_dev(gx, gy, rx, ry)
        flux[m] = dict(rms_pct_peak=dev[0], max_pct_peak=dev[1], mesh=finest)
        ax.plot(gx, c * gy, color=COLORS[m], lw=0.8, label=f"GARNET {m}, Marshak, $n={finest}$")
        sel = (gx >= rx.min()) & (gx <= rx.max())
        bx.plot(gx[sel], 100 * (c * gy[sel] - np.interp(gx[sel], rx, ry)) / ry.max(),
                color=COLORS[m], lw=0.8)
    unc = 100 * paper["uncertainty"]["y"] / ry.max()
    bx.axhspan(-unc, unc, color="0.85", label="extraction uncertainty")
    bx.set_ylabel("GARNET $-$ OpenMOC\n(% of peak)")
    bx.set_xlabel("$x$ (cm) at $y = 4.5$ cm")
    bx.legend(fontsize=6)
    ax.set_ylabel("scalar flux (paper units)")
    ax.legend(fontsize=6)
    fig.savefig(FIG / "bl_flux_line.pdf")
    plt.close(fig)
    out["flux_line"] = flux
    out["flux_extraction_uncertainty_pct_peak"] = unc
    rows = [[k, f"{v['rms_pct_peak']:.2f}", f"{v['max_pct_peak']:.2f}"] for k, v in flux.items()]
    write_table("bl_flux", ["Solution", "RMS (\\% of peak)", "max (\\% of peak)"], rows, "lrr")
    return out


def _scaled_dev(x, y, rx, ry):
    """Least-squares scale of (x, y) onto the reference curve, then the RMS and
    largest deviation in % of the reference peak, over the common x range."""
    sel = (x >= rx.min()) & (x <= rx.max())
    ref = np.interp(x[sel], rx, ry)
    c = float(np.dot(y[sel], ref) / np.dot(y[sel], y[sel]))
    d = (c * y[sel] - ref) / ry.max() * 100
    return c, (float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d))))


# ------------------------------------------------------------------------- C5G7

def fit_ladder(ss, ks):
    """Mesh limit of the pin-resolved C5G7 k.

    The pin-resolved model does not converge monotonically: how the fuel circle
    cuts the cells changes from mesh to mesh, and every method dips at s = 12.
    A power-law extrapolation through such points is arbitrary, so the estimate
    is the finest-mesh k. Its uncertainty is the larger of the largest
    difference from it among the other meshes with s >= 16 and its distance to
    a least-squares k_inf + b/s^2 over s >= 16."""
    ss, ks = np.asarray(ss, float), np.asarray(ks, float)
    order = np.argsort(ss)
    ss, ks = ss[order], ks[order]
    fine = ss >= 16
    if fine.sum() < 2:
        return None, None, {}
    central = ks[-1]
    unc = float(np.max(np.abs(ks[fine][:-1] - central)))
    A = np.vstack([np.ones(fine.sum()), 1 / ss[fine] ** 2]).T
    check = float(np.linalg.lstsq(A, ks[fine], rcond=None)[0][0])
    unc = max(unc, abs(check - central))
    return central, unc, {"finest_s": int(ss[-1]), "fit k_inf + b/s^2, s>=16": check}


def pin_metrics(P, ref):
    fuel = ref > 0
    e = (P[fuel] - ref[fuel]) / ref[fuel]
    asm = {"inner UO2": (slice(0, 17), slice(0, 17)), "MOX": (slice(0, 17), slice(17, 34)),
           "outer UO2": (slice(17, 34), slice(17, 34))}
    return dict(
        avg=100 * np.mean(np.abs(e)),                         # Eq. (50)
        rmse_paper=100 * np.sqrt(np.sum(e ** 2)) / e.size,    # what Table 7's RMSE matches
        rms=100 * np.sqrt(np.mean(e ** 2)),
        mre=100 * np.sum(np.abs(e) * ref[fuel]) / np.sum(ref[fuel]),
        max_abs=100 * np.max(np.abs(e)),
        p_max=float(P.max()), p_max_err=100 * (P.max() / ref.max() - 1),
        assemblies={k: 100 * (P[s].sum() / ref[s].sum() - 1) for k, s in asm.items()})


def c5g7_analysis(runs, figs, nea):
    ref = np.array(nea["pin_powers"])
    pr = {}
    for r in runs:
        if r["case"] != "c5g7":
            continue
        key = (r["geometry"], r["method"], r["vacuum"], r["mesh"], r.get("area", "exact"), r["device"])
        pr[key] = r
    out = {}

    def get(geom, m, v, s, area="exact"):
        """CPU record if present, else GPU."""
        return pr.get((geom, m, v, s, area, "cpu")) or pr.get((geom, m, v, s, area, "gpu"))

    # CPU/GPU agreement.
    agree = {}
    for (g, m, v, s, a, dev), r in pr.items():
        if dev == "gpu" and (g, m, v, s, a, "cpu") in pr:
            agree[f"{m}|{v}|s{s}"] = abs(r["k"] - pr[(g, m, v, s, a, "cpu")]["k"])
    out["cpu_gpu_max_dk"] = agree

    # Mesh ladder, pin-resolved, Marshak.
    ladder = {}
    for m in METHODS:
        pts = sorted({s for (g, mm, v, s, a, d) in pr if g == "pin-resolved" and mm == m
                      and v == "marshak" and a == "exact"})
        ks = [get("pin-resolved", m, "marshak", s)["k"] for s in pts]
        kinf, unc, alts = fit_ladder(pts, ks)
        ladder[m] = dict(s=pts, k=ks, kinf=kinf, unc=unc, alts=alts)
    out["ladder"] = ladder

    rows = []
    for m in METHODS:
        L = ladder[m]
        kd = dict(zip(L["s"], L["k"]))
        pk = PAPER_C5G7[m][0]
        rows.append([m, f"{pk:.5f} ({pcm(pk, K_MCNP):+.0f})"]
                    + [fmt(kd.get(s), ".5f") for s in (4, 8, 12, 16, 24, 32, 48)]
                    + [fmt(L["kinf"], ".5f"), fmt(None if L["unc"] is None else 1e5 * L["unc"], ".1f"),
                       fmt(None if L["kinf"] is None else pcm(L["kinf"], K_MCNP), "+.0f"),
                       fmt(None if L["kinf"] is None else pcm(L["kinf"], pk), "+.0f")])
    write_table("c5g7_keff", ["Method", "Paper (pcm)", "$s=4$", "$s=8$", "$s=12$", "$s=16$", "$s=24$",
                              "$s=32$", "$s=48$", "$k$ estimate", "$\\pm$ (pcm)", "$-$ MCNP", "$-$ paper"],
                rows, "llrrrrrrrrrrr")

    # Modelling effects at s = 8.
    eff = {}
    rows = []
    for m in METHODS:
        km = get("pin-resolved", m, "marshak", 8)
        kr = get("pin-resolved", m, "robin", 8)
        kz = get("pin-resolved", m, "zero-flux", 8)
        kh = get("homogenized", m, "zero-flux", 8)
        e = dict(robin=None if not (km and kr) or m == "SP1" else pcm(kr["k"], km["k"]),
                 zero_flux=None if not (km and kz) else pcm(kz["k"], km["k"]),
                 homogenized=None if not (kz and kh) else pcm(kh["k"], kz["k"]))
        eff[m] = e
        rows.append([m] + [fmt(e[k], "+.2f") for k in ("robin", "zero_flux")]
                    + [fmt(e["homogenized"], "+.0f")])
    write_table("c5g7_effects", ["Method", "Robin $-$ Marshak", "zero flux $-$ Marshak",
                                 "homogenized $-$ pin-resolved (zero flux)"], rows, "lrrr")
    area = {}
    for s in (4, 8, 16):
        a, b = get("pin-resolved", "SP1", "marshak", s, "sampled"), get("pin-resolved", "SP1", "marshak", s)
        if a and b:
            area[s] = pcm(a["k"], b["k"])
    eff["sampled_area_minus_exact_SP1"] = area
    out["effects_s8"] = eff

    # Homogenized pin cells (what tests/validation/test_c5g7.py runs), zero flux.
    hom = {}
    rows = []
    for m in METHODS:
        z = {s: get("homogenized", m, "zero-flux", s) for s in (1, 2, 4, 8)}
        kinf = None
        if z[4] and z[8]:
            kinf = z[8]["k"] + (z[8]["k"] - z[4]["k"]) / 3.0
        hom[m] = dict(zero_flux={s: r["k"] for s, r in z.items() if r}, kinf=kinf)
        rows.append([m] + [fmt(z[s] and z[s]["k"], ".5f") for s in (1, 2, 4, 8)]
                    + [fmt(kinf, ".5f"), fmt(z[1] and pcm(z[1]["k"], K_MCNP), "+.0f"),
                       fmt(None if kinf is None else pcm(kinf, K_MCNP), "+.0f"),
                       fmt(None if kinf is None else pcm(kinf, PAPER_C5G7[m][0]), "+.0f")])
    write_table("c5g7_homogenized", ["Method", "$s=1$ (test)", "$s=2$", "$s=4$", "$s=8$", "$k_\\infty$",
                                     "test $-$ MCNP", "$k_\\infty -$ MCNP", "$k_\\infty -$ paper"],
                rows, "lrrrrrrrr")
    out["homogenized"] = hom

    # SDP3 coefficient tables on C5G7, at s = 4.
    shift = {}
    base = get("pin-resolved", "SDP3", "marshak", 4)
    for m in ("SDP3-FFN", "SDP3-A6"):
        r = get("pin-resolved", m, "marshak", 4)
        if base and r:
            shift[m] = pcm(r["k"], base["k"])
    out["sdp3_table_shift_s4"] = shift

    # Pin powers against MCNP at the finest mesh.
    pins = {}
    rows = []
    for m in METHODS:
        L = ladder[m]
        if not L["s"]:
            continue
        s = L["s"][-1]
        r = get("pin-resolved", m, "marshak", s)
        pm = pin_metrics(np.array(r["pin_powers"]), ref)
        conv = {}
        for s2 in L["s"]:
            conv[s2] = pin_metrics(np.array(get("pin-resolved", m, "marshak", s2)["pin_powers"]), ref)["avg"]
        pm.update(mesh=s, avg_by_mesh=conv)
        pins[m] = pm
        a = pm["assemblies"]
        rows.append([m, str(s), f"{pm['avg']:.2f} ({PAPER_C5G7[m][1]:.2f})",
                     f"{pm['rmse_paper']:.3f} ({PAPER_C5G7[m][2]:.2f})", f"{pm['rms']:.2f}",
                     f"{pm['mre']:.2f}", f"{pm['max_abs']:.2f}", f"{pm['p_max']:.3f}",
                     f"{a['inner UO2']:+.2f}", f"{a['MOX']:+.2f}", f"{a['outer UO2']:+.2f}"])
    write_table("c5g7_pins", ["Method", "$s$", "AVG (paper)", "RMSE (paper)", "RMS", "MRE",
                              "max $|e|$", "$P_{\\max}$", "inner UO$_2$", "MOX", "outer UO$_2$"],
                rows, "lrrrrrrrrrr")
    out["pin_powers"] = pins

    # Fig.: k against mesh.
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for m in METHODS:
        L = ladder[m]
        if not L["s"]:
            continue
        ax.plot(L["s"], [pcm(k, K_MCNP) for k in L["k"]], "o-", ms=3, color=COLORS[m], label=m)
        if L["kinf"] is not None:
            ax.errorbar([62], [pcm(L["kinf"], K_MCNP)], yerr=[1e5 * L["unc"]], fmt="o", ms=3,
                        color=COLORS[m], capsize=2)
        ax.plot([80], [pcm(PAPER_C5G7[m][0], K_MCNP)], "s", ms=4, mfc="none", color=COLORS[m])
    ax.set_xscale("log")
    ticks = [4, 6, 8, 12, 16, 24, 32, 48, 62, 80]
    ax.set_xticks(ticks, [str(t) for t in ticks[:-2]] + ["est.", "paper"])
    ax.minorticks_off()
    ax.set_xlabel("cells per pin side $s$")
    ax.set_ylabel("$k - k_{\\rm MCNP}$ (pcm)")
    ax.legend(fontsize=6, ncol=4)
    fig.savefig(FIG / "c5g7_keff.pdf")
    plt.close(fig)

    # Fig.: pin power error maps.
    show = [m for m in ("SP1", "SP3", "SDP3") if m in pins]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(2.5 * len(show) + 0.8, 2.6))
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, show):
            s = pins[m]["mesh"]
            P = np.array(get("pin-resolved", m, "marshak", s)["pin_powers"])
            E = np.where(ref > 0, 100 * (P - ref) / np.where(ref > 0, ref, 1), np.nan)[:34, :34]
            im = ax.imshow(E, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
            ax.set_title(f"{m}, $s={s}$", fontsize=8)
            ax.set_xticks([0, 16, 33]); ax.set_yticks([0, 16, 33]); ax.grid(False)
        fig.colorbar(im, ax=axes, label="pin power error vs MCNP (%)", shrink=0.9)
        fig.savefig(FIG / "c5g7_pin_errors.pdf")
        plt.close(fig)

    # Figs. 6-7: flux along the diagonals.
    lines = {}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for ax, key in zip(axes.flat, ("fig6a", "fig6b", "fig7a", "fig7b")):
        f = figs[key]
        rx, ry = np.array(f["curves"]["OpenMC"]["x"]), np.array(f["curves"]["OpenMC"]["y"])
        ax.plot(rx, ry, color="k", lw=1.4, label="OpenMC (paper)")
        px, py = np.array(f["curves"]["SDP3"]["x"]), np.array(f["curves"]["SDP3"]["y"])
        c, dev = _scaled_dev(px, py, rx, ry)
        lines[f"{key} paper SDP3"] = dict(rms_pct_peak=dev[0], max_pct_peak=dev[1])
        ax.plot(px, py, ":", color="0.55", lw=1.0, label="paper SDP3")
        line = ("fig6" if key.startswith("fig6") else "fig7") + ("_g1" if key.endswith("a") else "_g7")
        for m in ("SP3", "SDP3"):
            L = ladder[m]
            if not L["s"]:
                continue
            r = get("pin-resolved", m, "marshak", L["s"][-1])
            if "lines" not in r:
                continue
            gx, gy = np.array(r["lines"]["d"]), np.array(r["lines"][line])
            c, dev = _scaled_dev(gx, gy, rx, ry)
            lines[f"{key} GARNET {m}"] = dict(rms_pct_peak=dev[0], max_pct_peak=dev[1], mesh=L["s"][-1])
            ax.plot(gx, c * gy, color={"SP3": "C1", "SDP3": "C3"}[m], lw=0.8,
                    label=f"GARNET {m}, $s={L['s'][-1]}$")
        ax.set_title(f"{key}: " + ("UO$_2$ main diagonal" if key.startswith("fig6") else "MOX anti-diagonal")
                     + (", group 1" if key.endswith("a") else ", group 7"), fontsize=8)
        ax.set_xlabel(f["xlabel"])
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "c5g7_flux_lines.pdf")
    plt.close(fig)
    out["flux_lines"] = lines
    rows = [[k.split()[0], " ".join(k.split()[1:]), f"{v['rms_pct_peak']:.2f}", f"{v['max_pct_peak']:.2f}"]
            for k, v in lines.items()]
    write_table("c5g7_flux", ["Figure", "Solution", "RMS (\\% of peak)", "max (\\% of peak)"], rows, "llrr")
    return out


KRYLOV = {"pcg": "CG", "gmres": "GMRES", "bicgstab": "BiCGStab"}


def cost_tables(runs, timing):
    rows = []
    for m in METHODS:
        for v in (("marshak",) if m == "SP1" else ("marshak", "robin")):
            t = timing.get(("bl", m, v, 160, "cpu"))
            if t:
                rows.append([m, v.capitalize() if m != "SP1" else "--", KRYLOV.get(t["linear_solver"], ""),
                             str(t["outers"]), str(t["inners"]), f"{t['solve_seconds']:.2f}",
                             f"{PAPER_BL_CPU[m]:.1f}"])
    write_table("bl_cost", ["Method", "Vacuum", "Krylov", "outers", "inners", "solve (s)", "paper CPU (s)"],
                rows, "lllrrrr")
    rows = []
    for m in METHODS:
        t = timing.get(("c5g7", m, "marshak", 4, "cpu"))
        g = [r for r in runs if r["case"] == "c5g7" and r["device"] == "gpu" and r["method"] == m
             and r["vacuum"] == "marshak" and r["mesh"] in (8, 24)]
        g = {r["mesh"]: r for r in g}
        rows.append([m, KRYLOV.get(t["linear_solver"], "--") if t else "--",
                     fmt(t and t["outers"], "d"), fmt(t and t["inners"], "d"), fmt(t and t["solve_seconds"], ".1f"),
                     fmt(g.get(8) and g[8]["solve_seconds"], ".1f"),
                     fmt(g.get(24) and g[24]["outers"], "d"), fmt(g.get(24) and g[24]["inners"], "d"),
                     fmt(g.get(24) and g[24]["solve_seconds"], ".1f")])
    write_table("c5g7_cost", ["Method", "Krylov", "outers", "inners", "CPU $s=4$ (s)", "GH200 $s=8$ (s)",
                              "outers $s=24$", "inners $s=24$", "GH200 $s=24$ (s)"], rows, "llrrrrrrr")


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    runs = load_runs()
    figs = json.loads((HERE / "data/paper-figures.json").read_text())["figures"]
    nea = json.loads((HERE / "data/nea-c5g7-mcnp-pin-powers.json").read_text())
    unconverged = [r["file"] for r in runs if not r["converged"]]
    summary = dict(runs=len(runs), unconverged=unconverged,
                   brantley_larsen=bl_analysis(runs, figs),
                   c5g7=c5g7_analysis(runs, figs, nea))
    cost_tables(runs, load_timing())
    (HERE / "data/summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"{len(runs)} runs, {len(unconverged)} not converged")


if __name__ == "__main__":
    main()
