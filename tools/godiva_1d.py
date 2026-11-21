"""Godiva (bare HEU sphere, R = 8.7407 cm) with GeN-Foam's 23-group data in
GARNET's one-dimensional spherical S_N: the converged k for those data, to
compare with GeN-Foam's Godiva_SN tutorial (regression value 0.980106 after 20
power iterations on its tetrahedral mesh) and with the experiment (k = 1).

Usage:
    python tools/godiva_1d.py NUCLEARDATA [OUT_DIR]

NUCLEARDATA is the tutorial's ``constant/neutroRegion/nuclearData`` (read in
place, not copied: GeN-Foam's data carry their own licence). Studies, all with
the order GeN-Foam's S_N uses (P2) unless stated:

1. mesh: S64, 25-400 equal shells, Richardson extrapolation (second order);
2. angle: 200 shells, Gauss-Legendre S8-S128, extrapolation in N^-2;
3. scattering: P0 with the true total, P0 transport-corrected (out-scatter and
   diagonal), P1, P2, P3, P5 at S64 / 200 shells.

Writes OUT_DIR/godiva_1d.json and prints a Markdown summary.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.genfoam import read_nuclear_data  # noqa: E402
from garnet.materials import transport_corrected  # noqa: E402
from garnet.sphere_sn import SphereSNSolver  # noqa: E402

R_GODIVA = 8.7407            # cm, HEU-MET-FAST-001 and the GeN-Foam mesh
K_GENFOAM = 0.980106         # GeN-Foam Godiva_SN regression value (not converged)


def solve(material, shells, n_mu):
    s = SphereSNSolver(np.linspace(0.0, R_GODIVA, shells + 1)[1:], material, n_mu=n_mu)
    r = s.solve(tol_k=1e-10, tol_source=1e-8)
    if not r.converged:
        raise RuntimeError(f"not converged: {shells} shells, S{n_mu}")
    return r


def pcm(k, ref):
    return 1e5 * (k - ref)


def main(path, out_dir):
    zone = read_nuclear_data(path)
    if len(zone) != 1:
        raise SystemExit(f"expected one (state, zone), found {list(zone)}")
    z = next(iter(zone.values()))
    L = z.solver_order
    out = {"file": str(path), "groups": z.energy_groups, "solver_order": L,
           "available_order": z.available_order, "chi_sum": z.chi_sum,
           "beta": z.beta_total}
    print(f"{z.energy_groups} groups, P0-P{z.available_order} in the file, GeN-Foam uses P{L}; "
          f"sum chi = {z.chi_sum:.8f}, beta = {z.beta_total * 1e5:.1f} pcm", flush=True)
    mat = z.material(L)

    rows = []
    for shells in (25, 50, 100, 200, 400):
        t = time.perf_counter()
        r = solve(mat, shells, 64)
        rows.append((shells, r.k_eff, time.perf_counter() - t))
        print(f"  mesh {shells:4d} shells S64: k = {r.k_eff:.7f} ({rows[-1][2]:.1f} s)", flush=True)
    k_mesh = rows[-1][1] + (rows[-1][1] - rows[-2][1]) / 3.0
    out["mesh"] = {"shells": [r[0] for r in rows], "k": [r[1] for r in rows],
                   "extrapolated": k_mesh}

    rows_a = []
    for n in (8, 16, 32, 64, 128):
        r = solve(mat, 200, n)
        rows_a.append((n, r.k_eff))
        print(f"  angle S{n:<3d} 200 shells: k = {r.k_eff:.7f}", flush=True)
    k_ang = rows_a[-1][1] + (rows_a[-1][1] - rows_a[-2][1]) / 3.0
    out["angle"] = {"n_mu": [r[0] for r in rows_a], "k": [r[1] for r in rows_a],
                    "extrapolated": k_ang}

    # converged estimate: S64/200 plus the mesh and angle corrections
    k64 = dict(rows_a)[64]
    k200 = {r[0]: r[1] for r in rows}[200]
    k_conv = k64 + (k_mesh - k200) + (k_ang - k64)
    out["k_converged"] = k_conv

    scat = {}
    variants = [(f"P{l}", z.material(l)) for l in range(z.available_order + 1)
                if l in (0, 1, 2, 3, z.available_order)]
    p1 = z.material(1)
    variants += [("P0 transport-corrected (out-scatter)", transport_corrected(p1, "outscatter")),
                 ("P0 transport-corrected (diagonal)", transport_corrected(p1, "diagonal"))]
    for name, m in variants:
        r = solve(m, 200, 64)
        scat[name] = r.k_eff
        print(f"  {name:38s} S64 200 shells: k = {r.k_eff:.7f}", flush=True)
    out["scattering"] = scat

    print("\n| Quantity | k | vs k = 1 (pcm) | vs GeN-Foam 0.980106 (pcm) |")
    print("|---|---|---|---|")
    for label, k in [(f"P{L}, converged (mesh + angle extrapolated)", k_conv),
                     (f"P{L}, S64, 200 shells", k64)]:
        print(f"| {label} | {k:.6f} | {pcm(k, 1.0):+.0f} | {pcm(k, K_GENFOAM):+.0f} |")
    print(f"\nmesh error at 200 shells: {pcm(k200, k_mesh):+.2f} pcm; "
          f"angular error at S64: {pcm(k64, k_ang):+.2f} pcm")
    print("\n| Scattering | k (S64, 200 shells) | vs P" + str(L) + " (pcm) |")
    print("|---|---|---|")
    for name, k in scat.items():
        print(f"| {name} | {k:.6f} | {pcm(k, scat[f'P{L}']):+.0f} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "godiva_1d.json").write_text(json.dumps(out, indent=1))
    print(f"\nwritten {out_dir / 'godiva_1d.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("godiva_1d_out"))
