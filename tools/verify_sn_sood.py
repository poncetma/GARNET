"""Verify the triangular S_N solver against the LA-13511 analytic criticality set.

    python tools/verify_sn_sood.py [PROBLEMS] [DEVICE] [SCHEME]

    PROBLEMS : comma list of problem numbers (default: all of
               garnet.benchmarks.sood.PROBLEMS)
    DEVICE   : auto (default), gpu or cpu
    SCHEME   : scb (default) or step

Each problem is exactly critical at its tabulated dimension, so k - 1 is the
solver's error. For each problem: a mesh series at quadrature (4, 16), then a
quadrature series (4, 16), (8, 32), (12, 48) on the finest mesh. The isotropic
problems (2, 12, 13, 54) calibrate the geometry and discretization; 34-37 and
71 carry P1 or P2 scattering. Slab problems also compare the scalar flux at the
tabulated r/rc (interior points by interpolation, the face by linear
extrapolation from the last two cells, which is first order).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from garnet.benchmarks.sood import PROBLEMS, build_sood_problem, slab_flux_edits  # noqa: E402
from garnet.tri_sn import TriSNTransportSolver  # noqa: E402

numbers = ([int(n) for n in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1]
           else sorted(PROBLEMS))
device = sys.argv[2] if len(sys.argv) > 2 else "auto"
scheme = sys.argv[3] if len(sys.argv) > 3 else "scb"
MESH = {"slab": (80, 160, 320, 640, 1280), "cylinder": (16, 32, 64, 128, 256)}
QUAD = ((4, 16), (8, 32), (12, 48))
TOL = dict(tol_k=1e-10, tol_source=1e-9)


def solve(num, cells, quad):
    c = build_sood_problem(num, cells)
    t0 = time.perf_counter()
    r = TriSNTransportSolver(c.grid, c.materials, c.material_map, active=c.active,
                             mix_material=c.mix_material, mix_weight=c.mix_weight,
                             n_polar=quad[0], n_azi=quad[1], bc=c.bc, scheme=scheme,
                             engine=c.engine, scattering_order=c.scattering_order,
                             device=device if c.engine == "levels" else "cpu").solve(**TOL)
    flux = ""
    if c.problem.flux:
        v, _ = slab_flux_edits(c, r.flux)
        e = 100 * (v / np.array(c.problem.flux) - 1)
        flux = (f"  flux err interior max {np.abs(e[:, :-1]).max():.3f} %, "
                f"face {e[0, -1]:+.2f} / {e[1, -1]:+.2f} %")
    n = int(c.active.sum()) if c.active is not None else int(np.prod(c.grid.shape))
    return r, n, time.perf_counter() - t0, flux


print(f"LA-13511 analytic criticality set, triangular S_N ({scheme}), device {device}\n",
      flush=True)
summary = []
for num in numbers:
    p = PROBLEMS[num]
    print(f"problem {num} {p.identifier}: {p.geometry}, r_c = {p.rc} cm, "
          f"scattering order {p.order}", flush=True)
    errs = []
    for cells in MESH[p.geometry]:
        r, n, sec, flux = solve(num, cells, QUAD[0])
        errs.append(1e5 * (r.k_eff - 1.0))
        print(f"  cells {cells:5d} ({n:7d} triangles) quad {QUAD[0]}: k - 1 = {errs[-1]:+9.2f} pcm, "
              f"outers {r.outer_iterations}, converged {bool(r.converged)} [{sec:.0f} s]{flux}",
              flush=True)
    d1, d2 = errs[-2] - errs[-3], errs[-1] - errs[-2]
    order = np.log2(d1 / d2) if d1 * d2 > 0 and abs(d2) > 1e-9 else float("nan")
    limit = errs[-1] + d2 / (2.0 ** order - 1.0) if np.isfinite(order) and order > 0.3 else float("nan")
    print(f"  observed order {order:.2f}, Richardson mesh limit at {QUAD[0]}: {limit:+.2f} pcm",
          flush=True)
    qerr = [errs[-1]]
    for quad in QUAD[1:]:
        r, n, sec, flux = solve(num, MESH[p.geometry][-1], quad)
        qerr.append(1e5 * (r.k_eff - 1.0))
        print(f"  cells {MESH[p.geometry][-1]:5d} quad {quad}: k - 1 = {qerr[-1]:+9.2f} pcm "
              f"[{sec:.0f} s]{flux}", flush=True)
    est = limit + (qerr[-1] - qerr[0]) if np.isfinite(limit) else float("nan")
    summary.append((num, p.identifier, p.order, errs[-1], order, limit, qerr[-1] - qerr[0], est))
    print(flush=True)

print("summary: k - 1 in pcm (finest mesh at (4, 16); mesh limit by Richardson; quadrature "
      "effect (12, 48) - (4, 16) on the finest mesh; their sum)")
print(f"{'problem':>8} {'identifier':>12} {'P':>2} {'finest':>9} {'order':>6} {'mesh lim':>9} "
      f"{'quad':>8} {'estimate':>9}")
for num, ident, order_l, fin, order, limit, dq, est in summary:
    print(f"{num:8d} {ident:>12} {order_l:2d} {fin:+9.2f} {order:6.2f} {limit:+9.2f} {dq:+8.2f} "
          f"{est:+9.2f}")
