"""Godiva on extruded triangular prisms (GARNET 3-D S_N) with GeN-Foam's
23-group data: the 3-D counterpart of tools/godiva_1d.py.

Usage:
    python tools/godiva_3d.py NUCLEARDATA SIDE [OPTIONS]

SIDE is the triangle side h (cm); prism height h sqrt(3)/2 unless DZ is set.
Options (KEY=VALUE):
    order=2              scattering order (GeN-Foam uses P2; 0 = transport-
                         corrected P0, out-scatter)
    quad=8x16            product set n_polar x n_azi, or a GeN-Foam
                         quadratureSet file path
    scheme=scb           scb or step
    treatment=blend      blend or staircase (garnet.benchmarks.sood.sphere_on_prisms)
    dz=                  prism height (cm)
    device=gpu
    corners=2            axial corners per prism (2: SCB in z; 1: step in z)
    cache=group          levels-engine corner tables: group or all
    cmfd=mg              3-D CMFD solver: mg (AMG) or lu
    graphs=1             CUDA-graph capture of the sweep (0 for debugging)
    reuse=0              CMFD lagged LU: max GMRES iterations before refactorizing
    accel=cmfd           outer acceleration: cmfd or power (no coarse solve)
    out=godiva_3d_out    directory for the JSON record

The 1D converged reference for the same data is k = 0.994256 (P2).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.benchmarks.sood import _void, sphere_on_prisms  # noqa: E402
from garnet.genfoam import read_nuclear_data, read_quadrature_set  # noqa: E402
from garnet.materials import transport_corrected  # noqa: E402
from garnet.tri_sn import TriSNTransportSolver  # noqa: E402

R_GODIVA = 8.7407
K_1D = {2: 0.994256, 0: 0.997619}          # tools/godiva_1d.py, converged


def main(path, side, opts):
    order = int(opts.get("order", 2))
    quad = opts.get("quad", "8x16")
    scheme = opts.get("scheme", "scb")
    treatment = opts.get("treatment", "blend")
    device = opts.get("device", "gpu")
    dz = float(opts["dz"]) if opts.get("dz") else None
    corners = int(opts.get("corners", 2))
    cache = opts.get("cache", "group")
    cmfd = opts.get("cmfd", "mg")
    graphs = opts.get("graphs", "1") != "0"
    reuse = int(opts.get("reuse", "0"))
    accel = opts.get("accel", "cmfd")
    out_dir = Path(opts.get("out", "godiva_3d_out"))

    z = next(iter(read_nuclear_data(path).values()))
    fuel = (z.material(order) if order > 0
            else transport_corrected(z.material(1), "outscatter"))
    G = fuel.n_groups
    t0 = time.perf_counter()
    sph = sphere_on_prisms(R_GODIVA, side, dz=dz, treatment=treatment)
    t_geo = time.perf_counter() - t0
    kw = {}
    if "x" in quad and not Path(quad).exists():
        npol, naz = (int(v) for v in quad.split("x"))
        kw.update(n_polar=npol, n_azi=naz)
    else:
        kw.update(quadrature=read_quadrature_set(quad), quadrature_check="warn")
    mixm = np.where(sph.mix_weight > 0.0, 1, -1)
    t1 = time.perf_counter()
    solver = TriSNTransportSolver(sph.grid, [fuel, _void(G)], np.zeros(sph.grid.shape, int),
                                  active=sph.active, mix_material=mixm,
                                  mix_weight=sph.mix_weight, bc="vacuum", scheme=scheme,
                                  engine="levels", device=device, scattering_order=order,
                                  axial_corners=corners, level_cache=cache,
                                  cmfd_solver=cmfd, graphs=graphs, cmfd_reuse=reuse,
                                  outer_acceleration=accel, **kw)
    t_setup = time.perf_counter() - t1
    r = solver.solve(tol_k=1e-7, tol_source=1e-6, verbose=True,
                     max_outer=int(opts.get("max_outer", "500")))
    rec = dict(side=side, dz=sph.grid.dz, order=order, quad=quad, scheme=scheme,
               corners=corners,
               treatment=treatment, directions=int(solver.M), shape=list(sph.grid.shape),
               active=int(sph.active.sum()), k=float(r.k_eff), converged=bool(r.converged),
               pcm_vs_1d=1e5 * (float(r.k_eff) - K_1D.get(order, np.nan)),
               geometry_s=t_geo, setup_s=t_setup, solve_s=float(r.solve_seconds
                                                               if hasattr(r, "solve_seconds")
                                                               else np.nan),
               gram_error=getattr(solver, "quadrature_gram_error", None),
               t_groups=getattr(solver, "t_groups", None), t_cmfd=getattr(solver, "t_cmfd", None),
               t_power=getattr(solver, "t_power", None),
               t_tables=getattr(solver, "t_tables", None), n_tables=getattr(solver, "n_tables", None),
               outers=int(getattr(r, "outer_iterations", -1)),
               t_cmfd_parts=getattr(solver, "t_cmfd_parts", None),
               accel=accel, cmfd_reuse=reuse, cmfd_reuse_stats=getattr(solver, "cmfd_reuse_stats", None))
    print(json.dumps(rec), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"h{side:g}_P{order}_{Path(quad).name}_{scheme}{corners}_{treatment}"
           + (f"_reuse{reuse}" if reuse else "") + (f"_{accel}" if accel != "cmfd" else "")
           + (f"_dz{dz:g}" if dz else ""))
    (out_dir / f"godiva_3d_{tag}.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), float(sys.argv[2]),
         dict(a.split("=", 1) for a in sys.argv[3:]))
