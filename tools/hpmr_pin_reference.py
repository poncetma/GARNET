"""Pin-resolved HP-MR 2-D references: diffusion, SP3 and tri-S_N, cached.

    python tools/hpmr_pin_reference.py REFINE ANGLES METHODS [DEVICE] [CACHE] [ABSORPTION]

    REFINE  : raster refine of build_hpmr2d_pins (8, 16, 24, ...)
    ANGLES  : comma list of drum angles in degrees (all 12 drums together)
    METHODS : comma list of
                diffusion, sp3, sn:NP:NA      isotropic scattering with the library's
                                              TRUE total (no transport correction;
                                              kept for comparison, physically wrong
                                              for this core)
                diffusion-tco, sp3-tco, sn-tco:NP:NA   out-scatter transport-
                                              corrected data (-tcd: within-group P1
                                              correction only)
                sn-pL:NP:NA                   anisotropic S_N, Legendre order L
                                              (Griffin's NA = L), true total
              S_N uses SCB differencing with n_polar NP and n_azi NA.
    DEVICE  : auto (default), gpu or cpu
    CACHE   : directory for the results (default
              /data/scratch/shared/poncet_m/garnet-sph-ref-net)
    ABSORPTION : net (default) or library, see
              :func:`garnet.benchmarks.hpmr_core_pins.hpmr_pin_core_materials`.
              The library absorption loses the Be (n,2n) gain; every result
              before 2026-11-16 used it, and those caches are in
              /data/scratch/shared/poncet_m/garnet-sph-ref.

The core is the faithful configuration of
:func:`garnet.benchmarks.hpmr_core_pins.build_hpmr2d_pins`: vendored 11-group
ENDF/B-8 data with their P0-P5 scattering moments and the net absorption that
keeps the Be (n,2n) gain, every pin layer blended at its exact area,
volume-mixed drum arc. In ``sn-tco`` the out-scatter correction exceeds the
YH moderator's within-group scattering in groups 7-8; the S_N solver keeps
that scattering negative so the balance holds (solves before this was fixed,
2026-11-16, clipped it and created neutrons: +1470 pcm at refine 16). Each solve is written to CACHE/<method>_r<refine>_a<angle>.npz with the
scalar flux, k, convergence, wall time and iteration counts, and an existing
file is reused rather than recomputed, so the SPH study tools can take S_N
references from the cache. One line per solve is printed, and S_N runs
verbose, so a long job can be watched in its log.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from garnet import TriDiffusionEigenSolver, TriSP3EigenSolver  # noqa: E402
from garnet.backend import asnumpy  # noqa: E402
from garnet.benchmarks.hpmr_core_pins import build_hpmr2d_pins  # noqa: E402
from garnet.tri_sn import TriSNTransportSolver  # noqa: E402

DEFAULT_CACHE = "/data/scratch/shared/poncet_m/garnet-sph-ref-net"
TOL = dict(tol_k=1e-8, tol_source=1e-7)
TOL_SN = dict(tol_k=1e-7, tol_source=1e-6)


def method_tag(method: str) -> str:
    return method.replace(":", "")


def cache_path(cache: str, method: str, refine: int, angle: float) -> str:
    return os.path.join(cache, f"{method_tag(method)}_r{refine}_a{angle:g}.npz")


def parse(method: str):
    """(solver kind, transport correction or None, scattering order, quadrature)."""
    head, *quad = method.split(":")
    kind, _, variant = head.partition("-")
    if kind not in ("diffusion", "sp3", "sn") or (kind == "sn") != bool(quad):
        raise ValueError(f"unknown method {method!r}")
    tc, order = None, 0
    if variant in ("tco", "tcd"):
        tc = "outscatter" if variant == "tco" else "diagonal"
    elif variant.startswith("p") and kind == "sn" and variant[1:].isdigit():
        order = int(variant[1:])
    elif variant:
        raise ValueError(f"unknown method variant {variant!r} in {method!r}")
    return kind, tc, order, tuple(int(q) for q in quad)


def problem_for(method: str, refine: int, angle: float, absorption: str = "net"):
    """The pin-resolved core with the data the method needs."""
    _, tc, _, _ = parse(method)
    p, _ = build_hpmr2d_pins(refine=refine, drum_angle_deg=angle,
                             transport_correction=tc, absorption=absorption)
    return p


def solve(problem, method: str, device: str):
    """(flux (G, *shape) host array, k, converged, seconds, outers)."""
    p = problem
    kind, tc, order, quad = parse(method)
    t0 = time.perf_counter()
    if kind in ("diffusion", "sp3"):
        cls = TriDiffusionEigenSolver if kind == "diffusion" else TriSP3EigenSolver
        r = cls(p.grid, p.materials, p.material_map, active=p.active,
                mask_bc=p.mask_bc, mix_material=p.mix_material,
                mix_weight=p.mix_weight, device=device).solve(**TOL)
        return (r.flux_numpy, r.k_eff, bool(r.converged), time.perf_counter() - t0,
                int(getattr(r, "outer_iterations", -1)))
    n_polar, n_azi = quad
    r = TriSNTransportSolver(p.grid, p.materials, p.material_map, active=p.active,
                             mix_material=p.mix_material, mix_weight=p.mix_weight,
                             n_polar=n_polar, n_azi=n_azi, bc="vacuum",
                             scheme="scb", device=device,
                             scattering_order=order).solve(verbose=True, **TOL_SN)
    return (asnumpy(r.flux), r.k_eff, bool(r.converged), time.perf_counter() - t0,
            int(r.outer_iterations))


def load_or_solve(refine: int, angle: float, method: str, device: str = "auto",
                  cache: str = DEFAULT_CACHE, problem=None):
    """Cached reference: dict(flux, k, converged, seconds, outers)."""
    path = cache_path(cache, method, refine, angle)
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    if problem is None:
        problem = problem_for(method, refine, angle)
    flux, k, ok, sec, outers = solve(problem, method, device)
    os.makedirs(cache, exist_ok=True)
    out = dict(flux=flux, k=k, converged=ok, seconds=sec, outers=outers)
    np.savez(path, **out)
    return out


def currents_path(cache: str, method: str, refine: int, angle: float,
                  per_hex: bool = False) -> str:
    tag = "_currents_hex" if per_hex else "_currents"
    return os.path.join(cache, f"{method_tag(method)}_r{refine}_a{angle:g}{tag}.npz")


def load_or_solve_currents(refine: int, angle: float, method: str, device: str = "auto",
                           cache: str = DEFAULT_CACHE, problem=None, per_hex: bool = False,
                           region_map=None, tag: str | None = None):
    """Reference partial currents on the assembly-zone surfaces (S_N only), cached.

    Solves the S_N reference again on the levels engine and tallies the
    outgoing and incoming currents through every zone interface and through
    each zone's vacuum boundary (TriSNTransportSolver.region_partial_currents,
    zones of garnet.benchmarks.hpmr.hpmr_assembly_zones without merged drums,
    pooled or ``per_hex``, cached separately).
    Returns dict(k, boundaries={zone: (G, 2)}, interfaces={(a, b): (G, 2)})
    with [J_out, J_in] in the normalization of the solve's flux, which is also
    stored in the flux cache when missing.
    """
    from garnet.benchmarks.hpmr import hpmr_assembly_zones

    path = (currents_path(cache, method, refine, angle, per_hex) if region_map is None
            else os.path.join(cache, f"{method_tag(method)}_r{refine}_a{angle:g}_currents_{tag}.npz"))
    if region_map is not None and not tag:
        raise ValueError("a custom region_map needs a cache tag")
    if os.path.exists(path):
        d = np.load(path)
        return dict(k=float(d["k"]),
                    boundaries={int(r): d["b_J"][i] for i, r in enumerate(d["b_regions"])},
                    interfaces={(int(a), int(b)): d["i_J"][i]
                                for i, (a, b) in enumerate(d["i_pairs"])})
    kind, tc, order, quad = parse(method)
    if kind != "sn":
        raise ValueError("reference currents need an S_N method")
    p = problem if problem is not None else problem_for(method, refine, angle)
    t0 = time.perf_counter()
    solver = TriSNTransportSolver(p.grid, p.materials, p.material_map, active=p.active,
                                  mix_material=p.mix_material, mix_weight=p.mix_weight,
                                  n_polar=quad[0], n_azi=quad[1], bc="vacuum",
                                  scheme="scb", device=device, engine="levels",
                                  scattering_order=order)
    r = solver.solve(verbose=True, **TOL_SN)
    rmap = (hpmr_assembly_zones(p, per_hex=per_hex).region_map if region_map is None
            else np.asarray(region_map))
    itf, bnd = solver.region_partial_currents(r, rmap)
    os.makedirs(cache, exist_ok=True)
    flux_file = cache_path(cache, method, refine, angle)
    if not os.path.exists(flux_file):
        np.savez(flux_file, flux=asnumpy(r.flux), k=r.k_eff, converged=bool(r.converged),
                 seconds=time.perf_counter() - t0, outers=int(r.outer_iterations))
    b_regions = sorted(bnd)
    i_pairs = sorted(itf)
    np.savez(path, k=r.k_eff, b_regions=np.array(b_regions, int),
             b_J=np.array([bnd[b] for b in b_regions]),
             i_pairs=np.array(i_pairs, int).reshape(-1, 2),
             i_J=np.array([itf[i] for i in i_pairs]))
    return dict(k=r.k_eff, boundaries=bnd, interfaces=itf)


def main():
    refine = int(sys.argv[1])
    angles = [float(a) for a in sys.argv[2].split(",")]
    methods = sys.argv[3].split(",")
    device = sys.argv[4] if len(sys.argv) > 4 else "auto"
    cache = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else DEFAULT_CACHE
    absorption = sys.argv[6] if len(sys.argv) > 6 else "net"
    print(f"pin-resolved HP-MR 2-D references, refine {refine}, angles {angles}, "
          f"methods {methods}, device {device}, cache {cache}, "
          f"absorption {absorption}", flush=True)
    for angle in angles:
        problems = {}
        for method in methods:
            tc = parse(method)[1]
            if tc not in problems:
                t0 = time.perf_counter()
                problems[tc] = problem_for(method, refine, angle, absorption)
                print(f"  angle {angle:g}, transport correction {tc}: "
                      f"{int(problems[tc].active.sum())} active cells, "
                      f"{len(problems[tc].materials)} materials, built in "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
            r = load_or_solve(refine, angle, method, device, cache, problems[tc])
            print(f"RESULT refine={refine} angle={angle:g} method={method} "
                  f"k={float(r['k']):.7f} converged={bool(r['converged'])} "
                  f"outers={int(r['outers'])} t={float(r['seconds']):.0f}s", flush=True)


if __name__ == "__main__":
    main()
