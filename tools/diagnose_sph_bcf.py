"""SPH with boundary discontinuity factors (Laboure et al. 2019 SPH-DF) on HP-MR.

    python tools/diagnose_sph_bcf.py [REFINE] [METHOD] [NODES] [ANGLES] [DEVICE] [CACHE] [ZONING]

    REFINE, METHOD : S_N reference of tools/hpmr_pin_reference.py (default 16,
                     sn-p2:4:16), its zone constants (transport-corrected,
                     ``pin_reference``) and its zone-surface currents
                     (``load_or_solve_currents``; solved and cached if missing)
    NODES          : raster or smeared node spec (default raster:8: 1.9 cm
                     cells, the drum arc resolved); the unstructured local mesh
                     has no boundary coefficient
    ANGLES         : comma list (default 0,180)
    DEVICE, CACHE  : as for the reference tools
    ZONING         : pooled (default: one zone each for the Be reflector, the
                     drum bodies and the drum arcs) or hex (one zone per hex of
                     the outer ring, each drum's body and arc separately)

Variants (select with SPH_VARIANTS="name,name"; all by default):

  FV interior   flux-volume SPH (one constant per group over the corrected
                zones) on fuel, central cell and drums
  FV all        the same on every zone, reflector included
  FK interior   fixed-k SPH (no normalization, k = k_ref; Laboure et al.'s
                HTTR choice) on fuel, central cell and drums
  FK all        the same on every zone
  SPH-DF all    SPH on every zone + a boundary discontinuity factor on the
                vacuum boundary of each zone that has one (Be reflector, drum
                Be): Laboure et al.'s method, which should reproduce k
  SPH-DF int.   the same with the reflector left uncorrected

Each line: k error, worst assembly-power error, solver iterations and
convergence, and the factors: fuel thermal (min-max), drum absorber and drum
Be and reflector (groups 1 / 6 / 11), and the boundary factors.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hpmr_pin_reference import DEFAULT_CACHE, load_or_solve_currents  # noqa: E402
from sph_hpmr_nodes import (merge_drum, node_model, pin_reference,  # noqa: E402
                            split_boundary_layer, transfer_boundary_layer)

from garnet import sph_bcf_correct, sph_correct  # noqa: E402
from garnet.power import fission_energy_xs  # noqa: E402

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 16
method = sys.argv[2] if len(sys.argv) > 2 else "sn-p2:4:16"
spec = sys.argv[3] if len(sys.argv) > 3 else "raster:8"
angles = [float(a) for a in (sys.argv[4] if len(sys.argv) > 4 else "0,180").split(",")]
device = sys.argv[5] if len(sys.argv) > 5 else "auto"
cache = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else DEFAULT_CACHE
PER_HEX = (sys.argv[7] if len(sys.argv) > 7 else "pooled") == "hex"
MERGE = merge_drum(spec)
# SPH_SPLIT=1: an inner and an outer zone in every hex on the vacuum boundary
SPLIT = os.environ.get("SPH_SPLIT", "") not in ("", "0")        # smeared nodes: one zone per drum, body and arc together
TOL = 1e-7
# SPH_VARIANTS: comma list of variant names to run (default all);
# SPH_VERBOSE=1 prints every factor iteration
ONLY = [v.strip() for v in os.environ.get("SPH_VARIANTS", "").split(",") if v.strip()]
VERBOSE = os.environ.get("SPH_VERBOSE", "") not in ("", "0")


def assembly_power(materials, flux, volume, fuel):
    table, _ = fission_energy_xs(materials)
    p = ((table * flux).sum(axis=1) * volume)[fuel]
    return p / p.mean()


def fmt(v):
    return "/".join(f"{x:.3g}" for x in v)


def rng(a):
    a = np.asarray(a, float).ravel()
    return f"{a.min():.3g}" if np.ptp(a) <= 1e-3 * abs(a).max() else f"{a.min():.3g}-{a.max():.3g}"


def cls(mu, zones, groups):
    """Factor ranges over a class of zones, groups 1 / 6 / 11."""
    if not zones:
        return "-"
    return "/".join(rng(mu[zones, g]) for g in groups)


print(f"SPH-DF on HP-MR 2-D: {method} reference at refine {refine}, node model {spec}, "
      f"{'per-hex' if PER_HEX else 'pooled'} zones{', boundary hexes split inner/outer' if SPLIT else ''}, "
      f"device {device}\n", flush=True)
for angle in angles:
    t0 = time.perf_counter()
    split = {}

    def node_hook(z, q):
        zs, outer, outer_of = split_boundary_layer(z, q)
        split.update(q=q, outer=outer, outer_of=outer_of)
        return zs

    node = node_model(spec, angle, device, per_hex=PER_HEX,
                      zones_hook=node_hook if SPLIT else None)
    ref = pin_reference(refine, method, angle, MERGE, device, cache, per_hex=PER_HEX,
                        zones_hook=(lambda z, p: transfer_boundary_layer(
                            z, p, split["q"], split["outer"], split["outer_of"]))
                        if SPLIT else None)
    cur = load_or_solve_currents(refine, angle, method, device, cache, per_hex=PER_HEX)
    if node["solve_bcf"] is None:
        raise SystemExit(f"node spec {spec} has no boundary coefficient")
    z = ref["zones"]
    names = list(z.names)
    G = ref["rflux"].shape[1]
    groups = (0, G // 2, G - 1)
    fuel = z.fuel
    of = {"be_reflector": [i for i, n in enumerate(names) if n.startswith("be_reflector")],
          "drum_absorber": [i for i, n in enumerate(names) if n.startswith("drum_absorber")],
          # drum body, or the whole hex-homogenized drum with merged zones
          "drum_be": [i for i, n in enumerate(names)
                      if n.startswith("drum") and not n.startswith("drum_absorber")]}
    central = names.index("central")
    interior = fuel.copy()
    interior[[central] + of["drum_be"] + of["drum_absorber"]] = True
    every = np.ones(len(names), bool)
    bregs = sorted(cur["boundaries"])
    Jref = np.array([cur["boundaries"][b][:, 0] for b in bregs])          # (B, G)
    if SPLIT:   # every vacuum face of a hex belongs to its outer zone
        bregs = [split["outer_of"][b] for b in bregs]
    # The currents are tallied on the unmerged zones; the boundary zones (Be
    # reflector and drum hexes) have the same indices with merged drums, since
    # only the absorber zones, which never touch the boundary, disappear.
    if not all(names[r].startswith(("be_reflector", "drum")) for r in bregs):
        raise RuntimeError("reference boundary zones do not match this zoning")
    b_be = [b for b, r in enumerate(bregs) if names[r].startswith("be_reflector")]
    b_drum = [b for b, r in enumerate(bregs) if names[r].startswith("drum")]
    # drum symmetry classes (per-hex zones): ring-3 corners, ring-4 mid-edges
    sym_zones = of["drum_absorber"] or of["drum_be"]
    corner, edge = sym_zones[:6], sym_zones[6:]
    vol = ref["rvol"]
    p_ref = assembly_power(ref["hm"], ref["rflux"], vol, fuel)
    phi0, k0 = node["solve"](ref["hm"])
    dp0 = 100 * np.abs(assembly_power(ref["hm"], np.asarray(phi0), vol, fuel) / p_ref - 1).max()
    print(f"angle {angle:g}: k_ref {ref['k']:.6f} (currents re-solve {cur['k']:.6f}); "
          f"{len(bregs)} vacuum-boundary zones; no SPH {1e5 * (k0 - ref['k']):+.0f} pcm, "
          f"dP {dp0:.2f} %", flush=True)
    print(f"  {'variant':<13} {'conv':>5} {'iter':>5} {'dk pcm':>9} {'dP %':>6} | "
          f"{'fuel thermal':>13} | absorber, drum Be, reflector (groups 1/6/11) | "
          f"boundary factors, reflector / drums (groups 1/6/11)", flush=True)

    def bcf_solve(mats, gam):
        return node["solve_bcf"](mats, gam, bregs)

    runs = {
        "FV interior": lambda: sph_correct(ref["hm"], z.region_map, ref["rflux"], node["solve"],
                                           region_volume=vol, normalization="flux-volume",
                                           sph_regions=interior, tol=TOL, max_iter=400),
        "FV all": lambda: sph_correct(ref["hm"], z.region_map, ref["rflux"], node["solve"],
                                      region_volume=vol, normalization="flux-volume",
                                      sph_regions=every, tol=TOL, max_iter=400),
        "FK interior": lambda: sph_correct(ref["hm"], z.region_map, ref["rflux"], node["solve"],
                                           normalization="none", k_ref=ref["k"],
                                           sph_regions=interior, tol=TOL, max_iter=400),
        "FK all": lambda: sph_correct(ref["hm"], z.region_map, ref["rflux"], node["solve"],
                                      normalization="none", k_ref=ref["k"],
                                      sph_regions=every, tol=TOL, max_iter=400),
        "SPH-DF all": lambda: sph_bcf_correct(ref["hm"], ref["rflux"], Jref, bcf_solve,
                                              region_volume=vol, sph_regions=every, tol=TOL,
                                              max_iter=400, verbose=VERBOSE),
        "SPH-DF int.": lambda: sph_bcf_correct(ref["hm"], ref["rflux"], Jref, bcf_solve,
                                               region_volume=vol, sph_regions=interior,
                                               tol=TOL, max_iter=400, verbose=VERBOSE),
    }
    for name, run in runs.items():
        if ONLY and name not in ONLY:
            continue
        t1 = time.perf_counter()
        out = run()
        mu = out.factors
        if out.bcf is not None:
            phi, k, _ = bcf_solve(out.corrected_materials, out.bcf)
            gam = (cls(out.bcf, b_be, groups) + "  " + cls(out.bcf, b_drum, groups))
        else:
            phi, k = node["solve"](out.corrected_materials)
            gam = "-"
        dp = 100 * np.abs(assembly_power(out.corrected_materials, np.asarray(phi), vol, fuel)
                          / p_ref - 1).max()
        th = mu[fuel, -1]
        sym = (f"  thermal {'drum' if MERGE else 'absorber'} corner/edge drums "
               f"{rng(mu[corner, -1])} / {rng(mu[edge, -1])}" if PER_HEX else "")
        print(f"  {name:<13} {str(bool(out.converged)):>5} {out.iterations:5d} "
              f"{1e5 * (k - ref['k']):+9.1f} {dp:6.2f} | {th.min():5.3f}-{th.max():5.3f} | "
              f"{cls(mu, of['drum_absorber'], groups)}, {cls(mu, of['drum_be'], groups)}, "
              f"{cls(mu, of['be_reflector'], groups)} | {gam}  "
              f"[{time.perf_counter() - t1:.0f} s]{sym}", flush=True)
    print(f"  [{time.perf_counter() - t0:.0f} s]\n", flush=True)
