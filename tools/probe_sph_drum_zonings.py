"""Which drum zonings admit an SPH-DF solution? Feasibility probe (HP-MR 2-D).

    python tools/probe_sph_drum_zonings.py [ANGLES] [NODES] [DEVICE] [CACHE] [VARIANTS] [RECIPES]

VARIANTS: comma-separated indices into the zoning list below (default all);
RECIPES: transport,flux (default both).

For drum zonings built from fixed ring sectors (tools/sph_hpmr_nodes.
sector_zoning_hooks; 8 sectors) and two recipes for the zone diffusion
coefficient, print the feasibility index of every boundary zone, as in
tools/probe_sph_bcf_feasibility.py: the coarse model's outgoing current per unit
zone-average flux with a zero-flux boundary (no SPH), over the reference's.
Below 1 no boundary discontinuity factor can reproduce the reference leakage
and SPH-DF has no solution.

Zonings: the drum hex's outer boundary layer pooled in one zone, the outer
layer split into the same sectors as the ring, and the drum hexes not split
(ring sectors reaching the boundary), plain and with aligned sectors. D
recipes: "transport", the default of garnet.flux_weighted_homogenize (flux-weight
1/(3D), invert), and "flux" (flux-weight D itself). Reference boundary currents
are the per-exposed-node-cell tallies of boundary_cell_currents (S_N
sn-p2:4:16 at refine 16), solved and cached when missing.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hpmr_pin_reference import DEFAULT_CACHE, load_or_solve  # noqa: E402
from sph_hpmr_nodes import (boundary_cell_currents, node_model,  # noqa: E402
                            sector_zoning_hooks, zone_boundary_currents)

from garnet import flux_weighted_homogenize  # noqa: E402
from garnet.benchmarks.hpmr_core_pins import (R_FUEL, build_hpmr2d_pins,  # noqa: E402
                                              hpmr_pin_layout)

angles = [float(a) for a in (sys.argv[1] if len(sys.argv) > 1 else "0,165,180").split(",")]
spec = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "raster:8"
device = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "auto"
cache = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else DEFAULT_CACHE
REFINE, METHOD = 16, "sn-p2:4:16"
PINS = hpmr_pin_layout()
VARIANTS = {
    "drum outer layer pooled": dict(),
    "drum outer layer in sectors": dict(outer_sectors=True),
    "drum hexes unsplit": dict(drum_outer=False),
    "drum hexes unsplit, aligned": dict(drum_outer=False, aligned=True),
    "outer layer Be only, aligned": dict(aligned=True, outer_arc_to_sector=True),
}
if len(sys.argv) > 5 and sys.argv[5]:
    keep = [int(i) for i in sys.argv[5].split(",")]
    VARIANTS = {k: v for i, (k, v) in enumerate(VARIANTS.items()) if i in keep}
RECIPES = (sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6]
           else ["transport", "flux"])


def homogenize(flux, p, z, recipe):
    kw = dict(cell_volume=p.grid.cell_volume, mix_material=p.mix_material,
              mix_weight=p.mix_weight, mix_region_map=z.mix_region_map)
    hm, rflux, rvol = flux_weighted_homogenize(flux, p.materials, p.material_map,
                                               z.region_map, **kw)
    if recipe == "flux":        # homogenizing 1/(9 D) returns 1/(9 <D>_flux)
        mats = [m.replace(diffusion=1.0 / (9.0 * np.asarray(m.diffusion))) for m in p.materials]
        hd, _, _ = flux_weighted_homogenize(flux, mats, p.material_map, z.region_map, **kw)
        hm = [a.replace(diffusion=1.0 / (9.0 * np.asarray(b.diffusion))) for a, b in zip(hm, hd)]
    return hm, rflux, rvol


print(f"SPH-DF feasibility of drum zonings, HP-MR 2-D: {METHOD} reference at refine {REFINE}, "
      f"node model {spec}, 8 drum sectors, device {device}\n", flush=True)
for angle in angles:
    p_solve, _ = build_hpmr2d_pins(refine=REFINE, drum_angle_deg=angle)
    flux = np.asarray(load_or_solve(REFINE, angle, METHOD, device, cache, p_solve)["flux"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p, _ = build_hpmr2d_pins(refine=REFINE, drum_angle_deg=angle,
                                 transport_correction="outscatter")
    cells = J = None
    print(f"angle {angle:g}: min index over boundary zones per group, infeasible conditions, "
          f"worst zone and group", flush=True)
    for label, kw in VARIANTS.items():
        t0 = time.perf_counter()
        nh, rh, st = sector_zoning_hooks(8, **kw)
        node = node_model(spec, angle, device, PINS.xy[PINS.fuel], R_FUEL, per_hex=True,
                          zones_hook=nh)
        z = rh(None, p)
        if cells is None:
            cells, J = boundary_cell_currents(REFINE, angle, METHOD, device, cache, st["q"],
                                              p.raster, tag=f"bcell-{spec.replace(':', '')}")
        bregs, Jref = zone_boundary_currents(node["zones"], cells, J)
        for recipe in RECIPES:
            hm, rflux, rvol = homogenize(flux, p, z, recipe)
            G = Jref.shape[1]
            phi, k, leak = node["solve_bcf"](hm, np.full((len(bregs), G), 1e8), bregs)
            idx = (leak / np.asarray(phi)[bregs]) / (Jref / rflux[bregs])
            b, gw = np.unravel_index(int(np.argmin(idx)), idx.shape)
            zz = bregs[b]
            print(f"  {label:28s} D {recipe:9s} {len(bregs):2d} zones | "
                  + " ".join(f"{idx[:, g].min():5.2f}" for g in range(G))
                  + f" | {int((idx < 1).sum()):3d}/{idx.size} | worst {z.names[zz]} g{gw + 1} "
                  f"(D {hm[zz].diffusion[gw]:.3f}, sa {hm[zz].sigma_a[gw]:.3f}) "
                  f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    print(flush=True)
