"""Can SPH + boundary discontinuity factors meet the reference leakage at all?

    python tools/probe_sph_bcf_feasibility.py [REFINE] [METHOD] [NODES] [ANGLES] [DEVICE] [CACHE] [ZONING]

Arguments as for tools/diagnose_sph_bcf.py.

A boundary factor gamma scales the Marshak albedo of a zone's vacuum faces. As
gamma -> infinity the face becomes zero-flux, and the leakage reaches the
largest value the diffusion model can carry across the boundary cells. A
uniform SPH factor mu scales D and every cross section of its zone together,
which leaves the flux shape inside the zone unchanged, so it cannot raise
that limit relative to the zone's average flux either. The normalization-free
feasibility index is therefore

    index = [L_coarse / Phi_coarse](zero-flux boundary) / [J_out,ref / phi_ref]

per boundary zone and group: outgoing current per unit zone-average flux,
coarse model at its largest against the reference. Below 1 the condition
c L = J_ref cannot be met by that zone's gamma (neighbouring zones' factors
change the shape only weakly). The same ratio at gamma = 1 (plain Marshak)
shows how far from the reference the uncorrected boundary is. Both use the
flux-weighted zone constants without SPH, and again with the flux-volume SPH
factors of all zones, since those set the converged neighbourhood.
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

from garnet import sph_correct  # noqa: E402

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 16
method = sys.argv[2] if len(sys.argv) > 2 else "sn-p2:4:16"
spec = sys.argv[3] if len(sys.argv) > 3 else "raster:8"
angles = [float(a) for a in (sys.argv[4] if len(sys.argv) > 4 else "0,180").split(",")]
device = sys.argv[5] if len(sys.argv) > 5 else "auto"
cache = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else DEFAULT_CACHE
PER_HEX = (sys.argv[7] if len(sys.argv) > 7 else "pooled") == "hex"
MERGE = merge_drum(spec)
# SPH_SPLIT=1: an inner and an outer zone in every hex on the vacuum boundary
SPLIT = os.environ.get("SPH_SPLIT", "") not in ("", "0")
BIG = 1e8          # gamma for the zero-flux limit

print(f"SPH-DF feasibility on HP-MR 2-D: {method} reference at refine {refine}, node model "
      f"{spec}, {'per-hex' if PER_HEX else 'pooled'} zones"
      f"{', boundary hexes split inner/outer' if SPLIT else ''}, device {device}\n", flush=True)
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
    names = list(ref["zones"].names)
    bregs = sorted(cur["boundaries"])
    B, G = len(bregs), ref["rflux"].shape[1]
    Jref = np.array([cur["boundaries"][b][:, 0] for b in bregs])          # (B, G)
    if SPLIT:
        bregs = [split["outer_of"][b] for b in bregs]
    ref_ratio = Jref / ref["rflux"][bregs]                                 # J / phi per zone
    fv = sph_correct(ref["hm"], ref["zones"].region_map, ref["rflux"], node["solve"],
                     region_volume=ref["rvol"], normalization="flux-volume", tol=1e-7,
                     max_iter=400)
    print(f"angle {angle:g}: {B} boundary zones; flux-volume SPH (all zones) "
          f"{'converged' if fv.converged else 'NOT converged'}", flush=True)
    for label, mats in (("no SPH", ref["hm"]), ("flux-volume SPH, all zones",
                                                fv.corrected_materials)):
        rows = {}
        for gam_value, key in ((1.0, "marshak"), (BIG, "zero-flux")):
            phi, k, leak = node["solve_bcf"](mats, np.full((B, G), gam_value), bregs)
            rows[key] = (leak / np.asarray(phi)[bregs]) / ref_ratio          # (B, G)
        idx, mar = rows["zero-flux"], rows["marshak"]
        print(f"  {label}: feasibility index (zero-flux max / reference), min over boundary "
              f"zones, and Marshak / reference, per group", flush=True)
        print("   group " + " ".join(f"{g + 1:>6d}" for g in range(G)), flush=True)
        print("   index " + " ".join(f"{idx[:, g].min():6.2f}" for g in range(G)), flush=True)
        print("   zone  " + " ".join(f"{names[bregs[int(np.argmin(idx[:, g]))]][:6]:>6}"
                                     for g in range(G)), flush=True)
        print("   Marsh " + " ".join(f"{mar[:, g].min():6.2f}" for g in range(G)), flush=True)
        bad = [(names[bregs[b]], g + 1, float(idx[b, g])) for b in range(B) for g in range(G)
               if idx[b, g] < 1.0]
        print(f"   infeasible conditions (index < 1): {len(bad)} of {B * G}"
              + ("" if not bad else ": " + ", ".join(f"{n} g{g} {v:.2f}"
                                                     for n, g, v in sorted(bad, key=lambda t: t[2])[:12])),
              flush=True)
    print(f"  [{time.perf_counter() - t0:.0f} s]\n", flush=True)
