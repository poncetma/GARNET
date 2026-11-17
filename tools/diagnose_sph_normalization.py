"""Fixed-k against flux-volume SPH on the pin-resolved HP-MR, zone by zone.

    python tools/diagnose_sph_normalization.py [REFINE] [METHOD] [NODES] [ANGLES] [DEVICE] [MASK]

    REFINE, METHOD : cached reference of tools/hpmr_pin_reference.py
                     (default 16, sn-p2:4:16), homogenized per assembly zone
                     with transport-corrected constants (``pin_reference``)
    NODES          : node spec of tools/sph_hpmr_nodes.py (default raster:4)
    ANGLES         : comma list of drum angles (default 0,180)
    DEVICE         : auto (default), gpu or cpu
    MASK           : fuel (default; Griffin's choice) or interior (fuel, central
                     cell and drum zones; the vacuum-facing Be reflector stays
                     uncorrected)

SPH on the MASK zones, three ways:
  A  fixed k (normalization="none", k = k_ref), the setup of the SPH studies;
  B  flux-volume (one constant per group over the fuel zones, k free);
  C  fixed k started from B's factors (a second branch of A, if any).
Each line: k error, worst assembly-power error, the fuel-volume-weighted
mean factor per group (fast, middle, thermal), then the thermal factors by
fuel ring (hex distance from the core centre). Fuel thermal factors near 0.1
in every ring (A) with factors near 1 and a small k error in B would mean
the fixed-k solve trades a small k mismatch for a large common rescaling.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sph_hpmr_nodes import node_model, pin_reference  # noqa: E402

from garnet import sph_correct  # noqa: E402
from garnet.power import fission_energy_xs  # noqa: E402

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 16
method = sys.argv[2] if len(sys.argv) > 2 else "sn-p2:4:16"
spec = sys.argv[3] if len(sys.argv) > 3 else "raster:4"
angles = [float(a) for a in (sys.argv[4] if len(sys.argv) > 4 else "0,180").split(",")]
device = sys.argv[5] if len(sys.argv) > 5 else "auto"
mask_name = sys.argv[6] if len(sys.argv) > 6 else "fuel"


def ring(site):
    q, r = site
    return (abs(q) + abs(r) + abs(q + r)) // 2


def assembly_power(materials, flux, volume, fuel):
    table, _ = fission_energy_xs(materials)
    p = ((table * flux).sum(axis=1) * volume)[fuel]
    return p / p.mean()


print(f"SPH on the {mask_name} zones of HP-MR 2-D: {method} reference at refine {refine}, "
      f"node model {spec}, device {device}\n", flush=True)
for angle in angles:
    t0 = time.perf_counter()
    ref = pin_reference(refine, method, angle, False, device)
    node = node_model(spec, angle, device)
    z = ref["zones"]
    fuel = z.fuel
    fz = np.flatnonzero(fuel)
    other = [i for i, n in enumerate(z.names) if n in ("central", "drum_be", "drum_absorber")]
    mask = fuel.copy()
    if mask_name == "interior":
        mask[other] = True
    elif mask_name != "fuel":
        raise ValueError(f"MASK must be fuel or interior, got {mask_name!r}")
    rings = np.array([ring(z.site[i]) for i in fz])
    G = ref["rflux"].shape[1]
    groups = (0, G // 2, G - 1)
    vol = ref["rvol"]
    p_ref = assembly_power(ref["hm"], ref["rflux"], vol, fuel)
    phi0, k0 = node["solve"](ref["hm"])
    dp0 = 100 * np.abs(assembly_power(ref["hm"], np.asarray(phi0), vol, fuel) / p_ref - 1).max()
    print(f"angle {angle:g}: k_ref {ref['k']:.6f}; rings {dict(zip(*np.unique(rings, return_counts=True)))}; "
          f"no SPH {1e5 * (k0 - ref['k']):+.0f} pcm, dP {dp0:.2f} %", flush=True)
    runs = {}
    runs["A fixed k"] = sph_correct(ref["hm"], z.region_map, ref["rflux"], node["solve"],
                                    normalization="none", k_ref=ref["k"], sph_regions=mask,
                                    tol=1e-7)
    runs["B flux-volume"] = sph_correct(ref["hm"], z.region_map, ref["rflux"], node["solve"],
                                        normalization="flux-volume", region_volume=vol,
                                        sph_regions=mask, tol=1e-7, max_iter=400)
    runs["C fixed k from B"] = sph_correct(ref["hm"], z.region_map, ref["rflux"], node["solve"],
                                           normalization="none", k_ref=ref["k"],
                                           sph_regions=mask, tol=1e-7,
                                           mu0=runs["B flux-volume"].factors)
    head = (f"  {'variant':<18} {'conv':>5} {'iter':>5} {'dk pcm':>8} {'dP %':>6} | "
            + " ".join(f"{'<mu> g' + str(g + 1):>9}" for g in groups) + " | thermal mu by ring: "
            + "  ".join(f"ring {r} (min-mean-max)" for r in np.unique(rings))
            + ("  | " + "  ".join(f"{z.names[i]} g1/g6/g11" for i in other) if mask_name == "interior" else ""))
    print(head)
    for name, out in runs.items():
        mu = out.factors
        phi, k = node["solve"](out.corrected_materials)
        dp = 100 * np.abs(assembly_power(out.corrected_materials, np.asarray(phi), vol, fuel)
                          / p_ref - 1).max()
        w = (vol[fz, None] * ref["rflux"][fz])                  # fuel flux-volume weights
        mean = (w * mu[fz]).sum(0) / w.sum(0)
        th = mu[fz, -1]
        by_ring = "  ".join(f"{th[rings == r].min():.3f}-{th[rings == r].mean():.3f}-{th[rings == r].max():.3f}"
                            .ljust(22) for r in np.unique(rings))
        print(f"  {name:<18} {str(bool(out.converged)):>5} {out.iterations:5d} "
              f"{1e5 * (k - ref['k']):+8.1f} {dp:6.2f} | "
              + " ".join(f"{mean[g]:9.3f}" for g in groups) + " | " + by_ring
              + ("| " + "  ".join("/".join(f"{mu[i, g]:.3g}" for g in groups) for i in other)
                 if mask_name == "interior" else ""), flush=True)
    print(f"  [{time.perf_counter() - t0:.0f} s]\n", flush=True)
