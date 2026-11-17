"""How finely must SPH factors be tabulated in drum angle? (HP-MR 2-D)

    python tools/assess_sph_angle_tables.py [reference] [node_spec] [device] [variants]

Plan step 5 of the SPH audit (docs/qa/sph_audit_2026-11-13.md, Sec. 9.3):
cross sections, factors and form functions are tabulated in drum angle and
interpolated linearly, as in Griffin (Kumar et al., Ann. Nucl. Energy 230 (2026)
112111, Sec. 3.5.1, where form functions needed a 5 degree grid). This measures
the cross-section and factor part on the assembly-zone workflow. ``reference``
is either an integer R (TriSP3 on the assembly-level raster of refine R with
the volume-mixed arc and two-group placeholder data, the first study) or
``pinR:METHOD`` (the faithful pin-resolved core at refine R, 11-group, exact
pin blending, solved by METHOD, a method of tools/hpmr_pin_reference.py such
as ``sn-p2:4:16`` (anisotropic S_N, the faithful reference), and cached by it;
zone constants are collapsed with transport-corrected data, see
``pin_reference`` in tools/sph_hpmr_nodes.py). A node model
given by ``node_spec`` (see tools/sph_hpmr_nodes.py: ``raster:R``,
``smeared:R``, ``local:R:L:polar|centroid``), one zone per fuel assembly,
fixed-k SPH. ``variants`` is a comma list of ``fuel`` (fuel zones only, the
default, Griffin's choice) and ``interior`` (every zone but the void and the
vacuum-facing Be reflector).

All 12 drums rotate together. Zone cross sections and factors are generated
every 7.5 degrees from 0 to 180. Tables of spacing 15, 30, 45, 60, 90 and 180
degrees are subsets of that grid; each is interpolated linearly (cross
sections and factors) at the 7.5-degree points that are not on it, and the
corrected node model is compared there with a reference solved at that angle.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sph_hpmr_nodes import describe, merge_drum, node_model, pin_reference  # noqa: E402

from garnet import (TriSP3EigenSolver, apply_sph_factors,  # noqa: E402
                    flux_weighted_homogenize, sph_correct)
from garnet.benchmarks.hpmr import build_hpmr2d, hpmr_assembly_zones  # noqa: E402
from garnet.materials import Material  # noqa: E402
from garnet.power import fission_energy_xs  # noqa: E402

ref_spec = sys.argv[1] if len(sys.argv) > 1 else "16"
PIN_REF = ref_spec.startswith("pin")
ref_refine = int(ref_spec[3:].split(":")[0]) if PIN_REF else int(ref_spec)
ref_method = ref_spec.split(":", 1)[1] if PIN_REF else "sp3" 
node_spec = sys.argv[2] if len(sys.argv) > 2 else "raster:4"
device = sys.argv[3] if len(sys.argv) > 3 else "auto"
VARIANTS = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("fuel",)
TIGHT = dict(tol_k=1e-9, tol_source=1e-8)
STEP = 7.5
GRID = np.arange(0.0, 180.0 + STEP / 2, STEP)
SPACINGS = (15.0, 30.0, 45.0, 60.0, 90.0, 180.0)
MERGE = merge_drum(node_spec)


def reference(angle):
    if PIN_REF:
        return pin_reference(ref_refine, ref_method, angle, MERGE, device)
    p = build_hpmr2d(refine=ref_refine, drum_angle_deg=angle, absorber="polar")
    z = hpmr_assembly_zones(p, merge_drum=MERGE)
    r = TriSP3EigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                          mask_bc=p.mask_bc, mix_material=p.mix_material,
                          mix_weight=p.mix_weight, device=device).solve(**TIGHT)
    assert r.converged
    hm, rflux, rvol = flux_weighted_homogenize(
        r.flux_numpy, p.materials, p.material_map, z.region_map,
        cell_volume=p.grid.cell_volume, mix_material=p.mix_material,
        mix_weight=p.mix_weight, mix_region_map=z.mix_region_map)
    return dict(k=r.k_eff, hm=hm, rflux=rflux, rvol=rvol, zones=z)


def assembly_power(materials, flux, volume, fuel):
    table, _ = fission_energy_xs(materials)
    p = ((table * flux).sum(axis=1) * volume)[fuel]
    return p / p.mean()


def errors(ref, solve, materials, fuel):
    phi, k = solve(materials)
    p_ref = assembly_power(ref["hm"], ref["rflux"], ref["rvol"], fuel)
    dp = np.abs(assembly_power(materials, np.asarray(phi), ref["rvol"], fuel) / p_ref - 1)
    return 1e5 * (k - ref["k"]), 100 * dp.max()


def blend(m0, m1, w):
    """Linear interpolation of two zone materials (a cross-section table)."""
    def mix(x, y):
        return (1 - w) * np.asarray(x, dtype=float) + w * np.asarray(y, dtype=float)
    kf = (None if m0.kappa_fission is None else mix(m0.kappa_fission, m1.kappa_fission))
    return Material(name=m0.name, diffusion=mix(m0.diffusion, m1.diffusion),
                    sigma_a=mix(m0.sigma_a, m1.sigma_a),
                    nu_sigma_f=mix(m0.nu_sigma_f, m1.nu_sigma_f),
                    sigma_s=mix(m0.sigma_s, m1.sigma_s), chi=m0.chi,
                    total=mix(m0.sigma_t, m1.sigma_t), kappa_fission=kf)


t0 = time.perf_counter()
refs, nodes = {}, {}
for a in GRID:
    refs[a] = reference(a)
    nodes[a] = node_model(node_spec, a, device)
zones = refs[GRID[0]]["zones"]
fuel = zones.fuel
interior = fuel | np.isin(np.arange(len(zones.names)),
                          [i for i, n in enumerate(zones.names)
                           if n in ("central", "drum_be", "drum_absorber", "drum")])
masks = {"fuel": fuel, "interior": interior}
print(f"HP-MR 2-D SPH tables in drum angle: "
      f"{'pin-resolved 11-group ' + ref_method if PIN_REF else 'assembly-level TriSP3'} "
      f"reference refine {ref_refine}; "
      f"node model {describe(node_spec)} ({nodes[GRID[0]]['n_cells']} cells); "
      f"device {device}; {len(GRID)} states "
      f"[{time.perf_counter() - t0:.0f} s for the references]\n", flush=True)

factors = {}
header = f"{'angle':>6} {'k_ref':>10} {'no-SPH dk':>10} {'dP':>6} |"
for v in VARIANTS:
    header += f" {v + ': conv':>14} {'nfev':>5} {'thermal mu':>12} |"
print(header)
for a in GRID:
    ref, solve = refs[a], nodes[a]["solve"]
    dk0, dp0 = errors(ref, solve, ref["hm"], fuel)
    line = f"{a:6.1f} {ref['k']:10.6f} {dk0:+9.1f}p {dp0:5.2f}% |"
    for v in VARIANTS:
        out = sph_correct(ref["hm"], zones.region_map, ref["rflux"], solve,
                          normalization="none", k_ref=ref["k"], verbose=True, sph_regions=masks[v],
                          tol=1e-7, mu0=factors.get((a - STEP, v)))
        factors[a, v] = out.factors
        mu = out.factors[fuel, -1]
        line += (f" {str(out.converged):>14} {out.iterations:5d} "
                 f"{mu.min():5.3f}-{mu.max():5.3f} |")
    print(line, flush=True)

print("\nLinear interpolation of zone cross sections and factors at angles off the "
      "table\n(max over the test angles; dk in pcm, dP = max assembly-power error):")
print(f"{'spacing':>8} {'tests':>5} | {'no SPH dk':>9} {'dP':>6} | "
      + " | ".join(f"{v + ' dk':>11} {'dP':>6}" for v in VARIANTS))
for spacing in SPACINGS:
    table = GRID[np.isclose(np.mod(GRID, spacing), 0.0)
                 | np.isclose(np.mod(GRID, spacing), spacing)]
    tests = [a for a in GRID if not np.any(np.isclose(table, a))]
    worst = {k: [0.0, 0.0] for k in ("none",) + VARIANTS}
    for a in tests:
        ref, solve = refs[a], nodes[a]["solve"]
        hi = table[np.searchsorted(table, a)]
        lo = table[np.searchsorted(table, a) - 1]
        w = (a - lo) / (hi - lo)
        hm = [blend(m0, m1, w) for m0, m1 in zip(refs[lo]["hm"], refs[hi]["hm"])]
        dk, dp = errors(ref, solve, hm, fuel)
        worst["none"] = [max(worst["none"][0], abs(dk)), max(worst["none"][1], dp)]
        for v in VARIANTS:
            mu = (1 - w) * factors[lo, v] + w * factors[hi, v]
            dk, dp = errors(ref, solve, apply_sph_factors(hm, mu), fuel)
            worst[v] = [max(worst[v][0], abs(dk)), max(worst[v][1], dp)]
    print(f"{spacing:7.1f}d {len(tests):5d} | {worst['none'][0]:8.1f}p "
          f"{worst['none'][1]:5.2f}% | "
          + " | ".join(f"{worst[v][0]:10.1f}p {worst[v][1]:5.2f}%" for v in VARIANTS),
          flush=True)
print(f"\ntotal {time.perf_counter() - t0:.0f} s")
