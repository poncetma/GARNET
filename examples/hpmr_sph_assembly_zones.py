"""HP-MR assembly-level SPH: one equivalence zone per fuel assembly on a node mesh.

The production workflow (b) of the SPH audit (docs/qa/sph_audit_2026-11-13.md,
Sec. 9.3), following Kumar et al., Ann. Nucl. Energy 230 (2026) 112111:

* reference: full-core TriSP3 on the fine raster (``ref_refine``), vacuum
  boundary, volume-mixed (polar) drum absorber;
* zones: one per fuel assembly plus one per other material
  (:func:`garnet.benchmarks.hpmr.hpmr_assembly_zones`), numbered identically
  on every raster;
* coarse model: TriDiffusion on a node raster (``node_refine``; 1 = six
  triangles per hexagon, Griffin's coarse mesh; default 2) with the zone
  materials;
* factors: fixed k, no normalization, either the fuel zones only (Griffin) or
  every zone except the void and the vacuum-facing Be reflector.

It then applies factors generated at one drum angle to the node model at
another angle, against the reference there. The zone cross sections do not
depend on the drum angle here (each zone is one material), so the transfer
error is the state dependence of the factors themselves: what plan step 5
(tables of factors in drum angle) has to capture.

    python examples/hpmr_sph_assembly_zones.py [ref_refine] [node_refine] [device]

Cross sections are placeholders, so the numbers are illustrative.
"""

import sys
import time

import numpy as np

from garnet import (TriDiffusionEigenSolver, TriSP3EigenSolver, apply_sph_factors,
                    flux_weighted_homogenize, region_average, sph_correct)
from garnet.benchmarks.hpmr import build_hpmr2d, hpmr_assembly_zones
from garnet.power import fission_energy_xs

ref_refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
node_refine = int(sys.argv[2]) if len(sys.argv) > 2 else 2
device = sys.argv[3] if len(sys.argv) > 3 else "auto"
TIGHT = dict(tol_k=1e-9, tol_source=1e-8)
ANGLES = (0.0, 90.0, 180.0)


def reference(angle):
    """Fine TriSP3 reference, homogenized per zone."""
    p = build_hpmr2d(refine=ref_refine, drum_angle_deg=angle, absorber="polar")
    z = hpmr_assembly_zones(p)
    r = TriSP3EigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                          mask_bc=p.mask_bc, mix_material=p.mix_material,
                          mix_weight=p.mix_weight, device=device).solve(**TIGHT)
    assert r.converged
    hm, rflux, rvol = flux_weighted_homogenize(
        r.flux_numpy, p.materials, p.material_map, z.region_map,
        cell_volume=p.grid.cell_volume, mix_material=p.mix_material,
        mix_weight=p.mix_weight, mix_region_map=z.mix_region_map)
    return dict(k=r.k_eff, hm=hm, rflux=rflux, rvol=rvol, zones=z)


def node_solver(angle):
    """solve(zone materials) -> (zone flux, k) on the node raster."""
    p = build_hpmr2d(refine=node_refine, drum_angle_deg=angle, absorber="polar")
    z = hpmr_assembly_zones(p)
    mix = dict(mix_material=z.mix_region_map, mix_weight=p.mix_weight)

    def solve(materials):
        r = TriDiffusionEigenSolver(p.grid, materials, z.region_map, active=p.active,
                                    mask_bc=p.mask_bc, device=device,
                                    **mix).solve(**TIGHT)
        return (region_average(r.flux_numpy, z.region_map,
                               cell_volume=p.grid.cell_volume, **mix), r.k_eff)
    return solve


def assembly_power(materials, flux, volume, fuel):
    """Fuel-zone power, normalized to mean 1."""
    table, _ = fission_energy_xs(materials)
    p = (table * flux).sum(axis=1) * volume
    p = p[fuel]
    return p / p.mean()


t0 = time.perf_counter()
refs = {a: reference(a) for a in ANGLES}
solvers = {a: node_solver(a) for a in ANGLES}
zones = refs[ANGLES[0]]["zones"]
fuel = zones.fuel
interior = fuel | np.isin(np.arange(len(zones.names)),
                          [zones.names.index(n) for n in ("central", "drum_be",
                                                          "drum_absorber")])
print(f"HP-MR 2-D assembly-level SPH: TriSP3 reference refine {ref_refine} "
      f"({6 * ref_refine ** 2} tri/hex), TriDiffusion nodes refine {node_refine} "
      f"({6 * node_refine ** 2} tri/hex), {int(fuel.sum())} fuel zones "
      f"[{time.perf_counter() - t0:.0f} s for the references]\n")

factors = {}
print("Generation (fixed k at each angle's reference):")
print(f"  {'angle':>5} {'variant':>9} {'k_ref':>10} {'no-SPH dk':>10} {'no-SPH dP':>10} "
      f"{'conv':>5} {'nfev':>5} {'SPH dk':>8} {'SPH dP':>8} {'fuel mu range (fast | thermal)':>34}")
for a in ANGLES:
    ref, solve = refs[a], solvers[a]
    p_ref = assembly_power(ref["hm"], ref["rflux"], ref["rvol"], fuel)
    phi0, k0 = solve(ref["hm"])
    dp0 = 100 * np.abs(assembly_power(ref["hm"], np.asarray(phi0), ref["rvol"], fuel)
                       / p_ref - 1).max()
    for variant, mask in (("fuel", fuel), ("interior", interior)):
        t = time.perf_counter()
        out = sph_correct(ref["hm"], zones.region_map, ref["rflux"], solve,
                          normalization="none", k_ref=ref["k"], sph_regions=mask,
                          tol=1e-7)
        phi, k = solve(out.corrected_materials)
        dp = 100 * np.abs(assembly_power(out.corrected_materials, np.asarray(phi),
                                         ref["rvol"], fuel) / p_ref - 1).max()
        mu = out.factors[fuel]
        factors[a, variant] = out.factors
        print(f"  {a:5.0f} {variant:>9} {ref['k']:10.6f} {1e5 * (k0 - ref['k']):+9.1f}p "
              f"{dp0:9.2f}% {str(out.converged):>5} {out.iterations:5d} "
              f"{1e5 * (k - ref['k']):+7.2f}p {dp:7.4f}% "
              f"  {mu[:, 0].min():.3f}-{mu[:, 0].max():.3f} | "
              f"{mu[:, 1].min():.3f}-{mu[:, 1].max():.3f}  "
              f"[{time.perf_counter() - t:.0f} s]")

print("\nTransfer: factors from one angle applied to the node model at another "
      "(dk in pcm, dP = max assembly-power error):")
print(f"  {'applied at':>10} {'factors':>22} {'fuel only':>20} {'interior':>20}")
for a in ANGLES:
    ref, solve = refs[a], solvers[a]
    p_ref = assembly_power(ref["hm"], ref["rflux"], ref["rvol"], fuel)
    sources = [(f"from {g:.0f} deg", lambda v, g=g: factors[g, v]) for g in ANGLES]
    sources.append(("linear 0-180 deg", lambda v, a=a:
                    factors[0.0, v] + (a / 180.0) * (factors[180.0, v] - factors[0.0, v])))
    phi0, k0 = solve(ref["hm"])
    dp0 = 100 * np.abs(assembly_power(ref["hm"], np.asarray(phi0), ref["rvol"], fuel)
                       / p_ref - 1).max()
    print(f"  {a:8.0f}deg {'none (diffusion)':>22} "
          f"{1e5 * (k0 - ref['k']):+8.1f}p {dp0:6.2f}%")
    for label, get in sources:
        cells = []
        for v in ("fuel", "interior"):
            mats = apply_sph_factors(ref["hm"], get(v))
            phi, k = solve(mats)
            dp = 100 * np.abs(assembly_power(mats, np.asarray(phi), ref["rvol"], fuel)
                              / p_ref - 1).max()
            cells.append(f"{1e5 * (k - ref['k']):+8.1f}p {dp:6.2f}%")
        print(f"  {'':>10} {label:>22}  {cells[0]:>18}  {cells[1]:>18}")
