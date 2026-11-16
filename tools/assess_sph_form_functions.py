"""Per-group form functions on the pin-resolved HP-MR (Kumar et al. 2026, Eqs. 4-5).

    python tools/assess_sph_form_functions.py [ref_refine] [node_spec] [device]

Plan step 4 of the SPH audit (docs/qa/sph_audit_2026-11-13.md, Sec. 9.3):

* reference: TriSP3 on the pin-resolved core of
  :func:`garnet.benchmarks.hpmr_core_pins.build_hpmr2d_pins` (``ref_refine``),
  homogenized per assembly zone; reference pin rates kappaSigma_f phi
  integrated over each fuel compact's exact footprint;
* coarse model: the node model ``node_spec`` (tools/sph_hpmr_nodes.py:
  ``raster:R``, ``smeared:R``, ``local:R:L:polar|centroid``; a bare integer
  means ``raster:R``) with the zone materials, fixed-k SPH on the fuel zones;
* form functions f_{p,g} = R^ref_{p,g} / (kappa~Sigma_f Phi_{p,g}) with
  Phi_{p,g} the coarse flux over the same footprint, and the reconstruction
  P_p = sum_g f kappa~Sigma_f Phi.

Tables of zone cross sections, factors and form functions are generated at
drum angles 0, 45, 90, 135, 180 and applied at every 22.5 degrees, exactly
(on the grid), by linear interpolation on the 45- and 90-degree grids, and by
the nearest 45-degree entry, against a pin-resolved reference solved at that
angle. Baselines: uncorrected diffusion with the reference's own zone cross
sections, and f = 1 (SPH alone, pin power = kappa~Sigma_f Phi over the
footprint). Pin powers are normalized to mean 1 over all fuel pins.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sph_hpmr_nodes import describe, merge_drum, node_model as make_node  # noqa: E402

from garnet import (TriSP3EigenSolver, apply_sph_factors,  # noqa: E402
                    flux_weighted_homogenize, sph_correct)
from garnet.benchmarks.hpmr import PITCH, hpmr_assembly_zones  # noqa: E402
from garnet.benchmarks.hpmr_assembly import FUEL as FUEL_COMPACT  # noqa: E402
from garnet.benchmarks.hpmr_core_pins import (PIN_OFFSET, R_FUEL,  # noqa: E402
                                              build_hpmr2d_pins)
from garnet.pin_power import (footprint_flux, group_form_functions,  # noqa: E402
                              pin_footprints, reconstruct_group_form_functions)
from garnet.power import fission_energy_xs  # noqa: E402
from garnet.materials import Material  # noqa: E402

ref_refine = int(sys.argv[1]) if len(sys.argv) > 1 else 16
node_spec = sys.argv[2] if len(sys.argv) > 2 else "raster:4"
if node_spec.isdigit():
    node_spec = f"raster:{node_spec}"
device = sys.argv[3] if len(sys.argv) > 3 else "auto"
TIGHT = dict(tol_k=1e-9, tol_source=1e-8)
TABLE = (0.0, 45.0, 90.0, 135.0, 180.0)
APPLY = tuple(np.arange(0.0, 180.0 + 1e-9, 22.5))


def reference(angle):
    p, pins = build_hpmr2d_pins(refine=ref_refine, drum_angle_deg=angle)
    z = hpmr_assembly_zones(p, merge_drum=merge_drum(node_spec))
    r = TriSP3EigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                          mask_bc=p.mask_bc, mix_material=p.mix_material,
                          mix_weight=p.mix_weight, device=device).solve(**TIGHT)
    assert r.converged
    hm, rflux, rvol = flux_weighted_homogenize(
        r.flux_numpy, p.materials, p.material_map, z.region_map,
        cell_volume=p.grid.cell_volume, mix_material=p.mix_material,
        mix_weight=p.mix_weight, mix_region_map=z.mix_region_map)
    fuel = pins.fuel
    table, key = fission_energy_xs(p.materials)
    W = pin_footprints(p.raster, pins.xy[fuel], R_FUEL)
    rate = table[PIN_OFFSET + FUEL_COMPACT] * footprint_flux(W, r.flux_numpy)
    return dict(k=r.k_eff, hm=hm, rflux=rflux, rvol=rvol, zones=z, rate=rate,
                key=key, pins=pins)


def node_model(angle, pins):
    return make_node(node_spec, angle, device, pins.xy[pins.fuel], R_FUEL)


def normalized(p):
    return p / p.mean()


def evaluate(ref, node, materials, form, zone_of_pin):
    """k error (pcm), assembly-power max error (%), pin-power RMS and max (%)."""
    flux, k = node["solve_full"](materials)
    table, key = fission_energy_xs(materials)
    assert key == ref["key"], (key, ref["key"])
    phi_p = footprint_flux(node["W"], flux)
    if form is None:
        pin = (table[zone_of_pin] * phi_p).sum(axis=1)
    else:
        pin = reconstruct_group_form_functions(form, zone_of_pin, table, phi_p)
    err = normalized(pin) / normalized(ref["rate"].sum(axis=1)) - 1.0
    assembly = np.bincount(zone_of_pin, weights=pin)[zone_of_pin]
    a_ref = np.bincount(zone_of_pin, weights=ref["rate"].sum(axis=1))[zone_of_pin]
    a_err = normalized(assembly) / normalized(a_ref) - 1.0
    return (1e5 * (k - ref["k"]), 100 * np.abs(a_err).max(),
            100 * np.sqrt((err ** 2).mean()), 100 * np.abs(err).max())


t0 = time.perf_counter()
refs = {a: reference(a) for a in APPLY}
pins = refs[APPLY[0]]["pins"]
nodes = {a: node_model(a, pins) for a in APPLY}
zones = refs[APPLY[0]]["zones"]
fuel_zones = zones.fuel
zone_of_pin = int(np.flatnonzero(fuel_zones)[0]) + pins.assembly[pins.fuel]
print(f"Pin-resolved HP-MR 2-D, TriSP3 refine {ref_refine} "
      f"(side {PITCH / (np.sqrt(3.0) * ref_refine):.3f} cm); node model "
      f"{describe(node_spec)} ({nodes[APPLY[0]]['n_cells']} cells); "
      f"{int(pins.fuel.sum())} fuel pins, device {device} "
      f"[{time.perf_counter() - t0:.0f} s for references and footprints]\n")

entries = {}
print("Generation at the table angles (fixed k, fuel zones):")
print(f"{'angle':>6} {'k_ref':>10} {'conv':>5} {'nfev':>5} {'mu thermal':>12} "
      f"{'f fast':>13} {'f thermal':>13} {'construction':>13}")
for a in TABLE:
    ref, node = refs[a], nodes[a]
    out = sph_correct(ref["hm"], zones.region_map, ref["rflux"], node["solve"],
                      normalization="none", k_ref=ref["k"], sph_regions=fuel_zones,
                      tol=1e-8)
    flux, _ = node["solve_full"](out.corrected_materials)
    table, _ = fission_energy_xs(out.corrected_materials)
    phi_p = footprint_flux(node["W"], flux)
    form = group_form_functions(ref["rate"], zone_of_pin, table, phi_p)
    back = reconstruct_group_form_functions(form, zone_of_pin, table, phi_p)
    exact = np.abs(back / ref["rate"].sum(axis=1) - 1).max()
    entries[a] = dict(hm=ref["hm"], mu=out.factors, form=form)
    mu = out.factors[fuel_zones, -1]
    print(f"{a:6.1f} {ref['k']:10.6f} {str(out.converged):>5} {out.iterations:5d} "
          f"{mu.min():5.3f}-{mu.max():5.3f} {form[:, 0].min():6.3f}-{form[:, 0].max():6.3f} "
          f"{form[:, 1].min():6.3f}-{form[:, 1].max():6.3f} {exact:13.1e}", flush=True)


def interpolate(a, grid):
    grid = np.asarray(grid)
    if np.any(np.isclose(grid, a)):
        e = entries[float(grid[np.isclose(grid, a)][0])]
        return e["hm"], e["mu"], e["form"]
    i = int(np.searchsorted(grid, a))
    lo, hi = entries[float(grid[i - 1])], entries[float(grid[i])]
    w = (a - grid[i - 1]) / (grid[i] - grid[i - 1])

    def mix(x, y):
        return (1 - w) * x + w * y
    hm = []
    for m0, m1 in zip(lo["hm"], hi["hm"]):
        kf = (None if m0.kappa_fission is None else
              mix(np.asarray(m0.kappa_fission), np.asarray(m1.kappa_fission)))
        hm.append(Material(name=m0.name, diffusion=mix(m0.diffusion, m1.diffusion),
                           sigma_a=mix(m0.sigma_a, m1.sigma_a),
                           nu_sigma_f=mix(m0.nu_sigma_f, m1.nu_sigma_f),
                           sigma_s=mix(m0.sigma_s, m1.sigma_s), chi=m0.chi,
                           total=mix(m0.sigma_t, m1.sigma_t), kappa_fission=kf))
    return hm, mix(lo["mu"], hi["mu"]), mix(lo["form"], hi["form"])


def nearest(a, grid):
    g = float(np.asarray(grid)[np.argmin(np.abs(np.asarray(grid) - a))])
    e = entries[g]
    return e["hm"], e["mu"], e["form"]


methods = {
    "linear 45": lambda a: interpolate(a, TABLE),
    "linear 90": lambda a: interpolate(a, TABLE[::2]),
    "nearest 45": lambda a: nearest(a, TABLE),
}
print("\nApplication (dk pcm | max assembly dP % | pin RMS % | pin max %):")
print(f"{'angle':>6} {'no SPH, own XS':>30} {'SPH, f = 1 (lin 45)':>30} "
      + " ".join(f"{m:>30}" for m in methods))
for a in APPLY:
    ref, node = refs[a], nodes[a]
    cells = [evaluate(ref, node, ref["hm"], None, zone_of_pin)]
    hm, mu, _ = interpolate(a, TABLE)
    cells.append(evaluate(ref, node, apply_sph_factors(hm, mu),
                          None, zone_of_pin))
    for get in methods.values():
        hm, mu, form = get(a)
        cells.append(evaluate(ref, node, apply_sph_factors(hm, mu),
                              form, zone_of_pin))
    print(f"{a:6.1f} " + " ".join(
        f"{c[0]:+8.1f} {c[1]:5.2f} {c[2]:5.2f} {c[3]:5.2f}".rjust(30) for c in cells),
        flush=True)
print(f"\ntotal {time.perf_counter() - t0:.0f} s")
