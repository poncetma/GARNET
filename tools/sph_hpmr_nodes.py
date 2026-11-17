"""Node models of the 2-D HP-MR for the SPH study tools.

A node model is the homogenized coarse solve of the assembly-level SPH workflow:
zone materials in, zone-average flux and k out, plus the full flux and the
fuel-compact footprints for pin-power reconstruction. The spec string picks the
mesh and the drum treatment:

    raster:R           uniform triangular raster, refine R, volume-mixed
                       (polar) absorber arc
    smeared:R          uniform raster, refine R, each drum one hex-homogenized
                       zone (drum body and arc together; Kumar et al.'s
                       homogenized-drum variant)
    local:R:L:MODE     raster refine R with the drum absorber band split L
                       times (drums at refine R * 2**L), unstructured finite
                       volumes; MODE = polar (volume-mixed arc) or centroid
                       (arc painted by cell centroid)

``merge_drum(spec)`` says whether the reference must be homogenized with the
merged drum zone so the zone sets match.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from garnet import TriDiffusionEigenSolver, region_average  # noqa: E402
from garnet.backend import asnumpy  # noqa: E402
from garnet.benchmarks.hpmr import (build_hpmr2d, hpmr_assembly_zones,  # noqa: E402
                                    hpmr_locally_refined_problem, hpmr_mesh_zones)
from garnet.mesh import UnstructuredDiffusionSolver  # noqa: E402
from garnet.pin_power import mesh_pin_footprints, pin_footprints  # noqa: E402

# Node solves inside an SPH factor solve: warm-started from the previous
# evaluation, and tight, because Jacobian-free Newton differences residuals.
TIGHT = dict(tol_k=1e-11, tol_source=1e-10)
# Fixed Wielandt shift for multigroup cores with upscatter (11-group HP-MR:
# ~250 -> ~10 outers, garnet.solver.DiffusionEigenSolver.solve).
WIELANDT_SHIFT = 0.3


def merge_drum(spec: str) -> bool:
    return spec.split(":")[0] == "smeared"


def describe(spec: str) -> str:
    kind, *rest = spec.split(":")
    if kind == "raster":
        return f"uniform raster refine {rest[0]}, volume-mixed arc"
    if kind == "smeared":
        return f"uniform raster refine {rest[0]}, hex-homogenized drum"
    r, levels, mode = rest
    return (f"raster refine {r}, drum band refined x{2 ** int(levels)} "
            f"(drums at refine {int(r) * 2 ** int(levels)}), {mode} arc")


def node_model(spec: str, angle: float, device: str = "auto", pin_xy=None,
               pin_radius=None):
    """dict(solve, solve_full, W, zones, n_cells) for one drum angle."""
    kind, *rest = spec.split(":")
    last = {"state": None, "k": 1.0}          # warm start for the next solve
    if kind in ("raster", "smeared"):
        q = build_hpmr2d(refine=int(rest[0]), drum_angle_deg=angle,
                         absorber="polar" if kind == "raster" else "smeared")
        z = hpmr_assembly_zones(q, merge_drum=(kind == "smeared"))
        mix = (dict(mix_material=z.mix_region_map, mix_weight=q.mix_weight)
               if z.mix_region_map is not None else {})

        def solve_full(materials):
            G = materials[0].n_groups
            shift = (dict(wielandt_shift=WIELANDT_SHIFT)
                     if G > 2 and WIELANDT_SHIFT else {})
            r = TriDiffusionEigenSolver(q.grid, materials, z.region_map,
                                        active=q.active, mask_bc=q.mask_bc,
                                        device=device, **mix).solve(
                state0=last["state"], k_guess=last["k"], **shift, **TIGHT)
            last["state"], last["k"] = [r.flux[g] for g in range(G)], r.k_eff
            return r.flux_numpy, r.k_eff

        cell_volume = q.grid.cell_volume
        W = (pin_footprints(q.raster, pin_xy, pin_radius)
             if pin_xy is not None else None)
        n_cells = int((q.material_map > 0).sum())
    elif kind == "local":
        r, levels, mode = int(rest[0]), int(rest[1]), rest[2]
        q = hpmr_locally_refined_problem(refine=r, drum_angle_deg=angle,
                                         levels=levels, absorber=mode)
        z = hpmr_mesh_zones(q)
        mix = (dict(mix_material=z.mix_region_map, mix_weight=q.mix_weight)
               if z.mix_region_map is not None else {})

        def solve_full(materials):
            mats, cells = q.solver_materials(materials, z.region_map, z.mix_region_map)
            res = UnstructuredDiffusionSolver(q.mesh, mats, cells, q.alpha_boundary,
                                              device=device).solve(
                flux0=last["state"], k_guess=last["k"], **TIGHT)
            flux = asnumpy(res.flux)
            last["state"], last["k"] = flux, res.k_eff
            return flux, res.k_eff

        cell_volume = q.mesh.area
        W = (mesh_pin_footprints(q.mesh, pin_xy, pin_radius)
             if pin_xy is not None else None)
        n_cells = q.mesh.n_cells
    else:
        raise ValueError(f"unknown node spec {spec!r}")

    def solve(materials):
        flux, k = solve_full(materials)
        return region_average(flux, z.region_map, cell_volume=cell_volume, **mix), k

    return dict(solve=solve, solve_full=solve_full, W=W, zones=z, n_cells=n_cells)


def pin_reference(refine: int, method: str, angle: float, merge: bool = False,
                  device: str = "auto", cache: str | None = None,
                  homogenize_tc: str | None = "auto"):
    """Pin-resolved reference of the faithful core (11-group, exact blending),
    from the cache of tools/hpmr_pin_reference.py (solved and stored if
    missing), homogenized per assembly zone.

    ``homogenize_tc`` picks the fine data the zone constants are collapsed
    from. Only the transport cross section 1/(3D) and the within-group
    scattering depend on it (absorption, fission and group transfers do not):
    ``"auto"`` uses the method's own transport correction for ``-tco``/``-tcd``
    references, the out-scatter correction for anisotropic ``sn-pL``
    references (the library's D = 1/(3 Sigma_t) has no transport correction,
    which is wrong for a forward-scattering moderator), and the raw data for
    the legacy isotropic ones (consistent with what they solved); otherwise
    None, ``"outscatter"`` or ``"diagonal"``.

    Returns dict(k, hm, rflux, rvol, zones, rate, key, pins): zone materials,
    zone fluxes and volumes, the zones, the fuel-compact rates
    kappaSigma_f phi over each compact's footprint (n_fuel_pins, G), the
    fission-energy key and the pin layout.
    """
    from hpmr_pin_reference import DEFAULT_CACHE, load_or_solve, parse

    from garnet import flux_weighted_homogenize
    from garnet.benchmarks.hpmr_assembly import FUEL as FUEL_COMPACT
    from garnet.benchmarks.hpmr_core_pins import PIN_OFFSET, R_FUEL, build_hpmr2d_pins
    from garnet.pin_power import footprint_flux
    from garnet.power import fission_energy_xs

    _, solve_tc, order, _ = parse(method)
    if homogenize_tc == "auto":
        homogenize_tc = solve_tc or ("outscatter" if order > 0 else None)
    p, pins = build_hpmr2d_pins(refine=refine, drum_angle_deg=angle,
                                transport_correction=solve_tc)
    r = load_or_solve(refine, angle, method, device, cache or DEFAULT_CACHE, p)
    if not bool(r["converged"]):
        raise RuntimeError(f"cached {method} reference at refine {refine}, "
                           f"{angle} deg did not converge")
    if homogenize_tc != solve_tc:
        import warnings
        with warnings.catch_warnings():       # clipped within-group P0 is reported
            warnings.simplefilter("ignore", RuntimeWarning)   # by the solve tools
            p, _ = build_hpmr2d_pins(refine=refine, drum_angle_deg=angle,
                                     transport_correction=homogenize_tc)
    flux = np.asarray(r["flux"])
    z = hpmr_assembly_zones(p, merge_drum=merge)
    hm, rflux, rvol = flux_weighted_homogenize(
        flux, p.materials, p.material_map, z.region_map,
        cell_volume=p.grid.cell_volume, mix_material=p.mix_material,
        mix_weight=p.mix_weight, mix_region_map=z.mix_region_map)
    table, key = fission_energy_xs(p.materials)
    W = pin_footprints(p.raster, pins.xy[pins.fuel], R_FUEL)
    rate = table[PIN_OFFSET + FUEL_COMPACT] * footprint_flux(W, flux)
    return dict(k=float(r["k"]), hm=hm, rflux=rflux, rvol=rvol, zones=z, rate=rate,
                key=key, pins=pins, homogenize_tc=homogenize_tc)
