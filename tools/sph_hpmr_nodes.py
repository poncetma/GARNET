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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from garnet import TriDiffusionEigenSolver, region_average  # noqa: E402
from garnet.backend import asnumpy  # noqa: E402
from garnet.benchmarks.hpmr import (build_hpmr2d, hpmr_assembly_zones,  # noqa: E402
                                    hpmr_locally_refined_problem, hpmr_mesh_zones)
from garnet.mesh import UnstructuredDiffusionSolver  # noqa: E402
from garnet.pin_power import mesh_pin_footprints, pin_footprints  # noqa: E402

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


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
    if kind in ("raster", "smeared"):
        q = build_hpmr2d(refine=int(rest[0]), drum_angle_deg=angle,
                         absorber="polar" if kind == "raster" else "smeared")
        z = hpmr_assembly_zones(q, merge_drum=(kind == "smeared"))
        mix = (dict(mix_material=z.mix_region_map, mix_weight=q.mix_weight)
               if z.mix_region_map is not None else {})

        def solve_full(materials):
            r = TriDiffusionEigenSolver(q.grid, materials, z.region_map,
                                        active=q.active, mask_bc=q.mask_bc,
                                        device=device, **mix).solve(**TIGHT)
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
                                              device=device).solve(**TIGHT)
            return asnumpy(res.flux), res.k_eff

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
