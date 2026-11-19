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
               pin_radius=None, per_hex: bool = False, zones_hook=None):
    """dict(solve, solve_full, solve_bcf, W, zones, n_cells) for one drum angle.

    ``solve_bcf(materials, gamma, boundary_regions)`` and ``solve_bcf_full``
    take boundary discontinuity factors for :func:`garnet.sph.sph_bcf_correct`,
    on every node kind (the drum-band mesh through
    :class:`garnet.equivalence.SphDfNode`). ``zones_hook(zones, problem)``
    rezones the node (e.g. :func:`sector_zoning_hooks`)."""
    kind, *rest = spec.split(":")
    last = {"state": None, "k": 1.0}          # warm start for the next solve
    if kind in ("raster", "smeared"):
        q = build_hpmr2d(refine=int(rest[0]), drum_angle_deg=angle,
                         absorber="polar" if kind == "raster" else "smeared")
        z = hpmr_assembly_zones(q, merge_drum=(kind == "smeared"), per_hex=per_hex)
        if zones_hook is not None:
            z = zones_hook(z, q)
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

        def solve_bcf_full(materials, gamma, boundary_regions):
            """Node solve with boundary discontinuity factors gamma (B, G) on
            the vacuum boundary of ``boundary_regions``: (full flux, k,
            boundary leakage (B, G))."""
            from garnet.tri import tri_boundary_leakage
            G = materials[0].n_groups
            bcf = np.ones((G,) + q.grid.shape)
            for b, reg in enumerate(boundary_regions):
                bcf[:, z.region_map == reg] = np.asarray(gamma[b], float)[:, None]
            shift = (dict(wielandt_shift=WIELANDT_SHIFT)
                     if G > 2 and WIELANDT_SHIFT else {})
            s = TriDiffusionEigenSolver(q.grid, materials, z.region_map,
                                        active=q.active, mask_bc=q.mask_bc,
                                        device=device, bcf=bcf, **mix)
            r = s.solve(state0=last["state"], k_guess=last["k"], **shift, **TIGHT)
            last["state"], last["k"] = [r.flux[g] for g in range(G)], r.k_eff
            flux = r.flux_numpy
            nreg = len(z.names)
            leak = np.array([tri_boundary_leakage(s.ops[g], flux[g], z.region_map, nreg,
                                                  q.active)[list(boundary_regions)]
                             for g in range(G)]).T
            return flux, r.k_eff, leak

        def solve_bcf(materials, gamma, boundary_regions):
            """As solve_bcf_full, with the zone-average flux (region flux, k,
            boundary leakage (B, G))."""
            flux, k, leak = solve_bcf_full(materials, gamma, boundary_regions)
            avg = region_average(flux, z.region_map, cell_volume=q.grid.cell_volume, **mix)
            return avg, k, leak

        cell_volume = q.grid.cell_volume
        W = (pin_footprints(q.raster, pin_xy, pin_radius)
             if pin_xy is not None else None)
        n_cells = int((q.material_map > 0).sum())
    else:
        solve_bcf = solve_bcf_full = None
    if kind == "local":
        r, levels, mode = int(rest[0]), int(rest[1]), rest[2]
        q = hpmr_locally_refined_problem(refine=r, drum_angle_deg=angle,
                                         levels=levels, absorber=mode)
        z = hpmr_mesh_zones(q, per_hex=per_hex)
        if zones_hook is not None:
            z = zones_hook(z, q)
        mix = (dict(mix_material=z.mix_region_map, mix_weight=q.mix_weight)
               if z.mix_region_map is not None else {})
        sph_nodes = {}

        def _sph_node(boundary_regions):
            from garnet.equivalence import SphDfNode
            key = tuple(int(b) for b in boundary_regions)
            if key not in sph_nodes:
                sph_nodes[key] = SphDfNode.mesh(z, list(key), q, device=device, tol=TIGHT)
            return sph_nodes[key]

        def solve_bcf_full(materials, gamma, boundary_regions):
            """Mesh node solve with boundary discontinuity factors: (cell flux,
            k, boundary leakage (B, G)); garnet.equivalence.SphDfNode."""
            return _sph_node(boundary_regions).solve_full(materials, gamma)

        def solve_bcf(materials, gamma, boundary_regions):
            return _sph_node(boundary_regions)(materials, gamma)

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
    elif kind not in ("raster", "smeared"):
        raise ValueError(f"unknown node spec {spec!r}")

    def solve(materials):
        flux, k = solve_full(materials)
        return region_average(flux, z.region_map, cell_volume=cell_volume, **mix), k

    return dict(solve=solve, solve_full=solve_full, solve_bcf=solve_bcf,
                solve_bcf_full=solve_bcf_full, W=W, zones=z, n_cells=n_cells)


def pin_reference(refine: int, method: str, angle: float, merge: bool = False,
                  device: str = "auto", cache: str | None = None,
                  homogenize_tc: str | None = "auto", per_hex: bool = False,
                  zones_hook=None):
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
    None, ``"outscatter"`` or ``"diagonal"``. ``per_hex`` homogenizes per hex of
    the outer ring (:func:`garnet.benchmarks.hpmr.hpmr_assembly_zones`).

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
    z = hpmr_assembly_zones(p, merge_drum=merge, per_hex=per_hex)
    if zones_hook is not None:
        z = zones_hook(z, p)
    hm, rflux, rvol = flux_weighted_homogenize(
        flux, p.materials, p.material_map, z.region_map,
        cell_volume=p.grid.cell_volume, mix_material=p.mix_material,
        mix_weight=p.mix_weight, mix_region_map=z.mix_region_map)
    table, key = fission_energy_xs(p.materials)
    W = pin_footprints(p.raster, pins.xy[pins.fuel], R_FUEL)
    rate = table[PIN_OFFSET + FUEL_COMPACT] * footprint_flux(W, flux)
    return dict(k=float(r["k"]), hm=hm, rflux=rflux, rvol=rvol, zones=z, rate=rate,
                key=key, pins=pins, homogenize_tc=homogenize_tc)


# --- inner/outer split of the boundary hexes -------------------------------
# Laboure et al. (Ann. Nucl. Energy 128 (2019) 443, HTTR) separate equivalence
# regions from homogenization regions. Here every hex that touches the vacuum
# boundary (Be reflector hexes, drum bodies) gets an extra "outer" zone: the
# first row of node cells along the boundary (every active node cell with a
# vertex on it). The layer is defined on the node raster and carried to the
# reference raster through the exact nesting of the two rasters, so both see the
# same physical region. All of a hex's vacuum faces then belong to its outer
# zone, whose reference boundary current is the hex's.

def _boundary_parent_zones(zones, drums=True):
    return [i for i, n in enumerate(zones.names)
            if (n.startswith("be_reflector") or (drums and n.startswith("drum")))
            and not n.startswith("drum_absorber")]


def split_boundary_layer(zones, problem, drums=True):
    """(zones with outer zones appended, outer node-cell mask, {parent: outer zone}).
    ``drums=False`` splits only the Be reflector hexes."""
    from dataclasses import replace

    from garnet.pin_power import tri_cell_vertices
    V = tri_cell_vertices(problem.raster)                       # (nr, nc, 2, 3, 2)
    act = np.asarray(problem.active, bool)
    scale = 1e6 / problem.raster.side
    keys = np.round(V * scale).astype(np.int64)
    key = keys[..., 0] * 4_000_000_007 + keys[..., 1]            # (nr, nc, 2, 3)
    boundary = np.intersect1d(key[act].ravel(), key[~act].ravel())
    touches = np.isin(key, boundary).any(axis=-1) & act
    parents = _boundary_parent_zones(zones, drums)
    region = zones.region_map.copy()
    R = len(zones.names)
    outer_of = {p: R + j for j, p in enumerate(parents)}
    outer = touches & np.isin(region, parents)
    new_region = region.copy()
    for p, o in outer_of.items():
        new_region[outer & (region == p)] = o
    mix = zones.mix_region_map
    if mix is not None:
        mix = np.where(outer & (mix == region), new_region, mix)
    names = tuple(zones.names) + tuple(f"{zones.names[p]}:outer" for p in parents)
    material = np.concatenate([zones.material, zones.material[parents]])
    z = replace(zones, region_map=new_region, mix_region_map=mix, names=names,
                material=material, fuel=np.concatenate([zones.fuel, np.zeros(len(parents), bool)]),
                site=tuple(zones.site) + tuple(zones.site[p] for p in parents))
    return z, outer, outer_of


def locate_parent_cells(fine_raster, coarse_raster):
    """Flat index of the coarse cell that contains each fine cell (nested rasters)."""
    from scipy.spatial import cKDTree

    from garnet.pin_power import tri_cell_vertices
    Vc = tri_cell_vertices(coarse_raster).reshape(-1, 3, 2)
    cf = tri_cell_vertices(fine_raster).reshape(-1, 3, 2).mean(axis=1)
    _, cand = cKDTree(Vc.mean(axis=1)).query(cf, k=6)

    def cross(u, v, w):
        return ((u[..., 0] - w[..., 0]) * (v[..., 1] - w[..., 1])
                - (v[..., 0] - w[..., 0]) * (u[..., 1] - w[..., 1]))

    hit = np.full(len(cf), -1)
    for j in range(cand.shape[1]):
        T = Vc[cand[:, j]]
        d = np.stack([cross(cf, T[:, 0], T[:, 1]), cross(cf, T[:, 1], T[:, 2]),
                      cross(cf, T[:, 2], T[:, 0])])
        ok = ~((d < -1e-12).any(0) & (d > 1e-12).any(0)) & (hit < 0)
        hit[ok] = cand[ok, j]
    return hit


def transfer_boundary_layer(zones, problem, node_problem, node_outer, outer_of):
    """Apply a node-raster split to the zones of a nested (finer) raster."""
    from dataclasses import replace
    parent_cell = locate_parent_cells(problem.raster, node_problem.raster)
    if np.any(parent_cell < 0):
        raise ValueError("the reference raster does not nest in the node raster")
    outer_fine = np.asarray(node_outer).reshape(-1)[parent_cell].reshape(zones.region_map.shape)
    region = zones.region_map.copy()
    new_region = region.copy()
    for p, o in outer_of.items():
        new_region[outer_fine & (region == p)] = o
    mix = zones.mix_region_map
    if mix is not None:
        mix = np.where(outer_fine & (mix == region), new_region, mix)
    parents = sorted(outer_of, key=outer_of.get)
    names = tuple(zones.names) + tuple(f"{zones.names[p]}:outer" for p in parents)
    material = np.concatenate([zones.material, zones.material[parents]])
    return replace(zones, region_map=new_region, mix_region_map=mix, names=names,
                   material=material,
                   fuel=np.concatenate([zones.fuel, np.zeros(len(parents), bool)]),
                   site=tuple(zones.site) + tuple(zones.site[p] for p in parents))


# --- drum ring sectors --------------------------------------------------------
# Instead of a thin absorber-arc zone, each drum gets a core zone (pure Be) and
# n fixed ring sectors containing the outer ring where the B4C arc lives. The
# sectors are fixed in space (relative to each drum's outward radial line), so
# as the drum rotates the arc moves through them and each sector's absorber
# content -- and its homogenized constants -- change continuously with angle.

def _replace_zones(zones, region, mix, names, material, fuel, site):
    from dataclasses import replace
    return replace(zones, region_map=region, mix_region_map=mix, names=tuple(names),
                   material=np.asarray(material), fuel=np.asarray(fuel, bool),
                   site=tuple(site))


def sector_zoning_hooks(n_sectors=8, r_inner=10.25, outer_sectors=False, aligned=False,
                        drum_outer=True, outer_arc_to_sector=False):
    """(node hook, reference hook, state) for node_model / pin_reference: the
    drum sector zonings of :func:`garnet.benchmarks.hpmr.hpmr_drum_sector_zones`
    (garnet.equivalence) on the node model, raster or drum-band mesh, carried
    to the nested reference raster by containment (``inherit_zones``).

    The baseline is ``aligned=True, outer_arc_to_sector=True``. ``state`` keeps
    the node and reference problems ("q", "p"), the node cells ("cells"), the
    boundary zones ("boundary") and ``outer_of``: {per-hex parent zone (Be
    reflector 2-13, drums 14-25): its outer zone}, so the per-hex reference
    boundary currents apply unchanged when no boundary zone is split further.
    """
    from garnet.benchmarks.hpmr import DRUM_ABSORBER, hpmr_drum_sector_zones
    from garnet.equivalence import inherit_zones
    from garnet.pin_power import tri_cell_centroids
    state = {}

    def node_hook(_z, q):
        zs, cells, bz, outer_of = hpmr_drum_sector_zones(
            q, n_sectors, r_inner, aligned, outer_sectors=outer_sectors,
            outer_arc_to_sector=outer_arc_to_sector, drum_outer=drum_outer)
        state.update(q=q, cells=cells, outer_of=outer_of, boundary=bz, node_zones=zs)
        return zs

    def ref_hook(_z, p):
        state["p"] = p
        return inherit_zones(state["node_zones"], state["cells"], tri_cell_centroids(p.raster),
                             np.asarray(p.active).shape, p.active, p.mix_material,
                             absorber_ids=[DRUM_ABSORBER])

    return node_hook, ref_hook, state


def exposed_cells(problem):
    """Mask of the active cells with a vacuum edge, in the problem's cell shape
    (raster (nr, nc, 2) or drum-band mesh (n_cells,))."""
    from garnet.equivalence import NodeCells
    cells = (NodeCells.from_mesh(problem.mesh) if hasattr(problem, "mesh")
             else NodeCells.from_raster(problem.raster, problem.active))
    return cells.exposed.reshape(cells.shape)


def boundary_cell_currents(refine, angle, method, device, cache, node_problem, ref_raster,
                           tag):
    """Reference outgoing current through the vacuum edges of every exposed
    node cell: (flat node-cell indices (N,), J_out (N, G)).

    The S_N reference is tallied once on a region map with one region per
    exposed node cell (cached as ..._currents_<tag>.npz), so the reference
    boundary current of any zoning made of whole node cells is the sum over
    its exposed cells (``zone_boundary_currents``). ``ref_raster`` only places
    the reference cells in the node cells: the S_N problem itself is built by
    ``load_or_solve_currents`` from ``method``. (Do not pass the problem of
    ``pin_reference``: it carries the transport-corrected data used for the
    homogenization, and an S_N solve on it is a different reference.)"""
    from hpmr_pin_reference import load_or_solve_currents
    exp = exposed_cells(node_problem).reshape(-1)
    cells = np.flatnonzero(exp)
    ids = np.zeros(exp.size, int)
    ids[cells] = np.arange(1, cells.size + 1)
    from garnet.equivalence import NodeCells, locate
    from garnet.pin_power import tri_cell_centroids
    node_cells = (NodeCells.from_mesh(node_problem.mesh) if hasattr(node_problem, "mesh")
                  else NodeCells.from_raster(node_problem.raster, node_problem.active))
    act = (np.asarray(ref_raster.material_map) > 0).reshape(-1)
    parent = np.full(act.size, -1)
    parent[act] = locate(tri_cell_centroids(ref_raster).reshape(-1, 2)[act], node_cells)
    if np.any(act & (parent < 0)):
        raise ValueError("the reference raster does not nest in the node cells")
    fine = np.where(parent >= 0, ids[np.maximum(parent, 0)], 0).reshape(
        np.asarray(ref_raster.material_map).shape)
    cur = load_or_solve_currents(refine, angle, method, device, cache, region_map=fine, tag=tag)
    stray = [b for b in cur["boundaries"] if b == 0 and np.any(cur["boundaries"][b])]
    if stray:
        raise ValueError("vacuum edges outside the exposed node cells")
    G = next(iter(cur["boundaries"].values())).shape[0]
    J = np.zeros((cells.size, G))
    for b, v in cur["boundaries"].items():
        if b > 0:
            J[b - 1] = v[:, 0]
    return cells, J


def zone_boundary_currents(zones, cells, J):
    """(boundary zones, J_out (B, G)) summed from per-cell currents."""
    zone = np.asarray(zones.region_map).reshape(-1)[cells]
    bregs = sorted(set(zone.tolist()))
    return bregs, np.array([J[zone == b].sum(axis=0) for b in bregs])


def absorber_split_zones(zones):
    """Zones with the absorber of every drum zone moved to a zone of its own.

    The mix component (the volume-mixed B4C) of each cell whose mix region is
    a drum zone z goes to a new zone "<name of z>:B4C"; everything else stays.
    Homogenizing on these zones gives, per drum zone, the non-absorber and the
    absorber constants and region fluxes, whose flux-volume weighted mix is
    the drum zone's constants (``shielded_mix``). Returns (zones, {z: zB}).
    """
    names, material = list(zones.names), list(zones.material)
    fuel, site = list(zones.fuel), list(zones.site)
    mix = np.array(zones.mix_region_map, copy=True)
    of = {}
    for z, n in enumerate(zones.names):
        if not n.startswith("drum") or not np.any(mix == z):
            continue
        of[z] = len(names)
        mix[mix == z] = len(names)
        names.append(f"{n}:B4C")
        material.append(zones.material[z])
        fuel.append(False)
        site.append(zones.site[z])
    return _replace_zones(zones, zones.region_map, mix, names, material, fuel, site), of


def shielded_mix(x_rest, x_abs, v_rest, v_abs, s):
    """Flux-volume weighted mix of two parts' constants, per group.

    x_* : (..., G) constants of the non-absorber and absorber parts (for a
    scattering matrix the leading axis is the source group, weighted alike);
    v_* : part volumes; s : (G,) absorber-to-rest average flux ratio.
    Returns (x_rest + rho x_abs) / (1 + rho), rho = v_abs s / v_rest."""
    rho = (v_abs / max(v_rest, 1e-300)) * np.asarray(s, float)
    if np.ndim(x_rest) == 2:                      # (G, G) scattering, by source group
        rho = rho[:, None]
    return (np.asarray(x_rest) + rho * np.asarray(x_abs)) / (1.0 + rho)
