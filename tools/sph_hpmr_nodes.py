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

    ``solve_bcf`` (raster and smeared nodes only; None for the unstructured
    local mesh, whose solver has no boundary coefficient) takes boundary
    discontinuity factors for :func:`garnet.sph.sph_bcf_correct`."""
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


def drum_ring_sectors(zones, problem, n_sectors=8, r_inner=10.25, outer_sectors=False,
                      aligned=False, outer_arc_to_sector=False):
    """Split every merged per-hex drum zone ("drum(R, C)", from
    ``hpmr_assembly_zones(merge_drum=True, per_hex=True)``) into a core zone
    (node-cell centroids within ``r_inner`` of the drum centre) and
    ``n_sectors`` ring sectors (the rest of the drum hex). Sector k spans
    outward azimuth + (k - 1/2, k + 1/2) * 360/n, so sector 0 faces away from
    the core and sector n/2 faces it. ``aligned=True`` turns them by half a
    sector, to outward azimuth + (k, k + 1) * 360/n: the sector boundaries then
    lie on the outward line and every 360/n from it, and the edges of a 90-deg
    arc (centred on the core-facing line at drum angle 0) cross them at drum
    angles that are multiples of 360/n -- on the table points for n = 8 and
    15- or 45-deg tables -- so a sector's absorber content has its kinks there
    and not between table points.

    ``outer_arc_to_sector=True`` keeps each drum's outer layer one zone of Be
    only: the absorber fraction of every outer-layer cell (its mix component)
    goes to the ring sector at the cell's azimuth. No zone with a vacuum face
    then holds B4C. (A boundary zone that holds part of the arc is smeared
    uniformly, so it cannot reproduce where the reference leaks, and SPH-DF
    has no solution at some drum angles; tools/probe_sph_drum_zonings.py.)

    ``outer_sectors=True`` splits each drum's outer boundary layer ("drum(R, C):outer",
    from :func:`split_boundary_layer`) into the same sectors as well,
    "drum(R, C):outer:s<k>", keeping only the sectors that hold node cells (the
    first one reuses the outer zone's number). A single outer zone smears the
    arc's share of the layer over the whole boundary of the drum hex; with the
    arc facing out (165-180 deg) the coarse model then cannot leak what the
    reference does, at any boundary factor."""
    from garnet.benchmarks.hpmr import PITCH
    from garnet.hexraster import hex_site_xy
    from garnet.pin_power import tri_cell_vertices
    cen = tri_cell_vertices(problem.raster).mean(axis=-2)        # (nr, nc, 2, 2)
    region = zones.region_map
    new = region.copy()
    names, material = list(zones.names), list(zones.material)
    fuel, site = list(zones.fuel), list(zones.site)
    dtheta = 2.0 * np.pi / n_sectors
    arc_cells = []                       # (outer zone, its cells, sector of each cell)
    first_sector = {}                    # drum site -> zone number of its sector 0
    for d, n in enumerate(zones.names):
        if not n.startswith("drum("):
            continue
        cx, cy = hex_site_xy(*zones.site[d], PITCH)
        out_az = np.arctan2(cy, cx)
        cells = region == d
        dx, dy = cen[..., 0] - cx, cen[..., 1] - cy
        r = np.hypot(dx, dy)
        shift = 0.0 if aligned else dtheta / 2.0
        k = np.floor(np.mod(np.arctan2(dy, dx) - out_az + shift, 2.0 * np.pi)
                     / dtheta).astype(int) % n_sectors
        if n.endswith(":outer"):
            if outer_arc_to_sector:
                arc_cells.append((d, cells, k))
            if outer_sectors:
                held = sorted(set(k[cells].tolist()))
                names[d] = f"{n}:s{held[0]}"
                for s in held[1:]:
                    new[cells & (k == s)] = len(names)
                    names.append(f"{n}:s{s}")
                    material.append(zones.material[d])
                    fuel.append(False)
                    site.append(zones.site[d])
            continue
        names[d] = f"{n}:core"
        base = len(names)
        first_sector[tuple(zones.site[d])] = base
        for s in range(n_sectors):
            names.append(f"{n}:s{s}")
            material.append(zones.material[d])
            fuel.append(False)
            site.append(zones.site[d])
        ring = cells & (r >= r_inner)
        new[ring] = base + k[ring]
    mix = zones.mix_region_map
    if mix is not None:
        mix = np.where((mix == region) & (new != region), new, mix)
        for o, cells, k in arc_cells:
            move = cells & (mix == o)
            mix = np.where(move, first_sector[tuple(zones.site[o])] + k, mix)
    return _replace_zones(zones, new, mix, names, material, fuel, site)


def transfer_zones(zones, problem, node_problem, node_old, node_zones, node_old_mix=None):
    """Carry a node-raster rezoning to the zones of a nested (finer) raster.

    ``node_old`` is the node region map before the rezoning and ``node_zones``
    the rezoned node zones; ``zones`` the fine raster's zones before it (same
    numbering as ``node_old``). Every fine cell whose node parent was rezoned
    takes the parent's new zone; with ``node_old_mix`` (the node mix map
    before the rezoning) the mix component is carried the same way."""
    parent = locate_parent_cells(problem.raster, node_problem.raster)
    if np.any(parent < 0):
        raise ValueError("the reference raster does not nest in the node raster")
    old_p = np.asarray(node_old).reshape(-1)[parent].reshape(zones.region_map.shape)
    new_p = np.asarray(node_zones.region_map).reshape(-1)[parent].reshape(zones.region_map.shape)
    region = zones.region_map
    changed = new_p != old_p
    if np.any(changed & (region != old_p)):
        raise ValueError("fine and node zonings disagree on rezoned cells")
    new = np.where(changed, new_p, region)
    mix = zones.mix_region_map
    if mix is not None:
        mix = np.where((mix == region) & changed, new, mix)
        if node_old_mix is not None and node_zones.mix_region_map is not None:
            old_m = np.asarray(node_old_mix).reshape(-1)[parent].reshape(region.shape)
            new_m = np.asarray(node_zones.mix_region_map).reshape(-1)[parent].reshape(region.shape)
            moved = (new_m != old_m) & ~changed
            mix = np.where(moved & (zones.mix_region_map == old_m), new_m, mix)
    return _replace_zones(zones, new, mix, node_zones.names, node_zones.material,
                          node_zones.fuel, node_zones.site)


def sector_zoning_hooks(n_sectors=8, r_inner=10.25, outer_sectors=False, aligned=False,
                        drum_outer=True, outer_arc_to_sector=False):
    """(node hook, reference hook, state) for node_model / pin_reference:
    per-hex merged drums -> inner/outer split of every hex on the vacuum
    boundary -> each drum's inner part into a core and ring sectors (and with
    ``outer_sectors`` its outer layer into sectors too), identical on node and
    reference. ``state["outer_of"]`` maps every boundary hex's zone (the
    per-hex numbering: Be reflector 2-13, drums 14-25) to its outer zone, so
    without ``outer_sectors`` the per-hex reference boundary currents apply unchanged;
    with it use :func:`boundary_cell_currents`. ``state`` also keeps the node
    and reference problems ("q", "p"). ``drum_outer=False`` leaves the drum
    hexes unsplit: their ring sectors reach the vacuum boundary and carry the
    boundary factors themselves."""
    from garnet.benchmarks.hpmr import hpmr_assembly_zones
    state = {}

    def node_hook(_z, q):
        zm = hpmr_assembly_zones(q, merge_drum=True, per_hex=True)
        zsplit, outer, outer_of = split_boundary_layer(zm, q, drums=drum_outer)
        zs = drum_ring_sectors(zsplit, q, n_sectors, r_inner, outer_sectors=outer_sectors,
                               aligned=aligned, outer_arc_to_sector=outer_arc_to_sector)
        state.update(q=q, outer=outer, outer_of=outer_of, node_split=zsplit.region_map,
                     node_split_mix=zsplit.mix_region_map, node_sectors=zs)
        return zs

    def ref_hook(_z, p):
        state["p"] = p
        zm = hpmr_assembly_zones(p, merge_drum=True, per_hex=True)
        zsplit = transfer_boundary_layer(zm, p, state["q"], state["outer"], state["outer_of"])
        return transfer_zones(zsplit, p, state["q"], state["node_split"], state["node_sectors"],
                              state["node_split_mix"])

    return node_hook, ref_hook, state


def exposed_cells(problem):
    """(nr, nc, 2) mask of the active cells with at least one vacuum edge."""
    from garnet.tri_sn import _EDGES
    act = np.asarray(problem.active, bool)
    nr, nc = act.shape[:2]
    I, J = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    out = np.zeros_like(act)
    for t, di, dj, tn, _ in _EDGES:
        ii, jj = I + di, J + dj
        inside = (ii >= 0) & (ii < nr) & (jj >= 0) & (jj < nc)
        nbr = np.zeros((nr, nc), bool)
        nbr[inside] = act[ii[inside], jj[inside], tn]
        out[:, :, t] |= act[:, :, t] & ~nbr
    return out


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
    parent = locate_parent_cells(ref_raster, node_problem.raster)
    if np.any(parent < 0):
        raise ValueError("the reference raster does not nest in the node raster")
    fine = ids[parent].reshape(np.asarray(ref_raster.material_map).shape)
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
