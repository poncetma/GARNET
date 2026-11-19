"""Equivalence zones and SPH-DF node models for hex cores with control drums.

The assembly-level SPH workflow homogenizes a fine reference into zones,
solves a coarse diffusion "node" model with the zone constants, and corrects
the zones with SPH factors (:mod:`garnet.sph`). For a core with rotating
control drums and a vacuum boundary, the method that reproduces k and every
zone rate, at every drum angle, is SPH with boundary discontinuity factors
(SPH-DF, Laboure et al., Ann. Nucl. Energy 128 (2019) 443;
:func:`garnet.sph.sph_bcf_correct`), on these zones:

* one zone per hex site, each drum hex one zone (body and arc together);
* every non-fuel hex on the vacuum boundary split into an inner zone and an
  outer zone, the first row of node cells (:func:`split_boundary_layer`):
  without the split the coarse model's zero-flux leakage limit falls below
  the reference and no boundary factor can match it;
* each drum hex's inner part split into a core and ``n`` ring sectors fixed
  in space (:func:`drum_ring_sectors`). The arc moves through the sectors as
  the drum turns, so their constants change continuously with angle, and
  with ``aligned=True`` the arc edges cross the sector boundaries at drum
  angles that are multiples of 360/n degrees (table points);
* each drum's outer layer kept free of absorber (``outer_arc_to_sector``):
  the absorber fraction of its cells goes to the ring sector at the same
  azimuth. A zone with a vacuum face that holds part of the arc is uniform in
  the coarse model and cannot place where the reference leaks; SPH-DF then
  has no solution at some angles.

Zones are built on the node cells, of a raster (:meth:`NodeCells.from_raster`)
or of an unstructured mesh with a refined drum band
(:meth:`NodeCells.from_mesh`, :mod:`garnet.drum_mesh`), and carried to a nested
reference by containment (:func:`inherit_zones`): every reference cell takes
the zone of the node cell it lies in, its absorber fraction the node cell's
absorber zone. :func:`drum_core_zones` does all of it for a
:class:`garnet.HexLattice`; :class:`SphDfNode` is the node solve that
:func:`garnet.sph.sph_bcf_correct` iterates.

Measured on the 2-D HP-MR (docs/qa/sph_audit_2026-11-13.md Sec. 10.6): SPH-DF
converges at all drum angles with k exact, and 5-degree tables of the zone
constants, SPH and boundary factors and form functions interpolate to
38 pcm and 0.22 % pin RMS at the midpoints between table points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .backend import asnumpy


@dataclass
class EquivalenceZones:
    """Homogenization / SPH zones of a 2-D core.

    region_map     : zone index of every cell (the raster's (nr, nc, 2) or a
                     mesh's (n_cells,)).
    mix_region_map : zone of every cell's volume-mixed component (the drum
                     absorber), -1 where the cell has none; None without
                     mixing. Pass it as ``mix_region_map`` to
                     :func:`garnet.sph.flux_weighted_homogenize` and
                     :func:`garnet.sph.region_average`.
    names          : zone names.
    material       : (Z,) base material index of each zone.
    fuel           : (Z,) True for fuel zones.
    site           : (Z,) hex site (R, C) of each zone, None for the void.
    """

    region_map: np.ndarray
    mix_region_map: np.ndarray | None
    names: tuple
    material: np.ndarray
    fuel: np.ndarray
    site: tuple

    @property
    def n_zones(self) -> int:
        return len(self.names)


def _rebuild(zones, region, mix, names, material, fuel, site):
    return replace(zones, region_map=region, mix_region_map=mix, names=tuple(names),
                   material=np.asarray(material), fuel=np.asarray(fuel, bool),
                   site=tuple(site))


# --------------------------------------------------------------------------
# Node-cell geometry
# --------------------------------------------------------------------------
@dataclass
class NodeCells:
    """Triangle geometry of a node model's cells, flattened.

    vertices : (N, 3, 2) vertex coordinates; area : (N,); active : (N,);
    exposed : (N,) cells with a vacuum edge; touches : (N,) cells with a
    vertex on the vacuum boundary (the outer boundary layer); shape : the
    shape zone maps take (a raster's (nr, nc, 2) or (N,)).
    """

    vertices: np.ndarray
    area: np.ndarray
    active: np.ndarray
    exposed: np.ndarray
    touches: np.ndarray
    shape: tuple

    @property
    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=1)

    @classmethod
    def from_raster(cls, raster, active=None):
        """Cells of a hex-lattice raster (:class:`garnet.hexraster.TriRaster`)."""
        from .pin_power import tri_cell_vertices
        from .tri_sn import _EDGES
        mmap = np.asarray(raster.material_map)
        act = (mmap > 0) if active is None else np.asarray(active, bool)
        nr, nc = act.shape[:2]
        V = tri_cell_vertices(raster)                               # (nr, nc, 2, 3, 2)
        exposed = np.zeros_like(act)
        I, J = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        for t, di, dj, tn, _ in _EDGES:
            ii, jj = I + di, J + dj
            inside = (ii >= 0) & (ii < nr) & (jj >= 0) & (jj < nc)
            nbr = np.zeros((nr, nc), bool)
            nbr[inside] = act[ii[inside], jj[inside], tn]
            exposed[:, :, t] |= act[:, :, t] & ~nbr
        scale = 1e6 / raster.side
        keys = np.round(V * scale).astype(np.int64)
        key = keys[..., 0] * 4_000_000_007 + keys[..., 1]            # (nr, nc, 2, 3)
        boundary = np.intersect1d(key[act].ravel(), key[~act].ravel())
        touches = np.isin(key, boundary).any(axis=-1) & act
        area = np.full(act.size, np.sqrt(3.0) / 4.0 * raster.side ** 2)
        return cls(vertices=V.reshape(-1, 3, 2), area=area, active=act.reshape(-1),
                   exposed=exposed.reshape(-1), touches=touches.reshape(-1),
                   shape=tuple(act.shape))

    @classmethod
    def from_mesh(cls, mesh):
        """Cells of a 2-D triangular :class:`garnet.mesh.Mesh` (e.g. from
        :func:`garnet.drum_mesh.refine_drum_band`); every cell is active."""
        cells = np.array([tuple(c) for c in mesh.cells])
        if cells.ndim != 2 or cells.shape[1] != 3:
            raise ValueError("NodeCells.from_mesh needs a triangular 2-D mesh")
        coords = np.asarray(mesh.coords, dtype=float)[:, :2]
        V = coords[cells]
        n = len(cells)
        exposed = np.zeros(n, bool)
        owners = np.array([b[0] for b in mesh.bfaces], dtype=np.int64)
        exposed[owners] = True
        # Boundary vertices: the end points of every boundary face.
        bn = np.asarray(mesh.bface_normal, dtype=float).reshape(len(owners), -1)[:, :2]
        bc = np.asarray(mesh.bface_center, dtype=float).reshape(len(owners), -1)[:, :2]
        half = np.array([b[1] for b in mesh.bfaces], dtype=float)[:, None] / 2.0
        tang = np.stack([bn[:, 1], -bn[:, 0]], axis=1)
        ends = np.concatenate([bc + half * tang, bc - half * tang])
        scale = 1e6 / max(float(np.sqrt(np.median(np.asarray(mesh.area)))), 1e-12)
        kb = np.round(ends * scale).astype(np.int64)
        kv = np.round(V * scale).astype(np.int64)
        bkey = kb[:, 0] * 4_000_000_007 + kb[:, 1]
        vkey = kv[..., 0] * 4_000_000_007 + kv[..., 1]
        touches = np.isin(vkey, bkey).any(axis=1)
        return cls(vertices=V, area=np.asarray(mesh.area, dtype=float), active=np.ones(n, bool),
                   exposed=exposed, touches=touches, shape=(n,))


def locate(points, cells: NodeCells, k: int = 12) -> np.ndarray:
    """Index of the node cell containing each point (-1 outside every cell)."""
    from scipy.spatial import cKDTree
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    V = cells.vertices

    def cross(u, v, w):
        return ((u[..., 0] - w[..., 0]) * (v[..., 1] - w[..., 1])
                - (v[..., 0] - w[..., 0]) * (u[..., 1] - w[..., 1]))

    def inside(p, idx):
        T = V[idx]
        d = np.stack([cross(p, T[..., 0, :], T[..., 1, :]), cross(p, T[..., 1, :], T[..., 2, :]),
                      cross(p, T[..., 2, :], T[..., 0, :])])
        return ~((d < -1e-12).any(0) & (d > 1e-12).any(0))

    hit = np.full(len(pts), -1)
    kk = min(k, len(V))
    _, cand = cKDTree(V.mean(axis=1)).query(pts, k=kk)
    cand = cand.reshape(len(pts), kk)
    for j in range(kk):
        ok = (hit < 0) & inside(pts, cand[:, j])
        hit[ok] = cand[ok, j]
    miss = np.flatnonzero(hit < 0)
    if miss.size:                                   # large coarse cells: search them all
        lo, hi = V.min(axis=1), V.max(axis=1)
        for i in miss:
            p = pts[i]
            box = np.flatnonzero(np.all((lo <= p + 1e-9) & (p - 1e-9 <= hi), axis=1))
            if box.size:
                ok = inside(np.broadcast_to(p, (box.size, 2)), box)
                if ok.any():
                    hit[i] = box[np.argmax(ok)]
    return hit


# --------------------------------------------------------------------------
# Zone construction
# --------------------------------------------------------------------------
def site_zones(cells: NodeCells, rows, cols, site_label: dict, fuel_sites=(),
               cell_material=None, mix_material=None, site_material=None) -> EquivalenceZones:
    """One zone per hex site: every cell of the site whatever its material
    (a drum's body and arc together), zone 0 the void (inactive cells).

    rows, cols : per-cell hex site (flat or in ``cells.shape``);
    site_label : {(R, C): label}, the zone is named ``f"{label}{(R, C)}"``;
    fuel_sites : sites whose zones are fuel; site_material : {(R, C): base
    material index} (else the most common base material of its cells);
    mix_material : per-cell mixed material (-1 none): its component joins the
    cell's own zone.
    """
    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    act = cells.active
    region = np.zeros(act.size, dtype=np.int64)
    names, material, fuel, site = ["void"], [0], [False], [None]
    fuel_sites = {tuple(s) for s in fuel_sites}
    cm = None if cell_material is None else np.asarray(cell_material).reshape(-1)
    for rc in site_label:
        sel = act & (rows == rc[0]) & (cols == rc[1])
        if not sel.any():
            continue
        region[sel] = len(names)
        names.append(f"{site_label[rc]}{tuple(rc)}")
        if site_material is not None and rc in site_material:
            material.append(int(site_material[rc]))
        elif cm is not None:
            material.append(int(np.bincount(cm[sel]).argmax()))
        else:
            material.append(-1)
        fuel.append(tuple(rc) in fuel_sites)
        site.append(tuple(rc))
    if np.any(act & (region == 0)):
        raise ValueError("active cells outside every labelled site")
    mix = None
    if mix_material is not None:
        mm = np.asarray(mix_material).reshape(-1)
        mix = np.where(mm >= 0, region, -1).astype(np.int64)
    return EquivalenceZones(region_map=region.reshape(cells.shape),
                            mix_region_map=None if mix is None else mix.reshape(cells.shape),
                            names=tuple(names), material=np.array(material),
                            fuel=np.array(fuel), site=tuple(site))


def split_boundary_layer(zones: EquivalenceZones, cells: NodeCells, split=None):
    """Split each zone on the vacuum boundary into an inner and an outer zone.

    The outer zone is every node cell of the zone with a vertex on the vacuum
    boundary (``cells.touches``), named ``"<name>:outer"`` and appended.
    ``split`` lists the zones to split (default: every non-fuel zone but the
    void that touches the boundary; a split fuel zone would break assembly
    powers). Returns (zones, {zone: its outer zone}).
    """
    region = np.asarray(zones.region_map).reshape(-1)
    touch = cells.touches
    if split is None:
        split = [z for z in range(zones.n_zones)
                 if z != 0 and not zones.fuel[z] and np.any(touch & (region == z))]
    new = region.copy()
    names, material = list(zones.names), list(zones.material)
    fuel, site = list(zones.fuel), list(zones.site)
    outer_of = {}
    for z in split:
        o = len(names)
        outer_of[z] = o
        new[touch & (region == z)] = o
        names.append(f"{zones.names[z]}:outer")
        material.append(zones.material[z])
        fuel.append(False)
        site.append(zones.site[z])
    mix = zones.mix_region_map
    if mix is not None:
        m = np.asarray(mix).reshape(-1)
        mix = np.where((m == region) & (new != region), new, m).reshape(cells.shape)
    return _rebuild(zones, new.reshape(cells.shape), mix, names, material, fuel, site), outer_of


def drum_ring_sectors(zones: EquivalenceZones, cells: NodeCells, drums, outer_of=None,
                      n_sectors: int = 8, r_inner=None, aligned: bool = True,
                      outer_arc_to_sector: bool = True, outer_sectors: bool = False,
                      core_centre=(0.0, 0.0)):
    """Split each drum zone into a core and ``n_sectors`` ring sectors fixed in space.

    drums : drum specs (``site``, ``centre``, ``inner``; as from
        :meth:`garnet.HexLattice.drum_specs`); the drum's zone is the
        non-outer zone of its site. Node cells of the zone with centroid at
        ``r_inner`` (default ``inner - 2 * (outer - inner)``) or more from the
        drum centre form the ring; the rest stays the core, renamed
        ``"<name>:core"``; sectors are appended as ``"<name>:s<k>"``.
    aligned : sector k spans outward azimuth + (k, k + 1) * 360/n, its
        boundaries on the drum's outward line (from ``core_centre``);
        ``False`` centres sector 0 on that line.
    outer_of : {zone: outer zone} of :func:`split_boundary_layer`; with
        ``outer_arc_to_sector`` the absorber fraction of the drum's outer-layer
        cells goes to the ring sector at the cell's azimuth, so the outer zone
        holds no absorber; with ``outer_sectors`` the outer layer is split
        into the same sectors instead (not recommended: with the arc facing
        out an outer sector is mostly absorber and SPH-DF has no solution).
    """
    outer_of = {} if outer_of is None else dict(outer_of)
    region = np.asarray(zones.region_map).reshape(-1)
    mix = None if zones.mix_region_map is None else np.asarray(zones.mix_region_map).reshape(-1).copy()
    new = region.copy()
    names, material = list(zones.names), list(zones.material)
    fuel, site = list(zones.fuel), list(zones.site)
    cen = cells.centroid
    dtheta = 2.0 * np.pi / n_sectors
    outer_zones = set(outer_of.values())
    moves = []
    for dm in drums:
        rc = tuple(dm["site"])
        cands = [z for z in range(zones.n_zones) if zones.site[z] == rc and z not in outer_zones]
        if len(cands) != 1:
            raise ValueError(f"drum at {rc}: expected one zone, found {len(cands)}")
        d = cands[0]
        cx, cy = dm["centre"]
        out_az = math.atan2(cy - core_centre[1], cx - core_centre[0])
        dx, dy = cen[:, 0] - cx, cen[:, 1] - cy
        r = np.hypot(dx, dy)
        shift = 0.0 if aligned else dtheta / 2.0
        k = (np.floor(np.mod(np.arctan2(dy, dx) - out_az + shift, 2.0 * np.pi) / dtheta)
             .astype(int) % n_sectors)
        ri = (dm["inner"] - 2.0 * (dm["outer"] - dm["inner"])) if r_inner is None else r_inner
        base = len(names)
        names[d] = f"{zones.names[d]}:core"
        for s in range(n_sectors):
            names.append(f"{zones.names[d]}:s{s}")
            material.append(zones.material[d])
            fuel.append(False)
            site.append(rc)
        ring = (region == d) & (r >= ri)
        new[ring] = base + k[ring]
        o = outer_of.get(d)
        if o is None:
            continue
        ocells = region == o
        if outer_sectors:
            held = sorted(set(k[ocells].tolist()))
            names[o] = f"{zones.names[o]}:s{held[0]}"
            for s in held[1:]:
                new[ocells & (k == s)] = len(names)
                names.append(f"{zones.names[o]}:s{s}")
                material.append(zones.material[o])
                fuel.append(False)
                site.append(rc)
        elif outer_arc_to_sector and mix is not None:
            moves.append((o, ocells, base + k))
    if mix is not None:
        mix = np.where((mix == region) & (new != region), new, mix)
        for o, ocells, sector in moves:
            mix = np.where(ocells & (mix == o), sector, mix)
    if np.any(np.bincount(new[cells.active], minlength=len(names))[1:] == 0):
        empty = [names[i] for i in np.flatnonzero(
            np.bincount(new[cells.active], minlength=len(names)) == 0) if i != 0]
        raise ValueError(f"empty zones {empty[:6]}: the node mesh is too coarse for "
                         f"{n_sectors} sectors (refine the drum band or use fewer)")
    return _rebuild(zones, new.reshape(cells.shape),
                    None if mix is None else mix.reshape(cells.shape),
                    names, material, fuel, site)


def drum_equivalence_zones(zones: EquivalenceZones, cells: NodeCells, drums,
                           n_sectors: int = 8, r_inner=None, aligned: bool = True,
                           core_centre=(0.0, 0.0)):
    """The baseline SPH-DF zoning from per-site zones: boundary split, then drum
    core and aligned ring sectors with absorber-free drum outer layers.
    Returns (zones, boundary zones (sorted list of the zones with vacuum
    edges), {zone: outer zone})."""
    zs, outer_of = split_boundary_layer(zones, cells)
    zs = drum_ring_sectors(zs, cells, drums, outer_of, n_sectors=n_sectors, r_inner=r_inner,
                           aligned=aligned, outer_arc_to_sector=True, core_centre=core_centre)
    return zs, boundary_zones(zs, cells), outer_of


def boundary_zones(zones: EquivalenceZones, cells: NodeCells) -> list:
    """Sorted zones that own a vacuum edge (where boundary factors act)."""
    region = np.asarray(zones.region_map).reshape(-1)
    return sorted(set(region[cells.exposed].tolist()))


def inherit_zones(node_zones: EquivalenceZones, node_cells: NodeCells, fine_centroids,
                  fine_shape, fine_active=None, fine_mix_material=None,
                  absorber_ids=()) -> EquivalenceZones:
    """Carry node zones to a nested fine (reference) mesh by containment.

    Every fine cell takes the zone of the node cell containing its centroid
    (inactive or uncontained cells the void, zone 0). A fine cell's mixed
    component takes the node cell's mix zone when it is one of
    ``absorber_ids`` (the drum absorber; so with absorber-free outer layers
    the reference's arc fraction lands in the same ring sector), its own
    zone otherwise. Exact when every fine cell lies in one node cell, as for
    a raster refined by an integer factor (a multiple of ``2**levels`` for a
    drum-band mesh).
    """
    cen = np.asarray(fine_centroids, dtype=float).reshape(-1, 2)
    act = (np.ones(len(cen), bool) if fine_active is None
           else np.asarray(fine_active, bool).reshape(-1))
    parent = np.full(len(cen), -1)
    parent[act] = locate(cen[act], node_cells)
    if np.any(act & (parent < 0)):
        raise ValueError(f"{int((act & (parent < 0)).sum())} active fine cells lie in no "
                         "node cell: the meshes are not nested")
    nreg = np.asarray(node_zones.region_map).reshape(-1)
    region = np.where(parent >= 0, nreg[np.maximum(parent, 0)], 0)
    region = np.where(act, region, 0)
    mix = None
    if fine_mix_material is not None:
        fm = np.asarray(fine_mix_material).reshape(-1)
        nmix = (nreg if node_zones.mix_region_map is None
                else np.asarray(node_zones.mix_region_map).reshape(-1))
        pm = np.where(parent >= 0, nmix[np.maximum(parent, 0)], -1)
        pm = np.where(pm >= 0, pm, region)
        is_abs = np.isin(fm, list(absorber_ids)) & act
        mix = np.where(is_abs, pm, np.where(fm >= 0, region, -1)).astype(np.int64)
    return replace(node_zones, region_map=region.reshape(fine_shape),
                   mix_region_map=None if mix is None else mix.reshape(fine_shape))


def drum_core_zones(lattice, model, n_sectors: int = 8, r_inner=None,
                    aligned: bool = True):
    """Baseline SPH-DF zones of a :class:`garnet.HexLattice` with control drums.

    model : the lattice's :meth:`~garnet.HexLattice.build` (raster) or
    :meth:`~garnet.HexLattice.build_mesh` (drum band refined). Sites are
    named by their material (``"fuel(0, 1)"``), drums ``"drum(R, C)"``; fuel
    sites are those whose material has fission. Returns (zones, NodeCells,
    boundary zones, drum specs).
    """
    from .hexraster import cell_sites, point_sites
    drums = lattice.drum_specs()
    if hasattr(model, "mesh"):
        cells = NodeCells.from_mesh(model.mesh)
        rows, cols = point_sites(cells.centroid, lattice.pitch)
        cm, mm = model.cell_material, model.mix_material
    else:
        cells = NodeCells.from_raster(model.raster, model.active)
        rows, cols = cell_sites(model.raster, lattice.pitch)
        cm, mm = model.material_map, model.mix_material
    drum_sites = {tuple(d["site"]) for d in drums}
    label = {rc: ("drum" if rc in drum_sites else mat.name) for rc, mat in lattice._sites.items()}
    fuel = [rc for rc, mat in lattice._sites.items()
            if rc not in drum_sites and np.any(np.asarray(mat.nu_sigma_f) > 0)]
    base = site_zones(cells, rows, cols, label, fuel, cell_material=cm, mix_material=mm)
    zones, bz, _ = drum_equivalence_zones(base, cells, drums, n_sectors=n_sectors,
                                          r_inner=r_inner, aligned=aligned)
    return zones, cells, bz, drums


# --------------------------------------------------------------------------
# The SPH-DF node solve
# --------------------------------------------------------------------------
class SphDfNode:
    """Coarse diffusion model for :func:`garnet.sph.sph_bcf_correct`.

    ``node(materials, gamma)`` solves the node model with zone ``materials``
    and boundary discontinuity factors ``gamma`` (B, G) on the vacuum edges
    of ``boundary`` zones, and returns (zone-average flux (R, G), k, boundary
    leakage (B, G)); :meth:`solve_full` returns the cell flux instead of the
    zone average. Warm-started from the previous solve. Build it with
    :meth:`raster` (``TriDiffusionEigenSolver``) or :meth:`mesh`
    (``UnstructuredDiffusionSolver``, e.g. a drum-band refined mesh).
    """

    def __init__(self, kind, zones, boundary, device="cpu", tol=None, **geo):
        self.kind, self.zones, self.boundary = kind, zones, list(boundary)
        self.device = device
        self.tol = dict(tol_k=1e-11, tol_source=1e-10) if tol is None else dict(tol)
        self.geo = geo
        self._last = {"state": None, "k": 1.0}

    @classmethod
    def raster(cls, zones, boundary, grid, active, mix_weight=None, mask_bc="vacuum",
               device="cpu", wielandt_shift=None, tol=None):
        return cls("raster", zones, boundary, device, tol, grid=grid, active=active,
                   mix_weight=mix_weight, mask_bc=mask_bc, wielandt_shift=wielandt_shift)

    @classmethod
    def mesh(cls, zones, boundary, problem, device="cpu", tol=None):
        """``problem``: a :class:`garnet.drum_mesh.TriMeshProblem`."""
        return cls("mesh", zones, boundary, device, tol, problem=problem)

    def _cell_volume(self):
        if self.kind == "raster":
            return self.geo["grid"].cell_volume
        return np.asarray(self.geo["problem"].mesh.area)

    def _mix(self):
        z = self.zones
        if z.mix_region_map is None:
            return {}
        w = self.geo["mix_weight"] if self.kind == "raster" else self.geo["problem"].mix_weight
        return dict(mix_material=z.mix_region_map, mix_weight=w)

    def solve_full(self, materials, gamma=None):
        """(cell flux (G, *shape), k, boundary leakage (B, G))."""
        z = self.zones
        G = materials[0].n_groups
        region = np.asarray(z.region_map)
        gamma = (np.ones((len(self.boundary), G)) if gamma is None
                 else np.asarray(gamma, dtype=float))
        bcf = np.ones((G,) + region.shape)
        for b, reg in enumerate(self.boundary):
            bcf[:, region == reg] = gamma[b][:, None]
        last = self._last
        if self.kind == "raster":
            from .tri import TriDiffusionEigenSolver, tri_boundary_leakage
            geo = self.geo
            shift = ({} if not geo["wielandt_shift"]
                     else dict(wielandt_shift=geo["wielandt_shift"]))
            s = TriDiffusionEigenSolver(geo["grid"], materials, region, active=geo["active"],
                                        mask_bc=geo["mask_bc"], device=self.device,
                                        bcf=bcf, **self._mix())
            r = s.solve(state0=last["state"], k_guess=last["k"], **shift, **self.tol)
            last["state"], last["k"] = [r.flux[g] for g in range(G)], r.k_eff
            flux = r.flux_numpy
            leak = np.array([tri_boundary_leakage(s.ops[g], flux[g], region, z.n_zones,
                                                  geo["active"])[self.boundary]
                             for g in range(G)]).T
            return flux, r.k_eff, leak
        from .mesh import UnstructuredDiffusionSolver
        q = self.geo["problem"]
        mats, cell_mat = q.solver_materials(materials, region, z.mix_region_map)
        s = UnstructuredDiffusionSolver(q.mesh, mats, cell_mat, q.alpha_boundary,
                                        device=self.device, bcf=bcf)
        r = s.solve(flux0=last["state"], k_guess=last["k"], **self.tol)
        flux = asnumpy(r.flux)
        last["state"], last["k"] = flux, r.k_eff
        leak = np.array([s.boundary_leakage(g, flux[g], region, z.n_zones)[self.boundary]
                         for g in range(G)]).T
        return flux, r.k_eff, leak

    def __call__(self, materials, gamma=None):
        from .sph import region_average
        flux, k, leak = self.solve_full(materials, gamma)
        avg = region_average(flux, self.zones.region_map, cell_volume=self._cell_volume(),
                             **self._mix())
        R = self.zones.n_zones
        if avg.shape[0] < R:
            avg = np.vstack([avg, np.zeros((R - avg.shape[0], avg.shape[1]))])
        return avg, k, leak


def reference_boundary_currents(sn_solver, sn_result, fine_zones: EquivalenceZones,
                                boundary) -> np.ndarray:
    """Reference outgoing current through the vacuum boundary of each zone in
    ``boundary``, (B, G), from a converged :class:`garnet.TriSNTransportSolver`
    solve on the levels engine (:meth:`~garnet.TriSNTransportSolver.region_partial_currents`)
    with the zones carried to the reference mesh (:func:`inherit_zones`). The
    target of the boundary factors in :func:`garnet.sph.sph_bcf_correct`."""
    _, bnd = sn_solver.region_partial_currents(sn_result, np.asarray(fine_zones.region_map))
    G = next(iter(bnd.values())).shape[0] if bnd else 0
    return np.array([bnd[b][:, 0] if b in bnd else np.zeros(G) for b in boundary])
