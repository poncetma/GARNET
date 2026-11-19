"""Control-drum band refinement: a hex-lattice raster made unstructured.

A control drum carries a thin absorber arc (HP-MR: 1 cm of B4C on a 13.25 cm
drum) that a raster coarse enough for the rest of the core cannot resolve.
:func:`refine_drum_band` keeps the coarse triangular raster everywhere and
splits every cell in the annular band the arc sweeps into ``4**levels``
sub-triangles, for all rotations at once, so the drums are resolved at
``refine * 2**levels``. Coarse-to-fine interfaces are hanging nodes, handled
conservatively by :func:`garnet.mesh.assemble_mesh`, and the result is solved
by :class:`garnet.mesh.UnstructuredDiffusionSolver`.

The arc is volume-mixed at its exact area fraction in every cell it crosses
(``absorber="polar"``, smooth in drum angle), or painted by cell centroid
(``"centroid"``, a staircase kept for comparison). The sub-triangles nest in
the coarse raster and in every raster refined by an integer factor that is a
multiple of ``2**levels``, so equivalence zones defined on this mesh carry
over exactly to such a reference (:func:`garnet.equivalence.inherit_zones`).

:meth:`garnet.HexLattice.build_mesh` builds one from a lattice;
:func:`garnet.benchmarks.hpmr.hpmr_locally_refined_problem` is the HP-MR case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_VACUUM_ALPHA = 0.5


@dataclass
class TriMeshProblem:
    """A 2-D core on an unstructured triangular mesh with volume-mixed drums.

    mesh          : :class:`garnet.mesh.Mesh` (triangles).
    cell_material : (n_cells,) base material index of every cell.
    materials     : material list (index 0 the void, as on the rasters).
    alpha_boundary: Robin coefficient of the outer boundary (0.5 = vacuum).
    mix_material  : (n_cells,) volume-mixed absorber material, -1 where none;
                    None without mixing.
    mix_weight    : (n_cells,) absorber area fraction, 0 where none.
    refine, levels: coarse raster refinement and drum-band split levels.
    parent        : (n_cells,) flat index of the coarse raster cell each mesh
                    cell came from.
    drums         : the drum specs the mesh was built for (centre, radii,
                    arc direction and half-angle, absorber material).
    """

    mesh: object
    cell_material: np.ndarray
    materials: list
    alpha_boundary: float
    mix_material: np.ndarray | None
    mix_weight: np.ndarray | None
    refine: int
    levels: int
    parent: np.ndarray = None
    drums: list = field(default_factory=list)

    def solver_materials(self, materials=None, cell_material=None, mix_material=None):
        """(materials, cell_material) for UnstructuredDiffusionSolver; pass zone
        materials and zone maps to solve a homogenized model on this mesh.
        Every mixed cell gets its own blended material by the raster solvers'
        rules (linear cross sections, harmonic D)."""
        from .blend import per_cell_blended_materials
        mats = self.materials if materials is None else list(materials)
        cm = self.cell_material if cell_material is None else cell_material
        mm = self.mix_material if mix_material is None else mix_material
        if mm is None:
            return mats, np.asarray(cm)
        return per_cell_blended_materials(mats, cm, mm, self.mix_weight)


def split_triangle(V, levels):
    """Midpoint subdivision, ``levels`` times: 4**levels sub-triangles."""
    tris = [tuple(np.asarray(v, dtype=float) for v in V)]
    for _ in range(levels):
        nxt = []
        for T in tris:
            M = [(T[i] + T[(i + 1) % 3]) / 2 for i in range(3)]
            nxt += [(T[0], M[0], M[2]), (M[0], T[1], M[1]),
                    (M[2], M[1], T[2]), (M[0], M[1], M[2])]
        tris = nxt
    return tris


def refine_drum_band(raster, materials, drums, levels: int = 1, band_margin: float = 1.5,
                     absorber: str = "polar", refine: int | None = None,
                     drum_cell=None) -> TriMeshProblem:
    """Split the drum absorber band of a hex-lattice raster ``levels`` times.

    raster    : :class:`garnet.hexraster.TriRaster` with the drum bodies
                painted (no absorber).
    materials : material list the raster indexes (0 = void).
    drums     : list of dicts with ``centre`` (x, y) cm, ``inner`` and
                ``outer`` absorber radii, ``arc_az`` (direction of the arc
                centre, rad), ``arc_half`` (half-angle, rad) and
                ``absorber_id`` (material index).
    levels    : sub-triangles per band cell are ``4**levels``.
    band_margin : the band is [inner - margin, outer + margin] from the
                nearest drum centre, by coarse-cell centroid.
    absorber  : "polar" (exact area fraction volume-mixed into every cell the
                arc crosses) or "centroid" (cell painted absorber when its
                centroid is in the arc).
    drum_cell : optional callable(base material) -> bool restricting the
                absorber to cells of the drum body material (default: any
                cell the arc touches).
    """
    from .mesh import assemble_mesh
    from .model import annular_sector_fraction

    if absorber not in ("polar", "centroid"):
        raise ValueError(f"absorber must be 'polar' or 'centroid', got {absorber!r}")
    if not drums:
        raise ValueError("refine_drum_band needs at least one drum")
    mm = raster.material_map
    xy = np.array([d["centre"] for d in drums], dtype=float)
    reach = max(d["outer"] for d in drums) + band_margin

    def nearest(cc):
        d2 = (xy[:, 0] - cc[0]) ** 2 + (xy[:, 1] - cc[1]) ** 2
        d = int(d2.argmin())
        return d, math.sqrt(d2[d])

    def in_arc(cc, d):
        dm = drums[d]
        rr = math.hypot(cc[0] - xy[d, 0], cc[1] - xy[d, 1])
        if dm["inner"] < rr <= dm["outer"]:
            dphi = (math.atan2(cc[1] - xy[d, 1], cc[0] - xy[d, 0])
                    - dm["arc_az"] + math.pi) % (2 * math.pi) - math.pi
            return abs(dphi) <= dm["arc_half"]
        return False

    node_at, coords = {}, []

    def node(p):
        k = (round(float(p[0]), 6), round(float(p[1]), 6))
        i = node_at.get(k)
        if i is None:
            i = len(coords)
            node_at[k] = i
            coords.append([float(p[0]), float(p[1])])
        return i

    cells, cmat, mix_w, mix_m, parent = [], [], [], [], []
    ni, nj, _ = mm.shape
    for a in range(ni):
        for b in range(nj):
            for t in (0, 1):
                base = int(mm[a, b, t])
                if base == 0:
                    continue
                V = raster.cell_vertices(a, b, t)
                d0, r0 = nearest(V.mean(0))
                dm0 = drums[d0]
                split = levels if dm0["inner"] - band_margin <= r0 <= dm0["outer"] + band_margin else 0
                for tri in split_triangle(V, split):
                    cells.append(tuple(node(p) for p in tri))
                    parent.append((a * nj + b) * 2 + t)
                    cc = np.mean(tri, 0)
                    d, rr = nearest(cc)
                    eligible = rr <= reach and (drum_cell is None or drum_cell(base))
                    if absorber == "centroid":
                        cmat.append(drums[d]["absorber_id"] if eligible and in_arc(cc, d) else base)
                        mix_w.append(0.0)
                        mix_m.append(-1)
                        continue
                    cmat.append(base)
                    f = (annular_sector_fraction(np.asarray(tri), xy[d], drums[d]["inner"],
                                                 drums[d]["outer"], drums[d]["arc_az"],
                                                 drums[d]["arc_half"]) if eligible else 0.0)
                    mix_w.append(f)
                    mix_m.append(drums[d]["absorber_id"] if f > 0.0 else -1)

    mesh = assemble_mesh(np.array(coords), cells, cmat)
    mix_material = mix_weight = None
    if absorber == "polar":
        mix_weight = np.array(mix_w)
        mix_material = np.array(mix_m, dtype=np.int64)
    return TriMeshProblem(mesh=mesh, cell_material=np.array(cmat, dtype=np.int64),
                          materials=list(materials), alpha_boundary=_VACUUM_ALPHA,
                          mix_material=mix_material, mix_weight=mix_weight,
                          refine=int(refine) if refine is not None else -1,
                          levels=int(levels), parent=np.array(parent, dtype=np.int64),
                          drums=[dict(d) for d in drums])
