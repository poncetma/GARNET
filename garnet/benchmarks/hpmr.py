"""HP-MR heat-pipe microreactor core on a body-fitted triangular mesh (2D
radial slice, or full 3D as extruded triangular prisms).

Assembly-level radial model of the ANL/INL HP-MR reference microreactor (the
NEAMS Virtual Test Bed model; Stauff et al., "High Fidelity Multiphysics
Modeling of a Heat-Pipe Microreactor using BlueCrab", NSE 2024): 2 MWt, 30
hexagonal TRISO fuel assemblies around a central shutdown-rod cell, ringed by
12 beryllium reflector hexes alternating with 12 rotating control drums, all
on a 26.752 cm flat-to-flat lattice. Geometry decoded from the VTB Serpent
model (microreactors/mrad/Serpent_Model/serpent_input.i): fuel fills hex
rings 1-2 plus the twelve ring-3 edge cells; the six ring-3 corner sites
(80.256 cm from centre) and six ring-4 mid-edge sites (92.67 cm) hold the
drums; twelve ring-4 sites hold the Be hexes. Each drum is a beryllium
cylinder of radius 13.25 cm carrying a 1 cm thick B4C absorber layer
(12.25-13.25 cm) spanning a 90 degree arc; rotating the arc toward the core
inserts negative reactivity. Vacuum boundary outside (the surrounding air is
treated as void).

The core is rasterized onto the structured triangular lattice exactly like
the VVER-440 benchmark (6 r^2 triangles per hex at refinement r); the drum
circle and its absorber arc are painted per-triangle inside the drum's hex
cell, so the arc resolution follows the mesh refinement (r >= 4 recommended,
r >= 6 for drum-worth studies).

Cross sections are two-group PLACEHOLDERS with plausible magnitudes for a
graphite/YH2-moderated TRISO core with Be reflector and B4C arcs -- good for
exercising geometry, symmetry and drum-worth behaviour, not for predicting
k_eff. Swap in SPH-corrected constants (e.g. read with garnet.femffusion) via
the ``materials`` argument, ordered as ``MATERIAL_NAMES``.

Reference concept and VTB geometry: Stauff et al., Nuclear Science and
Engineering (2024), doi:10.1080/00295639.2024.2375175.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..backend import asnumpy
from ..hexraster import (TriRaster, cell_sites, hex_site_xy, point_sites,
                         rasterize_hex_sites)
from ..materials import Kinetics, Material
from ..tri import TriGrid

PITCH = 26.752            # cm, assembly flat-to-flat / lattice spacing
DRUM_RADIUS = 13.25       # cm, drum outer radius
DRUM_ABSORBER_INNER = 12.25   # cm, B4C layer inner radius
DRUM_ARC_HALF_DEG = 45.0  # B4C arc spans +-45 deg about its facing direction
#: B4C arc area over the drum hex area. The 90 deg annular arc (12.25-13.25
#: cm) lies wholly inside the hex (13.25 cm < 13.376 cm inradius), so this is
#: the absorber fraction of a drum hex homogenized as one material.
DRUM_ARC_AREA_FRACTION = ((math.pi / 4.0) * (DRUM_RADIUS ** 2 - DRUM_ABSORBER_INNER ** 2)
                          / (math.sqrt(3.0) / 2.0 * PITCH ** 2))
CORE_HEIGHT = 160.0       # cm, fueled length
AXIAL_REFLECTOR_HEIGHT = 20.0  # cm, Be slab above and below the fuel
TOTAL_HEIGHT = CORE_HEIGHT + 2 * AXIAL_REFLECTOR_HEIGHT

# Axial (R, C) hex sites, decoded from the VTB Serpent 13x13 core lattice.
# x = PITCH * (C + R/2), y = PITCH * sqrt(3)/2 * R.
_FUEL_SITES = [
    (-3, 1), (-3, 2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
    (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (3, -2), (3, -1),
]
_BE_SITES = [
    (-4, 1), (-4, 3), (-3, -1), (-3, 4), (-1, -3), (-1, 4),
    (1, -4), (1, 3), (3, -4), (3, 1), (4, -3), (4, -1),
]
# Drum sites: 6 ring-3 corners (outward azimuth 0, 60, ... deg), then 6
# ring-4 mid-edges (30, 90, ... deg). Order fixes the per-drum angle indexing.
_DRUM_SITES = [
    (0, 3), (3, 0), (3, -3), (0, -3), (-3, 0), (-3, 3),
    (2, 2), (4, -2), (2, -4), (-2, -2), (-4, 2), (-2, 4),
]

# Stable public geometry metadata for examples and downstream diagnostics.
# Keep the private lists for compatibility with the internal raster builders,
# but expose immutable tuples so example code does not depend on internals.
HPMR_FUEL_SITES = tuple(_FUEL_SITES)
HPMR_BE_SITES = tuple(_BE_SITES)
HPMR_DRUM_SITES = tuple(_DRUM_SITES)

# material_map indices; keep in sync with _placeholder_materials().
VOID, FUEL, CENTRAL, BE_REFLECTOR, DRUM_BE, DRUM_ABSORBER, AXIAL_REFLECTOR = range(7)
MATERIAL_NAMES = ("void", "fuel", "central", "be_reflector", "drum_be",
                  "drum_absorber")
MATERIAL_NAMES_3D = MATERIAL_NAMES + ("axial_reflector",)

# Graphite-moderated thermal system, one delayed family (placeholder values).
HPMR_KINETICS = Kinetics(velocities=[1.1e7, 3.0e5], beta=[0.0065], decay=[0.078])

# Griffin/YakXs library material IDs for the HP-MR (VTB mrad Serpent model).
_XS_FUEL_COMPACT, _XS_MODERATOR, _XS_GRAPHITE = 801, 802, 803
_XS_BERYLLIUM, _XS_DRUM_BE, _XS_DRUM_B4C = 805, 810, 811
_XS_HEATPIPE, _XS_MOD_SHELL, _XS_HP_SHELL, _XS_CENTRAL = 815, 816, 817, 820

# Fuel-assembly volume fractions, from the Serpent 19x19 pin lattice (63 fuel,
# 27 moderator, 37 heat-pipe pins; pin radii 1.0 / 0.825(+0.92 shell) /
# 0.97(+1.07 shell) cm) over the 26.752 cm assembly hex; graphite monolith is
# the balance. Used to flat-flux-homogenize the pin materials into one
# assembly material (the pre-SPH homogenization).
_ASSEMBLY_VOLUME_FRACTIONS = {
    _XS_FUEL_COMPACT: 0.3193, _XS_MODERATOR: 0.0931, _XS_MOD_SHELL: 0.0227,
    _XS_HEATPIPE: 0.1765, _XS_HP_SHELL: 0.0383, _XS_GRAPHITE: 0.3501,
}


def hpmr_materials_builtin(fuel, three_d=False):
    """Core material list in MATERIAL_NAMES order, from VENDORED cross sections.

    The structural constants (graphite/central cell, Be reflector, drum body,
    B4C arc) are the same real 11-group ENDF/B-8 data
    :func:`hpmr_endfb8_materials` reads, extracted once into
    ``data/hpmr_core_xs_g11.npz`` -- so a full-core run needs no external
    library file. ``fuel`` is the assembly material, normally the FLUX-weighted
    homogenization of a heterogeneous assembly solve
    (:mod:`garnet.benchmarks.hpmr_assembly`) rather than a flat-flux volume mix.
    """
    import os
    path = os.path.join(os.path.dirname(__file__), "data",
                        "hpmr_core_xs_g11.npz")
    d = np.load(path, allow_pickle=False)

    def mat(mid, name):
        n = str(mid)
        chi = d[f"{n}.chi"]
        return Material(name=name, diffusion=d[f"{n}.D"], sigma_a=d[f"{n}.sa"],
                        nu_sigma_f=d[f"{n}.nsf"], sigma_s=d[f"{n}.ss"],
                        chi=chi if chi.sum() > 0 else None,
                        kappa_fission=(d[f"{n}.kf"]
                                       if d[f"{n}.kf"].sum() > 0 else None))

    G = int(d["G"])
    void = Material(name="void", diffusion=[1.0] * G, sigma_a=[0.0] * G,
                    nu_sigma_f=[0.0] * G, sigma_s=np.zeros((G, G)))
    mats = [void, fuel, mat(803, "central"), mat(805, "be_reflector"),
            mat(810, "drum_be"), mat(811, "drum_absorber")]
    if three_d:
        mats.append(mat(805, "axial_reflector"))
    return mats


def hpmr_endfb8_materials(xs_path, grid_index="3 3", three_d=True,
                          sph_fuel=None) -> list:
    """Real multigroup HP-MR materials from a Griffin/YakXs library.

    Reads the VTB HP-MR cross-section library (e.g.
    ``fullcore_xml_G11_endfb8_ss_tr.xml``, 11-group ENDF/B-8, transport
    corrected) and returns the material list in ``MATERIAL_NAMES_3D`` order,
    ready for ``build_hpmr3d(materials=...)`` / ``build_hpmr2d``.

    The fuel assembly is flat-flux-homogenized from its pin constituents (fuel
    compact, YH2 moderator + shell, heat pipe + shell, graphite monolith) via
    :data:`_ASSEMBLY_VOLUME_FRACTIONS`; reflector, drum body, B4C arc and
    central cell are taken from the library directly. This is the *pre-SPH*
    homogenization -- pass ``sph_fuel`` (per-group multipliers from a transport
    reference) to apply a superhomogenization correction to the fuel material.

    grid_index : temperature node of the library to read ("3 3" ~ 800 K, the
                 node nearest the 873 K operating point).
    """
    from ..griffin_xs import read_library, volume_homogenize

    ids = (_XS_FUEL_COMPACT, _XS_MODERATOR, _XS_GRAPHITE, _XS_BERYLLIUM,
           _XS_DRUM_BE, _XS_DRUM_B4C, _XS_HEATPIPE, _XS_MOD_SHELL,
           _XS_HP_SHELL, _XS_CENTRAL)
    lib = read_library(xs_path, ids, grid_index)
    G = lib[_XS_FUEL_COMPACT].n_groups

    void = Material(name="void", diffusion=[1.0] * G, sigma_a=[0.0] * G,
                    nu_sigma_f=[0.0] * G, sigma_s=np.zeros((G, G)))
    fuel = volume_homogenize(lib, _ASSEMBLY_VOLUME_FRACTIONS,
                             chi_from=_XS_FUEL_COMPACT, name="hpmr-fuel-asm",
                             sph_factors=sph_fuel)
    mats = [void, fuel,
            lib[_XS_GRAPHITE],      # central: graphite block, rod withdrawn
            lib[_XS_BERYLLIUM],     # radial Be reflector
            lib[_XS_DRUM_BE],       # drum body (Be)
            lib[_XS_DRUM_B4C],      # B4C absorber arc
            ]
    if three_d:
        mats.append(lib[_XS_BERYLLIUM])   # axial Be reflector
    else:
        mats = mats  # 2D uses the 6-material MATERIAL_NAMES order
    return mats


def _placeholder_materials(three_d: bool = False) -> list:
    """Two-group placeholder set (see module docstring); index = map id."""
    def mat(name, D, sa, nsf=(0.0, 0.0), s12=0.0, chi=None):
        return Material(name=f"hpmr-{name} (placeholder)", diffusion=D,
                        sigma_a=sa, nu_sigma_f=nsf,
                        sigma_s=[[0.0, s12], [0.0, 0.0]],
                        chi=chi if chi is not None else [1.0, 0.0])

    mats = [
        mat("void", [1.0, 1.0], [0.0, 0.0]),
        # TRISO fuel + graphite monolith + YH2 pins + heat pipes, homogenized;
        # nu_sigma_f trimmed so k_eff(drums out) ~ 1.03, the Griffin reference
        mat("fuel", [1.4, 0.9], [0.010, 0.100], nsf=[0.005, 0.1265], s12=0.025),
        # central shutdown-rod cell, rod withdrawn: graphite + SS tube + air
        mat("central", [1.1, 0.7], [0.0004, 0.004], s12=0.030),
        # beryllium reflector hex / drum body
        mat("be_reflector", [0.60, 0.45], [0.0006, 0.0012], s12=0.055),
        mat("drum_be", [0.60, 0.45], [0.0006, 0.0012], s12=0.055),
        # B4C absorber arc (strong thermal absorber; diffusion-level values)
        mat("drum_absorber", [0.60, 0.30], [0.10, 2.5], s12=0.008),
    ]
    if three_d:
        # axial Be slab; a separate entry so measured data can differ from
        # the radial ring
        mats.append(mat("axial_reflector", [0.60, 0.45], [0.0006, 0.0012],
                        s12=0.055))
    return mats


def hpmr_placeholder_materials(three_d: bool = False) -> list:
    """Return the illustrative HP-MR material set used by simple examples.

    The constants exercise geometry and solver behavior; they are not a
    predictive reactor library. Prefer :func:`hpmr_materials_builtin` for the
    vendored 11-group benchmark data.
    """
    return _placeholder_materials(three_d=three_d)


def _drum_geometry(drum_angle_deg):
    """(drum centres, per-drum absorber-arc centre azimuths) at these angles.

    Convention: angle 0 points each drum's B4C arc squarely at the core centre
    (fully inserted), 180 squarely away (fully withdrawn). atan2(y, x) is the
    drum's outward radial azimuth, so the arc centre is that direction rotated by
    180 + angle -- i.e. toward the centre at angle 0, back outward at angle 180.
    Every drum is measured from its own radial line, so a given angle is the same
    physical insertion for all 12 drums.
    """
    drum_xy = [hex_site_xy(R, C, PITCH) for R, C in _DRUM_SITES]
    arc_az = [math.atan2(y, x) + math.radians(180.0 + a)   # 0 = toward centre
              for (x, y), a in zip(drum_xy, drum_angle_deg)]
    return drum_xy, arc_az


def hpmr_raster(refine: int, drum_angle_deg, paint_absorber: bool = True) -> TriRaster:
    """Rasterize the radial core (see :mod:`garnet.hexraster`).

    drum_angle_deg : per-drum absorber rotation (12,); 0 = arc facing the core
    centre (inserted), 180 = facing radially outward (withdrawn).
    paint_absorber : if True (default) the B4C arc is stamped by centroid
    (staircase); if False the drum cells stay drum-body Be, for the polar
    volume-mixing path (see :func:`absorber_fraction_map`).
    """
    site_mat = {(0, 0): CENTRAL}
    site_mat.update({s: FUEL for s in _FUEL_SITES})
    site_mat.update({s: BE_REFLECTOR for s in _BE_SITES})
    site_mat.update({s: DRUM_BE for s in _DRUM_SITES})
    if not paint_absorber:
        return rasterize_hex_sites(site_mat, PITCH, refine)

    drum_index = {s: d for d, s in enumerate(_DRUM_SITES)}
    drum_xy, arc_az = _drum_geometry(drum_angle_deg)
    arc_half = math.radians(DRUM_ARC_HALF_DEG)

    def paint(cx, cy, site, mid):
        """B4C where the centroid falls inside a drum's absorber arc."""
        if mid != DRUM_BE:
            return mid
        d = drum_index[site]
        dx, dy = cx - drum_xy[d][0], cy - drum_xy[d][1]
        rr = math.hypot(dx, dy)
        # rr > DRUM_RADIUS: hex-corner slivers, kept as drum body.
        if DRUM_ABSORBER_INNER < rr <= DRUM_RADIUS:
            dphi = (math.atan2(dy, dx) - arc_az[d] + math.pi) \
                   % (2 * math.pi) - math.pi
            if abs(dphi) <= arc_half:
                return DRUM_ABSORBER
        return mid

    return rasterize_hex_sites(site_mat, PITCH, refine, paint=paint)


def absorber_fraction_map(raster: TriRaster, drum_angle_deg, samples: int = 10):
    """Per-cell B4C area fraction of each drum's absorber arc.

    The absorber is a polar region -- an annular sector r in
    [DRUM_ABSORBER_INNER, DRUM_RADIUS], azimuth within +-DRUM_ARC_HALF_DEG of
    the (rotated) arc centre. For every triangle within reach of a drum, the
    fraction of the cell area covered by that region is computed exactly
    (closed-form triangle / annular-sector intersection, see
    :func:`garnet.model.annular_sector_fraction`), so the total absorber area is
    the analytic sector area at every refinement. Unlike the centroid raster it
    is non-zero for cells the thin (1 cm) annulus only partly crosses -- so the
    arc is represented (diluted) even when it is thinner than a triangle, and
    the fraction varies smoothly as the drum rotates. Feed the result as
    ``mix_weight`` with ``mix_material`` = DRUM_ABSORBER to volume-mix the
    absorber into the drum-body cells.

    samples : retained for compatibility and ignored. It was the order of a
    barycentric sub-sampling lattice whose vertex/edge points biased the
    absorber area by up to -17%.
    """
    from ..model import annular_sector_fraction

    mmap = raster.material_map
    ni, nj, _ = mmap.shape
    frac = np.zeros((ni, nj, 2), dtype=float)
    drum_xy, arc_az = _drum_geometry(drum_angle_deg)
    drum_xy = np.asarray(drum_xy)
    arc_az = np.asarray(arc_az)
    arc_half = math.radians(DRUM_ARC_HALF_DEG)
    reach2 = (DRUM_RADIUS + raster.side) ** 2

    for a in range(ni):
        for b in range(nj):
            for t in (0, 1):
                if mmap[a, b, t] == 0:
                    continue
                V = raster.cell_vertices(a, b, t)          # (3, 2)
                cx, cy = V.mean(0)
                d2 = (drum_xy[:, 0] - cx) ** 2 + (drum_xy[:, 1] - cy) ** 2
                d = int(d2.argmin())
                if d2[d] > reach2:
                    continue
                frac[a, b, t] = annular_sector_fraction(
                    V, drum_xy[d], DRUM_ABSORBER_INNER, DRUM_RADIUS,
                    arc_az[d], arc_half)
    return frac


def smeared_drum_material(materials, fraction: float = DRUM_ARC_AREA_FRACTION,
                          name: str = "drum_smeared"):
    """One material for a whole drum hex: the B4C arc smeared into the drum body.

    Flat-flux (area-weighted) mix of ``materials[DRUM_ABSORBER]`` at area
    ``fraction`` into ``materials[DRUM_BE]``. Total, absorption, scattering and
    production are averaged and D = 1/(3 Sigma_t), so every cross section --
    the transport one included -- is linear in ``fraction``. The arc's rotation
    is invisible at this level: this is the hex-homogenized drum.
    """
    from ..griffin_xs import volume_homogenize

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must lie in [0, 1], got {fraction}")
    parts = {DRUM_BE: materials[DRUM_BE], DRUM_ABSORBER: materials[DRUM_ABSORBER]}
    return volume_homogenize(parts, {DRUM_BE: 1.0 - fraction, DRUM_ABSORBER: fraction},
                             name=name)


def _smear_drums(materials):
    """Copy of ``materials`` with the drum body replaced by the smeared drum."""
    mats = list(materials)
    mats[DRUM_BE] = smeared_drum_material(mats)
    return mats


@dataclass
class HpmrMeshProblem:
    """A 2-D HP-MR on an unstructured (locally refined) triangular mesh.

    cell_material is the base material of each cell; with the volume-mixed
    absorber, drum cells carry ``mix_material = DRUM_ABSORBER`` and the exact
    arc area fraction ``mix_weight`` (else -1 / 0). The unstructured solver
    takes one material per cell, so :meth:`solver_materials` expands every
    mixed cell into its own blended material by the raster solvers' rules.
    """

    mesh: object
    cell_material: np.ndarray
    materials: list
    alpha_boundary: float
    mix_material: np.ndarray | None
    mix_weight: np.ndarray | None
    drum_angle_deg: np.ndarray
    refine: int
    levels: int

    def solver_materials(self, materials=None, cell_material=None, mix_material=None):
        """(materials, cell_material) for UnstructuredDiffusionSolver; pass zone
        materials and zone maps to solve a homogenized model on this mesh."""
        from ..blend import per_cell_blended_materials
        mats = self.materials if materials is None else list(materials)
        cm = self.cell_material if cell_material is None else cell_material
        mm = self.mix_material if mix_material is None else mix_material
        if mm is None:
            return mats, np.asarray(cm)
        return per_cell_blended_materials(mats, cm, mm, self.mix_weight)


def _split_triangle(V, levels):
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


def hpmr_locally_refined_problem(refine: int = 3, drum_angle_deg=0.0, levels: int = 1,
                                 band_margin: float = 1.5, absorber: str = "polar",
                                 materials=None) -> HpmrMeshProblem:
    """Locally refined 2-D HP-MR: the drum absorber band split ``levels`` times.

    Coarse triangular lattice (``refine``) everywhere; each coarse cell whose
    centroid lies in the annular band the B4C arc sweeps (radius in
    [DRUM_ABSORBER_INNER - band_margin, DRUM_RADIUS + band_margin] of a drum)
    is split into 4**levels sub-triangles, so the drums are resolved at
    refine * 2**levels for any rotation. Coarse-to-fine interfaces are hanging
    nodes, handled conservatively by :func:`garnet.mesh.assemble_mesh`.

    absorber : "polar" (default) volume-mixes the exact arc area fraction into
        every drum cell (:func:`absorber_fraction_map`'s geometry), smooth in
        drum angle; "centroid" paints a cell B4C when its centroid is in the
        arc (the staircase of :func:`hpmr_locally_refined_mesh`).
    """
    from ..mesh import assemble_mesh
    from ..model import annular_sector_fraction

    if absorber not in ("polar", "centroid"):
        raise ValueError(f"absorber must be 'polar' or 'centroid', got {absorber!r}")
    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float),
                             (len(_DRUM_SITES),))
    mats = _placeholder_materials() if materials is None else list(materials)
    raster = hpmr_raster(refine, angles, paint_absorber=False)   # drums = DRUM_BE
    mm = raster.material_map
    drum_xy, arc_az = _drum_geometry(angles)
    drum_xy = np.asarray(drum_xy)
    arc_half = math.radians(DRUM_ARC_HALF_DEG)
    lo, hi = DRUM_ABSORBER_INNER - band_margin, DRUM_RADIUS + band_margin

    def nearest_drum(cc):
        d2 = (drum_xy[:, 0] - cc[0]) ** 2 + (drum_xy[:, 1] - cc[1]) ** 2
        d = int(d2.argmin())
        return d, math.sqrt(d2[d])

    def in_arc(cc):
        d, rr = nearest_drum(cc)
        if DRUM_ABSORBER_INNER < rr <= DRUM_RADIUS:
            dphi = (math.atan2(cc[1] - drum_xy[d, 1], cc[0] - drum_xy[d, 0])
                    - arc_az[d] + math.pi) % (2 * math.pi) - math.pi
            return abs(dphi) <= arc_half
        return False

    node_at, coords = {}, []

    def node(p):
        k = (round(float(p[0]), 6), round(float(p[1]), 6))
        i = node_at.get(k)
        if i is None:
            i = len(coords); node_at[k] = i; coords.append([float(p[0]), float(p[1])])
        return i

    cells, cmat, mix_w = [], [], []
    ni, nj, _ = mm.shape
    for a in range(ni):
        for b in range(nj):
            for t in (0, 1):
                if mm[a, b, t] == 0:
                    continue
                V = raster.cell_vertices(a, b, t)
                base = int(mm[a, b, t])
                rmin = nearest_drum(V.mean(0))[1]
                split = levels if lo <= rmin <= hi else 0
                for tri in _split_triangle(V, split):
                    cells.append(tuple(node(p) for p in tri))
                    cc = np.mean(tri, 0)
                    if base != DRUM_BE:
                        cmat.append(base); mix_w.append(0.0)
                    elif absorber == "centroid":
                        cmat.append(DRUM_ABSORBER if in_arc(cc) else DRUM_BE)
                        mix_w.append(0.0)
                    else:
                        d, _ = nearest_drum(cc)
                        cmat.append(DRUM_BE)
                        mix_w.append(annular_sector_fraction(
                            np.asarray(tri), drum_xy[d], DRUM_ABSORBER_INNER,
                            DRUM_RADIUS, arc_az[d], arc_half))

    mesh = assemble_mesh(np.array(coords), cells, cmat)
    cmat = np.array(cmat, dtype=np.int64)
    mix_material = mix_weight = None
    if absorber == "polar":
        mix_weight = np.array(mix_w)
        mix_material = np.where(mix_weight > 0.0, DRUM_ABSORBER, -1).astype(np.int64)
    return HpmrMeshProblem(mesh=mesh, cell_material=cmat, materials=mats,
                           alpha_boundary=0.5, mix_material=mix_material,
                           mix_weight=mix_weight, drum_angle_deg=np.array(angles),
                           refine=int(refine), levels=int(levels))


def hpmr_locally_refined_mesh(refine: int = 3, drum_angle_deg=0.0,
                              refine_drums: bool = True, band_margin: float = 1.5,
                              materials=None):
    """Locally-refined 2D HP-MR triangular finite-volume mesh.

    Coarse triangular lattice everywhere; each coarse cell whose centroid lies
    in the annular band the drum B4C arc occupies (radius in
    [DRUM_ABSORBER_INNER - band_margin, DRUM_RADIUS + band_margin] of any drum)
    is split one level into four sub-triangles, resolving the absorber directly
    at the fine scale -- no volume mixing needed there -- at a fraction of the
    cells a globally fine mesh would cost. The full radial band is refined (all
    azimuths), so the same mesh serves any drum rotation. The coarse-to-fine
    interface is a 2:1 hanging node, handled conservatively by
    :func:`garnet.mesh.assemble_mesh`.

    Returns ``(mesh, cell_material, materials, alpha_boundary)`` for
    :class:`garnet.mesh.UnstructuredDiffusionSolver`. With ``refine_drums=False``
    the mesh is uniform (the coarse-baseline / global-fine reference). This is
    :func:`hpmr_locally_refined_problem` with one level and the centroid
    (staircase) absorber; that function also offers more levels and the
    volume-mixed absorber.
    """
    p = hpmr_locally_refined_problem(refine, drum_angle_deg,
                                     levels=1 if refine_drums else 0,
                                     band_margin=band_margin, absorber="centroid",
                                     materials=materials)
    return p.mesh, p.cell_material, p.materials, p.alpha_boundary


@dataclass
class HpmrProblem:
    grid: TriGrid
    materials: list
    material_map: np.ndarray
    active: np.ndarray
    mask_bc: object
    bc: object = "reflective"   # z faces only (2D grids ignore it)
    kinetics: object = None
    drum_angle_deg: np.ndarray = None
    mix_material: np.ndarray = None   # polar absorber volume-mixing (optional)
    mix_weight: np.ndarray = None
    # The rasterizer's physical frame (side + lattice origin). Needed to place
    # anything given in CORE coordinates onto the mesh -- e.g. pin positions for
    # pin-power reconstruction (garnet.pin_power), which must land in the same
    # frame the raster used or the sampled flux is taken from the wrong cells.
    raster: object = None


def build_hpmr2d(refine: int = 4, drum_angle_deg=0.0,
                 materials: list | None = None,
                 absorber: str = "raster", samples: int = 10) -> HpmrProblem:
    """Assemble the 2D HP-MR core on a body-fitted triangular mesh.

    refine         : triangles per hex = 6 refine^2; also sets the drum-arc
                     rasterization fidelity (>= 4 recommended for "raster").
    drum_angle_deg : absorber-arc rotation, scalar or one value per drum (12).
                     0 = arc toward the core centre (inserted), 180 = outward
                     (withdrawn).
    materials      : optional replacement list ordered as MATERIAL_NAMES
                     (e.g. SPH-corrected sets read via garnet.femffusion).
    absorber       : "raster" (centroid staircase, one material per cell),
                     "polar" (exact polar area fraction volume-mixed into the
                     drum cells; sets ``mix_material``/``mix_weight`` on the
                     problem -- pass them to the solver), or "smeared" (every
                     drum hex is one material, :func:`smeared_drum_material`,
                     stored in the DRUM_BE slot; the drum angle has no effect).
                     The polar path represents the arc's area and rotation
                     smoothly and works below the raster's refine>=4 floor;
                     "smeared" is the fully hex-homogenized core.
    samples        : ignored (the polar area fractions are exact); kept for
                     compatibility.

    Use with TriDiffusionEigenSolver, e.g.::

        p = build_hpmr2d(refine=6, drum_angle_deg=120.0, absorber="polar")
        res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                      active=p.active, mask_bc=p.mask_bc,
                                      mix_material=p.mix_material,
                                      mix_weight=p.mix_weight).solve()
    """
    if absorber == "raster" and refine < 4:
        raise ValueError("refine >= 4 required for absorber='raster': below "
                         "that the 1 cm B4C annulus is thinner than a triangle "
                         "and no absorber is rasterized (use absorber='polar', "
                         "which volume-mixes the arc at any refinement)")
    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float),
                             (len(_DRUM_SITES),))
    mats = _placeholder_materials() if materials is None else list(materials)
    if len(mats) != len(MATERIAL_NAMES):
        raise ValueError(f"expected {len(MATERIAL_NAMES)} materials "
                         f"({', '.join(MATERIAL_NAMES)}), got {len(mats)}")

    mix_material = mix_weight = None
    if absorber == "raster":
        raster = hpmr_raster(refine, angles)
    elif absorber == "polar":
        raster = hpmr_raster(refine, angles, paint_absorber=False)
        frac = absorber_fraction_map(raster, angles, samples=samples)
        mix_weight = frac
        mix_material = np.where(frac > 0.0, DRUM_ABSORBER, -1).astype(np.int64)
    elif absorber == "smeared":
        raster = hpmr_raster(refine, angles, paint_absorber=False)
        mats = _smear_drums(mats)
    else:
        raise ValueError("absorber must be 'raster', 'polar' or 'smeared', "
                         f"got {absorber!r}")

    mmap, side = raster.material_map, raster.side
    return HpmrProblem(grid=TriGrid(shape=mmap.shape, side=side),
                       materials=mats, material_map=mmap, active=mmap > 0,
                       mask_bc="vacuum", kinetics=HPMR_KINETICS,
                       drum_angle_deg=np.array(angles),
                       mix_material=mix_material, mix_weight=mix_weight,
                       raster=raster)


@dataclass
class HpmrZones:
    """Homogenization / SPH zones of a 2-D HP-MR raster.

    region_map     : (nr, nc, 2) zone index of every cell.
    mix_region_map : (nr, nc, 2) zone of the volume-mixed absorber partner
                     (polar absorber), -1 elsewhere; None for a raster
                     without mixing. Pass it as ``mix_region_map`` to
                     :func:`garnet.sph.flux_weighted_homogenize`, and as the
                     solver's ``mix_material`` when the zone materials are the
                     solver's materials list.
    names          : zone names.
    material       : (Z,) base material index of each zone.
    fuel           : (Z,) True for the fuel-assembly zones.
    site           : (Z,) hex site (R, C) of each fuel zone, None otherwise.
    """

    region_map: np.ndarray
    mix_region_map: np.ndarray | None
    names: tuple
    material: np.ndarray
    fuel: np.ndarray
    site: tuple


def _zones_from_sites(rows, cols, mmap, mix_material, merge_drum):
    """Zone map from per-cell hex sites and base materials (see
    :func:`hpmr_assembly_zones`)."""
    mmap = np.asarray(mmap)
    if merge_drum:
        others = (VOID, CENTRAL, BE_REFLECTOR, DRUM_BE)
        names = ("void", "central", "be_reflector", "drum")
    else:
        others = (VOID, CENTRAL, BE_REFLECTOR, DRUM_BE, DRUM_ABSORBER)
        names = tuple(MATERIAL_NAMES[m] for m in others)
    zone_of_material = np.full(max(int(mmap.max()) + 1, len(MATERIAL_NAMES)), -1,
                               dtype=np.int64)
    zone_of_material[list(others)] = np.arange(len(others))
    if merge_drum:
        zone_of_material[DRUM_ABSORBER] = zone_of_material[DRUM_BE]
    region = zone_of_material[mmap]
    # site -> fuel zone, vectorized through an integer key per site
    keys = np.array([(R + 1000) * 4096 + (C + 1000) for R, C in _FUEL_SITES])
    order = np.argsort(keys)
    cell_key = (np.asarray(rows) + 1000) * 4096 + (np.asarray(cols) + 1000)
    pos = np.clip(np.searchsorted(keys[order], cell_key), 0, len(keys) - 1)
    hit = keys[order][pos] == cell_key
    fuel_cells = hit & (mmap != VOID)
    region[fuel_cells] = len(others) + order[pos[fuel_cells]]
    if np.any(region < 0):
        raise ValueError("cells outside the fuel sites carry a material with no "
                         "zone; mesh and site list are inconsistent")
    mix_region = None
    if mix_material is not None:
        mm = np.asarray(mix_material)
        mix_region = np.where(mm == DRUM_ABSORBER, zone_of_material[DRUM_ABSORBER],
                              np.where(mm >= 0, region, -1)).astype(np.int64)
    names = names + tuple(f"fuel{s}" for s in _FUEL_SITES)
    material = np.array(list(others) + [FUEL] * len(_FUEL_SITES), dtype=np.int64)
    return HpmrZones(region_map=region, mix_region_map=mix_region, names=names,
                     material=material, fuel=material == FUEL,
                     site=tuple([None] * len(others) + list(_FUEL_SITES)))


def hpmr_assembly_zones(problem: HpmrProblem, merge_drum: bool = False) -> HpmrZones:
    """One zone per fuel assembly and one per other material.

    Zones 0-4 are the void, central block, Be reflector, drum body and drum
    absorber (material order without the fuel); zones 5.. are the fuel
    assemblies in :data:`HPMR_FUEL_SITES` order. A fuel zone is every cell of
    its hexagon whatever its material, so a pin-resolved core
    (:mod:`garnet.benchmarks.hpmr_core_pins`) gets the same zones, and its
    volume-mixed pin constituents stay in their assembly's zone. The numbering
    depends only on the core layout, not on ``refine``, so a reference raster
    and a coarser node raster of the same core share it: this is the zone map
    of the assembly-level SPH workflow (one equivalence zone per fuel assembly,
    as in Kumar et al., Ann. Nucl. Energy 230 (2026) 112111).

    merge_drum : one "drum" zone for drum body and absorber together (zones
        0-3, fuel from 4): the hex-homogenized drum, whose flux-weighted
        constants a ``absorber="smeared"`` node raster uses (Kumar et al.'s
        homogenized-drum variant).
    """
    if problem.raster is None or np.asarray(problem.material_map).ndim != 3:
        raise ValueError("hpmr_assembly_zones needs a 2-D problem from build_hpmr2d")
    rows, cols = cell_sites(problem.raster, PITCH)
    return _zones_from_sites(rows, cols, problem.material_map, problem.mix_material,
                             merge_drum)


def hpmr_mesh_zones(problem, merge_drum: bool = False) -> HpmrZones:
    """:func:`hpmr_assembly_zones` for an unstructured HP-MR mesh
    (:func:`hpmr_locally_refined_problem`), same numbering; arrays are per cell
    (n_cells,)."""
    rows, cols = point_sites(np.asarray(problem.mesh.centroid)[:, :2], PITCH)
    return _zones_from_sites(rows, cols, problem.cell_material, problem.mix_material,
                             merge_drum)


def hpmr_transport_mask(problem: HpmrProblem, region: str = "drum") -> np.ndarray:
    """Cells that keep the full transport (SP3/SDPN) block in a hybrid solve;
    every other cell runs pure diffusion. Pass the result as ``hybrid_mask`` to
    a Tri SP3/SDPN eigen-solver.

    region
        "drum" (default): the whole rotating drum body -- the beryllium plus
            its B4C arc. Running transport over the entire drum gives a Be
            buffer around the strong absorber, so the reflective hybrid
            interface at the drum/reflector boundary sees an already-decayed
            second moment; this is the recommended, closure-insensitive choice.
        "absorber": only the B4C arc cells (raster: material == DRUM_ABSORBER;
            polar: the volume-mixed cells, mix_weight > 0). The tightest
            transport region -- fewest transport cells, but the reflective
            interface then sits right at the absorber edge.
    """
    mmap = np.asarray(problem.material_map)
    if region == "drum":
        mask = (mmap == DRUM_BE) | (mmap == DRUM_ABSORBER)
    elif region == "absorber":
        if problem.mix_weight is not None:         # polar: arc is volume-mixed
            mask = np.asarray(problem.mix_weight) > 0.0
        else:                                      # raster: arc is its own id
            mask = mmap == DRUM_ABSORBER
    else:
        raise ValueError(f"region must be 'drum' or 'absorber', got {region!r}")
    return mask & (mmap > 0)                        # never mark excised void cells


def build_hpmr3d(refine: int = 4, nz: int = 20, drum_angle_deg=0.0,
                 materials: list | None = None,
                 absorber: str = "raster", samples: int = 10) -> HpmrProblem:
    """Assemble the 3D HP-MR: the 2D radial core extruded to 200 cm.

    The fueled length is 160 cm with 20 cm beryllium axial reflectors above
    and below (fuel and central cells become AXIAL_REFLECTOR there); the
    radial Be ring and the drums -- including their B4C arcs -- run the full
    height, as in the VTB Serpent model. Vacuum on both z faces.

    refine, drum_angle_deg, absorber, samples : as in :func:`build_hpmr2d`.
    Because the drums (arc included) run the full height, the ``"polar"``
    absorber volume-mixing is the same in every axial layer, so the 2D
    ``mix_material``/``mix_weight`` are simply extruded over z.
    nz        : number of uniform axial layers over the 200 cm height; must be
                a multiple of 10 so layer boundaries fall on the 20/180 cm
                core-reflector interfaces (nz=20 -> dz=10 cm).
    materials : optional replacement list ordered as MATERIAL_NAMES_3D.

    Use with TriDiffusionEigenSolver, passing ``bc=p.bc`` for the vacuum z
    faces (and the mix arrays for absorber="polar")::

        p = build_hpmr3d(refine=4, nz=20, absorber="polar")
        res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                      active=p.active, mask_bc=p.mask_bc,
                                      bc=p.bc, mix_material=p.mix_material,
                                      mix_weight=p.mix_weight).solve()
    """
    if nz % 10 != 0:
        raise ValueError("nz must be a multiple of 10 so layer boundaries "
                         "align with the 20 cm axial reflectors")
    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float),
                             (len(_DRUM_SITES),))
    mats = _placeholder_materials(three_d=True) if materials is None \
        else list(materials)
    if len(mats) != len(MATERIAL_NAMES_3D):
        raise ValueError(f"expected {len(MATERIAL_NAMES_3D)} materials "
                         f"({', '.join(MATERIAL_NAMES_3D)}), got {len(mats)}")
    if absorber == "smeared":
        mats = _smear_drums(mats)
    p2d = build_hpmr2d(refine=refine, drum_angle_deg=angles,
                       absorber=absorber, samples=samples)

    def extrude(a):
        return None if a is None else np.repeat(a[..., None], nz, axis=3)

    mmap = extrude(p2d.material_map)
    mix_material = extrude(p2d.mix_material)
    mix_weight = extrude(p2d.mix_weight)

    dz = TOTAL_HEIGHT / nz
    zc = (np.arange(nz) + 0.5) * dz
    refl = (zc < AXIAL_REFLECTOR_HEIGHT) | (zc > TOTAL_HEIGHT - AXIAL_REFLECTOR_HEIGHT)
    fuelish = (mmap == FUEL) | (mmap == CENTRAL)
    mmap[fuelish & refl[None, None, None, :]] = AXIAL_REFLECTOR
    grid = TriGrid(shape=mmap.shape, side=p2d.grid.side, height=TOTAL_HEIGHT)
    return HpmrProblem(grid=grid, materials=mats, material_map=mmap,
                       active=mmap > 0, mask_bc="vacuum",
                       bc=("reflective", "reflective", "vacuum"),
                       kinetics=HPMR_KINETICS, drum_angle_deg=np.array(angles),
                       mix_material=mix_material, mix_weight=mix_weight,
                       # The extrusion keeps the 2D raster's physical frame, so
                       # anything given in core coordinates (pin positions,
                       # coupling-mesh vertices) can be placed in 3D too.
                       raster=p2d.raster)


def hpmr_sn_homogenization(refine: int = 6, n_polar: int = 2, n_azi: int = 12,
                           scheme: str = "scb", device: str = "auto",
                           tol_k: float = 1e-7, tol_source: float = 1e-6,
                           materials=None):
    """Transport-weighted homogenization of the HP-MR assembly: one tri-S_N solve
    of the pin-resolved geometry, collapsed two ways.

    The assembly constants for :func:`build_hpmr2d` are normally taken from a
    *diffusion* solve of the pin lattice. The pin cell is precisely where
    diffusion is least trustworthy, and it shows: S_N gives k_inf = 1.177297
    against diffusion's 1.186813 at refine 6, a ~950 pcm difference that the flux
    shape carries into the homogenized cross sections too. This routes the
    homogenization through a transport reference instead, for ~91 s of CPU.

      ``assembly``       1 material, flux-weighted over the whole assembly. Feed
                         to :func:`hpmr_materials_builtin` for a core built on
                         transport-weighted constants.
      ``pin_materials``  the same flux collapsed per *pin type* instead -- 4
                         materials (graphite, fuel_compact, moderator_pin,
                         heatpipe_pin), each clad sharing its pin's region so the
                         homogenization performs the clad smearing. Diagnostic
                         here, and the natural starting point for a per-pin SPH
                         correction.

    Note this is plain flux weighting, NOT SPH: reaction rates are exact only at
    the reference flux, and a coarse solve has a different one. See
    :func:`~garnet.sph.sph_correct` for the correction that closes that gap
    (``method="jfnk"`` -- Picard runs away).

    Returns a dict with keys: k_inf, pin_materials, assembly, region_flux,
    region_volume, problem, flux.
    """
    from ..sph import flux_weighted_homogenize
    from ..tri_sn import TriSNTransportSolver
    from .hpmr_assembly import build_hpmr_assembly2d

    asm = build_hpmr_assembly2d(refine=refine, materials=materials)
    r = TriSNTransportSolver(
        asm.grid, asm.materials, asm.material_map, bc="periodic", scheme=scheme,
        device=device, n_polar=n_polar, n_azi=n_azi,
        mix_material=asm.mix_material, mix_weight=asm.mix_weight,
    ).solve(tol_k=tol_k, tol_source=tol_source)
    if not r.converged:
        raise RuntimeError(f"S_N assembly reference did not converge: {r}")
    flux = asnumpy(r.flux)

    # Pin types, not materials: each clad shares its pin's region, which is what
    # turns the homogenization into the smearing we want.
    #   0 graphite | 1 fuel | 2 moderator+clad | 3 heat pipe+wall
    region_of = np.array([0, 1, 2, 2, 3, 3])
    mm = np.asarray(asm.mix_material)
    kw = dict(cell_volume=asm.grid.cell_volume, mix_material=asm.mix_material,
              mix_weight=asm.mix_weight)
    pins, rflux, rvol = flux_weighted_homogenize(
        flux, asm.materials, asm.material_map, region_of[asm.material_map],
        mix_region_map=region_of[np.clip(mm, 0, None)], **kw)
    whole, _, _ = flux_weighted_homogenize(
        flux, asm.materials, asm.material_map,
        np.zeros(asm.material_map.shape, dtype=np.int64),
        mix_region_map=np.zeros(mm.shape, dtype=np.int64), **kw)

    return dict(k_inf=float(r.k_eff), pin_materials=pins, assembly=whole[0],
                region_flux=rflux, region_volume=rvol, problem=asm, flux=flux)
