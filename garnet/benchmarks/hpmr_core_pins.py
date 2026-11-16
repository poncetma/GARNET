"""Pin-resolved 2-D HP-MR core: the 127-pin lattice in every fuel assembly.

:func:`garnet.benchmarks.hpmr.build_hpmr2d` models each fuel hex as one
homogenized material; :mod:`garnet.benchmarks.hpmr_assembly` resolves the pins
of a single assembly in an infinite lattice. This module puts the pins of
:func:`~garnet.benchmarks.hpmr_assembly.vtb_pin_lattice` into every fuel site of
the full core, on the core's own triangular raster, so a full-core reference
carries the pin-wise reaction rates that per-group form functions are built
from (Kumar et al., Ann. Nucl. Energy 230 (2026) 112111, Eq. 4) -- the data
the assembly-level SPH workflow needs and a single-assembly lattice cannot
give.

Geometry. Pin centres are the assembly centre (:func:`hex_site_xy`) plus the
VTB lattice offsets, which already carry the deck's 30 degree rotation and so
fit the core's flat-x hexagons. Every triangle of a fuel hex gets the exact
area fraction of each pin layer (closed-form polygon/circle intersection,
:func:`~garnet.benchmarks.hpmr_assembly.pin_layer_fractions`); the graphite
monolith is the remainder. The solvers blend at most two materials per cell,
so a cell keeps its two dominant constituents at their exact fractions and a
third (e.g. a thin shell beside its pin and the monolith) goes to the dominant
one -- the representation of the single-assembly model. The thin shells lose
most of their volume that way (measured at refine 16: moderator shell 0.0089
and heat-pipe shell 0.0143 of the assembly against 0.0227 and 0.0383; fuel
compact 0.3151 against 0.3193), so the core is a self-consistent test case,
not a faithful HP-MR. The drum absorber is
the volume-mixed polar arc of ``absorber="polar"``; drum and fuel cells never
coincide, so the two blends share the ``mix_material``/``mix_weight`` arrays.

Materials. The assembly-level list (``MATERIAL_NAMES`` order, the homogenized
fuel entry kept but unused so the indices match) followed by the six pin
materials at ``PIN_OFFSET + m``. The default pin data are the two-group
placeholders of :mod:`hpmr_assembly`, consistent in group structure with the
core placeholders; they exercise the form-function machinery, they do not
predict k.

    p, pins = build_hpmr2d_pins(refine=16, drum_angle_deg=0.0)
    TriSP3EigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                      mask_bc=p.mask_bc, mix_material=p.mix_material,
                      mix_weight=p.mix_weight).solve()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hexraster import hex_site_xy
from ..tri import TriGrid
from .hpmr import (DRUM_ABSORBER, FUEL, HPMR_FUEL_SITES, HPMR_KINETICS,
                   MATERIAL_NAMES, PITCH, HpmrProblem, _placeholder_materials,
                   absorber_fraction_map, hpmr_raster)
from .hpmr_assembly import (_PIN_LAYERS, GRAPHITE, PIN_MATERIAL_NAMES, R_FUEL,
                            _placeholder_pin_materials, pin_layer_fractions,
                            vtb_pin_lattice)

PIN_OFFSET = len(MATERIAL_NAMES)          # pin material m -> PIN_OFFSET + m
PIN_CORE_MATERIAL_NAMES = MATERIAL_NAMES + PIN_MATERIAL_NAMES


@dataclass
class PinLayout:
    """Pins of the full core.

    xy       : (n_pins, 2) pin centres in the raster frame.
    kind     : "fuel" | "mod" | "hp" per pin.
    assembly : (n_pins,) index of the pin's fuel site in ``HPMR_FUEL_SITES``.
    radius   : (n_pins,) radius of the pin's innermost layer (the fuel
               compact for fuel pins).
    """

    xy: np.ndarray
    kind: tuple
    assembly: np.ndarray
    radius: np.ndarray

    @property
    def fuel(self) -> np.ndarray:
        return np.array([k == "fuel" for k in self.kind])


def hpmr_pin_layout() -> PinLayout:
    """All 30 x 127 pins of the core, assembly by assembly."""
    offsets, kinds = vtb_pin_lattice()
    xy, kind, assembly = [], [], []
    for i, (R, C) in enumerate(HPMR_FUEL_SITES):
        xy.append(np.asarray(hex_site_xy(R, C, PITCH)) + offsets)
        kind.extend(kinds)
        assembly.extend([i] * len(kinds))
    radius = np.array([_PIN_LAYERS[k][0][0] for k in kind])
    return PinLayout(xy=np.concatenate(xy), kind=tuple(kind),
                     assembly=np.asarray(assembly, dtype=np.int64), radius=radius)


def build_hpmr2d_pins(refine: int = 16, drum_angle_deg=0.0,
                      pin_materials: list | None = None,
                      core_materials: list | None = None):
    """Pin-resolved 2-D HP-MR core on the triangular raster.

    refine : triangles per hex = 6 refine^2; the triangle side is
             PITCH / (sqrt3 refine) (0.97 cm at 16, 0.64 cm at 24, 0.48 cm at
             32) against a 1.0 cm fuel-compact radius.

    Returns (problem, pins): an :class:`~garnet.benchmarks.hpmr.HpmrProblem`
    whose ``materials`` are ``PIN_CORE_MATERIAL_NAMES`` and a
    :class:`PinLayout`.
    """
    from scipy.spatial import cKDTree

    from ..pin_power import tri_cell_centroids, tri_cell_vertices

    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float), (12,))
    raster = hpmr_raster(refine, angles, paint_absorber=False)
    core = _placeholder_materials() if core_materials is None else list(core_materials)
    pinm = _placeholder_pin_materials() if pin_materials is None else list(pin_materials)
    if len(core) != len(MATERIAL_NAMES) or len(pinm) != len(PIN_MATERIAL_NAMES):
        raise ValueError("expected the assembly-level and the pin material lists")

    mmap = raster.material_map.copy()
    shape = mmap.shape
    flat = mmap.reshape(-1)
    absorber = absorber_fraction_map(raster, angles).reshape(-1)
    mix_material = np.where(absorber > 0.0, DRUM_ABSORBER, -1).astype(np.int64)
    mix_weight = absorber.copy()

    # exact pin-layer area fractions of every fuel-hex cell
    fuel_cells = np.flatnonzero(flat == FUEL)
    frac = np.zeros((fuel_cells.size, len(PIN_MATERIAL_NAMES)))
    position = {c: i for i, c in enumerate(fuel_cells)}
    cen = tri_cell_centroids(raster).reshape(-1, 2)
    V = tri_cell_vertices(raster).reshape(-1, 3, 2)
    reach = raster.side / np.sqrt(3.0)
    tree = cKDTree(cen[fuel_cells])
    pins = hpmr_pin_layout()
    for xy, kind in zip(pins.xy, pins.kind):
        outer = _PIN_LAYERS[kind][-1][0]
        inner_r, inner_m = _PIN_LAYERS[kind][0]
        for j in tree.query_ball_point(xy, outer + reach):
            cell = fuel_cells[j]
            d = np.hypot(*(cen[cell] - xy))
            parts = ([(inner_m, 1.0)] if d + reach <= inner_r
                     else pin_layer_fractions(V[cell], xy, kind))
            for material, f in parts:
                frac[position[cell], material] += f
    if np.any(frac.sum(axis=1) > 1.0 + 1e-9):
        raise RuntimeError("pin layers overlap in some cell")
    frac[:, GRAPHITE] = 1.0 - frac.sum(axis=1)       # the monolith is the rest
    frac[frac < 1e-12] = 0.0
    order = np.argsort(-frac, axis=1)
    rows = np.arange(fuel_cells.size)
    second_w = frac[rows, order[:, 1]]
    flat[fuel_cells] = PIN_OFFSET + order[:, 0]
    mix_material[fuel_cells] = np.where(second_w > 0.0, PIN_OFFSET + order[:, 1], -1)
    mix_weight[fuel_cells] = np.where(second_w > 0.0, second_w, 0.0)

    mmap = flat.reshape(shape)
    problem = HpmrProblem(grid=TriGrid(shape=shape, side=raster.side),
                          materials=core + pinm, material_map=mmap,
                          active=mmap > 0, mask_bc="vacuum",
                          kinetics=HPMR_KINETICS, drum_angle_deg=np.array(angles),
                          mix_material=mix_material.reshape(shape),
                          mix_weight=mix_weight.reshape(shape), raster=raster)
    return problem, pins


__all__ = ["PIN_OFFSET", "PIN_CORE_MATERIAL_NAMES", "PinLayout",
           "build_hpmr2d_pins", "hpmr_pin_layout", "R_FUEL"]
