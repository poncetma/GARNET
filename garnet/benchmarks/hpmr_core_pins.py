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
monolith is the remainder. By default (``mixing="exact"``) every cell with
more than one constituent gets its own flat-flux blend of all of them
(:func:`garnet.blend.blend_materials`: linear cross sections, harmonic D), so
every layer keeps its exact area; the solvers' per-cell fields, the S_N
solver's included, are built from the materials list and take any number of
them. ``mixing="two"`` keeps the solvers' two-material blend instead: the two
dominant constituents at their exact fractions, a third folded into the
dominant one, which loses most of the thin shells (measured at refine 16:
moderator shell 0.0089 and heat-pipe shell 0.0143 of the assembly against
0.0227 and 0.0383). The drum absorber is the volume-mixed polar arc of
``absorber="polar"`` in ``mix_material``/``mix_weight``.

Materials. The assembly-level list (``MATERIAL_NAMES`` order, the homogenized
fuel entry kept but unused so the indices match), the six pin materials at
``PIN_OFFSET + m``, then the per-cell blends. The default data are the
vendored 11-group ENDF/B-8 constants (:func:`hpmr_pin_core_materials`);
``data="placeholder"`` gives the two-group stand-ins used for the first
form-function study.

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


def hpmr_pin_core_materials(data: str = "g11", transport_correction: str | None = None,
                            absorption: str = "net"):
    """(core, pin) material lists for :func:`build_hpmr2d_pins`.

    absorption : "net" (default) replaces the library absorption with
        Sigma_t - sum_g' Sigma_s0[g -> g'] (:func:`garnet.materials.
        net_absorption`), which keeps the Be (n,2n) gain that the library
        absorption loses in every GARNET solver: the library's P0 matrix counts
        (n,2n) twice and its total once, so in Be's fastest group
        Sigma_a + sum P0 exceeds Sigma_t by 0.0146 /cm (2.4 times Sigma_a).
        This is the balance a transport code given Sigma_t and the P0 matrix
        solves (Griffin). With the library absorption, refine-4 diffusion is
        141 pcm low with the drums in and 732 pcm low with them out.
        "library" keeps the stored absorption (the convention of every result
        before 2026-11-16).

    transport_correction : None (default) keeps the library data as they are:
        the true total cross section plus the Legendre scattering moments
        P0..P5, for an anisotropic S_N solve (``scattering_order`` of
        :class:`~garnet.tri_sn.TriSNTransportSolver`). An isotropic solver
        given these data scatters isotropically with the uncorrected total,
        which is wrong for this core (the YH moderator's P1 moment reaches
        56 % of its total). "outscatter" or "diagonal" return
        :func:`garnet.materials.transport_corrected` data for diffusion, SPN
        and isotropic S_N.

    data="g11": the vendored 11-group ENDF/B-8 constants (VTB Griffin library,
    ~800 K) for the pins (:func:`~garnet.benchmarks.hpmr_assembly.
    pin_materials_builtin`) and for the central cell, reflector, drum body and
    B4C arc (:func:`~garnet.benchmarks.hpmr.hpmr_materials_builtin`); the unused
    homogenized-fuel slot holds the flat-flux mix of the pins at the documented
    volume fractions. data="placeholder": the two-group stand-ins.
    """
    from ..blend import blend_materials
    from .hpmr import _ASSEMBLY_VOLUME_FRACTIONS, hpmr_materials_builtin
    from .hpmr_assembly import PIN_XS_IDS, pin_materials_builtin

    if absorption not in ("net", "library"):
        raise ValueError(f"absorption must be 'net' or 'library', got {absorption!r}")
    if data == "placeholder":
        if transport_correction is not None:
            raise ValueError("the placeholder data carry no scattering moments")
        return _placeholder_materials(), _placeholder_pin_materials()
    if data != "g11":
        raise ValueError(f"data must be 'g11' or 'placeholder', got {data!r}")
    pins = pin_materials_builtin()
    frac = np.array([_ASSEMBLY_VOLUME_FRACTIONS[i] for i in PIN_XS_IDS])
    fuel = blend_materials(pins, frac / frac.sum(), name="hpmr-fuel-flat-mix")
    core = hpmr_materials_builtin(fuel)
    if absorption == "net":
        from ..materials import net_absorption
        # the void keeps its zero data; the flat-mix fuel slot is linear in
        # its constituents' absorption, so it is corrected in place too
        core = [m if m.name == "void" else net_absorption(m) for m in core]
        pins = [net_absorption(m) for m in pins]
    if transport_correction is not None:
        import warnings

        from ..materials import transport_corrected
        with warnings.catch_warnings():
            # the out-scatter correction clips the YH moderator's group-8
            # within-group scattering at zero; documented, not news per call
            warnings.simplefilter("ignore", RuntimeWarning)
            core = [transport_corrected(m, transport_correction)
                    if m.scattering_order >= 1 else m for m in core]
            pins = [transport_corrected(m, transport_correction) for m in pins]
    return core, pins


def build_hpmr2d_pins(refine: int = 16, drum_angle_deg=0.0,
                      pin_materials: list | None = None,
                      core_materials: list | None = None,
                      data: str = "g11", mixing: str = "exact",
                      transport_correction: str | None = None,
                      absorption: str = "net"):
    """Pin-resolved 2-D HP-MR core on the triangular raster.

    refine : triangles per hex = 6 refine^2; the triangle side is
             PITCH / (sqrt3 refine) (0.97 cm at 16, 0.64 cm at 24, 0.48 cm at
             32) against a 1.0 cm fuel-compact radius.
    data   : "g11" (default) or "placeholder", see :func:`hpmr_pin_core_materials`;
             ignored for the lists passed explicitly.
    transport_correction : see :func:`hpmr_pin_core_materials`; None (default)
             gives the raw data with scattering moments for anisotropic S_N.
    absorption : "net" (default, keeps the Be (n,2n) gain) or "library"; see
             :func:`hpmr_pin_core_materials`.
    mixing : "exact" (default): every fuel-hex cell with more than one
             constituent gets its own flat-flux blend of all of them
             (:func:`garnet.blend.blend_materials`), so every pin layer keeps
             its exact area. "two": the two dominant constituents through
             ``mix_material``/``mix_weight``, the rest folded into the dominant
             one (thin shells lose most of their volume).

    Returns (problem, pins): an :class:`~garnet.benchmarks.hpmr.HpmrProblem`
    whose ``materials`` are ``PIN_CORE_MATERIAL_NAMES`` and a
    :class:`PinLayout`.
    """
    from scipy.spatial import cKDTree

    from ..pin_power import tri_cell_centroids, tri_cell_vertices

    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float), (12,))
    raster = hpmr_raster(refine, angles, paint_absorber=False)
    if mixing not in ("exact", "two"):
        raise ValueError(f"mixing must be 'exact' or 'two', got {mixing!r}")
    default_core, default_pins = (hpmr_pin_core_materials(data, transport_correction,
                                                          absorption)
                                  if core_materials is None or pin_materials is None
                                  else (None, None))
    core = default_core if core_materials is None else list(core_materials)
    pinm = default_pins if pin_materials is None else list(pin_materials)
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
    materials = core + pinm
    if mixing == "two":
        flat[fuel_cells] = PIN_OFFSET + order[:, 0]
        mix_material[fuel_cells] = np.where(second_w > 0.0, PIN_OFFSET + order[:, 1], -1)
        mix_weight[fuel_cells] = np.where(second_w > 0.0, second_w, 0.0)
    else:
        from ..blend import blend_materials
        pure = second_w <= 0.0
        flat[fuel_cells[pure]] = PIN_OFFSET + order[pure, 0]
        blends = {}
        for i in np.flatnonzero(~pure):
            key = tuple(np.round(frac[i], 12))
            index = blends.get(key)
            if index is None:
                index = blends[key] = len(materials)
                materials.append(blend_materials(pinm, frac[i] / frac[i].sum(),
                                                 name=f"pin-blend-{index}"))
            flat[fuel_cells[i]] = index
        mix_material[fuel_cells] = -1
        mix_weight[fuel_cells] = 0.0

    mmap = flat.reshape(shape)
    problem = HpmrProblem(grid=TriGrid(shape=shape, side=raster.side),
                          materials=materials, material_map=mmap,
                          active=mmap > 0, mask_bc="vacuum",
                          kinetics=HPMR_KINETICS, drum_angle_deg=np.array(angles),
                          mix_material=mix_material.reshape(shape),
                          mix_weight=mix_weight.reshape(shape), raster=raster)
    return problem, pins


__all__ = ["PIN_OFFSET", "PIN_CORE_MATERIAL_NAMES", "PinLayout",
           "build_hpmr2d_pins", "hpmr_pin_core_materials", "hpmr_pin_layout",
           "R_FUEL"]
