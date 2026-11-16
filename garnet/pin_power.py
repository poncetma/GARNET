"""Pin-power reconstruction: homogeneous core flux x heterogeneous form function.

A full-core diffusion solve on assembly-homogenized constants knows the power
of each assembly but nothing about the 127 pins inside it -- homogenization
discarded that shape, and equivalence factoring (SPH, GET) does not bring it
back, because it corrects assembly-integrated reaction rates rather than the
distribution within a region. Reconstruction supplies the missing shape from a
separate heterogeneous calculation:

    P_pin  =  S(x_pin)  *  f(pin)

``f`` is the intra-assembly FORM FUNCTION, from a single-assembly heterogeneous
S_N solve in an infinite lattice (:mod:`garnet.benchmarks.hpmr_assembly`),
normalized to mean 1 over the fuel pins. ``S`` is the homogeneous power shape
from the core solve, evaluated AT the pin's position rather than averaged over
the assembly -- which is the point of "spanning the assembly": across a core
with drums and a reflector the coarse flux tilts substantially from one side of
an assembly to the other, and a single per-assembly scalar throws that tilt
away. The two factors are separable to the extent that the local spectrum
resembles the infinite-lattice one; that assumption weakens next to a
reflector, a drum, or a strong flux gradient, which is exactly where the
reconstruction error concentrates.

The form function is generated once per assembly TYPE and per state (drum angle
changes it), then applied everywhere that type appears.
"""

from __future__ import annotations

from .backend import asnumpy
from .power import power_density
import numpy as np



def tri_cell_centroids(raster):
    """(nr, nc, 2, 2) Cartesian centroids of a rasterized core.

    Vectorizes ``TriRaster.cell_centroid`` exactly -- same basis
    ``av = side (sqrt3/2, 1/2)``, ``bv = side (0, 1)`` and the same
    ``(i0, j0)`` lattice origin, with the down triangle's centroid at 1/3 of
    the rhombus diagonal and the up triangle's at 2/3. It must be the raster's
    own frame: a generic tri-lattice convention differs by a rotation and an
    offset, which would silently sample the flux from the wrong cells.
    """
    mmap = np.asarray(raster.material_map)
    nr, nc = mmap.shape[0], mmap.shape[1]
    h = raster.side
    av = np.array([h * np.sqrt(3.0) / 2.0, h * 0.5])
    bv = np.array([0.0, h])
    i0, j0 = raster.origin
    aa, bb = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    O = ((i0 + aa - 1)[..., None] * bv + (j0 + bb - 1)[..., None] * av)
    cen = np.empty((nr, nc, 2, 2))
    cen[:, :, 0, :] = O + (av + bv) / 3.0
    cen[:, :, 1, :] = O + 2.0 * (av + bv) / 3.0
    return cen


def _layer_density(dens, raster, layer):
    """Select the (nr, nc, 2[, nz]) power density the centroids index.

    A 2-D density is returned as (n_cells,); a 3-D prismatic one as
    (n_cells, nz) for ``layer=None``, (n_cells,) for an integer layer or the
    axial sum for ``layer="integrate"``. Any other shape is rejected rather
    than flattened against the 2-D centroid indices.
    """
    nr, nc = np.asarray(raster.material_map).shape[:2]
    if dens.shape[:3] != (nr, nc, 2) or dens.ndim not in (3, 4):
        raise ValueError(f"flux cell shape {dens.shape} matches neither the 2-D "
                         f"raster ({nr}, {nc}, 2) nor its 3-D extrusion "
                         f"({nr}, {nc}, 2, nz)")
    if dens.ndim == 3:
        if layer is not None:
            raise ValueError("layer applies only to a 3-D (prismatic) flux")
        return dens.reshape(-1)
    nz = dens.shape[3]
    if layer is None:
        return dens.reshape(-1, nz)
    if isinstance(layer, str):
        if layer != "integrate":
            raise ValueError(f"layer must be None, an int or 'integrate', got {layer!r}")
        # Uniform extrusion: the layer sum is the axial integral / dz.
        return dens.sum(axis=3).reshape(-1)
    if isinstance(layer, bool) or int(layer) != layer or not -nz <= layer < nz:
        raise ValueError(f"layer must be an integer in [0, {nz}), got {layer!r}")
    return dens[..., int(layer)].reshape(-1)


def sample_power_shape(raster, flux, materials, material_map, points,
                       mix_material=None, mix_weight=None, active=None,
                       layer=None):
    """Homogeneous fission-power density sampled at arbitrary points.

    Nearest-cell lookup on the triangular mesh. Returns one value per point;
    points falling on an inactive cell return 0. This is ``S`` above -- the
    smooth shape the reconstruction rides on, and it varies WITHIN an assembly
    because the core mesh resolves sub-assembly detail.

    A 3-D (prismatic, ``(G, nr, nc, 2, nz)``) flux is sampled in-plane:
    ``layer=None`` returns ``(n_points, nz)`` (one radial shape per axial
    layer), an integer selects that layer, and ``"integrate"`` sums the
    layers (the axially integrated shape on the uniform extrusion). For a 2-D
    flux ``layer`` must be None.
    """
    # kappa*Sigma_f, not nu*Sigma_f -- see garnet.power.fission_energy_xs. Power
    # lives only in the fuel (both vanish elsewhere); the difference is the
    # group weighting. Unnormalized: this is a shape, and it is the caller's
    # form-function normalization that fixes the scale.
    dens = asnumpy(power_density(flux, materials, material_map,
                                 mix_material=mix_material, mix_weight=mix_weight,
                                 active=active))
    val = _layer_density(np.asarray(dens), raster, layer)

    cen = tri_cell_centroids(raster).reshape(-1, 2)
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(cen).query(np.asarray(points))
    except ImportError:                                    # pragma: no cover
        d = np.linalg.norm(np.asarray(points)[:, None, :] - cen[None, :, :], axis=2)
        idx = d.argmin(axis=1)
    return val[idx]


def reconstruct_pin_powers(raster, flux, materials, material_map, sites,
                           pin_centres, form, pin_kind=None,
                           mix_material=None, mix_weight=None, active=None,
                           layer=None):
    """Reconstruct pin powers across a whole core.

    sites       : (n_sites, 2) xy of each assembly centre to reconstruct.
    pin_centres : (n_pins, 2) pin positions relative to an assembly centre.
    form        : (n_pins,) intra-assembly form function, mean 1 over fuel pins.
    layer       : 3-D flux only, as in :func:`sample_power_shape`; None gives
                  per-layer pin powers with a trailing ``nz`` axis.

    Returns (power, xy) with shape (n_sites, n_pins) -- or
    (n_sites, n_pins, nz) for a 3-D flux with ``layer=None`` -- and
    (n_sites, n_pins, 2) -- absolute pin positions, so the result can be
    mapped or reduced without recomputing geometry.

    The core flux is sampled at every pin's own position, so an assembly
    sitting in a flux gradient gets a tilted reconstruction rather than a
    uniform scaling of the form function.
    """
    sites = np.asarray(sites, dtype=float)
    pin_centres = np.asarray(pin_centres, dtype=float)
    form = np.asarray(form, dtype=float)
    xy = sites[:, None, :] + pin_centres[None, :, :]
    flat = xy.reshape(-1, 2)
    S = sample_power_shape(raster, flux, materials, material_map, flat,
                           mix_material=mix_material, mix_weight=mix_weight,
                           active=active, layer=layer)
    S = S.reshape((len(sites), len(pin_centres)) + S.shape[1:])
    extra = (None,) * (S.ndim - 2)
    power = S * form[(None, slice(None)) + extra]
    if pin_kind is not None:
        fuel = np.array([k == "fuel" for k in pin_kind])
        power = np.where(fuel[(None, slice(None)) + extra], power, 0.0)
    return power, xy


def peaking(power, pin_kind=None):
    """Pin peaking factor: max pin power over the mean across fuel pins."""
    p = np.asarray(power, dtype=float)
    if pin_kind is not None:
        fuel = np.array([k == "fuel" for k in pin_kind])
        vals = p[:, fuel]
    else:
        vals = p[p > 0].reshape(1, -1) if p.ndim == 2 else p
    m = vals[vals > 0].mean()
    return float(vals.max() / m), float(vals.max()), float(m)


# --- full-core, per-group form functions (Kumar et al. 2026) -----------------
#
# The reconstruction above multiplies a single-assembly, group-collapsed form
# function by the coarse power density at the pin centre. Griffin's assembly-
# level workflow (S. Kumar et al., "Transient multiphysics simulations with pin
# power reconstruction in the Griffin reactor physics code", Ann. Nucl. Energy
# 230 (2026) 112111, Eqs. 4-5) instead builds the form function per pin and per
# group from a full-core reference and the SPH-corrected coarse solution of the
# same state:
#
#     f_{p,g} = R^ref_{p,g} / (kappa~Sigma_f,{z(p),g} Phi_{p,g})
#     P_p     = sum_g f_{p,g} kappa~Sigma_f,{z(p),g} Phi_{p,g}
#
# where Phi_{p,g} is the coarse flux integrated over the pin's footprint and
# z(p) the pin's equivalence zone. Because the denominator comes from the coarse
# solve itself, f also absorbs the coarse model's intra-zone shape error, and
# the generation state is reproduced exactly.

def tri_cell_vertices(raster):
    """(nr, nc, 2, 3, 2) Cartesian vertices of every cell, in the raster's own
    frame (the vectorized ``TriRaster.cell_vertices``)."""
    mmap = np.asarray(raster.material_map)
    nr, nc = mmap.shape[0], mmap.shape[1]
    h = raster.side
    av = np.array([h * np.sqrt(3.0) / 2.0, h * 0.5])
    bv = np.array([0.0, h])
    i0, j0 = raster.origin
    aa, bb = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    O = ((i0 + aa - 1)[..., None] * bv + (j0 + bb - 1)[..., None] * av)
    V = np.empty((nr, nc, 2, 3, 2))
    V[:, :, 0] = O[:, :, None, :] + np.array([np.zeros(2), av, bv])
    V[:, :, 1] = O[:, :, None, :] + np.array([av, bv, av + bv])
    return V


def pin_footprints(raster, pin_xy, radius):
    """Exact area of each pin's disk in each cell, as a sparse (n_pins, n_cells)
    CSR matrix (cells in ``material_map.reshape(-1)`` order).

    pin_xy : (n_pins, 2) absolute pin centres in the raster frame.
    radius : scalar or (n_pins,) disk radius (e.g. the fuel-compact radius).

    Each row sums to pi r^2 for a disk inside the meshed region. The areas
    come from the closed-form polygon/circle intersection
    (:func:`garnet.model.annular_sector_fraction`), so a pin smaller than a
    cell still gets its exact share of every cell it touches.
    """
    V = tri_cell_vertices(raster).reshape(-1, 3, 2)
    return cell_footprints(V, pin_xy, radius)


def mesh_pin_footprints(mesh, pin_xy, radius):
    """:func:`pin_footprints` on a triangular unstructured mesh
    (:class:`garnet.mesh.Mesh`), cells in mesh order."""
    coords = np.asarray(mesh.coords, dtype=float)[:, :2]
    V = coords[np.asarray(mesh.cells, dtype=np.int64)]
    if V.shape[1] != 3:
        raise ValueError("mesh_pin_footprints needs a triangle mesh")
    return cell_footprints(V, pin_xy, radius)


def cell_footprints(vertices, pin_xy, radius):
    """Pin-disk areas in triangles given by their vertices, (n_cells, 3, 2)."""
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    from .model import annular_sector_fraction

    V = np.asarray(vertices, dtype=float)
    cen = V.mean(axis=1)
    e = V - cen[:, None, :]
    reach = np.sqrt((e ** 2).sum(axis=2)).max(axis=1)        # centroid-to-vertex
    area = 0.5 * np.abs((V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1])
                        - (V[:, 2, 0] - V[:, 0, 0]) * (V[:, 1, 1] - V[:, 0, 1]))
    pin_xy = np.asarray(pin_xy, dtype=float).reshape(-1, 2)
    radius = np.broadcast_to(np.asarray(radius, dtype=float), (len(pin_xy),))
    tree = cKDTree(cen)
    rmax = float(reach.max())
    rows, cols, vals = [], [], []
    for p, (xy, r) in enumerate(zip(pin_xy, radius)):
        for c in tree.query_ball_point(xy, r + rmax):
            d = np.hypot(*(cen[c] - xy))
            if d - reach[c] >= r:
                continue
            frac = (1.0 if d + reach[c] <= r else
                    annular_sector_fraction(V[c], xy, 0.0, r, 0.0, np.pi))
            if frac > 0.0:
                rows.append(p)
                cols.append(c)
                vals.append(frac * area[c])
    return csr_matrix((vals, (rows, cols)), shape=(len(pin_xy), len(cen)))


def footprint_flux(footprints, flux):
    """Flux integrated over each pin footprint, (n_pins, G): sum_c A_pc phi_gc."""
    f = np.asarray(asnumpy(flux), dtype=float)
    G = f.shape[0]
    return np.asarray(footprints @ f.reshape(G, -1).T)


def group_form_functions(reference_pin_rate, zone_of_pin, kappa_fission, pin_flux):
    """Per-pin, per-group form functions, Kumar et al. 2026 Eq. (4).

    reference_pin_rate : (n_pins, G) reference kappa*Sigma_f*phi integrated over
                         each pin, from the full-core reference of the state.
    zone_of_pin        : (n_pins,) equivalence zone of each pin.
    kappa_fission      : (Z, G) kappa*Sigma_f of the zone materials the coarse
                         solve used (SPH-corrected).
    pin_flux           : (n_pins, G) coarse flux integrated over each pin
                         footprint (:func:`footprint_flux`), same state.

    Returns f with shape (n_pins, G); 0 where the coarse rate vanishes.
    """
    rate = np.asarray(reference_pin_rate, dtype=float)
    denom = np.asarray(kappa_fission, dtype=float)[np.asarray(zone_of_pin)] \
        * np.asarray(pin_flux, dtype=float)
    return np.divide(rate, denom, out=np.zeros_like(rate), where=denom > 0.0)


def reconstruct_group_form_functions(form, zone_of_pin, kappa_fission, pin_flux):
    """Pin powers from per-group form functions, Kumar et al. 2026 Eq. (5):
    P_p = sum_g f_{p,g} kappa~Sigma_f,{z(p),g} Phi_{p,g}, shape (n_pins,)."""
    k = np.asarray(kappa_fission, dtype=float)[np.asarray(zone_of_pin)]
    return (np.asarray(form) * k * np.asarray(pin_flux)).sum(axis=1)
