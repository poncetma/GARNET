"""Per-group form functions (Kumar et al. 2026, Eqs. 4-5) and the pin-resolved
HP-MR core they are generated from.

* pin footprints are exact disk areas on any raster, so the reference raster
  and the node raster integrate the flux over the same geometry;
* the reconstruction reproduces the reference pin rates at the generation
  state by construction;
* the pin-resolved core places every pin inside its own assembly zone with
  the right fuel-compact area.
"""

import numpy as np
import pytest

from garnet.benchmarks.hpmr import (HPMR_FUEL_SITES, PITCH, build_hpmr2d,
                                    hpmr_assembly_zones)
from garnet.benchmarks.hpmr_assembly import FUEL as FUEL_COMPACT
from garnet.benchmarks.hpmr_core_pins import (PIN_OFFSET, R_FUEL, build_hpmr2d_pins,
                                              hpmr_pin_layout)
from garnet.pin_power import (footprint_flux, group_form_functions, pin_footprints,
                              reconstruct_group_form_functions, tri_cell_vertices)


def test_cell_vertices_match_the_raster_method():
    p = build_hpmr2d(refine=2, absorber="polar")
    V = tri_cell_vertices(p.raster)
    for a, b, t in ((3, 4, 0), (10, 7, 1), (V.shape[0] - 2, 5, 1)):
        np.testing.assert_allclose(V[a, b, t], p.raster.cell_vertices(a, b, t),
                                   rtol=0, atol=1e-12)


@pytest.mark.parametrize("refine", [1, 3])
def test_footprints_are_exact_disk_areas_inside_their_assembly(refine):
    p = build_hpmr2d(refine=refine, absorber="polar")
    pins = hpmr_pin_layout()
    W = pin_footprints(p.raster, pins.xy[pins.fuel], R_FUEL)
    np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel(), np.pi * R_FUEL ** 2,
                               rtol=1e-12)
    zone = hpmr_assembly_zones(p).region_map.reshape(-1)
    owner = 5 + pins.assembly[pins.fuel]
    rows, cols = W.nonzero()
    np.testing.assert_array_equal(zone[cols], owner[rows])


def test_footprint_flux_of_a_uniform_flux_is_flux_times_area():
    p = build_hpmr2d(refine=2, absorber="polar")
    pins = hpmr_pin_layout()
    W = pin_footprints(p.raster, pins.xy[:50], pins.radius[:50])
    flux = np.stack([np.full(p.material_map.shape, 2.0),
                     np.full(p.material_map.shape, 0.5)])
    got = footprint_flux(W, flux)
    np.testing.assert_allclose(got, np.pi * pins.radius[:50, None] ** 2 * [2.0, 0.5],
                               rtol=1e-12)


def test_form_functions_reproduce_the_reference_at_the_generation_state():
    rng = np.random.default_rng(7)
    n_pins, G, Z = 40, 2, 4
    zone = rng.integers(0, Z, n_pins)
    kappa = rng.uniform(0.01, 0.2, (Z, G))
    pin_flux = rng.uniform(0.5, 2.0, (n_pins, G))
    rate = rng.uniform(0.1, 1.0, (n_pins, G))
    f = group_form_functions(rate, zone, kappa, pin_flux)
    np.testing.assert_allclose(reconstruct_group_form_functions(f, zone, kappa, pin_flux),
                               rate.sum(axis=1), rtol=1e-13)
    # a zero coarse rate gives f = 0 rather than inf
    kappa[0] = 0.0
    f = group_form_functions(rate, zone, kappa, pin_flux)
    assert np.all(np.isfinite(f)) and np.all(f[zone == 0] == 0.0)


def test_pin_resolved_core_puts_the_lattice_in_every_fuel_assembly():
    p, pins = build_hpmr2d_pins(refine=8, data="placeholder", mixing="two")
    assert len(pins.kind) == 127 * len(HPMR_FUEL_SITES)
    assert int(pins.fuel.sum()) == 63 * len(HPMR_FUEL_SITES)
    z = hpmr_assembly_zones(p)
    zone = z.region_map.reshape(-1)
    mm, mx = p.material_map.reshape(-1), p.mix_material.reshape(-1)
    w = p.mix_weight.reshape(-1)
    # only pin materials inside the fuel zones, and none outside
    fuel_cells = np.isin(zone, np.flatnonzero(z.fuel))
    assert np.all(mm[fuel_cells] >= PIN_OFFSET)
    assert not np.any(mm[~fuel_cells] >= PIN_OFFSET)
    # every assembly carries the fuel-compact area of 63 compacts, to the
    # two-constituent representation (exact cells, third constituents folded)
    area = np.sqrt(3.0) / 4.0 * p.raster.side ** 2
    compact = PIN_OFFSET + FUEL_COMPACT
    for zi in np.flatnonzero(z.fuel)[:3]:
        sel = zone == zi
        a = ((mm[sel] == compact) * (1 - w[sel])).sum() + ((mx[sel] == compact) * w[sel]).sum()
        assert a * area == pytest.approx(63 * np.pi * R_FUEL ** 2, rel=0.03)
    # pin mix partners stay in their own assembly zone
    mixed = (mx >= PIN_OFFSET)
    np.testing.assert_array_equal(z.mix_region_map.reshape(-1)[mixed], zone[mixed])
    hex_area = np.sqrt(3.0) / 2.0 * PITCH ** 2
    counts = np.bincount(zone, minlength=len(z.names))[z.fuel]
    np.testing.assert_allclose(counts * area, hex_area, rtol=1e-12)


def test_exact_mixing_keeps_every_pin_layer_at_its_documented_volume():
    # Flat-flux homogenization of one assembly must equal the documented
    # volume mix of the six pin materials in every cross section: that holds
    # only if every constituent, thin shells included, keeps its exact area.
    from garnet import flux_weighted_homogenize
    from garnet.benchmarks.hpmr import _ASSEMBLY_VOLUME_FRACTIONS
    from garnet.benchmarks.hpmr_assembly import PIN_XS_IDS
    from garnet.blend import blend_materials

    p, _ = build_hpmr2d_pins(refine=8, drum_angle_deg=0.0)
    z = hpmr_assembly_zones(p)
    G = p.materials[0].n_groups
    assert G == 11
    hm, _, _ = flux_weighted_homogenize(
        np.ones((G,) + p.material_map.shape), p.materials, p.material_map,
        z.region_map, cell_volume=p.grid.cell_volume, mix_material=p.mix_material,
        mix_weight=p.mix_weight, mix_region_map=z.mix_region_map)
    pins = p.materials[PIN_OFFSET:PIN_OFFSET + 6]
    frac = np.array([_ASSEMBLY_VOLUME_FRACTIONS[i] for i in PIN_XS_IDS])
    want = blend_materials(pins, frac / frac.sum())
    for zi in np.flatnonzero(z.fuel)[:4]:
        for attr in ("sigma_a", "nu_sigma_f", "sigma_t"):
            np.testing.assert_allclose(getattr(hm[zi], attr), getattr(want, attr),
                                       rtol=2e-3)
    # the drum arc stays a two-material blend
    assert np.all(p.mix_material[p.mix_material >= 0] == 5)


def test_blend_materials_follows_the_two_material_rules():
    from garnet.blend import blend_material, blend_materials
    from garnet.benchmarks.hpmr_core_pins import hpmr_pin_core_materials

    _, pins = hpmr_pin_core_materials("g11")
    a = blend_materials([pins[0], pins[1]], [0.7, 0.3])
    b = blend_material(pins[0], pins[1], 0.3)
    np.testing.assert_allclose(a.diffusion, b.diffusion, rtol=1e-14)
    np.testing.assert_allclose(a.sigma_s, b.sigma_s, rtol=1e-14)
    np.testing.assert_allclose(1.0 / a.diffusion,
                               0.7 / pins[0].diffusion + 0.3 / pins[1].diffusion,
                               rtol=1e-14)
    assert blend_materials(pins[:3], [0.0, 1.0, 0.0]) is pins[1]
    with pytest.raises(ValueError, match="fractions"):
        blend_materials(pins[:2], [0.6, 0.6])
