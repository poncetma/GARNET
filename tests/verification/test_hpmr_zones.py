"""Assembly zones of the 2-D HP-MR raster (SPH workflow (b)).

The zone map must put every cell in the assembly it was painted from, give each
fuel assembly exactly its 6 refine^2 triangles, and number the assemblies the
same way on every refinement, so factors generated on one raster index the
zones of another.
"""

import numpy as np
import pytest

from garnet import flux_weighted_homogenize
from garnet.benchmarks.hpmr import (BE_REFLECTOR, CENTRAL, DRUM_ABSORBER, DRUM_BE,
                                    FUEL, HPMR_BE_SITES, HPMR_DRUM_SITES,
                                    HPMR_FUEL_SITES, PITCH, build_hpmr2d,
                                    hpmr_assembly_zones)
from garnet.hexraster import cell_sites


@pytest.mark.parametrize("refine", [1, 2, 4])
def test_cell_sites_recover_the_painted_site_of_every_cell(refine):
    p = build_hpmr2d(refine=refine, absorber="polar")
    rows, cols = cell_sites(p.raster, PITCH)
    mmap = p.material_map
    expected = {FUEL: set(HPMR_FUEL_SITES), CENTRAL: {(0, 0)},
                BE_REFLECTOR: set(HPMR_BE_SITES), DRUM_BE: set(HPMR_DRUM_SITES)}
    for material, sites in expected.items():
        cells = mmap == material
        got = set(zip(rows[cells].tolist(), cols[cells].tolist()))
        assert got == sites, material


def test_raster_absorber_cells_sit_in_drum_sites():
    p = build_hpmr2d(refine=4, absorber="raster")
    rows, cols = cell_sites(p.raster, PITCH)
    cells = p.material_map == DRUM_ABSORBER
    assert cells.any()
    assert set(zip(rows[cells].tolist(), cols[cells].tolist())) <= set(HPMR_DRUM_SITES)


@pytest.mark.parametrize("refine", [1, 2, 4])
def test_each_fuel_zone_is_one_whole_assembly(refine):
    p = build_hpmr2d(refine=refine, absorber="polar")
    z = hpmr_assembly_zones(p)
    counts = np.bincount(z.region_map.ravel(), minlength=len(z.names))
    assert len(z.names) == 5 + len(HPMR_FUEL_SITES)
    np.testing.assert_array_equal(counts[z.fuel], 6 * refine ** 2)
    assert z.site[5:] == HPMR_FUEL_SITES
    # the polar absorber has no base cells: its zone exists through mixing only
    assert counts[z.names.index("drum_absorber")] == 0
    assert (z.mix_region_map >= 0).sum() == (p.mix_weight > 0).sum()


def test_zone_volumes_agree_between_rasters():
    # Exact hexagon areas for the assemblies and exact polar areas for the
    # absorber: a reference raster and a node raster have the same zone
    # volumes, which the SPH comparison of zone-average fluxes relies on.
    volumes = []
    for refine in (2, 4):
        p = build_hpmr2d(refine=refine, drum_angle_deg=30.0, absorber="polar")
        z = hpmr_assembly_zones(p)
        _, _, vol = flux_weighted_homogenize(
            np.ones((2,) + p.material_map.shape), p.materials, p.material_map,
            z.region_map, cell_volume=p.grid.cell_volume, mix_material=p.mix_material,
            mix_weight=p.mix_weight, mix_region_map=z.mix_region_map)
        volumes.append(vol)
    names = hpmr_assembly_zones(build_hpmr2d(refine=1, absorber="polar")).names
    keep = [i for i, n in enumerate(names) if n != "void"]
    np.testing.assert_allclose(volumes[0][keep], volumes[1][keep], rtol=1e-10)
    hex_area = np.sqrt(3.0) / 2.0 * PITCH ** 2
    np.testing.assert_allclose(volumes[1][5:], hex_area, rtol=1e-12)


def test_per_hex_zones_split_the_outer_ring_and_keep_the_fuel():
    # One zone per Be reflector hex and per drum body and arc (Laboure et al.
    # 2019: one equivalence region per hexagon); the fuel zones are those of
    # the pooled map, shifted past the 36 outer-ring zones.
    p = build_hpmr2d(refine=4, drum_angle_deg=30.0, absorber="polar")
    pooled = hpmr_assembly_zones(p)
    z = hpmr_assembly_zones(p, per_hex=True)
    assert len(z.names) == 2 + 12 + 12 + 12 + 30
    act = np.asarray(p.active)
    cells = np.bincount(z.region_map[act].ravel(), minlength=len(z.names))
    of = {c: [i for i, n in enumerate(z.names) if n.startswith(c)]
          for c in ("be_reflector", "drum_be", "drum_absorber", "fuel")}
    assert len(set(cells[of["be_reflector"]])) == 1 and len(set(cells[of["drum_be"]])) == 1
    pooled_cells = np.bincount(pooled.region_map[act].ravel(), minlength=len(pooled.names))
    assert cells[of["be_reflector"]].sum() == pooled_cells[pooled.names.index("be_reflector")]
    assert cells[of["drum_be"]].sum() == pooled_cells[pooled.names.index("drum_be")]
    # every drum carries the same absorber volume, as its own mix-partner zone
    w = np.asarray(p.mix_weight)
    m = (z.mix_region_map >= 0) & act & (w > 0)
    arc = np.bincount(z.mix_region_map[m], weights=w[m], minlength=len(z.names))
    np.testing.assert_allclose(arc[of["drum_absorber"]], arc[of["drum_absorber"]][0], rtol=1e-12)
    assert arc[of["drum_absorber"]][0] > 0
    fuel = z.region_map >= of["fuel"][0]
    np.testing.assert_array_equal(z.region_map[fuel] - of["fuel"][0],
                                  pooled.region_map[fuel] - int(pooled.fuel.argmax()))
    assert all(z.site[i] is not None for i in of["drum_absorber"] + of["be_reflector"])
