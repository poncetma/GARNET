"""Drum-band meshes, equivalence zones and SPH-DF node models (garnet.equivalence,
garnet.drum_mesh) on a small HexLattice core with six control drums.

The reference is a fine raster diffusion solve, the node models a coarser
raster and a drum-band refined mesh; zones are built on the node cells and
carried to the reference by containment. SPH with boundary discontinuity
factors must then reproduce k and every zone's rate and boundary leakage on
both node models.
"""

import numpy as np
import pytest

from garnet import (PWR_TWO_GROUP, HexLattice, Material, TriDiffusionEigenSolver,
                    flux_weighted_homogenize, sph_bcf_correct, sph_correct)
from garnet.equivalence import NodeCells, SphDfNode, drum_core_zones, inherit_zones
from garnet.mesh import UnstructuredDiffusionSolver
from garnet.pin_power import tri_cell_centroids
from garnet.tri import tri_boundary_leakage

REFL = Material(name="refl", diffusion=[1.3, 0.3], sigma_a=[0.001, 0.02],
                nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.03], [0.0, 0.0]])
ABS = Material(name="b4c", diffusion=[1.0, 0.3], sigma_a=[0.05, 0.5],
               nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
CORNERS = [(2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2), (2, -2)]
TIGHT = dict(tol_k=1e-11, tol_source=1e-10)
ARC_AREA = 6 * np.pi / 4 * (4.5 ** 2 - 3.5 ** 2)          # six 90-degree arcs


def lattice(refine, angle=30.0):
    lat = HexLattice(pitch=10.0, refine=refine).set_disk(1, PWR_TWO_GROUP).set_ring(2, REFL)
    for rc in CORNERS:
        lat.set_drum(rc, REFL, ABS, 3.5, 4.5, 90.0, angle)
    return lat


def test_drum_band_mesh_matches_the_raster_and_closes_the_balance_with_boundary_factors():
    lat = lattice(4)
    rx = lat.build()
    k_raster = rx.steady(**TIGHT).k_eff
    q = lat.build_mesh(levels=0)                  # the raster itself, unstructured
    mats, cells = q.solver_materials()
    assert UnstructuredDiffusionSolver(q.mesh, mats, cells, device="cpu").solve(
        **TIGHT).k_eff == pytest.approx(k_raster, abs=1e-10)
    q = lat.build_mesh(levels=1)
    assert (q.mix_weight * np.asarray(q.mesh.area)).sum() == pytest.approx(ARC_AREA, rel=1e-12)
    assert q.mesh.n_cells == 3606 and len(set(q.parent)) == int(rx.active.sum())
    mats, cells = q.solver_materials()
    G, n = 2, q.mesh.n_cells
    plain = UnstructuredDiffusionSolver(q.mesh, mats, cells, device="cpu").solve(**TIGHT)
    s = UnstructuredDiffusionSolver(q.mesh, mats, cells, device="cpu", bcf=np.ones((G, n)))
    r = s.solve(**TIGHT)
    assert r.k_eff == plain.k_eff
    A = np.asarray(q.mesh.area)
    prod = sum((np.array([mats[m].nu_sigma_f[g] for m in cells]) * r.flux[g] * A).sum()
               for g in range(G)) / r.k_eff
    absorption = sum((np.array([mats[m].sigma_a[g] for m in cells]) * r.flux[g] * A).sum()
                     for g in range(G))
    leak = sum(s.boundary_leakage(g, r.flux[g], np.zeros(n, int), 1)[0] for g in range(G))
    assert prod - absorption == pytest.approx(leak, rel=1e-8)
    # a larger boundary coefficient (Robin albedo) leaks more, so k falls
    k3 = UnstructuredDiffusionSolver(q.mesh, mats, cells, device="cpu",
                                     bcf=np.full((G, n), 3.0)).solve(**TIGHT).k_eff
    assert k3 < r.k_eff


@pytest.fixture(scope="module")
def reference():
    fine = lattice(16).build()
    r = TriDiffusionEigenSolver(fine.grid, fine.materials, fine.material_map, active=fine.active,
                                mask_bc=fine.mask_bc, mix_material=fine.mix_material,
                                mix_weight=fine.mix_weight, device="cpu").solve(**TIGHT)
    return fine, r


def _node(kind):
    lat = lattice(8 if kind == "raster" else 4)
    model = lat.build() if kind == "raster" else lat.build_mesh(levels=1)
    zones, cells, boundary, drums = drum_core_zones(lat, model, n_sectors=8)
    return lat, model, zones, cells, boundary, drums


@pytest.mark.parametrize("kind", ["raster", "mesh"])
def test_zones_nest_exactly_and_keep_the_absorber_out_of_the_boundary_zones(kind, reference):
    fine, _ = reference
    lat, model, zones, cells, boundary, drums = _node(kind)
    assert zones.n_zones == 80 and len(boundary) == 12          # 7 fuel, 12 outer ring hexes
    assert all(zones.names[b].endswith(":outer") for b in boundary)
    fz = inherit_zones(zones, cells, tri_cell_centroids(fine.raster), fine.active.shape,
                       fine.active, fine.mix_material,
                       absorber_ids=[d["absorber_id"] for d in drums])

    def volumes(region, mix, w, area, act):
        region, mix = np.asarray(region).reshape(-1), np.asarray(mix).reshape(-1)
        w, area, act = np.asarray(w).reshape(-1), np.broadcast_to(area, region.shape), act.reshape(-1)
        vol = np.bincount(region[act], weights=area[act], minlength=zones.n_zones)
        m = act & (w > 0)
        ab = np.bincount(mix[m], weights=(w * area)[m], minlength=zones.n_zones)
        return vol, ab

    vn, an = volumes(zones.region_map, zones.mix_region_map,
                     model.mix_weight, cells.area, cells.active)
    vf, af = volumes(fz.region_map, fz.mix_region_map, fine.mix_weight,
                     fine.grid.cell_volume, np.asarray(fine.active))
    np.testing.assert_allclose(vn, vf, atol=1e-9)
    np.testing.assert_allclose(an, af, atol=1e-9)
    assert an.sum() == pytest.approx(ARC_AREA, rel=1e-12)
    assert an[boundary].sum() == 0.0 and af[boundary].sum() == 0.0
    fine_cells = NodeCells.from_raster(fine.raster, fine.active)
    assert np.isin(np.asarray(fz.region_map).reshape(-1)[fine_cells.exposed], boundary).all()


@pytest.mark.parametrize("kind", ["raster", "mesh"])
def test_sph_df_reproduces_k_rates_and_leakage_on_raster_and_mesh_nodes(kind, reference):
    fine, ref = reference
    lat, model, zones, cells, boundary, drums = _node(kind)
    fz = inherit_zones(zones, cells, tri_cell_centroids(fine.raster), fine.active.shape,
                       fine.active, fine.mix_material,
                       absorber_ids=[d["absorber_id"] for d in drums])
    hm, rflux, rvol = flux_weighted_homogenize(
        ref.flux_numpy, fine.materials, fine.material_map, fz.region_map,
        cell_volume=fine.grid.cell_volume, mix_material=fine.mix_material,
        mix_weight=fine.mix_weight, mix_region_map=fz.mix_region_map)
    s = TriDiffusionEigenSolver(fine.grid, fine.materials, fine.material_map, active=fine.active,
                                mask_bc=fine.mask_bc, mix_material=fine.mix_material,
                                mix_weight=fine.mix_weight, device="cpu")
    G = rflux.shape[1]
    Jref = np.array([tri_boundary_leakage(s.ops[g], ref.flux_numpy[g], fz.region_map,
                                          zones.n_zones, fine.active)[boundary]
                     for g in range(G)]).T
    node = (SphDfNode.raster(zones, boundary, model.grid, model.active, model.mix_weight,
                             model.mask_bc) if kind == "raster"
            else SphDfNode.mesh(zones, boundary, model))
    fv = sph_correct(hm, zones.region_map, rflux, lambda m: node(m)[:2], region_volume=rvol,
                     normalization="flux-volume", tol=1e-8)
    assert abs(fv.k_eff - ref.k_eff) > 5e-5                       # plain SPH misses k
    out = sph_bcf_correct(hm, rflux, Jref, node, region_volume=rvol, tol=1e-8, max_iter=400)
    assert out.converged
    phi, k, leak = node(out.corrected_materials, out.bcf)
    assert k == pytest.approx(ref.k_eff, abs=1e-8)
    cg = (rvol[:, None] * rflux).sum(0) / (rvol[:, None] * phi).sum(0)
    live = rflux > rflux.max() * 1e-12
    np.testing.assert_allclose((out.factors * phi * cg)[live], rflux[live], rtol=1e-6)
    np.testing.assert_allclose(leak * cg, Jref, rtol=1e-6)
