"""SPH with boundary discontinuity factors (Laboure et al. 2019 SPH-DF).

With a vacuum boundary, flux-volume normalized SPH does not reproduce k: the
normalization is not a gauge there. With SPH factors on every zone and a
boundary factor on every zone that has vacuum faces, the corrected problem is
invariant under a per-group rescaling of flux, factors and boundary factors,
so the normalization only fixes that gauge and k is reproduced together with
every zone's reaction rate and boundary leakage -- provided a solution exists.
On HP-MR it exists once every hex on the vacuum boundary is split into an inner
and an outer zone (tools/sph_hpmr_nodes.split_boundary_layer); without the split
the zero-flux limit of the coarse boundary leakage is below the reference in
some groups (tools/probe_sph_bcf_feasibility.py).
"""

import numpy as np
import pytest

from garnet import (TriDiffusionEigenSolver, flux_weighted_homogenize, region_average,
                    sph_bcf_correct, sph_correct)
from garnet.benchmarks.hpmr import build_hpmr2d, hpmr_assembly_zones
from garnet.benchmarks.hpmr_core_pins import build_hpmr2d_pins
from garnet.tri import tri_boundary_leakage
from tools.sph_hpmr_nodes import (exposed_cells, sector_zoning_hooks, split_boundary_layer,
                                  transfer_boundary_layer)

TIGHT = dict(tol_k=1e-11, tol_source=1e-10)


@pytest.fixture(scope="module")
def case():
    """Pin-resolved placeholder core (reference) and a split-zone node raster."""
    q = build_hpmr2d(refine=4, drum_angle_deg=30.0, absorber="polar")
    zn, outer, outer_of = split_boundary_layer(hpmr_assembly_zones(q, per_hex=True), q)
    p, _ = build_hpmr2d_pins(refine=8, drum_angle_deg=30.0, data="placeholder", mixing="two")
    zr = transfer_boundary_layer(hpmr_assembly_zones(p, per_hex=True), p, q, outer, outer_of)
    ref = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                                  mask_bc=p.mask_bc, mix_material=p.mix_material,
                                  mix_weight=p.mix_weight, device="cpu").solve(**TIGHT)
    hm, rflux, rvol = flux_weighted_homogenize(
        ref.flux_numpy, p.materials, p.material_map, zr.region_map,
        cell_volume=p.grid.cell_volume, mix_material=p.mix_material, mix_weight=p.mix_weight,
        mix_region_map=zr.mix_region_map)
    bregs = sorted(outer_of.values())
    G = rflux.shape[1]
    # reference boundary current of each outer zone: the reference solve's own leakage
    solver = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                                     mask_bc=p.mask_bc, mix_material=p.mix_material,
                                     mix_weight=p.mix_weight, device="cpu")
    Jref = np.array([tri_boundary_leakage(solver.ops[g], ref.flux_numpy[g], zr.region_map,
                                          len(zr.names), p.active)[bregs] for g in range(G)]).T
    mix = dict(mix_material=zn.mix_region_map, mix_weight=q.mix_weight)

    def solve_bcf(mats, gam):
        bcf = np.ones((G,) + q.grid.shape)
        for b, reg in enumerate(bregs):
            bcf[:, zn.region_map == reg] = np.asarray(gam[b], float)[:, None]
        s = TriDiffusionEigenSolver(q.grid, mats, zn.region_map, active=q.active,
                                    mask_bc=q.mask_bc, device="cpu", bcf=bcf, **mix)
        r = s.solve(**TIGHT)
        leak = np.array([tri_boundary_leakage(s.ops[g], r.flux_numpy[g], zn.region_map,
                                              len(zn.names), q.active)[bregs]
                         for g in range(G)]).T
        return (region_average(r.flux_numpy, zn.region_map, cell_volume=q.grid.cell_volume,
                               **mix), r.k_eff, leak)

    def solve(mats):
        phi, k, _ = solve_bcf(mats, np.ones((len(bregs), G)))
        return phi, k

    return dict(k=ref.k_eff, hm=hm, rflux=rflux, rvol=rvol, Jref=Jref, zones=zn,
                solve=solve, solve_bcf=solve_bcf)


def test_split_layer_is_the_same_region_on_nested_rasters_and_owns_the_boundary():
    from garnet.tri_sn import _EDGES
    q = build_hpmr2d(refine=4, drum_angle_deg=180.0, absorber="polar")
    zn, outer, outer_of = split_boundary_layer(hpmr_assembly_zones(q, per_hex=True), q)
    p, _ = build_hpmr2d_pins(refine=8, drum_angle_deg=180.0)
    zr = transfer_boundary_layer(hpmr_assembly_zones(p, per_hex=True), p, q, outer, outer_of)
    assert zn.names == zr.names and len(outer_of) == 24
    o = list(outer_of.values())
    area_n = np.bincount(zn.region_map[q.active].ravel(),
                         minlength=len(zn.names)) * q.grid.cell_volume
    area_r = np.bincount(zr.region_map[p.active].ravel(),
                         minlength=len(zr.names)) * p.grid.cell_volume
    np.testing.assert_allclose(area_n[o], area_r[o], rtol=1e-12)
    for prob, z in ((q, zn), (p, zr)):
        act = np.asarray(prob.active)
        nr, nc = act.shape[:2]
        I, J = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        for t, di, dj, tn, _ in _EDGES:
            ii, jj = I + di, J + dj
            inside = (ii >= 0) & (ii < nr) & (jj >= 0) & (jj < nc)
            nbr = np.zeros((nr, nc), bool)
            nbr[inside] = act[ii[inside], jj[inside], tn]
            exposed = act[:, :, t] & ~nbr
            assert np.isin(z.region_map[:, :, t][exposed], o).all()


@pytest.mark.parametrize("variant", [dict(), dict(aligned=True, outer_arc_to_sector=True)],
                         ids=["sectors", "aligned, Be-only outer layer"])
def test_drum_sectors_are_the_same_regions_on_nested_rasters_and_keep_the_absorber(variant):
    """Drum core + ring-sector zoning (sector_zoning_hooks): identical zone areas
    and per-zone absorber areas on the node raster and the nested reference
    raster, the analytic absorber area in total, and the boundary still owned by
    the outer zones. On refine 8 (1.9 cm) nodes every sector is non-empty; on
    refine 4 the outer layer swallows the sectors that face away from the core.
    With ``outer_arc_to_sector`` no outer (boundary) zone holds absorber."""
    from garnet.benchmarks.hpmr import DRUM_ABSORBER_INNER, DRUM_RADIUS
    q = build_hpmr2d(refine=8, drum_angle_deg=165.0, absorber="polar")
    p, _ = build_hpmr2d_pins(refine=16, drum_angle_deg=165.0, data="placeholder")
    node_hook, ref_hook, state = sector_zoning_hooks(8, **variant)
    zn, zr = node_hook(None, q), ref_hook(None, p)
    assert zn.names == zr.names and len(zn.names) == 176

    def areas(prob, z):
        act = np.asarray(prob.active)
        area = np.bincount(z.region_map[act], minlength=len(z.names)) * prob.grid.cell_volume
        w = np.where(act, np.asarray(prob.mix_weight), 0.0)
        m = w > 0
        drum = np.isin(z.mix_region_map[m],
                       [i for i, n in enumerate(z.names) if n.startswith("drum")])
        absorber = np.bincount(z.mix_region_map[m][drum], weights=w[m][drum],
                               minlength=len(z.names)) * prob.grid.cell_volume
        return area, absorber

    an, abn = areas(q, zn)
    ar, abr = areas(p, zr)
    live = np.array([n != "void" for n in zn.names])
    assert (an[live] > 10.0).all()
    np.testing.assert_allclose(an, ar, rtol=1e-12)
    np.testing.assert_allclose(abn, abr, atol=1e-9)
    assert abn.sum() == pytest.approx(12 * np.pi / 4 * (DRUM_RADIUS ** 2 - DRUM_ABSORBER_INNER ** 2),
                                      rel=1e-12)
    o = list(state["outer_of"].values())
    drum_outer = [i for i in o if zn.names[i].startswith("drum")]
    if variant.get("outer_arc_to_sector"):
        assert abn[drum_outer].sum() == 0.0 and abr[drum_outer].sum() == 0.0
    else:
        assert abn[drum_outer].sum() > 1.0          # the arc faces out at 165 deg
    for prob, z in ((q, zn), (p, zr)):
        assert np.isin(z.region_map[exposed_cells(prob)], o).all()


def test_sph_df_reproduces_k_and_rates_where_flux_volume_sph_does_not(case):
    c = case
    fv = sph_correct(c["hm"], c["zones"].region_map, c["rflux"], c["solve"],
                     region_volume=c["rvol"], normalization="flux-volume", tol=1e-8)
    assert fv.converged and abs(fv.k_eff - c["k"]) > 20e-5          # tens of pcm off
    out = sph_bcf_correct(c["hm"], c["rflux"], c["Jref"], c["solve_bcf"],
                          region_volume=c["rvol"], tol=1e-8, max_iter=400)
    assert out.converged
    phi, k, leak = c["solve_bcf"](out.corrected_materials, out.bcf)
    assert k == pytest.approx(c["k"], abs=1e-7)
    # every zone's reaction rate, up to the per-group normalization
    cg = (c["rvol"][:, None] * c["rflux"]).sum(0) / (c["rvol"][:, None] * phi).sum(0)
    live = c["rflux"] > c["rflux"].max() * 1e-12
    np.testing.assert_allclose((out.factors * phi * cg)[live], c["rflux"][live], rtol=1e-6)
    np.testing.assert_allclose(leak * cg, c["Jref"], rtol=1e-6)
