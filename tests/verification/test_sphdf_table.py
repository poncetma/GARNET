"""Tabulated SPH-DF model in drum angle (garnet.equivalence.SphDfTable) on the
six-drum HexLattice of test_equivalence, with the drum-band mesh as the node.

SPH-DF entries are generated at two drum angles against a fine raster
reference. Checks:

* at a table angle the tabulated state (SPH-corrected zone materials, zone
  maps, absorber blend, boundary coefficients) reproduces the reference k and
  zone rates on the mesh eigen solver, and holds a transient steady;
* between the angles every quantity is the linear interpolation of the two
  entries, the boundary factors sit on the boundary zones' cells only, and
  problem_at returns the same tuple while the angle holds;
* a ramp between the two angles runs on the mesh transient and ends at the
  power level of a step to the final state.

Broken on purpose: a state that drops the boundary factors misses k by
13.9 pcm against 1e-6 pcm when complete (asserted below at 5 pcm).
"""

import numpy as np
import pytest

from garnet import (Kinetics, TransientSolver, TriDiffusionEigenSolver,
                    flux_weighted_homogenize, sph_bcf_correct)
from garnet.equivalence import (SphDfNode, SphDfTable, drum_core_zones, inherit_zones,
                                zone_material_arrays, ZONE_KEYS)
from garnet.mesh import MeshDiffusionEigenSolver, MeshGrid, MeshGroupOperator
from garnet.pin_power import tri_cell_centroids
from garnet.sph import region_average
from garnet.tri import tri_boundary_leakage

from test_equivalence import TIGHT, lattice

KIN = Kinetics(velocities=[1.25e7, 2.5e5], beta=[0.0065], decay=[0.0784])
ANGLES = (20.0, 40.0)


def _node(angle):
    lat = lattice(4, angle)
    q = lat.build_mesh(levels=1)
    zones, cells, boundary, drums = drum_core_zones(lat, q, n_sectors=8)
    return lat, q, zones, cells, boundary, drums


def geometry_at(angle):
    _, q, zones, _, _, _ = _node(angle)
    return zones, q.mix_weight


def _entry(angle):
    fine = lattice(16, angle).build()
    ref = TriDiffusionEigenSolver(fine.grid, fine.materials, fine.material_map,
                                  active=fine.active, mask_bc=fine.mask_bc,
                                  mix_material=fine.mix_material, mix_weight=fine.mix_weight,
                                  device="cpu")
    r = ref.solve(**TIGHT)
    _, q, zones, cells, boundary, drums = _node(angle)
    fz = inherit_zones(zones, cells, tri_cell_centroids(fine.raster), fine.active.shape,
                       fine.active, fine.mix_material,
                       absorber_ids=[d["absorber_id"] for d in drums])
    hm, rflux, rvol = flux_weighted_homogenize(
        r.flux_numpy, fine.materials, fine.material_map, fz.region_map,
        cell_volume=fine.grid.cell_volume, mix_material=fine.mix_material,
        mix_weight=fine.mix_weight, mix_region_map=fz.mix_region_map)
    G = rflux.shape[1]
    Jref = np.array([tri_boundary_leakage(ref.ops[g], r.flux_numpy[g], fz.region_map,
                                          zones.n_zones, fine.active)[boundary]
                     for g in range(G)]).T
    node = SphDfNode.mesh(zones, boundary, q)
    out = sph_bcf_correct(hm, rflux, Jref, node, region_volume=rvol, tol=1e-9, max_iter=400)
    assert out.converged
    _, k, _ = node(out.corrected_materials, out.bcf)
    return dict(k=k, k_ref=r.k_eff, mu=out.factors, gamma=out.bcf, bregs=np.array(boundary),
                names=np.array(zones.names), converged=out.converged, rflux=rflux, rvol=rvol,
                **zone_material_arrays(hm))


@pytest.fixture(scope="module")
def table():
    return SphDfTable({a: _entry(a) for a in ANGLES}, geometry_at)


def _solve(state, q):
    return MeshDiffusionEigenSolver(MeshGrid(q.mesh, q.alpha_boundary), state.materials,
                                    state.region_map, mix_material=state.mix_region_map,
                                    mix_weight=state.mix_weight, bcf=state.bcf, device="cpu",
                                    linear_solver="gmres").solve(**TIGHT)


def test_table_state_reproduces_the_reference_at_a_table_angle(table):
    a = ANGLES[0]
    e = table.entries[a]
    st = table.state(a)
    assert st.k_table == pytest.approx(e["k"])
    _, q, zones, _, _, _ = _node(a)
    r = _solve(st, q)
    assert r.k_eff == pytest.approx(e["k_ref"], abs=1e-8)
    avg = region_average(r.flux_numpy, st.region_map, cell_volume=np.asarray(q.mesh.area),
                         mix_material=st.mix_region_map, mix_weight=st.mix_weight)
    phi = st.mu * avg[:len(st.mu)]
    rflux = e["rflux"]
    phi *= (e["rvol"][:, None] * rflux).sum(0) / (e["rvol"][:, None] * phi).sum(0)
    live = rflux > rflux.max() * 1e-12
    np.testing.assert_allclose(phi[live], rflux[live], rtol=1e-6)


def test_broken_state_without_boundary_factors_misses_k(table):
    a = ANGLES[0]
    st = table.state(a)
    st.bcf = np.ones_like(st.bcf)
    _, q, _, _, _, _ = _node(a)
    assert abs(_solve(st, q).k_eff - table.entries[a]["k_ref"]) > 5e-5       # 13.9 pcm


def test_between_table_angles_everything_is_linear_and_held_constant(table):
    st = table.state(30.0)
    lo, hi = (table.entries[a] for a in ANGLES)
    np.testing.assert_allclose(st.mu, 0.5 * (lo["mu"] + hi["mu"]), rtol=1e-14)
    np.testing.assert_allclose(st.gamma, 0.5 * (lo["gamma"] + hi["gamma"]), rtol=1e-14)
    D = 0.5 * (lo["D"] + hi["D"])
    np.testing.assert_allclose([m.diffusion for m in st.materials], D * st.mu, rtol=1e-12)
    assert st.k_table is None
    on_boundary = np.isin(st.region_map, table.bregs)
    assert np.all(st.bcf[:, ~on_boundary] == 1.0)
    for b, reg in enumerate(table.bregs):
        np.testing.assert_array_equal(st.bcf[:, st.region_map == reg],
                                      np.repeat(st.gamma[b][:, None],
                                                (st.region_map == reg).sum(), axis=1))
    assert set(ZONE_KEYS) <= set(lo)
    at = table.problem_at(lambda t: 30.0 if t < 1.0 else 35.0)
    assert at(0.0) is at(0.5)
    assert at(2.0) is not at(0.5)


def _transient(table, angle_of_t, t_end, dt):
    q = _node(ANGLES[0])[1]
    s = TransientSolver(MeshGrid(q.mesh, q.alpha_boundary), table.problem_at(angle_of_t),
                        KIN, group_operator=MeshGroupOperator,
                        eig_solver=MeshDiffusionEigenSolver, device="cpu",
                        linear_solver="gmres")
    return s.solve(t_end, dt, tol_step=1e-10, steady_kwargs=dict(tol_k=1e-11, tol_source=1e-10))


def test_tabulated_transient_holds_steady_and_follows_a_ramp(table):
    hold = _transient(table, lambda t: ANGLES[0], 0.1, 0.05)
    np.testing.assert_allclose(hold.power, 1.0, atol=1e-9)
    ramp = _transient(table, lambda t: min(ANGLES[0] + 100.0 * t, 30.0), 0.2, 0.025)
    step = _transient(table, lambda t: ANGLES[0] if t <= 0 else 30.0, 0.2, 0.025)
    assert abs(step.power[-1] - 1.0) > 1e-2                     # 20 -> 30 deg moves the power
    assert ramp.power[-1] == pytest.approx(step.power[-1], rel=2e-2)
    # mid-ramp (25 deg at t = 0.05 s) the ramp lags the step
    assert abs(ramp.power[2] - 1.0) < abs(step.power[2] - 1.0)
