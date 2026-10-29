"""The steady coupling's stopping test, its returned state, and the drivers
around it, on a slab small enough to iterate to tight tolerances.

* Convergence is the fixed-point residual max|G(T) - T|, not the relaxed step
  (which under Picard relaxation omega is omega times smaller).
* The returned (k, flux, power, T) is ONE evaluated state.
* The power edit follows the feedback's change to Sigma_f.
* ``uncoupled_k`` puts every material at its own reference temperature.
* ``criticality_search`` refuses to return an unconverged angle.
* ``coupling_vertices`` covers every cell when there is no active mask.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import garnet.coupling as coupling
from garnet import DiffusionEigenSolver, Grid, Kinetics, Material
from garnet.coupling import (CoupledSolver, CouplingContext, coupled_map,
                             coupled_transient, coupling_vertices,
                             criticality_search, neutronics_step, uncoupled_k)
from garnet.feedback import ThermalFeedback
from garnet.thermal import ThermalMaterial

GRID = Grid(shape=(16, 1, 1), size=(80.0, 1.0, 1.0))
REFL = Material(name="refl", diffusion=[1.2, 0.9], sigma_a=[2e-4, 6e-3],
                nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.04], [0.0, 0.0]])
FUEL = Material(name="fuel", diffusion=[1.3, 0.4], sigma_a=[0.009, 0.08],
                nu_sigma_f=[0.006, 0.13], sigma_s=[[0.0, 0.02], [0.0, 0.0]],
                chi=[1.0, 0.0], kappa_fission=[0.0025, 0.052])
MMAP = np.zeros(GRID.shape, dtype=int)
MMAP[3:13] = 1
KIN = Kinetics(velocities=[1.5e7, 2.5e5], beta=[0.0065], decay=[0.08])


def _ctx(feedback=None, **kw):
    if feedback is None:
        feedback = ThermalFeedback(t_ref=[300.0, 800.0], doppler=[0.0, 4e-3])
    thermal = [ThermalMaterial(0.3, heat_capacity=3.0, name="refl"),
               ThermalMaterial(0.05, sink_coeff=0.01, sink_temperature=700.0,
                               heat_capacity=3.5, name="fuel")]
    args = dict(grid=GRID, materials=[REFL, FUEL], material_map=MMAP,
                thermal_materials=thermal, feedback=feedback, total_power=60.0,
                bc=("vacuum", "reflective", "reflective"),
                thermal_bc="adiabatic", ambient_temperature=300.0,
                kinetics=KIN, solver_cls=DiffusionEigenSolver, device="cpu",
                warm_start=False,
                eigen_kwargs={"tol_k": 1e-13, "tol_source": 1e-12})
    args.update(kw)
    return CouplingContext(**args)


def _span(T):
    return max(float(np.ptp(T)), 1.0)


def test_relaxed_picard_converges_on_the_fixed_point_residual():
    ctx = _ctx()
    tol = 1e-6
    res = CoupledSolver(ctx).solve(tol=tol, relaxation=0.1, max_iter=600)
    assert res.converged
    # G is a pure function of T here (warm start off), so re-applying it to
    # the returned field must reproduce the reported state exactly and show a
    # residual inside the tolerance. Stopping on the relaxed step let the
    # residual sit ~1/omega = 10x above tol.
    t_next, info = coupled_map(res.temperature, ctx)
    assert np.max(np.abs(t_next - res.temperature)) / _span(t_next) < tol
    assert res.residual_history[-1] / _span(t_next) < tol


def test_the_returned_state_is_one_evaluated_state():
    ctx = _ctx()
    res = CoupledSolver(ctx).solve(tol=1e-5, relaxation=0.5, max_iter=200)
    _, info = coupled_map(res.temperature, ctx)
    # k, flux and power are the neutronics AT the returned temperature.
    assert info["neutronics"].k_eff == pytest.approx(res.k_eff, rel=1e-12)
    np.testing.assert_allclose(info["power"], res.power_density, rtol=1e-9,
                               atol=1e-12 * float(np.max(res.power_density)))


def test_anderson_stops_on_the_residual_too():
    ctx = _ctx()
    tol = 1e-7
    res = CoupledSolver(ctx).solve(tol=tol, anderson_depth=3, relaxation=0.5,
                                   max_iter=200)
    assert res.converged
    t_next, _ = coupled_map(res.temperature, ctx)
    assert np.max(np.abs(t_next - res.temperature)) / _span(t_next) < tol


def test_power_edit_follows_the_feedback_change_in_fission():
    """Expansion scales Sigma_f by f(T) per cell; kappa*Sigma_f must move with
    it. With a non-uniform temperature the power SHAPE changes, which the
    rated-power normalization cannot cancel."""
    beta = 2e-4
    fb = ThermalFeedback(t_ref=[300.0, 800.0], doppler=[0.0, 0.0],
                         expansion=[0.0, beta])
    ctx = _ctx(fb)
    x = GRID.cell_centers(0).reshape(GRID.shape)
    T = 800.0 + 300.0 * np.sin(np.pi * x / 80.0)        # 800 .. 1100 K
    q, res = neutronics_step(T, ctx)

    flux = np.asarray(res.flux)
    f = np.where(MMAP == 1, 1.0 - beta * (T - 800.0), 1.0)
    kappa = np.stack([np.where(MMAP == 1, FUEL.kappa_fission[g], 0.0)
                      for g in range(2)])
    raw = (kappa * f * flux).sum(axis=0)
    expected = raw * ctx.total_power / float(np.sum(raw * GRID.cell_volume))
    np.testing.assert_allclose(q, expected, rtol=1e-10,
                               atol=1e-12 * float(expected.max()))
    # And the difference is not a rounding matter: f spans 0.94 .. 1.0.
    static = (kappa * flux).sum(axis=0)
    static *= ctx.total_power / float(np.sum(static * GRID.cell_volume))
    assert np.max(np.abs(static - q)) / q.max() > 1e-3


def test_coupled_transient_holds_the_steady_state_with_density_feedback():
    """The transient's device-side power edit must use the same
    feedback-moved kappa as the steady one, or t = 0 is not an equilibrium."""
    fb = ThermalFeedback(t_ref=[300.0, 800.0], doppler=[0.0, 3e-3],
                         expansion=[0.0, 2e-4])
    ctx = _ctx(fb, warm_start=True)
    steady = CoupledSolver(ctx).solve(tol=1e-10, anderson_depth=3)
    r = coupled_transient(ctx, t_end=2.0, dt=0.5, dt_thermal=1.0,
                          initial_coupled_steady=steady)
    np.testing.assert_allclose(r.power, 1.0, atol=2e-6)
    np.testing.assert_allclose(r.temperature, steady.temperature, rtol=0,
                               atol=2e-5)


def test_uncoupled_k_puts_each_material_at_its_own_reference():
    # The reflector responds (c_D != 0) and its reference is 300 K, the fuel's
    # 800 K. "The cross sections exactly as tabulated" is no hook at all; the
    # old max(t_ref) evaluated the reflector 500 K hot.
    fb = ThermalFeedback(t_ref=[300.0, 800.0], doppler=[5e-3, 4e-3])
    ctx = _ctx(fb)
    plain = DiffusionEigenSolver(GRID, ctx.materials, MMAP, bc=ctx.bc,
                                 device="cpu").solve(**ctx.eigen_kwargs).k_eff
    assert uncoupled_k(ctx) == pytest.approx(plain, rel=1e-12)
    # An explicit temperature is still honoured.
    assert uncoupled_k(ctx, np.full(GRID.shape, 800.0)) < plain


def _fake_search(monkeypatch, k_of_angle):
    class FakeSolver:
        def __init__(self, ctx):
            self.angle = ctx

        def solve(self, **_):
            return SimpleNamespace(k_eff=k_of_angle(self.angle), converged=True,
                                   peak_temperature=0.0)

    monkeypatch.setattr(coupling, "CoupledSolver", FakeSolver)
    return lambda angle: angle                          # rebuild: ctx = angle


def test_criticality_search_converges_on_a_smooth_curve(monkeypatch):
    rebuild = _fake_search(monkeypatch, lambda a: 0.99 + 2e-4 * a)
    angle, res = criticality_search(rebuild, {}, tol=1e-9)
    assert angle == pytest.approx(50.0, abs=1e-4)
    assert abs(res.k_eff - 1.0) < 1e-9


def test_criticality_search_refuses_a_stalled_secant(monkeypatch):
    # A coarse-mesh staircase: the secant lands on two angles of one flat
    # step, |fb - fa| vanishes, and the old loop returned 112.5 deg at
    # k = 1.001 as if it were the answer.
    rebuild = _fake_search(monkeypatch,
                           lambda a: 1.0 + (1e-3 if a > 100.0 else -1e-3))
    with pytest.raises(RuntimeError, match="did not converge"):
        criticality_search(rebuild, {}, tol=1e-6)


def test_criticality_search_refuses_when_iterations_run_out(monkeypatch):
    rebuild = _fake_search(monkeypatch, lambda a: 0.9 + 0.2 * (a / 180.0) ** 3)
    with pytest.raises(RuntimeError, match="did not converge"):
        criticality_search(rebuild, {}, tol=1e-12, max_iter=2)


def test_coupling_vertices_cover_every_cell_without_an_active_mask():
    problem = SimpleNamespace(grid=GRID, raster=None, active=None)
    coords, idx = coupling_vertices(problem)
    np.testing.assert_array_equal(idx, np.arange(GRID.shape[0]))
    assert coords.shape == (GRID.shape[0], 3)
    masked = SimpleNamespace(grid=GRID, raster=None,
                             active=(MMAP == 1))
    _, idx = coupling_vertices(masked)
    np.testing.assert_array_equal(idx, np.flatnonzero((MMAP == 1).ravel()))
