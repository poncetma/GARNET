"""Transient triangular S_N on the level-scheduled sweep (``engine="levels"``):
P_L scattering, volume-mixed cells that move in time, in-place material
updates and the whole-core rebalance of ``TransientSNSolver``.

Two-group core (fuel, reflector, an absorber), 8 x 8 x 2 triangles, vacuum,
2 x 8 product quadrature. Checks, each broken on purpose (the broken variants
run without CMFD: with it they stop converging, since the coarse balance
assumes the time source integrates to theta * phi_old):

* isotropic: the levels engine reproduces the LU engine (the same discrete
  equations, independently swept) to 5e-13 after a nu*Sigma_f step, step and
  SCB; the levels engine takes the step through ``t_update_materials``, the LU
  engine through a rebuild. Broken (SCB time source averaged over the three
  corners of each cell) it misses by 3.1e-3;
* P2: the critically adjusted steady state holds (power 1 to 3e-12, step and
  SCB). Broken (the first step's angular flux seeded without the anisotropic
  scattering source) the power drifts by 2.3e-3 (step) / 2.6e-3 (SCB);
* P2 dynamics: for a uniform nu*Sigma_f step the shape stays the transport
  mode, so the power follows exact point kinetics with beta_eff = beta and
  the generation time of ``sn_effective_kinetics`` (P_L moments). Slow
  neutrons (Lambda 1.6 ms) put the prompt rise on the time steps; BDF2 at
  dt = 0.01 s matches to 1.27e-3 (1.36e-3 / 1.25e-3 at dt 0.02 / 0.005 s:
  the floor is the time eigenmode's departure from the k-mode, not the time
  step), better than with the scalar-flux Lambda (1.82e-3). Broken (the time
  source made isotropic, theta * phi_old / W) it misses by 117 %;
* a moving volume blend (an absorber strip whose weight grows every step):
  the in-place update equals rebuilding the engine every step. Broken (the
  sweep's corner inverses not refreshed) it misses by 0.24;
* rebalance: the whole-core rebalance with Picard reaches the same P(t) as
  the Anderson fixed point;
* transient CMFD (``t_cmfd_factor``, the default step acceleration on this
  engine): the same P(t) as the unaccelerated fixed point (1e-10 / 4e-10 for
  P0 / P2) in 6-8 sweeps per step instead of 15-19. Broken (the coarse
  operator fitted to a sweep without the time source) the step does not
  converge (amplitude error 2.5e-3 / 2.2e-3 after 200 sweeps);

The CUDA-graph path of the same sweeps is checked in test_gpu_paths.py.
"""

import numpy as np
import pytest

from garnet import Kinetics, Material, TransientSNSolver
from garnet.tri import TriGrid
from garnet.tri_sn import TriSNTransportSolver

KIN = Kinetics(velocities=[2.0e7, 3.0e5], beta=[0.0065], decay=[0.08])
N = 8
STEADY = dict(tol_k=1e-12, tol_source=1e-11)


def _mat(name, st, s0, s1, nsf, chi=(1.0, 0.0)):
    st, s0, s1 = np.array(st), np.array(s0), np.array(s1)
    return Material(name=name, diffusion=list(1 / (3 * st)), sigma_a=list(st - s0.sum(1)),
                    nu_sigma_f=list(nsf), sigma_s=s0.tolist(), total=list(st), chi=list(chi),
                    sigma_s_moments=np.array([s0, s1]))


FUEL_S0, FUEL_S1 = [[0.40, 0.05], [0.0, 1.05]], [[0.12, 0.01], [0.0, 0.25]]
FUEL = _mat("fuel", [0.5, 1.2], FUEL_S0, FUEL_S1, [0.008, 0.20])
FUEL_UP = _mat("fuel+", [0.5, 1.2], FUEL_S0, FUEL_S1, [0.008, 0.2008])
REFL = _mat("refl", [0.6, 1.8], [[0.50, 0.09], [0.0, 1.76]], [[0.20, 0.02], [0.0, 0.50]],
            [0.0, 0.0], chi=(0.0, 0.0))
ABS = _mat("abs", [0.6, 3.0], [[0.45, 0.05], [0.0, 1.5]], [[0.10, 0.01], [0.0, 0.2]],
           [0.0, 0.0], chi=(0.0, 0.0))
BASE, PERT = [FUEL, REFL, ABS], [FUEL_UP, REFL, ABS]
GRID = TriGrid(shape=(N, N, 2), side=1.5)
MMAP = np.zeros((N, N, 2), int)
MMAP[:2] = 1
MMAP[:, :2] = 1


def _step(t):
    return (BASE if t <= 0 else PERT, MMAP)


def _run(problem, engine="levels", scheme="scb", order=0, t_end=0.05, dt=0.01,
         accel="auto", **kw):
    s = TransientSNSolver(GRID, problem, KIN, bc="vacuum", engine_cls=TriSNTransportSolver,
                          n_polar=2, n_azi=8, scheme=scheme, engine=engine,
                          scattering_order=order, step_acceleration=accel)
    return s.solve(t_end, dt, tol_step=1e-10, steady_kwargs=STEADY, **kw)


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_levels_engine_reproduces_the_lu_engine(scheme):
    lu, lv = _run(_step, "lu", scheme), _run(_step, "levels", scheme)
    assert lv.power[-1] > 1.5                                   # the step matters
    # LU: plain fixed point; levels: CMFD. Each converges to tol_step 1e-10.
    np.testing.assert_allclose(lv.power, lu.power, rtol=0, atol=1e-9)


def test_broken_corner_averaged_time_source_misses_the_lu_engine(monkeypatch):
    lu = _run(_step, "lu", "scb")
    write = TriSNTransportSolver._t_time_source

    def averaged(self, g, psi_old):
        write(self, g, np.repeat(psi_old.mean(axis=2, keepdims=True), 3, axis=2))

    monkeypatch.setattr(TriSNTransportSolver, "_t_time_source", averaged)
    lv = _run(_step, "levels", "scb", accel="none")
    assert np.abs(lv.power - lu.power).max() > 1e-6


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_p2_steady_state_holds(scheme):
    r = _run(lambda t: (BASE, MMAP), scheme=scheme, order=2)
    np.testing.assert_allclose(r.power, 1.0, rtol=0, atol=1e-10)


def test_broken_isotropic_seed_drifts(monkeypatch):
    seed = TriSNTransportSolver.t_seed_psi

    def without_anisotropic_source(self, g, src, state):
        saved = self._last_moments
        self._last_moments = [np.zeros_like(m) for m in saved]
        try:
            return seed(self, g, src, state)
        finally:
            self._last_moments = saved

    monkeypatch.setattr(TriSNTransportSolver, "t_seed_psi", without_anisotropic_source)
    r = _run(lambda t: (BASE, MMAP), order=2, accel="none")
    assert np.abs(r.power - 1.0).max() > 1e-6


MIXM = np.full((N, N, 2), -1)
MIXM[5, 2:6] = 2                                         # an absorber strip


def _moving_blend(t):
    w = np.zeros((N, N, 2))
    w[5, 2:6] = 0.1 + 4.0 * max(t, 0.0)                  # a new array every step
    return BASE, MMAP, MIXM, w


def test_in_place_update_equals_rebuilding_the_engine():
    fast = _run(_moving_blend, order=2, update_in_place=True)
    slow = _run(_moving_blend, order=2, update_in_place=False)
    assert fast.power[-1] < 0.99                         # the absorber bites
    np.testing.assert_allclose(fast.power, slow.power, rtol=0, atol=1e-10)


def test_broken_stale_sweep_tables_miss_the_rebuild(monkeypatch):
    slow = _run(_moving_blend, order=2, update_in_place=False)
    monkeypatch.setattr(TriSNTransportSolver, "_refresh_lv_tables", lambda self: None)
    fast = _run(_moving_blend, order=2, update_in_place=True, accel="none")
    assert np.abs(fast.power - slow.power).max() > 1e-6


def test_rebalance_reaches_the_same_transient():
    plain = _run(_step, order=2)
    reb = _run(_step, order=2, rebalance=True, anderson_depth=1)
    np.testing.assert_allclose(reb.power, plain.power, rtol=0, atol=1e-8)


SLOW = Kinetics(velocities=[2.0e5, 3.0e3], beta=[0.0065], decay=[0.08])
EPS = 2e-3


def _p2_against_point_kinetics(dt, accel="auto"):
    """Max relative deviation of the P2 power from exact point kinetics with
    the moments and the scalar-flux generation time, uniform nuSigma_f step."""
    from scipy.integrate import solve_ivp

    from garnet.tri_sn import sn_effective_kinetics
    up = _mat("fuel*", [0.5, 1.2], FUEL_S0, FUEL_S1, [0.008 * (1 + EPS), 0.20 * (1 + EPS)])
    kw = dict(n_polar=2, n_azi=8, scheme="step", scattering_order=2, engine="levels")
    fs = TriSNTransportSolver(GRID, BASE, MMAP, bc="vacuum", **kw)
    ad = TriSNTransportSolver(GRID, BASE, MMAP, bc="vacuum", adjoint=True, **kw)
    rf, ra = fs.solve(**STEADY), ad.solve(**STEADY)
    beta, lam, rho = float(SLOW.beta[0]), float(SLOW.decay[0]), EPS / (1 + EPS)
    s = TransientSNSolver(GRID, lambda t: (BASE if t <= 0 else [up, REFL, ABS], MMAP), SLOW,
                          bc="vacuum", engine_cls=TriSNTransportSolver,
                          step_acceleration=accel, **kw)
    r = s.solve(0.4, dt, tol_step=1e-11, time_scheme="bdf2", steady_kwargs=STEADY)
    n = r.power / (1 + EPS)
    out = []
    for angular in (True, False):
        L = sn_effective_kinetics(fs, rf, ad, ra, SLOW, angular=angular).generation_time
        f = lambda t, y: [((rho - beta) / L) * y[0] + lam * y[1], (beta / L) * y[0] - lam * y[1]]
        ref = solve_ivp(f, (0, r.times[-1]), [1.0, beta / (L * lam)], method="Radau",
                        t_eval=r.times, rtol=1e-12, atol=1e-14).y[0]
        out.append(float(np.abs(n[1:] / ref[1:] - 1).max()))
    return out


def test_p2_transient_follows_exact_point_kinetics():
    moments, scalar = _p2_against_point_kinetics(0.01)
    assert moments < 2e-3                                    # 1.27e-3
    assert moments < scalar                                  # 1.82e-3


def test_broken_isotropic_time_source_misses_point_kinetics(monkeypatch):
    write = TriSNTransportSolver._t_time_source

    def isotropic(self, g, psi_old):
        write(self, g, np.repeat((self.w @ psi_old / self.w.sum())[None], self.M, axis=0))

    monkeypatch.setattr(TriSNTransportSolver, "_t_time_source", isotropic)
    assert _p2_against_point_kinetics(0.01, accel="none")[0] > 0.1   # 1.17


def _accelerated(order, **kw):
    s = TransientSNSolver(GRID, _step, KIN, bc="vacuum", engine_cls=TriSNTransportSolver,
                          n_polar=2, n_azi=8, scheme="scb", engine="levels",
                          scattering_order=order, **kw)
    return s.solve(0.05, 0.01, tol_step=1e-10, steady_kwargs=STEADY)


@pytest.mark.parametrize("order", [0, 2])
def test_transient_cmfd_reaches_the_same_steps_in_fewer_sweeps(order):
    plain = _accelerated(order, step_acceleration="none")
    cmfd = _accelerated(order, step_acceleration="cmfd")
    np.testing.assert_allclose(cmfd.power, plain.power, rtol=0, atol=2e-9)
    assert max(cmfd.step_iterations) <= 10 < min(plain.step_iterations)


def test_broken_cmfd_without_the_time_source_does_not_converge(monkeypatch):
    fit = TriSNTransportSolver.t_cmfd_factor
    write = TriSNTransportSolver._t_time_source

    def without_time_source(self, g, src, psi_old, state):
        monkeypatch.setattr(TriSNTransportSolver, "_t_time_source",
                            lambda self_, g_, p_: write(self_, g_, 0.0 * p_))
        try:
            return fit(self, g, src, psi_old, state)
        finally:
            monkeypatch.setattr(TriSNTransportSolver, "_t_time_source", write)

    monkeypatch.setattr(TriSNTransportSolver, "t_cmfd_factor", without_time_source)
    with pytest.raises(RuntimeError, match="did not converge"):
        _accelerated(0, step_acceleration="cmfd")
