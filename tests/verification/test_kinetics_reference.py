"""Diffusion space-time kinetics against exact infinite-medium references.

A homogeneous core with reflective boundaries on every face has a spatially
flat fundamental mode, and a uniform cross-section change keeps it flat: the
space-time diffusion equations collapse *exactly* onto the multigroup
infinite-medium kinetics ODEs (group fluxes plus precursor families). Those
ODEs are integrated here independently of garnet -- ``scipy.linalg.expm`` for
piecewise-constant data (steps) and a tight Radau solve for a ramp -- so the
comparison checks the whole transient stack (precursor elimination, delayed
spectra, critical adjustment, time-scheme coefficients, history restarts)
against something other than itself.

``res.power`` is the volume-integrated ``nu*Sigma_f*phi / k0`` normalised to
one at t = 0. The perturbations change absorption only, so ``res.power`` tracks
the reference production rate with no (1 + eps) bookkeeping.

The tests at the end reproduce audit findings; each failed for the stated
reason before its fix.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from garnet import Grid, Kinetics, Material, TransientSNSolver, TransientSolver

# Four cells are enough: the solution is flat, and the reflective problem has
# no spatial discretisation error at all.
GRID = Grid(shape=(2, 2, 1), size=(10.0, 10.0, 1.0))

D, SIGMA_A, NU_SIGMA_F = 1.3, 0.030, 0.035
V = 2.2e5
BETA = np.array([0.00025, 0.0016, 0.0014, 0.003, 0.0009, 0.0003])
LAM = np.array([0.0127, 0.0317, 0.115, 0.311, 1.4, 3.87])
KIN = Kinetics(velocities=[V], beta=BETA, decay=LAM)

T_END = 0.2
# Sigma_a -> Sigma_a - 1.2e-4 is rho = 1.2e-4 / 0.030 = 0.004 = 0.615 $.
STEP = 0.00012
# Ramp of the same total worth over the window: d(Sigma_a)/dt = -6e-4 1/s.
RAMP_RATE = 0.0006


def _one_group(delta_sigma_a=0.0):
    return Material(name="1g", diffusion=[D], sigma_a=[SIGMA_A - delta_sigma_a],
                    nu_sigma_f=[NU_SIGMA_F])


def _two_group(delta_sigma_a2=0.0):
    return Material(name="2g", diffusion=[1.4, 0.4],
                    sigma_a=[0.010, 0.08 - delta_sigma_a2],
                    nu_sigma_f=[0.007, 0.135],
                    sigma_s=[[0.0, 0.018], [0.0, 0.0]], chi=[0.98, 0.02])


def _in_scatter(material):
    S = material.sigma_s.T.copy()
    np.fill_diagonal(S, 0.0)  # self-scatter cancels against the removal term
    return S


def _initial_state(material, kinetics):
    """k_inf and the equilibrium [phi_g, C_i] with production rate one."""
    phi = np.linalg.solve(np.diag(material.removal) - _in_scatter(material),
                          material.chi)
    k_inf = float(material.nu_sigma_f @ phi)
    phi *= k_inf / float(material.nu_sigma_f @ phi)
    return k_inf, np.r_[phi, kinetics.beta / kinetics.decay]


def _kinetics_matrix(material, kinetics, k0):
    """d/dt [phi_g, C_i] = A [phi_g, C_i] for the infinite medium.

    Built directly from the textbook equations: Material.chi is the total
    spectrum, so the prompt spectrum is chi - sum_i beta_i chi_d,i.
    """
    G, n_fam = len(material.sigma_a), kinetics.n_families
    v, beta, lam = kinetics.velocities, kinetics.beta, kinetics.decay
    if kinetics.chi_delayed is None:
        chi_d = np.tile(material.chi, (n_fam, 1))
    else:
        chi_d = np.broadcast_to(kinetics.chi_delayed, (n_fam, G))
    chi_p = material.chi - beta @ chi_d
    production = material.nu_sigma_f / k0
    A = np.zeros((G + n_fam, G + n_fam))
    A[:G, :G] = v[:, None] * (-np.diag(material.removal) + _in_scatter(material)
                              + np.outer(chi_p, production))
    A[:G, G:] = v[:, None] * (chi_d.T * lam)
    A[G:, :G] = np.outer(beta, production)
    A[G:, G:] = -np.diag(lam)
    return A


def _step_reference(base, perturbed, kinetics, times):
    """Exact P(t) for base -> perturbed at t = 0+."""
    k0, x0 = _initial_state(base, kinetics)
    A = _kinetics_matrix(perturbed, kinetics, k0)
    G = len(base.sigma_a)
    return np.array([perturbed.nu_sigma_f @ (expm(A * t) @ x0)[:G] / k0
                     for t in times])


def _ramp_problem():
    cache = {}

    def problem_at(t):
        key = round(float(t), 15)
        if key not in cache:
            cache[key] = [_one_group(RAMP_RATE * max(float(t), 0.0))]
        return cache[key], None
    return problem_at


def _ramp_reference():
    k0, x0 = _initial_state(_one_group(), KIN)
    sol = solve_ivp(
        lambda t, x: _kinetics_matrix(_one_group(RAMP_RATE * t), KIN, k0) @ x,
        (0.0, T_END), x0, method="Radau", rtol=1e-12, atol=1e-14)
    return float(sol.y[0, -1] * NU_SIGMA_F / k0)


def _step_problem(base, perturbed):
    return lambda t: (([base] if t <= 0 else [perturbed]), None)


def _solve(problem_at, kinetics=KIN, t_end_override=None, **kwargs):
    t_end = T_END if t_end_override is None else t_end_override
    return TransientSolver(GRID, problem_at, kinetics, bc="reflective",
                           device="cpu").solve(t_end=t_end, tol_step=1e-12,
                                               **kwargs)


def _ramp_error(n_steps, **kwargs):
    result = _solve(_ramp_problem(), dt=T_END / n_steps, **kwargs)
    return abs(result.power[-1] / RAMP_REFERENCE - 1.0)


def _observed_order(errors):
    """Order from the two finest of a sequence of halvings."""
    return float(np.log2(errors[-2] / errors[-1]))


RAMP_REFERENCE = _ramp_reference()  # 1.8485070085738


# Measured relative errors (fixed schemes at dt = T/40 = 5 ms; adaptive at
# rtol=1e-4), as (at t_end, max over the trajectory):
#   backward Euler   1.32e-3, 1.33e-2      BDF2             4.9e-5, 5.9e-3
#   Crank-Nicolson   1.40e-3, 5.5e-2       adaptive BDF2    9.0e-5, 1.9e-4
#   adaptive BDF3    3.3e-5,  5.6e-5
# The fixed-step trajectory maxima sit in the prompt jump (t <= 30 ms, prompt
# period Lambda/(beta - rho) ~ 60 ms), which 5 ms steps resolve only
# coarsely; the adaptive controller refines there. Crank-Nicolson is first
# order *for a step at t = 0*: its first trapezoidal half evaluates
# problem_at(0), the pre-step material by construction (the steady state is
# computed from it). Its second order is checked on the smooth ramp below.
_ADAPTIVE = dict(dt=1e-3, step_solver="monolithic",
                 multigroup_kwargs={"rtol": 1e-13}, adaptive_bdf={"rtol": 1e-4})


@pytest.mark.parametrize("label,kwargs,end_tol,trajectory_tol", [
    ("backward-euler", dict(dt=T_END / 40, time_scheme="backward-euler"),
     2.5e-3, 2.5e-2),
    ("bdf2", dict(dt=T_END / 40, time_scheme="bdf2"), 1e-4, 1.2e-2),
    ("crank-nicolson", dict(dt=T_END / 40, time_scheme="crank-nicolson"),
     3e-3, 0.1),
    ("adaptive-bdf2", dict(_ADAPTIVE, time_scheme="bdf2"), 2e-4, 4e-4),
    ("adaptive-bdf3", dict(_ADAPTIVE, time_scheme="bdf3"), 7e-5, 1.2e-4),
])
def test_step_insertion_matches_exact_point_kinetics(label, kwargs, end_tol,
                                                     trajectory_tol):
    """0.615 $ step, six delayed families: trajectory vs expm."""
    result = _solve(_step_problem(_one_group(), _one_group(STEP)), **kwargs)
    assert result.times[-1] == pytest.approx(T_END, abs=1e-14)
    reference = _step_reference(_one_group(), _one_group(STEP), KIN,
                                result.times)
    assert reference[-1] == pytest.approx(2.2749664483, rel=1e-9)
    error = np.abs(result.power / reference - 1.0)
    assert error[-1] < end_tol, f"{label}: P(t_end) error {error[-1]:.3e}"
    assert error.max() < trajectory_tol, f"{label}: max error {error.max():.3e}"


# The three spectra give reference P(T) = 2.8518, 2.4449, 2.4275, so a solver
# that ignored chi_delayed would be off by ~15%. Measured BDF2 errors
# (dt = T/80) at t_end: 5.3e-7, 2.2e-7, 2.3e-7; max over t >= T/2: 4.0e-5,
# 1.5e-5, 1.4e-5. (The first steps cannot resolve the fast-group prompt
# response at v1 = 1e7 cm/s, ~1.6e-2 at t = 2.5 ms; L-stable BDF2 damps it.)
@pytest.mark.parametrize("chi_delayed", [
    None, [0.6, 0.4], [[0.7, 0.3], [0.5, 0.5]]],
    ids=["chi", "single-chi_d", "per-family-chi_d"])
def test_two_group_step_with_delayed_spectrum_matches_exact_kinetics(chi_delayed):
    kinetics = Kinetics(velocities=[1e7, 3e5], beta=[0.0025, 0.004],
                        decay=[0.03, 0.3], chi_delayed=chi_delayed)
    base, perturbed = _two_group(), _two_group(0.0004)
    result = _solve(_step_problem(base, perturbed), kinetics,
                    dt=T_END / 80, time_scheme="bdf2")
    assert result.k0 == pytest.approx(_initial_state(base, kinetics)[0],
                                      rel=1e-12)
    reference = _step_reference(base, perturbed, kinetics, result.times)
    assert result.power[-1] == pytest.approx(reference[-1], rel=1.5e-6)
    late = result.times >= 0.5 * T_END - 1e-12
    np.testing.assert_allclose(result.power[late], reference[late],
                               rtol=1e-4, atol=0)
    if chi_delayed is not None:
        plain = Kinetics(velocities=[1e7, 3e5], beta=[0.0025, 0.004],
                         decay=[0.03, 0.3])
        ignored = _step_reference(base, perturbed, plain, [T_END])[0]
        assert abs(ignored / reference[-1] - 1.0) > 0.1


# Measured ramp errors at T/40, T/80, T/160 and order from the last pair:
# backward Euler 3.36e-3, 1.68e-3, 8.40e-4 (1.00); BDF2 1.20e-4, 3.07e-5,
# 7.75e-6 (1.98); Crank-Nicolson 2.58e-5, 6.44e-6, 1.61e-6 (2.00).
@pytest.mark.parametrize("time_scheme,order", [
    ("backward-euler", 1.0), ("bdf2", 2.0), ("crank-nicolson", 2.0)])
def test_observed_temporal_order_on_smooth_ramp(time_scheme, order):
    errors = [_ramp_error(n, time_scheme=time_scheme) for n in (40, 80, 160)]
    assert errors[-1] < 1e-3
    assert _observed_order(errors) == pytest.approx(order, abs=0.2), errors


# --------------------------------------------------------------------------
# Audit findings, fixed: each test failed for the stated reason before.
# --------------------------------------------------------------------------

def test_adaptive_bdf_survives_fast_neutron_step_at_default_rtol():
    """Audit H12: the first step (no history) used a constant predictor, an
    O(h) defect, and rejected down to min_dt ('adaptive BDF error 2.5e+01
    exceeds one at min_dt = 1e-7 s'). The explicit-Euler startup predictor
    resolves the ~5 us prompt jump instead: 53 accepted steps, 13 rejections
    (all on the first step), P(t_end) error 3.7e-5 with the default blockwise
    norm (49 steps and 1.8e-4 with error_norm='concatenated', audit O11)."""
    kinetics = Kinetics(velocities=[2.2e9], beta=BETA, decay=LAM)
    result = _solve(_step_problem(_one_group(), _one_group(STEP)), kinetics,
                    dt=1e-3, time_scheme="bdf2", step_solver="monolithic",
                    multigroup_kwargs={"rtol": 1e-13}, adaptive_bdf={})
    reference = _step_reference(_one_group(), _one_group(STEP), kinetics,
                                [T_END])[0]
    assert result.power[-1] == pytest.approx(reference, rel=1e-3)
    assert result.step_widths[0] < 1e-6        # the prompt jump is resolved
    assert len(result.step_widths) < 100


def test_adaptive_bdf_startup_defect_is_second_order():
    """The startup defect itself: halving the first width quarters it."""
    errors = []
    for width in (2e-3, 1e-3, 5e-4):
        result = _solve(_step_problem(_one_group(), _one_group(STEP)),
                        dt=width, t_end_override=width, time_scheme="bdf2",
                        step_solver="monolithic",
                        multigroup_kwargs={"rtol": 1e-13},
                        adaptive_bdf={"rtol": 1.0, "max_dt": width})
        errors.append(result.local_errors[0])
    assert _observed_order(errors) == pytest.approx(2.0, abs=0.15), errors


def _event_problem(base, first, second, t_event):
    def problem_at(t):
        if t <= 0:
            return [base], None
        return ([first] if t <= t_event * (1 + 1e-12) else [second]), None
    return problem_at


def test_fixed_bdf2_restart_at_interior_event_keeps_second_order():
    """Audit M-event: fixed steps restarted the step *ending* at the event,
    so BDF2 differenced across the kink: 8.12e-4, 4.17e-4, 2.10e-4 (order
    0.99). Restarting the step leaving it, as the adaptive path does, gives
    3.74e-5, 8.63e-6, 2.07e-6 (order 2.06)."""
    t_event, d1, d2 = 0.5 * T_END, STEP, -0.0002
    first, second, base = _one_group(d1), _one_group(d2), _one_group()
    problem_at = _event_problem(base, first, second, t_event)
    k0, x0 = _initial_state(base, KIN)
    x_event = expm(_kinetics_matrix(first, KIN, k0) * t_event) @ x0
    x_end = expm(_kinetics_matrix(second, KIN, k0) * (T_END - t_event)) @ x_event
    reference = x_end[0] * NU_SIGMA_F / k0
    errors = []
    for n in (80, 160, 320):
        result = _solve(problem_at, dt=T_END / n, time_scheme="bdf2",
                        bdf_restart_times=[t_event])
        errors.append(abs(result.power[-1] / reference - 1.0))
        orders = result.time_orders
        assert orders[n // 2 - 1] == 2 and orders[n // 2] == 1, orders
    assert _observed_order(errors) >= 1.8, errors
    assert errors[-1] < 4e-6, errors


def test_fixed_and_adaptive_restart_conventions_agree():
    """Replaying the adaptive widths on the fixed path reproduces it."""
    t_event = 0.5 * T_END
    problem_at = _event_problem(_one_group(), _one_group(STEP),
                                _one_group(-0.0002), t_event)
    adaptive = _solve(problem_at, dt=1e-3, time_scheme="bdf2",
                      step_solver="monolithic",
                      multigroup_kwargs={"rtol": 1e-13},
                      adaptive_bdf={"rtol": 1e-4}, bdf_restart_times=[t_event])
    fixed = _solve(problem_at, dt=np.asarray(adaptive.step_widths),
                   time_scheme="bdf2", step_solver="monolithic",
                   multigroup_kwargs={"rtol": 1e-13},
                   bdf_restart_times=[t_event])
    assert fixed.time_orders == adaptive.time_orders
    assert fixed.power[-1] == pytest.approx(adaptive.power[-1], rel=1e-9)


def test_crank_nicolson_restart_keeps_second_order_on_ramp():
    """Audit M-CN-restart: the restart set Psi = u, dropping the (1-w) F^n
    half of the trapezoid: 3.37e-4, 1.84e-4, 9.61e-5 (order 0.94). Re-seeded
    with F^n: 1.13e-4, 2.77e-5, 6.85e-6 (order 2.02)."""
    errors = [_ramp_error(n, time_scheme="crank-nicolson",
                          bdf_restart_times=[0.5 * T_END])
              for n in (40, 80, 160)]
    assert _observed_order(errors) >= 1.8, errors


def test_crank_nicolson_restart_is_second_order_across_a_step_event():
    """F^n is re-seeded from the post-event fields, so a restart makes CN
    second order across a genuine discontinuity (equilibrium until t_e):
    3.31e-5, 8.28e-6, 2.07e-6, against order 1.00 (2.09e-3 .. 5.21e-4)
    without the restart. Data continuous at t=0 avoid CN's documented
    first-order step at a t=0 perturbation."""
    t_event = 0.5 * T_END
    base, perturbed = _one_group(), _one_group(STEP)
    problem_at = _event_problem(base, base, perturbed, t_event)
    k0, x0 = _initial_state(base, KIN)
    x_end = expm(_kinetics_matrix(perturbed, KIN, k0) * (T_END - t_event)) @ x0
    reference = x_end[0] * NU_SIGMA_F / k0
    errors = []
    for n in (80, 160, 320):
        result = _solve(problem_at, dt=T_END / n, time_scheme="crank-nicolson",
                        bdf_restart_times=[t_event])
        errors.append(abs(result.power[-1] / reference - 1.0))
    assert _observed_order(errors) >= 1.8, errors


def _graded_widths(n_steps, order, ratio=1.5):
    """Startup widths from h*sqrt(h/T)**(order-1) growing to h, then uniform."""
    h = T_END / n_steps
    width, widths = h * (h / T_END) ** ((order - 1) / 2), []
    while width < h:
        widths.append(width)
        width *= ratio
    rest = T_END - sum(widths)
    count = int(np.ceil(rest / h))
    return np.array(widths + [rest / count] * count)


def test_fixed_bdf3_order_cap_and_graded_startup():
    """Audit M-BDF3, documented order cap (see timescheme.BDF): full-width
    BDF1/BDF2 startup steps make constant-step BDF3 second order (2.40e-5,
    5.39e-6, 1.27e-6; order 2.09). A schedule graded from h*sqrt(h/T) gives
    5.05e-6, 6.78e-7, 7.13e-8 (order 3.25)."""
    uniform = [_ramp_error(n, time_scheme="bdf3") for n in (40, 80, 160)]
    assert _observed_order(uniform) == pytest.approx(2.0, abs=0.25), uniform
    graded = []
    for n in (40, 80, 160):
        result = _solve(_ramp_problem(), dt=_graded_widths(n, 3),
                        time_scheme="bdf3")
        graded.append(abs(result.power[-1] / RAMP_REFERENCE - 1.0))
    assert _observed_order(graded) >= 2.7, graded
    assert graded[-1] < 0.1 * uniform[-1]


def test_sn_bdf3_runs_to_completion():
    """Audit H13: the S_N driver kept only [C^n, C^{n-1}] and raised
    'insufficient BDF history for requested order' at step 3."""
    slab = Grid(shape=(20, 1, 1), size=(500.0, 1e5, 1.0))
    kinetics = Kinetics(velocities=[2.2e5], beta=[0.0065], decay=[0.08])
    base = _one_group()
    perturbed = Material(name="p", diffusion=[D], sigma_a=[SIGMA_A],
                         nu_sigma_f=[NU_SIGMA_F * 1.001])
    powers = {}
    for scheme in ("bdf2", "bdf3", "bdf6"):
        result = TransientSNSolver(
            slab, _step_problem(base, perturbed), kinetics, n_polar=2,
            n_azi=4, acceleration="dsa").solve(t_end=0.008, dt=0.001,
                                               time_scheme=scheme)
        assert len(result.power) == 9
        assert np.all(np.isfinite(result.power))
        assert np.all(np.diff(result.power) > 0.0)
        powers[scheme] = result.power
    # Identical BDF1/BDF2 startup, then close higher-order continuations.
    np.testing.assert_array_equal(powers["bdf3"][:3], powers["bdf2"][:3])
    np.testing.assert_allclose(powers["bdf6"], powers["bdf3"], rtol=1e-5)


# Audit O12: near criticality the change between fission-source sweeps
# understates the amplitude error by 1 / (1 - k_prompt), ~100x here, and the
# first sweep's group solves ran at 1e-6 of the group source. A slow step could
# therefore pass on its first sweep with the warm start untouched, at any
# tol_step. At the default tol_step the 0.002 $ Anderson run was 9.6e-5 off
# the tight solve and the 0.02 $ rebalanced-Picard run 3.7e-4 off, at
# 100-200 sweeps a step; both are now within 1.4e-11.
@pytest.mark.parametrize("dollars", [0.002, 0.02])
@pytest.mark.parametrize("outer", [{}, dict(rebalance=True, anderson_depth=1)],
                         ids=["anderson", "rebalanced-picard"])
def test_slow_steps_are_solved_not_accepted_unchanged(dollars, outer):
    problem = _step_problem(_one_group(), _one_group(dollars * BETA.sum() * SIGMA_A))
    power = {tol: TransientSolver(GRID, problem, KIN, bc="reflective",
                                  device="cpu").solve(
                 t_end=10.0, dt=0.05, tol_step=tol, time_scheme="backward-euler",
                 **outer).power
             for tol in (1e-6, 1e-12)}
    assert np.all(np.diff(power[1e-6]) > 0.0)   # every step moves
    assert np.abs(power[1e-6] / power[1e-12] - 1.0).max() < 1e-5
