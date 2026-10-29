"""FTM2 precursor production controls: linear transformed versus physical.

The quadratic physical option follows the PARCS v2.6 precursor integration
(physical fission source through three accepted endpoints), at the same
transformed-theta neutron equation and frequency samples as the default.
"""

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from garnet import DiffusionEigenSolver, Grid, Kinetics, Material, TransientSolver
from garnet.timescheme import ExponentialTheta, make_time_scheme
from tests.verification.test_exponential_adaptive import GENERATION, FUEL, point_core
from tests.verification.test_exponential_damping import core
from tests.verification.test_exponential_theta import _solve

PRODUCTIONS = ExponentialTheta.precursor_productions


def _two_group_mapped():
    grid = Grid((6, 4, 1), (60.0, 40.0, 8.0))
    fuel = Material(diffusion=[1.4, 0.4], sigma_a=[0.01, 0.15],
                    nu_sigma_f=[0.007, 0.2], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
    materials = [fuel, fuel.replace(sigma_a=[0.012, 0.14])]
    mapping = np.zeros(grid.shape, dtype=int)
    mapping[3:] = 1
    kinetics = Kinetics(velocities=[[1e7, 2e5], [0.9e7, 2.1e5]],
                        beta=[[0.005, 0.0025], [0.006, 0.002]], decay=[0.08, 0.4],
                        chi_delayed=[[0.95, 0.05], [0.85, 0.15]])
    return grid, materials, mapping, kinetics, fuel


def test_quadratic_update_integrates_three_mapped_production_products_exactly():
    beta0 = np.array([0.0065, 0.004])
    scheme = ExponentialTheta(frequency=0.7, precursor_production="quadratic-physical")
    scheme.start([np.array([1.0, 1.0])])
    scheme.seed_derivative([np.array([0.0, 0.0])], [beta0, 2*beta0])
    beta0[:] = 0.9  # Accepted history must be an independent snapshot.
    decay = [0.4, 3.0]
    nodes = [(0.0, np.array([1.2, 0.8]), [np.array([0.0065, 0.004]), np.array([0.013, 0.008])]),
             (0.07, np.array([1.5, 0.9]), [np.array([0.007, 0.003]), np.array([0.012, 0.009])]),
             (0.27, np.array([1.1, 1.4]), [np.array([0.008, 0.005]), np.array([0.011, 0.01])])]
    C = [np.array([0.08, 0.05]), np.array([0.02, 0.03])]
    # First step: no older node, so production is linear in physical time.
    scheme.prepare_step(1, 0.07)
    assert scheme.production_interpolation == "linear-physical"
    C = scheme.precursor_update(1, [C], nodes[1][1], nodes[0][1], nodes[1][2], decay, 0.07)
    scheme.push([np.array([1.0, 1.0])])
    scheme.prepare_step(2, 0.2)
    assert scheme.production_interpolation == "quadratic-physical"
    decayed = scheme.precursor_decayed(2, [C], nodes[1][1], nodes[2][2], decay, 0.2)
    removed = scheme.precursor_bcoef(2, [1.0, 1.0], decay, 0.2)
    actual = scheme.precursor_update(2, [C], nodes[2][1], nodes[1][1], nodes[2][2], decay, 0.2)
    times = np.array([t for t, _, _ in nodes])
    for family, lam in enumerate(decay):
        for cell in range(2):
            products = [b[family][cell]*S[cell] for _, S, b in nodes]
            poly = np.polyfit(times, products, 2)
            integral = quad(lambda t: np.exp(-lam*(0.27-t))*np.polyval(poly, t), 0.07, 0.27,
                            epsabs=0, epsrel=2e-14)[0]
            expected = np.exp(-lam*0.2)*C[family][cell] + integral
            assert actual[family][cell] == pytest.approx(expected, rel=5e-14)
            # The eliminated form splits the same update into decayed history
            # and the end-of-step production retained by the fission weight.
            new_weight = (1 - removed[family])/lam
            assert decayed[family][cell]/lam + new_weight*products[2] == pytest.approx(
                actual[family][cell], rel=2e-14)


@pytest.mark.parametrize("frequency", [0.0, None])
def test_zero_frequency_linear_physical_is_the_linear_transformed_control(frequency):
    runs = {mode: _solve(dt=0.01, frequency=0.0 if frequency == 0.0 else None, beta=0.0065,
                         precursor_fraction=0.7, ftm_kwargs={"precursor_production": mode})
            for mode in ("linear-transformed", "linear-physical")}
    a, b = runs["linear-transformed"], runs["linear-physical"]
    if frequency == 0.0:
        np.testing.assert_allclose(b.power, a.power, rtol=1e-13)
        np.testing.assert_allclose(b.precursors, a.precursors, rtol=1e-13)
    else:
        # A lagged frequency changes only the transformed production frame.
        assert not np.allclose(b.precursors, a.precursors, rtol=1e-13, atol=0)
    assert b.step_precursor_productions == ["linear-physical"] * 20
    assert a.step_precursor_productions == ["linear-transformed"] * 20


@pytest.mark.parametrize("production", PRODUCTIONS)
@pytest.mark.parametrize("frequency", [None, 0.0, 0.5])
def test_nonequilibrium_point_kinetics_is_second_order(production, frequency):
    from scipy.linalg import expm
    beta, decay, generation = 0.0065, 0.08, 1/6600
    matrix = np.array([[-beta/generation, decay], [beta/generation, -decay]])
    initial = np.array([1.0, 0.7*beta/(generation*decay)])
    errors = []
    for width in (0.005, 0.0025, 0.00125):
        result = _solve(dt=width, frequency=frequency, beta=beta, precursor_fraction=0.7,
                        ftm_kwargs={"precursor_production": production})
        exact = np.array([expm(matrix*t) @ initial for t in result.times])
        errors.append(max(np.max(np.abs(result.power/exact[:, 0]-1)),
                          abs(result.precursors.ravel()[0]/(exact[-1, 1]*generation)-1)))
    assert errors[0]/errors[1] > 3.8, errors
    assert errors[1]/errors[2] > 3.8, errors


@pytest.mark.parametrize("production", PRODUCTIONS)
def test_two_family_changing_fission_against_independent_ode(production):
    grid = Grid((1, 1, 1), (1.0, 1.0, 1.0))
    absorption, fission = np.array([0.02, 0.1]), np.array([0.005, 0.15])
    scatter = np.array([[0.0, 0.03], [0.002, 0.0]])
    spectrum, velocities = np.array([0.9, 0.1]), np.array([200.0, 60.0])
    beta, decay = np.array([0.005, 0.002]), np.array([0.08, 3.0])
    delayed = np.array([[0.95, 0.05], [0.8, 0.2]])
    base = Material(diffusion=[1.4, 0.4], sigma_a=absorption, nu_sigma_f=fission,
                    sigma_s=scatter, chi=spectrum)

    def coefficients(t):
        return (absorption*(1-np.array([0.04, 0.02])*np.sin(8*t)),
                fission*(1+np.array([0.02, -0.01])*t))

    def problem(t):
        sigma_a, nu_f = coefficients(t)
        return [base.replace(sigma_a=sigma_a, nu_sigma_f=nu_f)], None

    steady = DiffusionEigenSolver(grid, [base], bc="reflective", device="cpu").solve(
        tol_k=1e-13, tol_source=1e-12)
    phi0 = steady.flux[:, 0, 0, 0]
    phi0 = phi0/(np.dot(fission, phi0)/steady.k_eff)
    prompt = spectrum - beta @ delayed

    def rhs(t, state):
        sigma_a, nu_f = coefficients(t)
        phi, precursors = state[:2], state[2:]
        source = np.dot(nu_f, phi)/steady.k_eff
        neutron = (-(sigma_a+scatter.sum(axis=1))*phi + scatter.T @ phi
                   + prompt*source + (decay*precursors) @ delayed)
        return np.r_[velocities*neutron, beta*source - decay*precursors]

    reference = solve_ivp(rhs, (0, 0.4), np.r_[phi0, beta/decay], method="Radau",
                          rtol=2e-12, atol=1e-13)
    assert reference.success
    kinetics = Kinetics(velocities=velocities, beta=beta, decay=decay, chi_delayed=delayed)
    errors = []
    for width in (0.04, 0.02, 0.01):
        result = TransientSolver(grid, problem, kinetics, bc="reflective", device="cpu").solve(
            t_end=0.4, dt=width, time_scheme="ftm2", initial_steady=steady,
            ftm_kwargs={"precursor_production": production}, step_solver="monolithic",
            multigroup_kwargs={"rtol": 1e-13}, tol_step=1e-12)
        actual = np.r_[result.flux.ravel(), result.precursors.ravel()]
        errors.append(np.max(np.abs(actual/reference.y[:, -1]-1)))
    assert errors[0]/errors[1] > 3.7, errors
    assert errors[1]/errors[2] > 3.7, errors


@pytest.mark.parametrize("production", ["linear-physical", "quadratic-physical"])
def test_mapped_kinetics_equilibrium_and_spatial_solver_agreement(production):
    grid, materials, mapping, kinetics, fuel = _two_group_mapped()
    steady = DiffusionEigenSolver(grid, materials, mapping, device="cpu").solve(
        tol_k=1e-13, tol_source=1e-12)
    widths = [0.02, 0.01, 0.03, 0.005, 0.02, 0.02]
    options = dict(t_end=0.105, dt=widths, time_scheme="ftm2", initial_steady=steady,
                   tol_step=1e-11, max_sweeps=2000,
                   ftm_kwargs={"precursor_production": production})
    still = TransientSolver(grid, lambda t: (materials, mapping), kinetics, device="cpu").solve(
        **options, step_solver="monolithic", multigroup_kwargs={"rtol": 1e-12})
    np.testing.assert_allclose(still.power, 1.0, atol=1e-9, rtol=0)
    changed = [fuel.replace(sigma_a=[0.01, 0.1499]), materials[1]]
    problem = lambda t: (materials if t <= 0 else changed, mapping)
    groups = TransientSolver(grid, problem, kinetics, device="cpu").solve(**options)
    block = TransientSolver(grid, problem, kinetics, device="cpu").solve(
        **options, step_solver="monolithic", multigroup_kwargs={"rtol": 1e-12})
    np.testing.assert_allclose(groups.power, block.power, rtol=3e-8)
    np.testing.assert_allclose(groups.precursors, block.precursors, rtol=3e-8)
    assert block.power[-1] > 1.0 + 1e-4


@pytest.mark.parametrize("startup,expected", [
    (0, (["linear-physical"] + ["quadratic-physical"] * 4) * 2),
    (1, (["backward-euler"] + ["quadratic-physical"] * 4) * 2),
])
def test_quadratic_history_starts_and_restarts_without_spanning_events(startup, expected):
    result = _solve(dt=0.02, frequency=0.7, beta=0.0065, restarts=[0.1],
                    ftm_kwargs={"precursor_production": "quadratic-physical",
                                "startup_be_steps": startup})
    # BE fallback steps still record physical production for the next node.
    assert result.step_precursor_productions == expected


@pytest.mark.parametrize("estimator", ["step-doubling", "history"])
def test_adaptive_rollback_replays_quadratic_production_histories(estimator):
    model = core("cartesian")
    changed = FUEL.replace(sigma_a=[0.03 * (1 - 0.002)])
    options = dict(device="cpu", time_scheme="ftm2", tol_step=1e-11,
                   step_solver="monolithic", multigroup_kwargs={"rtol": 1e-12},
                   ftm_kwargs={"precursor_production": "quadratic-physical"},
                   materials_at=lambda t: {"fuel": changed} if t > 0.02 else {})
    adaptive = model.transient(0.06, 0.01, adaptive_ftm={
        "rtol": 1e-5, "min_dt": 1e-11, "max_dt": 0.02, "estimator": estimator},
        event_times=[0.02], **options)
    assert adaptive.rejected_steps > 0
    for trials in adaptive.ftm_trial_diagnostics:
        assert all(t["precursor_production"] in ("linear-physical", "quadratic-physical")
                   for t in trials)
    if estimator == "step-doubling":
        # Replay ONLY the accepted half steps: leaked coarse or rejected
        # production nodes would change the quadratic precursor history.
        half_widths = np.repeat(np.diff(adaptive.times) / 2, 2)
        replay = model.transient(0.06, half_widths, event_times=[0.02], **options)
        np.testing.assert_allclose(adaptive.power, replay.power[::2], rtol=2e-9)
        np.testing.assert_allclose(adaptive.precursors_numpy, replay.precursors_numpy, rtol=2e-9)
        assert replay.step_precursor_productions[0] == "linear-physical"
        event = 2 * int(np.flatnonzero(adaptive.times == 0.02)[0])
        assert replay.step_precursor_productions[event] == "linear-physical"
        assert replay.step_precursor_productions[1] == "quadratic-physical"


@pytest.mark.parametrize("production", ["quadratic-physical", "linear-transformed"])
@pytest.mark.parametrize("estimator", ["step-doubling", "history"])
def test_adaptive_six_family_precursors_against_radau(production, estimator):
    beta = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
    decay = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
    initial = np.r_[1.0, 0.7 * beta / (GENERATION * decay)]

    def rhs(t, state):
        return np.r_[(0.001 * t / 0.05 - beta.sum()) / GENERATION * state[0]
                     + decay @ state[1:], beta / GENERATION * state[0] - decay * state[1:]]

    reference = solve_ivp(rhs, (0, 0.05), initial, method="Radau",
                          rtol=2e-12, atol=1e-13, dense_output=True)
    assert reference.success
    result = point_core(beta, decay).transient(0.05, 0.01, device="cpu", time_scheme="ftm2",
        adaptive_ftm={"rtol": 1e-6, "estimator": estimator, "min_dt": 1e-11},
        ftm_kwargs={"precursor_production": production}, tol_step=1e-12,
        multigroup_kwargs={"rtol": 1e-13},
        initial_precursors=(initial[1:] * GENERATION).reshape(6, 1, 1, 1),
        materials_at=lambda t: {"fuel": FUEL.replace(sigma_a=[0.03 * (1 - 0.001 * t / 0.05)])})
    expected = reference.sol(result.times)
    assert np.max(np.abs(result.power / expected[0] - 1)) < 1e-4
    np.testing.assert_allclose(result.precursors_numpy.ravel() / GENERATION,
                               expected[1:, -1], rtol=1e-4)


def test_invalid_precursor_production_is_refused():
    with pytest.raises(ValueError, match="precursor_production"):
        make_time_scheme("ftm2", ftm_kwargs={"precursor_production": "cubic"})
    for scheme in ("ftm", "ftm-bdf2"):
        with pytest.raises(TypeError, match="precursor_production"):
            make_time_scheme(scheme, ftm_kwargs={"precursor_production": "quadratic-physical"})
