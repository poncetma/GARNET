"""Diffusion FTM against analytic modes, BE and conservative thermal coupling."""

import numpy as np
import pytest

from garnet import DiffusionEigenSolver, Grid, Kinetics, Material, TransientSolver
from garnet.timescheme import FrequencyTransform, TimeScheme, make_time_scheme


def _uniform(alpha=1.0, beta=0.0):
    grid = Grid(shape=(1, 1, 1), size=(1.0, 1.0, 1.0))
    velocity = 2.2e5
    base = Material(diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    changed = base.replace(sigma_a=base.sigma_a - alpha/velocity)
    initial, final = [base], [changed]
    problem = lambda t: (initial if t <= 0.0 else final, None)
    kinetics = Kinetics(velocities=[velocity], beta=[beta], decay=[0.08])
    return grid, initial, problem, kinetics


def _solve(alpha=1.0, beta=0.0, **kwargs):
    grid, _, problem, kinetics = _uniform(alpha, beta)
    return TransientSolver(grid, problem, kinetics, bc="reflective", device="cpu").solve(
        t_end=kwargs.pop("t_end", 0.3), dt=kwargs.pop("dt", 0.05),
        tol_step=1e-11, step_solver=kwargs.pop("step_solver", "monolithic"),
        multigroup_kwargs=kwargs.pop("multigroup_kwargs", {"rtol": 1e-12}),
        max_sweeps=2000, **kwargs)


@pytest.mark.parametrize("alpha", [-1.0, 1.0])
def test_ftm_exact_prompt_diffusion_mode_and_source_quadrature(alpha):
    result = _solve(alpha, time_scheme="ftm", ftm_kwargs={"frequency": alpha})
    np.testing.assert_allclose(result.power, np.exp(alpha*result.times), rtol=2e-10)
    np.testing.assert_allclose(result.step_frequencies, alpha)
    energy = np.dot(result.power[1:], result.step_source_weights)
    assert energy == pytest.approx(np.expm1(alpha*result.times[-1])/alpha, rel=2e-10)


def test_ftm_estimates_only_from_accepted_population_and_previous_width():
    widths = [0.02, 0.04, 0.01, 0.03]
    result = _solve(0.2, t_end=0.1, dt=widths, time_scheme="ftm")
    assert result.step_frequencies[0] == 0.0
    np.testing.assert_allclose(
        result.step_frequencies[1:],
        np.log(result.power[1:-1]/result.power[:-2])/np.asarray(widths[:-1]),
        rtol=1e-8, atol=1e-9)


def test_ftm_restart_discards_population_frequency():
    result = _solve(0.2, t_end=0.1, dt=[0.02, 0.04, 0.01, 0.03],
                    time_scheme="ftm", bdf_restart_times=[0.06])
    assert result.step_frequencies[1] > 0.0
    assert result.step_frequencies[2] == 0.0


@pytest.mark.parametrize("geometry", ["cylindrical", "triangular", "prismatic"])
def test_ftm_endpoint_source_matches_geometry_weights(geometry):
    from garnet.tri import TriDiffusionEigenSolver, TriGrid, TriGroupOperator

    _, initial, problem, kin = _uniform()
    if geometry == "cylindrical":
        grid = Grid((5, 1, 3), (10.0, 1.0, 6.0), geometry="cylindrical")
        extra = {}
    else:
        grid = TriGrid((3, 3, 2) if geometry == "triangular" else (3, 3, 2, 3),
                       side=5.0, height=12.0)
        extra = {"group_operator": TriGroupOperator, "eig_solver": TriDiffusionEigenSolver}
    result = TransientSolver(grid, problem, kin, bc="reflective", device="cpu", **extra).solve(
        t_end=0.2, dt=0.05, time_scheme="ftm", ftm_kwargs={"frequency": 1.0},
        step_solver="monolithic", multigroup_kwargs={"rtol": 1e-12},
        steady_kwargs={"tol_k": 1e-13, "tol_source": 1e-12, "inner_rtol_floor": 1e-14})
    np.testing.assert_allclose(result.power, np.exp(result.times), rtol=2e-9)


def test_unsafe_frequency_falls_back_to_actual_be_including_precursors():
    be = _solve(beta=0.0065)
    ftm = _solve(beta=0.0065, time_scheme="ftm", ftm_kwargs={"frequency": 100.0})
    np.testing.assert_allclose(ftm.power, be.power, rtol=3e-11)
    np.testing.assert_allclose(ftm.precursors, be.precursors, rtol=3e-11)
    assert ftm.ftm_fallback_reasons == ["frequency-bound"] * 6
    np.testing.assert_allclose(ftm.step_source_weights, 0.05)


def test_ftm_groupwise_and_monolithic_solve_identical_step_equations():
    common = dict(alpha=0.3, beta=0.0065, t_end=0.006, dt=0.002, time_scheme="ftm")
    block = _solve(**common)
    groups = _solve(**common, step_solver="fixed-point", multigroup_kwargs=None)
    np.testing.assert_allclose(groups.power, block.power, rtol=3e-8)
    np.testing.assert_allclose(groups.precursors, block.precursors, rtol=3e-8)


@pytest.mark.parametrize("scheme", ["ftm", "ftm2"])
def test_frequency_transform_multigroup_kinetics_agree_between_spatial_solvers(scheme):
    grid = Grid((6, 4, 1), (60.0, 40.0, 8.0))
    fuel = Material(diffusion=[1.4, 0.4], sigma_a=[0.01, 0.15],
                    nu_sigma_f=[0.007, 0.2], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
    materials = [fuel, fuel.replace(sigma_a=[0.012, 0.14])]
    changed = [fuel.replace(sigma_a=[0.01, 0.1499]), materials[1]]
    mapping = np.zeros(grid.shape, dtype=int)
    mapping[3:] = 1
    kin = Kinetics(velocities=[[1e7, 2e5], [0.9e7, 2.1e5]],
                   beta=[[0.005, 0.0025], [0.006, 0.002]], decay=[0.08, 0.4],
                   chi_delayed=[[0.95, 0.05], [0.85, 0.15]])
    steady = DiffusionEigenSolver(grid, materials, mapping, device="cpu").solve(
        tol_k=1e-11, tol_source=1e-10)
    problem = lambda t: (materials if t <= 0 else changed, mapping)
    common = dict(t_end=0.02, dt=0.005, time_scheme=scheme, initial_steady=steady,
                  tol_step=1e-11, max_sweeps=2000)
    groups = TransientSolver(grid, problem, kin, device="cpu").solve(**common)
    block = TransientSolver(grid, problem, kin, device="cpu").solve(
        **common, step_solver="monolithic", multigroup_kwargs={"rtol": 1e-12})
    np.testing.assert_allclose(groups.power, block.power, rtol=3e-8)
    np.testing.assert_allclose(groups.precursors, block.precursors, rtol=3e-8)


@pytest.mark.parametrize("scheme", ["ftm", "ftm2"])
def test_frequency_transform_preserves_multigroup_equilibrium(scheme):
    grid = Grid(shape=(8, 6, 1), size=(80.0, 60.0, 8.0))
    fuel = Material(diffusion=[1.4, 0.4], sigma_a=[0.01, 0.15],
                    nu_sigma_f=[0.007, 0.2], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
    materials = [fuel, fuel.replace(sigma_a=[0.012, 0.14])]
    mapping = np.zeros(grid.shape, dtype=int)
    mapping[4:] = 1
    kin = Kinetics(velocities=np.tile([1e7, 2e5], (2, 1)),
                   beta=[[0.005, 0.0025], [0.006, 0.002]], decay=[0.08, 0.4],
                   chi_delayed=[[0.95, 0.05], [0.85, 0.15]])
    # A tight initial state: since audit O12 the steps are actually solved, and
    # from the default steady tolerance the power drifts 3e-7 to 5e-7 (5e-12
    # from this one). Before, the warm start was accepted untouched.
    result = TransientSolver(grid, lambda t: (materials, mapping), kin, device="cpu").solve(
        t_end=0.1, dt=0.02, time_scheme=scheme, tol_step=1e-9,
        steady_kwargs=dict(tol_k=1e-12, tol_source=1e-11))
    np.testing.assert_allclose(result.power, 1.0, atol=2e-7, rtol=0)


def test_delayed_ftm_converges_and_be_bdf_share_the_point_kinetics_reference():
    from scipy.linalg import expm

    beta, decay, generation = 0.0065, 0.08, 1/(2.2e5*0.03)
    matrix = np.array([[13.2-beta/generation, decay],
                       [beta/generation, -decay]])
    initial = np.array([1.0, beta/(generation*decay)])

    def error(result):
        exact = np.array([(expm(matrix*t) @ initial)[0] for t in result.times])
        return np.max(np.abs(result.power/exact-1))

    errors = [error(_solve(13.2, beta, t_end=0.2, dt=h, time_scheme="ftm"))
              for h in (0.01, 0.005, 0.0025)]
    assert errors[-1] < errors[0]/2.5
    be = _solve(13.2, beta, t_end=0.2, dt=0.001)
    bdf = _solve(13.2, beta, t_end=0.2, dt=1e-5, time_scheme="bdf5",
                 adaptive_bdf={"rtol": 1e-4, "max_dt": 0.01, "automatic_order": True})
    assert error(be) < 0.01
    assert error(bdf) < 0.01


def test_ftm_adaptive_bdf_and_wrong_scheme_kwargs_are_refused():
    with pytest.raises(ValueError, match="requires bdf"):
        _solve(time_scheme="ftm", adaptive_bdf={"rtol": 1e-3})
    with pytest.raises(ValueError, match="ftm_kwargs"):
        make_time_scheme("bdf2", ftm_kwargs={"frequency": 0.0})


def test_zero_frequency_is_exact_exponential_precursor_control_not_be():
    scheme = FrequencyTransform(frequency=0.0)
    scheme.start([np.array(1.0)])
    scheme.prepare_step(1, 2.0)
    args = (1, [[np.array(2.0)]], np.array(1.0), np.array(1.0), [0.0065], [0.08], 2.0)
    value = scheme.precursor_update(*args)[0]
    exact = 2*np.exp(-0.16) + 0.0065 * (-np.expm1(-0.16))/0.08
    assert value == pytest.approx(exact)
    assert not np.isclose(value, TimeScheme().precursor_update(*args)[0])


@pytest.mark.parametrize("widths", [0.1, [0.05, 0.15, 0.03, 0.17, 0.05, 0.15]])
def test_ftm_coupling_accumulates_exponential_heat_across_thermal_window(widths):
    from garnet.coupling import CouplingContext, coupled_transient
    from garnet.feedback import ThermalFeedback
    from garnet.thermal import ThermalMaterial

    grid, initial, problem, kin = _uniform()
    ctx = CouplingContext(
        grid, initial, np.zeros(grid.shape, dtype=int),
        [ThermalMaterial(1.0, sink_coeff=1.0, sink_temperature=300.0, heat_capacity=10.0)],
        ThermalFeedback(t_ref=[310.0], doppler=[0.0]), total_power=10.0,
        solver_cls=DiffusionEigenSolver, kinetics=kin, device="cpu")
    common = dict(t_end=0.6, dt=widths, dt_thermal=0.2, problem_at=problem,
                  transient_kwargs={"time_scheme": "ftm", "ftm_kwargs": {"frequency": 1.0},
                                    "step_solver": "monolithic", "tol_step": 1e-11,
                                    "multigroup_kwargs": {"rtol": 1e-12}})
    result = coupled_transient(ctx, **common)
    expected = 310.0
    for k in range(3):
        energy = 10.0*(np.exp((k+1)*0.2)-np.exp(k*0.2))
        expected = (expected + energy/10.0 + 0.2*0.1*300.0)/(1 + 0.2*0.1)
    np.testing.assert_allclose(result.temperature, expected, atol=2e-8)
    rectangles = coupled_transient(ctx, **common, power_quadrature="endpoint")
    assert float(np.max(rectangles.temperature - result.temperature)) > 1e-3
    with pytest.raises(ValueError, match="every thermal exchange"):
        coupled_transient(ctx, **{**common, "dt": [0.1, 0.3, 0.2]})
