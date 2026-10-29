"""IQS accuracy against a fine-step full spatial transient.

The other quasi-static tests use a one-group homogeneous slab, where the
fission-normalized shape barely moves. Here a two-group heterogeneous core
receives a localized thermal-absorption decrease (power rises about threefold
in Cartesian geometry), so the spatial shape changes during the prompt jump and
the shape-time treatment controls the error.

With the predictor-corrector IQS the point kinetics over each macro interval
use rho, Lambda and beta interpolated between the start and end shapes, and the
interval is re-marched with that correction. Measured maximum relative power
errors against full diffusion at the same fine dt=0.25 ms (over 20 ms;
the test uses the first 10 ms, which contains the prompt excursion):

    geometry  shape_dt   IQS        adiabatic   (explicit-shape IQS before)
    x-z       1.0 ms     ~2.0e-3    ~1.0e-2     ~9.0e-3
    x-z       0.5 ms     ~4.8e-4    ~6.8e-3     ~4.7e-3
    r-z       1.0 ms     ~7.9e-4    ~3.4e-3
    r-z       0.5 ms     ~2.1e-4    ~2.0e-3

The explicit-shape scheme converged only first order in shape_dt (ratio ~1.9
per halving) and was barely better than adiabatic; the error ratio per halving
is now about four. With shape_dt=dt the corrector is one full backward-Euler
step, so IQS reproduces the full transient up to its iteration tolerances.
"""

import numpy as np
import pytest

from garnet import (DiffusionEigenSolver, Grid, Kinetics, Material,
                    PWR_TWO_GROUP, ThermalFeedback, ThermalMaterial,
                    quasistatic_coupled_transient)
from garnet.coupling import CouplingContext, coupled_transient


BASE = PWR_TWO_GROUP
KIN = Kinetics(velocities=[1e7, 2.2e5], beta=[0.002, 0.004],
               decay=[0.08, 1.0], chi_delayed=[0.8, 0.2])
INSERTED = Material(
    name="local thermal-absorption decrease", diffusion=BASE.diffusion,
    sigma_a=BASE.sigma_a * np.array([1.0, 0.985]),
    nu_sigma_f=BASE.nu_sigma_f, sigma_s=BASE.sigma_s, chi=BASE.chi)
DT = 2.5e-4
T_END = 0.01


def grid_for(geometry):
    if geometry == "rz":
        return Grid(shape=(8, 1, 6), size=(60.0, 1.0, 60.0),
                    geometry="cylindrical")
    return Grid(shape=(8, 1, 6), size=(60.0, 1.0, 60.0))


def context(grid):
    thermal = [ThermalMaterial(
        conductivity=0.2, sink_coeff=0.04, sink_temperature=600.0,
        heat_capacity=2.0, name=f"fuel {i}") for i in range(2)]
    return CouplingContext(
        grid=grid, materials=[BASE, BASE],
        material_map=np.zeros(grid.shape, np.int32),
        thermal_materials=thermal,
        feedback=ThermalFeedback(t_ref=[600.0, 600.0],
                                 doppler=[[1e-5, 0.0], [1e-5, 0.0]]),
        total_power=120.0, bc="reflective", thermal_bc="adiabatic",
        device="cpu", kinetics=KIN, solver_cls=DiffusionEigenSolver,
        eigen_kwargs={"tol_k": 1e-11, "tol_source": 1e-11})


def problem(grid):
    base_map = np.zeros(grid.shape, np.int32)
    inserted_map = base_map.copy()
    inserted_map[:3] = 1
    states = (([BASE, BASE], base_map), ([BASE, INSERTED], inserted_map))
    return lambda t: states[0] if t <= 0.0 else states[1]


@pytest.mark.parametrize("geometry, coarse_bound, fine_bound",
                         [("xz", 4e-3, 1e-3), ("rz", 1.6e-3, 5e-4)])
def test_iqs_converges_faster_than_first_order_in_shape_dt(
        geometry, coarse_bound, fine_bound):
    grid = grid_for(geometry)
    options = dict(t_end=T_END, dt=DT, dt_thermal=DT,
                   problem_at=problem(grid))
    full = coupled_transient(context(grid), **options)
    assert full.power[-1] > 1.3   # a real prompt excursion

    def error(result):
        return np.max(np.abs(result.power / full.power - 1.0))

    coarse = quasistatic_coupled_transient(
        context(grid), shape_dt=4 * DT, shape_method="iqs", **options)
    fine = quasistatic_coupled_transient(
        context(grid), shape_dt=2 * DT, shape_method="iqs", **options)
    adiabatic = quasistatic_coupled_transient(
        context(grid), shape_dt=4 * DT, shape_method="adiabatic", **options)

    # Bounds are about twice the measured errors (module docstring).
    assert error(coarse) < coarse_bound, error(coarse)
    assert error(fine) < fine_bound, error(fine)
    # First order would give a ratio of two per halving; measured ~3.8-4.1.
    assert error(coarse) / error(fine) > 2.8, (error(coarse), error(fine))
    assert error(coarse) < 0.3 * error(adiabatic), (
        error(coarse), error(adiabatic))
    # Every accepted interval was re-marched with the end-shape correction
    # and converged without rejection.
    for result in (coarse, fine):
        counters = result.counters
        assert counters["iqs_corrector_iterations"] >= 1
        assert counters["iqs_unconverged_accepts"] == 0
        assert counters["iqs_interval_rejections"] == 0
        assert counters["iqs_shape_solves"] == counters["shape_updates"]


def test_iqs_with_shape_dt_equal_to_dt_reproduces_full_transient():
    """One corrector step is one full backward-Euler step: with consistent
    end-shape kinetics the projected amplitude must reproduce it."""
    grid = grid_for("xz")
    options = dict(t_end=4 * 2e-3, dt=2e-3, dt_thermal=2e-3,
                   problem_at=problem(grid))
    tight = {"tol_step": 1e-11}
    full = coupled_transient(context(grid), transient_kwargs=tight,
                             **options)
    iqs = quasistatic_coupled_transient(
        context(grid), shape_dt=2e-3, shape_method="iqs", iqs_kwargs=tight,
        **options)
    assert full.power[-1] > 1.5
    np.testing.assert_allclose(iqs.power, full.power, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(iqs.predictor_disagreement, 0.0, atol=1e-7)
