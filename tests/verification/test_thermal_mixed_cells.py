"""Heat-pipe sink in volume-mixed cells.

The sink term in a cell blended from materials i with weights w_i is
sum_i w_i h_i (T - T_sink,i), so its source part is blend(h * T_sink), not
blend(h) * blend(T_sink). With no power and no conduction a cell must relax to
the h-weighted sink temperature sum w_i h_i T_sink,i / sum w_i h_i -- in
particular, a fuel cell half-blended with a sink-free reflector must still sit
at the fuel's heat-pipe temperature, not halfway to the reflector's unused
(zero) T_sink.
"""

import numpy as np
import pytest

from garnet.grid import Grid
from garnet.thermal import ConductionSolver, ThermalMaterial


def _one_cell(mats, weight):
    grid = Grid(shape=(1, 1, 1), size=(1.0, 1.0, 1.0))
    return ConductionSolver(grid, mats, np.zeros(grid.shape, dtype=int),
                            mix_material=np.ones(grid.shape, dtype=int),
                            mix_weight=np.full(grid.shape, weight), device="cpu")


def test_mixed_cell_without_power_relaxes_to_the_sink_temperature():
    fuel = ThermalMaterial(0.25, sink_coeff=0.02, sink_temperature=750.0)
    refl = ThermalMaterial(0.9)                     # no heat pipes, T_sink unset
    solver = _one_cell([fuel, refl], 0.5)
    res = solver.solve(np.zeros((1, 1, 1)))
    assert res.temperature_numpy.item() == pytest.approx(750.0, rel=1e-12)
    assert solver.t_sink.item() == pytest.approx(750.0, rel=1e-12)
    assert solver.h.item() == pytest.approx(0.01, rel=1e-14)


def test_two_heat_pipe_materials_reach_the_h_weighted_equilibrium():
    a = ThermalMaterial(0.25, sink_coeff=0.02, sink_temperature=750.0)
    b = ThermalMaterial(0.25, sink_coeff=0.06, sink_temperature=900.0)
    w = 0.3                                          # 70% a, 30% b
    expected = ((1 - w) * 0.02 * 750.0 + w * 0.06 * 900.0) / ((1 - w) * 0.02 + w * 0.06)
    solver = _one_cell([a, b], w)
    res = solver.solve(np.zeros((1, 1, 1)))
    assert res.temperature_numpy.item() == pytest.approx(expected, rel=1e-12)
    # With a source the cell sits q/h above that, and the balance closes.
    q = 3.0
    res = solver.solve(np.full((1, 1, 1), q))
    h = (1 - w) * 0.02 + w * 0.06
    assert res.temperature_numpy.item() == pytest.approx(expected + q / h, rel=1e-12)
    assert res.sink_watts == pytest.approx(res.source_watts, rel=1e-10)


def test_mixed_sink_in_a_conducting_grid_closes_the_energy_balance():
    grid = Grid(shape=(6, 1, 1), size=(6.0, 1.0, 1.0))
    mats = [ThermalMaterial(0.3, sink_coeff=0.02, sink_temperature=700.0),
            ThermalMaterial(0.5, sink_coeff=0.05, sink_temperature=850.0),
            ThermalMaterial(0.9)]
    mmap = np.array([0, 0, 1, 1, 2, 2]).reshape(grid.shape)
    mix_m = np.array([-1, 1, 2, -1, 0, -1]).reshape(grid.shape)
    mix_w = np.array([0.0, 0.4, 0.5, 0.0, 0.25, 0.0]).reshape(grid.shape)
    solver = ConductionSolver(grid, mats, mmap, mix_material=mix_m,
                              mix_weight=mix_w, device="cpu")
    # No power: every cell, sink or not, ends at a temperature between the two
    # sink temperatures, and the zero-power balance is exactly zero.
    res = solver.solve(np.zeros(grid.shape))
    T = res.temperature_numpy.ravel()
    assert np.all(T > 700.0 - 1e-9) and np.all(T < 850.0 + 1e-9)
    assert abs(res.sink_watts) < 1e-9
    res = solver.solve(np.linspace(1.0, 2.0, 6).reshape(grid.shape))
    assert res.balance_residual < 1e-10
