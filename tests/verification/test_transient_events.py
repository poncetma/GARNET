"""Fixed-schedule events, restart placement and the accepted-step observer.

An event endpoint belongs to the old smooth segment: ``problem_at(t_e)`` is
the pre-event state and BDF history is discarded before the step *leaving*
the event, on fixed and adaptive schedules alike. Fixed grids snap a nearby
endpoint onto an event instead of creating a sliver step.
"""

import numpy as np
import pytest

from garnet import Grid, Kinetics, Material, Model
from garnet._model_support import transient_schedule
from garnet.transient import TransientSolver

FUEL = Material(name="fuel", diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
PERTURBED = FUEL.replace(sigma_a=FUEL.sigma_a - 0.00012)
KINETICS = Kinetics([2.2e5], [0.0065], [0.08])
GRID = Grid(shape=(2, 2, 1), size=(10.0, 10.0, 1.0))


def _core():
    return Model(20, 2).fill(FUEL).set_boundary("reflective").set_kinetics(KINETICS)


@pytest.mark.parametrize("offset", [1e-10, -1e-10, 0.0, 5e-6])
def test_event_near_a_grid_endpoint_is_snapped_not_split(offset):
    """An event at 0.1 + 1e-10 used to leave a 1e-10 s step beside 0.1."""
    event = 0.1 + offset
    widths, options = transient_schedule(0.2, 0.01, [event],
                                         dict(time_scheme="bdf2"))
    ends = np.cumsum(widths)
    assert len(widths) == 20
    assert widths.min() > 0.99 * 0.01
    assert np.any(ends == event)
    assert ends[-1] == pytest.approx(0.2, abs=1e-15)
    np.testing.assert_array_equal(options["bdf_restart_times"], [event])


def test_event_away_from_the_grid_still_splits_the_step():
    widths, _ = transient_schedule(0.2, 0.01, [0.105], dict(time_scheme="bdf2"))
    assert len(widths) == 21
    np.testing.assert_allclose(np.sort(widths)[:2], 0.005, rtol=1e-12)


def test_explicit_widths_snap_and_keep_t_end():
    widths, _ = transient_schedule(0.2, np.full(20, 0.01), [0.1 + 1e-10,
                                                            0.2 - 1e-12],
                                   dict(time_scheme="bdf2"))
    ends = np.cumsum(widths)
    assert ends[-1] == pytest.approx(0.2, abs=1e-15)
    assert np.any(ends == 0.1 + 1e-10)
    # t_end is never moved, so an event that close to it keeps its sliver.
    assert np.sum(widths < 1e-6) == 1


def test_fixed_event_restarts_the_step_leaving_the_event():
    """The step ending at t_e still sees BDF history; the next one is BDF1."""
    result = _core().transient(0.1, 0.01, device="cpu", time_scheme="bdf3",
        materials_at=lambda t: {} if t <= 0.05 else {"fuel": PERTURBED},
        event_times=[0.05])
    assert result.time_orders == [1, 2, 3, 3, 3, 1, 2, 3, 3, 3]
    np.testing.assert_allclose(result.power[:6], 1.0, atol=1e-9)
    assert result.power[-1] > 1.0


def test_sliver_event_keeps_bdf_accuracy():
    """Snapping removes the 1e-10 s step; the result matches the on-grid event."""
    def run(event):
        return _core().transient(
            0.1, 0.01, device="cpu", time_scheme="bdf2", tol_step=1e-10,
            materials_at=lambda t: {} if t <= event else {"fuel": PERTURBED},
            event_times=[event])
    on_grid, near = run(0.05), run(0.05 + 1e-10)
    assert len(near.times) == len(on_grid.times)
    assert near.power[-1] == pytest.approx(on_grid.power[-1], rel=1e-7)


@pytest.mark.parametrize("step_solver", ["fixed-point", "monolithic"])
def test_on_step_receives_a_retainable_snapshot(step_solver):
    """Callbacks may keep the flux: the solver no longer overwrites it."""
    kept, copied = [], []

    def on_step(_t, flux, _power):
        kept.append(flux)
        copied.append(np.array([np.asarray(f).copy() for f in flux]))

    result = TransientSolver(
        GRID, lambda t: ([FUEL] if t <= 0 else [PERTURBED], None), KINETICS,
        bc="reflective", device="cpu", on_step=on_step).solve(
            t_end=0.04, dt=0.01, time_scheme="bdf2", step_solver=step_solver)
    assert len(kept) == 4
    for snapshot, copy in zip(kept, copied):
        np.testing.assert_array_equal(np.array([np.asarray(f) for f in snapshot]),
                                      copy)
    assert not np.allclose(copied[0], copied[-1])
    np.testing.assert_allclose(copied[-1], result.flux, rtol=0, atol=0)
