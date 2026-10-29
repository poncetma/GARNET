"""Transient model workflows, accepted-step histories and physical r-z volumes."""

import numpy as np
import pytest

from garnet import (CylindricalModel, Grid, HexLattice, Kinetics, Material,
                    Model, ReactorResult, ThermalMaterial, TransientModelResult,
                    power_density)


FUEL = Material(name="fuel", diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035],
                kappa_fission=[0.1])
REFLECTOR = Material(name="reflector", diffusion=[1.0], sigma_a=[0.001], nu_sigma_f=[0.0])
KINETICS = Kinetics([2.2e5], [0.0065], [0.08])


def cartesian():
    return Model(80, 8).fill(FUEL).set_boundary("reflective").set_kinetics(KINETICS)


def cylindrical():
    return (CylindricalModel(radius=10, height=20, cells=(4, 4))
            .fill(FUEL).set_boundary("reflective").set_kinetics(KINETICS))


def triangular():
    return (HexLattice(pitch=20, refine=1).set_disk(0, FUEL)
            .set_boundary("reflective").set_kinetics(KINETICS).build())


@pytest.mark.parametrize("build", [cartesian, cylindrical, triangular])
def test_common_transient_result_and_accepted_step_observer(build):
    core = build()
    steps = []
    result = core.transient(0.06, 0.02, device="cpu", time_scheme="bdf3",
        on_step=lambda t, flux, p: steps.append((t, np.asarray(flux).copy(), float(p))))
    assert isinstance(result, TransientModelResult)
    assert isinstance(result.steady, ReactorResult)
    assert result.model is result.reactor is core
    assert result.raw.time_orders == result.time_orders == [1, 2, 3]
    assert result.n_steps == len(steps) == 3
    np.testing.assert_allclose(result.step_widths, 0.02, atol=1e-16)
    np.testing.assert_allclose(result.power, 1.0, atol=1e-9)
    np.testing.assert_array_equal(result.flux, result.flux_numpy)
    np.testing.assert_array_equal(result.steady.flux, result.steady.flux_numpy)
    np.testing.assert_allclose([s[0] for s in steps], result.times[1:])
    assert result.precursors_numpy.shape == (1,) + tuple(core.grid.shape)
    assert "time_orders" in dir(result) and "bdf3" in result.summary()
    assert result.power_at(0.03) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(result.power_at([0.0, 0.03, 0.06]), 1.0, atol=1e-9)
    for invalid in (-0.01, 0.07, np.nan):
        with pytest.raises(ValueError, match="recorded time span"):
            result.power_at(invalid)


@pytest.mark.parametrize("build", [cartesian, cylindrical, triangular])
def test_events_split_fixed_steps_and_restart_bdf(build):
    events = [0.045, 0.075]
    accepted = []
    result = build().transient(0.10, 0.03, device="cpu", time_scheme="bdf3",
        event_times=events, on_step=lambda t, _flux, _p: accepted.append(t))
    np.testing.assert_allclose(result.times, [0, 0.03, 0.045, 0.06, 0.075, 0.09, 0.1], atol=1e-16)
    np.testing.assert_allclose(accepted, result.times[1:])
    assert np.all(result.step_widths > 0) and np.max(result.step_widths) <= 0.03 + 1e-15
    for event in events:
        # History restarts on the step leaving the event (times[index]).
        index = int(np.flatnonzero(np.isclose(result.times, event))[0])
        assert result.time_orders[index] == 1
    np.testing.assert_allclose(result.power, 1.0, atol=1e-9)


@pytest.mark.parametrize("build", [cartesian, cylindrical, triangular])
def test_adaptive_bdf_observer_omits_rejections_and_hits_events(build):
    perturbed = FUEL.replace(sigma_a=FUEL.sigma_a - 0.00005)
    accepted = []
    result = build().transient(0.02, 0.008, device="cpu", time_scheme="bdf3",
        materials_at=lambda t: {} if t == 0 else {"fuel": perturbed},
        adaptive_bdf={"rtol": 1e-3, "min_dt": 1e-7, "max_dt": 0.01},
        event_times=[0.013], on_step=lambda t, _flux, _p: accepted.append(t))
    assert result.step_method == "monolithic"
    assert result.rejected_steps > 0
    assert len(result.local_errors) == result.n_steps == len(accepted)
    np.testing.assert_allclose(accepted, result.times[1:], atol=1e-15)
    assert np.any(np.isclose(result.times, 0.013, atol=1e-15))
    assert result.times[-1] == pytest.approx(0.02, abs=1e-15)
    assert result.final_power > 1.0
    assert max(result.local_errors) <= 1.0


@pytest.mark.parametrize("build", [cartesian, cylindrical, triangular])
def test_public_steady_reuse_checks_time_zero_physics(build):
    core = build()
    steady = core.steady(device="cpu", tol_k=1e-9, tol_source=1e-8)
    unchanged = core.with_materials({"fuel": FUEL.replace(name="same constants")})
    reused = unchanged.transient(0.02, 0.02, device="cpu", initial_steady=steady)
    assert reused.initial_state_reused
    wrong = FUEL.replace(sigma_a=FUEL.sigma_a * 1.01)
    with pytest.raises(ValueError, match="time-zero materials"):
        core.transient(0.02, 0.02, device="cpu", initial_steady=steady,
                       materials_at=lambda _t: {"fuel": wrong})
    adjoint = core.steady(adjoint=True, device="cpu")
    with pytest.raises(ValueError, match="forward diffusion"):
        core.transient(0.02, 0.02, device="cpu", initial_steady=adjoint)
    other = build()
    if hasattr(other, "set_boundary"):
        other.set_boundary("vacuum")
    else:
        other.mask_bc = "vacuum"
    with pytest.raises(ValueError, match="grid or boundary"):
        other.transient(0.02, 0.02, device="cpu", initial_steady=steady)


def test_cylindrical_regions_use_annular_volumes_and_physical_power():
    core = (CylindricalModel(radius=80, height=100, cells=(8, 10)).fill(FUEL)
            .add_region(REFLECTOR, r=(40, 80)).set_boundary("vacuum"))
    assert core.material_map.shape == (8, 10)
    assert core.grid.shape == (8, 1, 10)
    assert core.grid.cell_volumes.sum() == pytest.approx(np.pi * 80**2 * 100, rel=1e-14)
    np.testing.assert_allclose(core.radial_centers, np.arange(5, 80, 10))
    np.testing.assert_allclose(core.axial_centers, np.arange(5, 100, 10))
    result = core.steady(device="cpu")
    rows = {row.name: row for row in result.material_summary}
    assert rows["reflector"].volume_fraction == pytest.approx(0.75)
    assert rows["fuel"].fission_fraction == pytest.approx(1.0)
    q = result.power_density(2e6)
    assert np.sum(q * result.cell_volumes) == pytest.approx(2e6, rel=1e-12)
    expected_peaking = result.flux.max() / np.average(result.flux[0], weights=result.cell_volumes)
    assert result.peaking == pytest.approx(expected_peaking, rel=1e-13)


@pytest.mark.parametrize("build", [cartesian, cylindrical, triangular])
def test_public_reuse_rejects_untracked_field_hooks(build):
    core = build()
    steady = core.steady(device="cpu", solver_kwargs={"xs_update": lambda _fields: None})
    with pytest.raises(ValueError, match="untracked cross-section updates"):
        core.transient(0.02, 0.02, device="cpu", initial_steady=steady)
    history = core.transient(0.02, 0.02, device="cpu",
        solver_kwargs={"xs_update_at": lambda _t: (lambda _fields: None)})
    with pytest.raises(ValueError, match="untracked cross-section updates"):
        core.transient(0.02, 0.02, device="cpu", initial_steady=history.steady)
    plain = core.steady(device="cpu")
    with pytest.raises(ValueError, match="untracked cross-section updates"):
        core.transient(0.02, 0.02, device="cpu", initial_steady=plain,
            solver_kwargs={"xs_update_at": lambda _t: (lambda _fields: None)})


def test_cylindrical_from_grid_ignores_dummy_y_extent_in_volumes():
    grid = Grid((4, 1, 6), (10, 7, 30), geometry="cylindrical")
    core = Model.from_grid(grid, FUEL, bc="reflective", kinetics=KINETICS)
    assert isinstance(core, CylindricalModel) and core.grid is grid
    assert core.grid.cell_volumes.sum() == pytest.approx(np.pi * 10**2 * 30, rel=1e-14)
    q = power_density(np.ones((1,) + grid.shape), [FUEL],
                       total_power=100, cell_volume=grid.cell_volumes)
    assert np.sum(q * grid.cell_volumes) == pytest.approx(100.0, rel=1e-13)
    with pytest.raises(ValueError, match="cylindrical geometry"):
        CylindricalModel.from_grid(Grid((4, 1, 6), (10, 1, 30)), FUEL)


def test_from_grid_copies_maps_and_validates_indices():
    grid = Grid((4, 1, 1), (40, 1, 1))
    mmap = np.zeros(grid.shape, int)
    core = Model.from_grid(grid, [FUEL, REFLECTOR], mmap)
    mmap[...] = 1
    assert not core.material_map.any()
    with pytest.raises(ValueError, match="integer indices"):
        Model.from_grid(grid, FUEL, np.full(grid.shape, 0.1))
    with pytest.raises(ValueError, match="required"):
        Model.from_grid(grid, [FUEL, REFLECTOR])
    with pytest.raises(ValueError, match="grid shape"):
        Model.from_grid(grid, FUEL, np.zeros(4))


@pytest.mark.parametrize("events", [[0], [0.1], [-0.1], [np.nan], [[0.05]]])
def test_bad_event_times_fail_clearly(events):
    with pytest.raises(ValueError, match="event_times"):
        cartesian().transient(0.1, 0.01, event_times=events, device="cpu")


def test_event_and_callback_option_conflicts_are_explicit():
    core = cartesian()
    with pytest.raises(ValueError, match="not both"):
        core.transient(0.1, 0.01, event_times=[0.05], bdf_restart_times=[0.05])
    with pytest.raises(NotImplementedError, match="BDF"):
        core.transient(0.1, 0.01, event_times=[0.05], time_scheme="crank-nicolson")
    with pytest.raises(ValueError, match="only one"):
        core.transient(0.1, 0.01, materials_at=lambda _t: {}, state_at=lambda _t: core)
    with pytest.raises(TypeError, match="on_step must be callable"):
        core.transient(0.1, 0.01, on_step=42)


def test_explicit_widths_can_be_split_at_events():
    result = cartesian().transient(0.1, [0.04, 0.06], device="cpu",
                                   event_times=[0.025], time_scheme="bdf2")
    np.testing.assert_allclose(result.times, [0, 0.025, 0.04, 0.1], atol=1e-16)


def test_adaptive_defaults_keep_backward_euler_and_choose_a_supported_step_solver():
    result = cartesian().transient(0.04, 0.01, device="cpu",
        adaptive_bdf={"rtol": 1e-3, "max_dt": 0.02})
    assert result.time_scheme == "backward-euler"
    assert result.step_method == "monolithic"
    assert set(result.time_orders) == {1}
    np.testing.assert_allclose(result.power, 1.0, atol=1e-9)


def test_cylindrical_adaptive_coupling_and_quasistatics_preserve_hot_equilibrium():
    core = cylindrical().configure_thermal(
        {"fuel": ThermalMaterial(0.25, sink_coeff=0.02,
                                 sink_temperature=650, heat_capacity=2.5)}, total_power=100)
    result = core.coupled_transient(0.05, 0.01, dt_thermal=0.023, device="cpu",
        transient_kwargs={"adaptive_bdf": {"rtol": 1e-3, "max_dt": 0.02}})
    assert result.counters["neutron_monolithic_steps"] == result.steps
    assert np.any(np.isclose(result.times, 0.023))
    assert np.any(np.isclose(result.times, 0.046))
    np.testing.assert_allclose(result.power, 1.0, atol=1e-7)
    qs = core.quasistatic_transient(0.04, 0.02, shape_dt=0.04, dt_thermal=0.04,
                                    device="cpu", materials_at=lambda _t: {})
    np.testing.assert_allclose(qs.power, 1.0, atol=1e-7)


@pytest.mark.parametrize("build", [cartesian, cylindrical, triangular])
def test_coupled_material_callbacks_and_progress_use_the_public_api(build):
    core = build().configure_thermal(
        {"fuel": ThermalMaterial(0.25, sink_coeff=0.02,
                                 sink_temperature=650, heat_capacity=2.5)},
        total_power=100)
    hot = core.coupled_steady(device="cpu")
    assert hot.thermal.source_watts == pytest.approx(100.0, rel=1e-12)
    assert hot.thermal.balance_residual < 1e-9
    steps = []
    stationary = core.coupled_transient(0.06, 0.02, dt_thermal=0.04,
        initial_coupled_steady=hot, device="cpu", materials_at=lambda _t: {},
        on_step=lambda t, flux, p, temp: steps.append((t, float(p), np.asarray(temp).copy())))
    np.testing.assert_allclose(stationary.power, 1.0, atol=1e-7)
    np.testing.assert_allclose([s[0] for s in steps], stationary.times[1:])
    assert steps[-1][2].shape == tuple(core.grid.shape)
    np.testing.assert_allclose(steps[-1][2], stationary.temperature)
    poisoned = FUEL.replace(sigma_a=FUEL.sigma_a * 1.001)
    frame = core.with_materials({"fuel": poisoned})
    named = core.coupled_transient(0.04, 0.02, device="cpu", initial_coupled_steady=hot,
        materials_at=lambda t: {} if t == 0 else {"fuel": poisoned})
    framed = core.coupled_transient(0.04, 0.02, device="cpu", initial_coupled_steady=hot,
        state_at=lambda t: core if t == 0 else frame)
    np.testing.assert_allclose(named.power, framed.power, rtol=1e-12, atol=1e-14)
    assert named.power[-1] < 1.0
