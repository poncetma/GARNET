"""Distinct delayed spectra against an independently assembled kinetic system."""

import numpy as np
import pytest

from garnet import Grid, Kinetics, Material, Model, TransientSolver


@pytest.mark.parametrize("step_solver", ["fixed-point", "monolithic"])
@pytest.mark.parametrize("time_scheme", ["backward-euler", "bdf2"])
@pytest.mark.parametrize("per_family", [False, True])
def test_delayed_spectra_match_full_neutron_precursor_system(
        step_solver, time_scheme, per_family):
    """A perturbed two-group medium must reproduce the full coupled equations.

    Reflective single-cell geometry removes leakage. The reference retains
    both neutron groups and both precursor families in one dense system; it
    does not eliminate precursors or use the solver's spectrum helpers.
    """
    base = Material(diffusion=[1.2, 0.4], sigma_a=[0.02, 0.08],
                    nu_sigma_f=[0.01, 0.12], chi=[0.8, 0.2],
                    sigma_s=[[0.0, 0.025], [0.002, 0.0]])
    perturbed = Material(diffusion=base.diffusion, sigma_a=[0.019, 0.079],
                         nu_sigma_f=base.nu_sigma_f, chi=base.chi,
                         sigma_s=base.sigma_s)
    spectra = np.array([[0.25, 0.75], [0.6, 0.4]])
    supplied_spectra = spectra if per_family else spectra[0]
    kinetics = Kinetics(velocities=[100.0, 20.0], beta=[0.02, 0.01],
                        decay=[0.1, 0.8], chi_delayed=supplied_spectra)
    core = Model(size=(1, 1, 1), cells=(1, 1, 1)).fill(base)
    core.set_boundary("reflective").set_kinetics(kinetics)
    steady = core.steady(device="cpu", tol_k=1e-12, tol_source=1e-12)
    initial_flux = steady.flux.reshape(2) * steady.k_eff
    initial_precursors = kinetics.beta / kinetics.decay
    initial_state = np.r_[initial_flux, initial_precursors]

    family_spectra = (spectra if per_family
                      else np.tile(supplied_spectra, (2, 1)))
    prompt_weights = base.chi - kinetics.beta @ family_spectra
    matrix = np.zeros((4, 4))
    matrix[:2, :2] = (
        -np.diag(perturbed.removal) + perturbed.sigma_s.T
        + np.outer(prompt_weights, base.nu_sigma_f) / steady.k_eff)
    matrix[:2, 2:] = family_spectra.T * kinetics.decay
    matrix[:2] *= kinetics.velocities[:, None]
    matrix[2:, :2] = np.outer(kinetics.beta, base.nu_sigma_f) / steady.k_eff
    matrix[2:, 2:] = -np.diag(kinetics.decay)

    width, steps = 0.01, 8
    history = [initial_state]
    reference_power = [1.0]
    for step in range(steps):
        if time_scheme == "bdf2" and step:
            reference = np.linalg.solve(
                1.5 * np.eye(4) - width * matrix,
                2.0 * history[-1] - 0.5 * history[-2])
        else:
            reference = np.linalg.solve(np.eye(4) - width * matrix, history[-1])
        history.append(reference)
        reference_power.append(base.nu_sigma_f @ reference[:2] / steady.k_eff)

    result = core.transient(
        at=lambda time: ([base if time == 0 else perturbed], None),
        device="cpu", t_end=steps * width, dt=width, initial_steady=steady,
        step_solver=step_solver, time_scheme=time_scheme, tol_step=1e-11,
        multigroup_kwargs={"rtol": 1e-12})
    np.testing.assert_allclose(result.power, reference_power, rtol=2e-9, atol=1e-11)
    np.testing.assert_allclose(result.flux_numpy.reshape(2), history[-1][:2],
                               rtol=2e-9, atol=1e-11)
    np.testing.assert_allclose(result.raw.precursors.reshape(2), history[-1][2:],
                               rtol=2e-9, atol=1e-11)

    # The perturbation exercises a non-equilibrium inventory and spectral shift.
    assert abs(result.power[-1] - 1.0) > 1e-3


def test_low_level_diffusion_rejects_wrong_delayed_group_count():
    material = Material(diffusion=[1.0, 0.4], sigma_a=[0.02, 0.08],
                        nu_sigma_f=[0.01, 0.1])
    kinetics = Kinetics([100.0, 20.0], [0.006], [0.08], chi_delayed=[1.0])
    solver = TransientSolver(
        Grid(shape=(1, 1, 1), size=(1, 1, 1)),
        lambda _time: ([material], None), kinetics, bc="reflective", device="cpu")
    with pytest.raises(ValueError, match="chi_delayed.*per group"):
        solver.solve(t_end=0.01, dt=0.01)
