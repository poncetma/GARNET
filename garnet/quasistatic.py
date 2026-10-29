"""Adjoint projection and point-kinetics foundation for quasi-static solves.

The spatial flux is factored as ``phi(r, g, t) = A(t) psi(r, g, t)``.  This
module implements the inexpensive half of that split: project a diffusion
operator onto a forward/adjoint shape and march the resulting amplitude and
effective delayed-neutron precursors.  Adaptive spatial shape correction is
built on top of these primitives; keeping the primitives standalone makes
their normalization and kinetics testable against exact homogeneous limits.

Background: Devooght and Mund, "Generalized Quasi-Static Method for Nuclear
Reactor Space-Time Kinetics", Nuclear Science and Engineering 76 (1980),
10-17, doi:10.13182/NSE80-A19288. GARNET uses a single amplitude with
adjoint-projected kinetics and spatial precursor inventories.
"""

from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .backend import asnumpy, synchronize
from .materials import Kinetics
from .solver import DiffusionEigenSolver, Result


@dataclass(frozen=True)
class EffectiveKinetics:
    """Adjoint-projected point-kinetics parameters at one spatial state.

    ``rho`` is reactivity relative to the transient's time-zero critical
    adjustment ``k0``. ``generation_time`` is Lambda in seconds. ``beta`` and
    ``decay`` contain one value per delayed-neutron family.
    """

    rho: float
    generation_time: float
    beta: np.ndarray
    decay: np.ndarray
    time_importance: float
    fission_importance: float

    def __post_init__(self):
        beta = np.asarray(self.beta, dtype=float)
        decay = np.asarray(self.decay, dtype=float)
        if beta.ndim != 1 or decay.ndim != 1 or beta.shape != decay.shape:
            raise ValueError("effective beta and decay must be same-length vectors")
        if (not np.isfinite(self.rho)
                or not np.isfinite(self.generation_time)
                or self.generation_time <= 0.0):
            raise ValueError("rho must be finite and generation_time positive")
        if (np.any(~np.isfinite(beta)) or np.any(beta < 0.0)
                or np.any(~np.isfinite(decay)) or np.any(decay <= 0.0)):
            raise ValueError("effective delayed-neutron data are invalid")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "decay", decay)

    @property
    def beta_total(self) -> float:
        return float(self.beta.sum())


@dataclass
class PointKineticsResult:
    """Amplitude and effective precursor history from a point-kinetics march."""

    times: np.ndarray
    amplitude: np.ndarray
    precursors: np.ndarray
    rho: np.ndarray

    @property
    def power(self) -> np.ndarray:
        return self.amplitude


@dataclass
class QuasiStaticResult:
    """Coupled fixed-shape, adiabatic, or improved quasi-static result.

    ``rho`` records the reactivity projected from each current control and
    temperature operator. ``amplitude`` is the adjoint-weighted neutron
    population marched by point kinetics, while ``power`` includes the
    current shape's fission integral and is therefore the physical
    ``P(t)/P(0)``. They are identical for a fixed shape. IQS results
    additionally retain the accepted spatial precursor field,
    ``power_shape_factor``, and predictor-disagreement counters.
    """

    times: np.ndarray
    power: np.ndarray
    rho: np.ndarray
    peak_temperature: np.ndarray
    mean_temperature: np.ndarray
    temperature: np.ndarray
    flux: object
    adjoint_flux: object
    k0: float
    steady: object
    seconds: float
    initialization_seconds: float
    device: str
    steps: int
    amplitude: np.ndarray | None = None
    power_shape_factor: np.ndarray | None = None
    effective_precursors: np.ndarray | None = None
    spatial_precursors: object = None
    shape_update_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    shape_update_reasons: list = field(default_factory=list)
    shape_k_eff: np.ndarray = field(default_factory=lambda: np.empty(0))
    residual_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    shape_residual: np.ndarray = field(default_factory=lambda: np.empty(0))
    adjoint_residual_times: np.ndarray = field(
        default_factory=lambda: np.empty(0))
    adjoint_residual: np.ndarray = field(default_factory=lambda: np.empty(0))
    predictor_disagreement_times: np.ndarray = field(
        default_factory=lambda: np.empty(0))
    predictor_disagreement: np.ndarray = field(
        default_factory=lambda: np.empty(0))
    fallback_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    phase_seconds: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"QuasiStaticResult(P/P0 {self.power.min():.4f} .. "
                f"{self.power.max():.4f}, peak T "
                f"{self.peak_temperature.max():.1f} K, {self.steps} steps in "
                f"{self.seconds:.1f} s)")


# Compatibility name used by the Phase-1 public API.
FixedShapeQuasiStaticResult = QuasiStaticResult


def _flux_array(value, name: str, groups: int, shape: tuple[int, ...], xp,
                dtype):
    value = value.flux if isinstance(value, Result) else value
    out = xp.asarray(value, dtype=dtype)
    expected = (groups,) + shape
    if out.shape != expected:
        raise ValueError(f"{name} shape {out.shape} != {expected}")
    if not bool(xp.all(xp.isfinite(out))):
        raise ValueError(f"{name} must be finite")
    return out


def _require_scalar_flux(solver, role, caller):
    """Reject solvers whose group state is not a scalar flux.

    The projections below apply diffusion loss/fission operators to the
    group-first scalar flux a :class:`Result` exposes. The SP3/SPN/SDPN block
    solvers carry a moment-vector state instead, so applying their operators
    to a scalar flux would silently produce a meaningless projection.
    """
    if not isinstance(solver, DiffusionEigenSolver):
        raise TypeError(f"{caller} supports diffusion-family solvers only; "
                        f"{role} is {type(solver).__name__}")


def _source_weight(solver):
    """Weight matching the discrete operator's right-hand-side convention."""
    weight = getattr(solver, "_src_weight", None)
    return 1.0 if weight is None else weight


def _loss_apply(solver, flux, *, adjoint=False):
    """Apply the loss operator using the eigensolver's transpose convention."""
    weight = _source_weight(solver)
    out = []
    for g in range(solver.n_groups):
        apply_A, _ = solver._linear_components(g, adjoint)
        value = apply_A(flux[g])
        for gf in range(solver.n_groups):
            scatter = (solver.sigma_s[g][gf] if adjoint
                       else solver.sigma_s[gf][g])
            if gf != g and scatter is not None:
                value = value - weight * scatter * flux[gf]
        out.append(value)
    return out


def _fission_apply(solver, flux):
    """Apply the fission production operator, weighted like ``_loss_apply``:
    (F phi)_g = w chi_g sum_g' nuSigma_f,g' phi_g' (w = r-z rhs weight)."""
    weight = _source_weight(solver)
    production = solver.nu_sigma_f[0] * flux[0]
    for g in range(1, solver.n_groups):
        production = production + solver.nu_sigma_f[g] * flux[g]
    return [weight * solver.chi[g] * production for g in range(solver.n_groups)]


def _mapped_kinetics(solver, kinetics: Kinetics):
    """Map material-dependent speeds/beta through the solver's volume blend."""
    fields = solver.fields
    groups = solver.n_groups
    if kinetics.velocities.shape[-1] != groups:
        raise ValueError(f"kinetics needs {groups} group velocities")
    if kinetics.velocities.ndim == 2:
        inv_velocity = [fields.map_table(1.0 / kinetics.velocities[:, g])
                        for g in range(groups)]
    else:
        inv_velocity = [1.0 / float(kinetics.velocities[g])
                        for g in range(groups)]
    if kinetics.beta.ndim == 2:
        beta = [fields.map_table_fission_weighted(kinetics.beta[:, i])
                for i in range(kinetics.n_families)]
    else:
        beta = [float(kinetics.beta[i]) for i in range(kinetics.n_families)]
    return inv_velocity, beta


def _shape_time_importance(solver, forward, adjoint, kinetics: Kinetics) -> float:
    """Return ``<phi*, V^-1 psi>`` in the solver's discrete inner product.

    This is the normalization functional for the IQS amplitude.  Keeping its
    value continuous when a corrected shape replaces the old one is the
    finite-interval form of the adjoint-weighted shape-derivative term in the
    amplitude equation.
    """
    _require_scalar_flux(solver, "solver", "quasi-static projection")
    xp = solver.xp
    groups = solver.n_groups
    shape = tuple(solver.grid.shape)
    forward_flux = _flux_array(forward, "forward flux", groups, shape, xp, solver.dtype)
    adjoint_flux = _flux_array(adjoint, "adjoint flux", groups, shape, xp, solver.dtype)
    weight = _source_weight(solver)
    inv_velocity, _ = _mapped_kinetics(solver, kinetics)
    value = 0.0
    for g in range(groups):
        value = value + xp.sum(adjoint_flux[g] * weight * inv_velocity[g] * forward_flux[g])
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("forward/adjoint shapes have non-positive time importance")
    return value


def project_effective_kinetics(solver, forward, adjoint, kinetics: Kinetics,
                               *, k0: float | None = None) -> EffectiveKinetics:
    """Project the current diffusion operator onto a forward/adjoint shape.

    ``solver`` supplies the *current* material/control/temperature operator.
    ``forward`` and ``adjoint`` are the most recent shape anchor and may be
    either solver :class:`Result` objects or group-first arrays. ``k0`` is the
    physical eigenvalue used for the transient critical adjustment; when
    omitted it is read from a forward ``Result``.

    Reusing an anchor against a nearby operator gives the local Rayleigh
    reactivity estimate used between shape corrections. Large black-absorber
    moves must be split into small cached frames and periodically re-anchored.
    """
    _require_scalar_flux(solver, "solver", "project_effective_kinetics")
    if not isinstance(kinetics, Kinetics):
        raise TypeError("kinetics must be a Kinetics object")
    if k0 is None:
        if not isinstance(forward, Result):
            raise ValueError("k0 is required when forward is not a Result")
        k0 = float(forward.k_eff)
    k0 = float(k0)
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ValueError("k0 must be finite and positive")

    xp = solver.xp
    groups = solver.n_groups
    shape = tuple(solver.grid.shape)
    forward_flux = _flux_array(forward, "forward flux", groups, shape, xp, solver.dtype)
    adjoint_flux = _flux_array(adjoint, "adjoint flux", groups, shape, xp, solver.dtype)
    weight = _source_weight(solver)
    inv_velocity, beta_fields = _mapped_kinetics(solver, kinetics)

    fission_source = solver.nu_sigma_f[0] * forward_flux[0]
    for g in range(1, groups):
        fission_source = fission_source + solver.nu_sigma_f[g] * forward_flux[g]

    loss = _loss_apply(solver, forward_flux)
    loss_importance = 0.0
    fission_importance = 0.0
    time_importance = 0.0
    for g in range(groups):
        loss_importance = loss_importance + xp.sum(adjoint_flux[g] * loss[g])
        fission_importance = fission_importance + xp.sum(
            adjoint_flux[g] * weight * solver.chi[g] * fission_source)
        time_importance = time_importance + xp.sum(
            adjoint_flux[g] * weight * inv_velocity[g] * forward_flux[g])
    loss_importance = float(loss_importance)
    fission_importance = float(fission_importance)
    time_importance = float(time_importance)
    if fission_importance <= 0.0 or time_importance <= 0.0:
        raise ValueError("forward/adjoint shapes have non-positive importance")

    beta_eff = []
    chi_delayed = kinetics.chi_delayed
    for i, beta_i in enumerate(beta_fields):
        delayed_importance = 0.0
        for g in range(groups):
            if chi_delayed is None:
                spectrum = solver.chi[g]
            elif chi_delayed.ndim == 1:
                spectrum = float(chi_delayed[g])
            else:
                spectrum = float(chi_delayed[i, g])
            delayed_importance = delayed_importance + xp.sum(
                adjoint_flux[g] * weight * spectrum * beta_i * fission_source)
        beta_eff.append(float(delayed_importance) / fission_importance)

    rho = 1.0 - k0 * loss_importance / fission_importance
    generation_time = k0 * time_importance / fission_importance
    return EffectiveKinetics(
        rho=float(rho), generation_time=float(generation_time),
        beta=np.asarray(beta_eff), decay=kinetics.decay.copy(),
        time_importance=time_importance,
        fission_importance=fission_importance)


def projected_shape_residual(solver, forward, adjoint) -> float:
    """Relative eigen-shape defect with the amplitude mode projected out.

    The current operator is applied to ``forward`` and its fission component is
    scaled so the residual is orthogonal to ``adjoint``. A pure normalization
    or eigenvalue change therefore reports zero; only spatial/energy-shape
    mismatch remains. This is the trigger used by the quasi-static driver.
    """
    _require_scalar_flux(solver, "solver", "projected_shape_residual")
    xp = solver.xp
    groups = solver.n_groups
    shape = tuple(solver.grid.shape)
    forward_flux = _flux_array(forward, "forward flux", groups, shape, xp, solver.dtype)
    adjoint_flux = _flux_array(adjoint, "adjoint flux", groups, shape, xp, solver.dtype)
    loss = _loss_apply(solver, forward_flux)
    fission = _fission_apply(solver, forward_flux)
    loss_importance = sum(xp.sum(adjoint_flux[g] * loss[g]) for g in range(groups))
    fission_importance = sum(
        xp.sum(adjoint_flux[g] * fission[g]) for g in range(groups))
    fission_importance = float(fission_importance)
    if fission_importance <= 0.0:
        raise ValueError("forward/adjoint shapes have non-positive importance")
    alpha = float(loss_importance) / fission_importance
    residual_norm = 0.0
    reference_norm = 0.0
    for g in range(groups):
        residual = loss[g] - alpha * fission[g]
        residual_norm = residual_norm + xp.sum(residual * residual)
        reference_norm = reference_norm + xp.sum(loss[g] * loss[g])
    reference_norm = float(reference_norm)
    if reference_norm <= 0.0:
        raise ValueError("shape has a non-positive operator norm")
    return float(np.sqrt(float(residual_norm) / reference_norm))


def projected_adjoint_residual(solver, forward, adjoint) -> float:
    """Relative adjoint eigen-shape defect with its amplitude mode removed.

    This is the transposed counterpart of :func:`projected_shape_residual`.
    It measures whether a reused adjoint still satisfies the current operator,
    without treating arbitrary adjoint normalization or eigenvalue change as a
    shape error. The transpose follows the eigen solver's discrete convention:
    Cartesian ADF leakage uses its algebraic transpose, and scattering and
    fission exchange their energy-group production/emission roles.
    """
    _require_scalar_flux(solver, "solver", "projected_adjoint_residual")
    xp = solver.xp
    groups = solver.n_groups
    shape = tuple(solver.grid.shape)
    forward_flux = _flux_array(forward, "forward flux", groups, shape, xp, solver.dtype)
    adjoint_flux = _flux_array(adjoint, "adjoint flux", groups, shape, xp, solver.dtype)
    weight = _source_weight(solver)
    loss = _loss_apply(solver, adjoint_flux, adjoint=True)
    importance_source = solver.chi[0] * adjoint_flux[0]
    for g in range(1, groups):
        importance_source = importance_source + solver.chi[g] * adjoint_flux[g]
    fission = [weight * solver.nu_sigma_f[g] * importance_source
               for g in range(groups)]
    loss_importance = sum(xp.sum(forward_flux[g] * loss[g]) for g in range(groups))
    fission_importance = sum(
        xp.sum(forward_flux[g] * fission[g]) for g in range(groups))
    fission_importance = float(fission_importance)
    if fission_importance <= 0.0:
        raise ValueError("forward/adjoint shapes have non-positive importance")
    alpha = float(loss_importance) / fission_importance
    residual_norm = 0.0
    reference_norm = 0.0
    for g in range(groups):
        residual = loss[g] - alpha * fission[g]
        residual_norm = residual_norm + xp.sum(residual * residual)
        reference_norm = reference_norm + xp.sum(loss[g] * loss[g])
    reference_norm = float(reference_norm)
    if reference_norm <= 0.0:
        raise ValueError("adjoint shape has a non-positive operator norm")
    return float(np.sqrt(float(residual_norm) / reference_norm))


def _project_spatial_precursors(solver, adjoint, spatial_precursors,
                                kinetics, time_importance):
    """Adjoint-project spatial precursor fields into amplitude coordinates."""
    xp = solver.xp
    star = _flux_array(adjoint, "adjoint flux", solver.n_groups,
                       tuple(solver.grid.shape), xp, solver.dtype)
    fields = xp.asarray(spatial_precursors, dtype=solver.dtype)
    expected = (kinetics.n_families,) + tuple(solver.grid.shape)
    if fields.shape != expected:
        raise ValueError(f"spatial precursor shape {fields.shape} != {expected}")
    weight = _source_weight(solver)
    out = []
    for i in range(kinetics.n_families):
        importance = 0.0
        for g in range(solver.n_groups):
            if kinetics.chi_delayed is None:
                spectrum = solver.chi[g]
            elif kinetics.chi_delayed.ndim == 1:
                spectrum = float(kinetics.chi_delayed[g])
            else:
                spectrum = float(kinetics.chi_delayed[i, g])
            importance = importance + xp.sum(
                star[g] * weight * spectrum * fields[i])
        out.append(float(importance) / time_importance)
    projected = np.asarray(out)
    if np.any(~np.isfinite(projected)) or np.any(projected < 0.0):
        raise RuntimeError("spatial precursor projection is invalid")
    return projected


def _match_spatial_precursor_importance(solver, adjoint, candidate, target,
                                        kinetics, time_importance):
    """Rescale each candidate precursor field to an accepted projection.

    An IQS macro corrector supplies useful *spatial* delayed-source evolution,
    but its fields follow the corrector's independent coarse amplitude.  A
    family-wise positive rescaling retains that spatial distribution while
    making its adjoint projection exactly equal to the accepted fine-step
    precursor inventory.  Thus the corrector cannot reintroduce an amplitude
    or inventory jump.
    """
    xp = solver.xp
    fields = xp.asarray(candidate, dtype=solver.dtype)
    expected = (kinetics.n_families,) + tuple(solver.grid.shape)
    if fields.shape != expected:
        raise ValueError(f"candidate precursor shape {fields.shape} != {expected}")
    target = np.asarray(target, dtype=float)
    if (target.shape != (kinetics.n_families,)
            or np.any(~np.isfinite(target)) or np.any(target < 0.0)):
        raise ValueError("target precursor projection is invalid")
    projected = _project_spatial_precursors(
        solver, adjoint, fields, kinetics, time_importance)
    factors = np.ones_like(target)
    positive = projected > 0.0
    if np.any(~positive & (target > 0.0)):
        raise RuntimeError("IQS precursor shape has zero accepted importance")
    factors[positive] = target[positive] / projected[positive]
    reshape = (kinetics.n_families,) + (1,) * len(solver.grid.shape)
    matched = fields * xp.asarray(factors, dtype=solver.dtype).reshape(reshape)
    check = _project_spatial_precursors(
        solver, adjoint, matched, kinetics, time_importance)
    if not np.allclose(check, target, rtol=5e-12, atol=1e-14):
        raise RuntimeError("IQS precursor importance matching failed")
    return matched, factors


def equilibrium_precursors(parameters: EffectiveKinetics,
                           amplitude: float = 1.0) -> np.ndarray:
    """Effective precursor amplitudes for a critical steady population."""
    amplitude = float(amplitude)
    return (parameters.beta / (parameters.generation_time * parameters.decay)
            * amplitude)


def _interpolated_kinetics(start: EffectiveKinetics, delta, weight: float
                           ) -> EffectiveKinetics:
    """Add the fraction ``weight`` of an end-shape correction to ``start``.

    ``start`` is projected with the interval's start shape on the current
    operator and ``delta`` is ``(d_rho, d_Lambda, d_beta)``, the end-shape
    minus start-shape projection on the interval-end operator. Interpolating
    only this shape correction keeps every fine-step control and feedback
    change while the shape varies linearly between the two anchors.
    """
    if delta is None or weight == 0.0:
        return start
    d_rho, d_generation_time, d_beta = delta
    return replace(
        start, rho=start.rho + weight * d_rho,
        generation_time=start.generation_time + weight * d_generation_time,
        beta=start.beta + weight * d_beta)


def _step_count(t_end: float, dt: float) -> int:
    if not np.isfinite(t_end) or not np.isfinite(dt) or t_end < 0.0 or dt <= 0.0:
        raise ValueError("t_end must be non-negative and dt positive")
    steps = int(round(t_end / dt))
    if not np.isclose(steps * dt, t_end, rtol=1e-12,
                      atol=1e-14 * max(1.0, t_end)):
        raise ValueError("t_end must be an integer multiple of dt")
    return steps


def advance_point_kinetics(parameters: EffectiveKinetics, dt: float,
                           amplitude: float, precursors) -> tuple[float, np.ndarray]:
    """Advance one backward-Euler point-kinetics step.

    The state is deliberately host-resident: even with dozens of precursor
    families this is a tiny dense system, while the projected spatial fields
    and coupled temperature remain on the accelerator.
    """
    if not isinstance(parameters, EffectiveKinetics):
        raise TypeError("parameters must be EffectiveKinetics")
    dt = float(dt)
    amplitude = float(amplitude)
    precursors = np.asarray(precursors, dtype=float)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude must be finite and positive")
    if (precursors.shape != parameters.beta.shape
            or np.any(~np.isfinite(precursors))):
        raise ValueError("precursors must match the delayed-neutron families")

    families = len(parameters.beta)
    matrix = np.zeros((1 + families, 1 + families))
    matrix[0, 0] = ((parameters.rho - parameters.beta_total)
                    / parameters.generation_time)
    matrix[0, 1:] = parameters.decay
    matrix[1:, 0] = parameters.beta / parameters.generation_time
    matrix[1:, 1:] = -np.diag(parameters.decay)
    old = np.concatenate(([amplitude], precursors))
    state = np.linalg.solve(np.eye(len(old)) - dt * matrix, old)
    if np.any(~np.isfinite(state)) or state[0] <= 0.0:
        raise RuntimeError("point kinetics produced an invalid state")
    return float(state[0]), state[1:]


def integrate_point_kinetics(parameters_at: EffectiveKinetics | Callable,
                             t_end: float, dt: float, *, amplitude0: float = 1.0,
                             precursors0=None) -> PointKineticsResult:
    """Backward-Euler march of the projected amplitude/precursor equations.

    ``parameters_at`` is either one :class:`EffectiveKinetics` object or an
    end-of-step callable. Parameter changes do not reset precursors. This is
    intentionally the same first-order time convention as ``TransientSolver``
    so fixed-shape comparisons isolate the spatial approximation rather than a
    time-integrator difference.
    """
    steps = _step_count(t_end, dt)
    get = (parameters_at if callable(parameters_at)
           else lambda _t: parameters_at)
    p0 = get(0.0)
    if not isinstance(p0, EffectiveKinetics):
        raise TypeError("parameters_at must return EffectiveKinetics")
    amplitude0 = float(amplitude0)
    if not np.isfinite(amplitude0) or amplitude0 <= 0.0:
        raise ValueError("amplitude0 must be finite and positive")
    precursors = (equilibrium_precursors(p0, amplitude0) if precursors0 is None
                  else np.asarray(precursors0, dtype=float).copy())
    if precursors.shape != p0.beta.shape or np.any(~np.isfinite(precursors)):
        raise ValueError("precursors0 must match the delayed-neutron families")

    y = np.concatenate(([amplitude0], precursors))
    times = np.arange(steps + 1, dtype=float) * dt
    amplitude = [amplitude0]
    precursor_history = [precursors.copy()]
    rho_history = [p0.rho]
    for n in range(1, steps + 1):
        p = get(float(times[n]))
        if not isinstance(p, EffectiveKinetics):
            raise TypeError("parameters_at must return EffectiveKinetics")
        if p.beta.shape != p0.beta.shape or not np.array_equal(p.decay, p0.decay):
            raise ValueError("delayed family count/decay constants changed in time")
        next_amplitude, next_precursors = advance_point_kinetics(
            p, dt, y[0], y[1:])
        y = np.concatenate(([next_amplitude], next_precursors))
        amplitude.append(float(y[0]))
        precursor_history.append(y[1:].copy())
        rho_history.append(p.rho)
    return PointKineticsResult(
        times=times, amplitude=np.asarray(amplitude),
        precursors=np.asarray(precursor_history), rho=np.asarray(rho_history))


def _unpack_problem(spec, mix_material, mix_weight):
    if len(spec) == 2:
        materials, material_map = spec
        return materials, material_map, mix_material, mix_weight
    if len(spec) == 4:
        return tuple(spec)
    raise ValueError("problem_at must return (materials, material_map) or "
                     "(materials, material_map, mix_material, mix_weight), "
                     f"got {len(spec)} elements")


def _coupled_quasistatic(
        ctx, t_end, dt, *, problem_at=None, dt_thermal=None,
        initial_coupled=None, steady_kwargs=None, shape_kwargs=None,
        adjoint_kwargs=None,
        thermal_rtol=1e-8, thermal_maxiter=20000,
        thermal_check_every=4, thermal_precond_degree=0,
        thermal_diagnostics_every=0, profile=False, verbose=False,
        shape_dt=None,
        adjoint_every=1, shape_on_final=True, shape_method="adiabatic",
        adjoint_residual_tol=None,
        residual_tol=None, fallback_residual=None, iqs_kwargs=None,
        iqs_precond_degree=1, iqs_substeps=1,
        iqs_predictor_tol=None, iqs_iteration_tol=1e-4,
        iqs_max_iterations=4):
    """Shared fixed-shape, adiabatic, and IQS coupled implementation.

    An initial adjoint eigenvalue solve establishes the shape importance. At
    changed control or thermal-feedback states, the current operator is
    projected onto the anchor and the resulting point-kinetics amplitude
    drives the backward-Euler conduction solver. Adiabatic mode periodically
    replaces the anchor with an eigen shape; IQS mode instead advances the
    time-dependent spatial shape and precursor fields over each macro step.
    Optional projected-residual thresholds can shorten that macro step or
    replace a fine interval with the full spatial diffusion solve.

    IQS is a predictor-corrector iteration over each macro interval. The
    predictor marches the amplitude with start-shape kinetics; the corrector
    advances the time-dependent shape to the interval end, and the amplitude,
    precursors, power edits and thermal windows are then re-marched from the
    accepted start state with the end-shape correction to rho, Lambda and beta
    interpolated linearly in time. The end shape is re-solved when feedback
    changes the end operator. The interval is accepted once the end power of
    the re-projected march changes by at most ``iqs_iteration_tol``; it is
    rejected and retried with half the width after ``iqs_max_iterations``
    re-marches or when the corrector's own end power disagrees with the
    accepted one by more than ``iqs_predictor_tol``.

    ``problem_at(0)`` must describe the same physical core as ``ctx`` because
    the converged coupled state supplies the time-zero forward shape. Return
    cached material/map/blend objects between actual control changes so the
    driver can avoid redundant operator construction. ``initial_coupled`` can
    reuse a previously computed converged :class:`CoupledResult`.

    With ``shape_dt=None`` the time-zero spatial shape remains fixed. Otherwise
    ``shape_method`` selects either periodic adiabatic eigen re-anchoring or a
    time-dependent IQS corrector.
    """
    from .coupling import (CoupledResult, CoupledSolver, _CoupledPhaseProfiler,
                           _fuel_mask)
    from .thermal import ConductionSolver

    if ctx.kinetics is None:
        raise ValueError("coupled quasi-static transients require ctx.kinetics")
    steps = _step_count(float(t_end), float(dt))
    t_end, dt = float(t_end), float(dt)
    dt_thermal = float(dt if dt_thermal is None else dt_thermal)
    if not np.isfinite(dt_thermal) or dt_thermal <= 0.0:
        raise ValueError("dt_thermal must be finite and positive")
    every = int(round(dt_thermal / dt))
    if every < 1 or not np.isclose(every * dt, dt_thermal, rtol=1e-12,
                                   atol=1e-14 * max(1.0, dt_thermal)):
        raise ValueError("dt_thermal must be an integer multiple of dt")
    if shape_dt is None:
        shape_every = None
    else:
        shape_dt = float(shape_dt)
        if not np.isfinite(shape_dt) or shape_dt <= 0.0:
            raise ValueError("shape_dt must be finite and positive")
        shape_every = int(round(shape_dt / dt))
        if (shape_every < 1
                or not np.isclose(shape_every * dt, shape_dt, rtol=1e-12,
                                  atol=1e-14 * max(1.0, shape_dt))):
            raise ValueError("shape_dt must be an integer multiple of dt")
    adjoint_every = int(adjoint_every)
    if adjoint_every < 1:
        raise ValueError("adjoint_every must be positive")
    shape_on_final = bool(shape_on_final)
    shape_method = str(shape_method).lower()
    if shape_method not in ("adiabatic", "iqs"):
        raise ValueError("shape_method must be 'adiabatic' or 'iqs'")
    if shape_every is None and shape_method == "iqs":
        raise ValueError("IQS treatment requires shape_dt")
    residual_tol = (None if residual_tol is None else float(residual_tol))
    fallback_residual = (None if fallback_residual is None
                         else float(fallback_residual))
    adjoint_residual_tol = (None if adjoint_residual_tol is None
                            else float(adjoint_residual_tol))
    iqs_predictor_tol = (None if iqs_predictor_tol is None
                         else float(iqs_predictor_tol))
    for name, value in (("residual_tol", residual_tol),
                        ("fallback_residual", fallback_residual),
                        ("adjoint_residual_tol", adjoint_residual_tol),
                        ("iqs_predictor_tol", iqs_predictor_tol)):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if (residual_tol is not None and fallback_residual is not None
            and fallback_residual < residual_tol):
        raise ValueError("fallback_residual must be >= residual_tol")
    iqs_precond_degree = int(iqs_precond_degree)
    iqs_substeps = int(iqs_substeps)
    if iqs_precond_degree < 0:
        raise ValueError("iqs_precond_degree must be non-negative")
    if iqs_substeps < 1:
        raise ValueError("iqs_substeps must be positive")
    iqs_iteration_tol = float(iqs_iteration_tol)
    iqs_max_iterations = int(iqs_max_iterations)
    if not np.isfinite(iqs_iteration_tol) or iqs_iteration_tol <= 0.0:
        raise ValueError("iqs_iteration_tol must be finite and positive")
    if iqs_max_iterations < 1:
        raise ValueError("iqs_max_iterations must be positive")
    if (isinstance(ctx.solver_cls, type)
            and not issubclass(ctx.solver_cls, DiffusionEigenSolver)):
        raise TypeError("quasi-static transients support diffusion-family "
                        f"solvers only; ctx.solver_cls is "
                        f"{ctx.solver_cls.__name__}")
    thermal_maxiter = int(thermal_maxiter)
    thermal_check_every = int(thermal_check_every)
    thermal_precond_degree = int(thermal_precond_degree)
    thermal_diagnostics_every = int(thermal_diagnostics_every)
    if thermal_maxiter < 1 or thermal_check_every < 1:
        raise ValueError("thermal iteration controls must be positive")
    if thermal_precond_degree < 0 or thermal_diagnostics_every < 0:
        raise ValueError("thermal precondition/diagnostic controls must be non-negative")
    if not np.isfinite(thermal_rtol) or thermal_rtol <= 0.0:
        raise ValueError("thermal_rtol must be finite and positive")
    reserved = {"adjoint", "state0"} & (shape_kwargs or {}).keys()
    if reserved:
        raise ValueError("shape_kwargs cannot set "
                         + ", ".join(sorted(reserved)))

    xp = ctx.thermal_solver().xp
    profiler = _CoupledPhaseProfiler(xp, enabled=profile)
    counters = profiler.counters
    if initial_coupled is None:
        with profiler.region("initial_coupled_solve"):
            options = {"tol": 1e-8, "anderson_depth": 5}
            options.update(steady_kwargs or {})
            steady = CoupledSolver(ctx).solve(**options)
    else:
        if not isinstance(initial_coupled, CoupledResult):
            raise TypeError("initial_coupled must be a CoupledResult")
        steady = initial_coupled
        counters["initial_state_reuses"] += 1
    if not steady.converged:
        raise RuntimeError(f"coupled steady state did not converge: {steady}")

    synchronize(xp)
    initialization_start = time.perf_counter()

    stationary = lambda _t: (ctx.materials, ctx.material_map,
                              ctx.mix_material, ctx.mix_weight)
    get_problem = problem_at or stationary
    spec0 = _unpack_problem(get_problem(0.0), ctx.mix_material, ctx.mix_weight)
    temperature = xp.asarray(steady.temperature, dtype=ctx.dtype)
    feedback_hook = ctx.feedback.hook(temperature)

    def build_solver(spec, hook):
        mats, mmap, mix_m, mix_w = spec
        return ctx.solver_cls(
            ctx.grid, mats, mmap, bc=ctx.bc, active=ctx.active,
            mask_bc=ctx.mask_bc, mix_material=mix_m, mix_weight=mix_w,
            device=ctx.device, dtype=ctx.dtype, xs_update=hook)

    with profiler.region("operator_rebuild"):
        anchor_solver = build_solver(spec0, feedback_hook)
    counters["operator_rebuilds"] += 1
    _require_scalar_flux(anchor_solver, "ctx.solver_cls",
                         "quasi-static transients")
    adjoint_options = dict(ctx.eigen_kwargs)
    adjoint_options.update(adjoint_kwargs or {})
    reserved = {"adjoint", "state0"} & adjoint_options.keys()
    if reserved:
        raise ValueError("adjoint_kwargs cannot set "
                         + ", ".join(sorted(reserved)))
    with profiler.region("initial_adjoint_solve"):
        adjoint = anchor_solver.solve(adjoint=True, **adjoint_options)
    counters["adjoint_eigen_solves"] += 1
    if not adjoint.converged:
        raise RuntimeError(f"initial adjoint solve did not converge: {adjoint}")

    forward_flux = steady.neutronics.flux
    with profiler.region("reactivity_projection"):
        anchor_raw = project_effective_kinetics(
            anchor_solver, forward_flux, adjoint.flux, ctx.kinetics,
            k0=steady.k_eff)
    counters["reactivity_projections"] += 1
    # The final fixed-point temperature and the neutron result retained by the
    # coupled steady solver differ only by its convergence tolerance. Remove
    # that tiny projected mismatch so the supplied equilibrium is exactly the
    # zero-reactivity amplitude initial condition.
    rho_reference = anchor_raw.rho
    parameters = replace(anchor_raw, rho=0.0)
    amplitude = 1.0
    precursors = equilibrium_precursors(parameters, amplitude)

    metric = getattr(ctx.grid, "cylindrical_metrics", lambda: None)()
    volume_weight = (None if metric is None
                     else xp.asarray(metric[0], dtype=ctx.dtype))

    def normalized_neutron_shape(solver, flux):
        phi = xp.asarray(flux, dtype=ctx.dtype)
        source = solver.fields.fission_source([phi[g]
                                               for g in range(solver.n_groups)])
        source = source / steady.k_eff
        total = xp.sum(source if volume_weight is None
                       else source * volume_weight)
        total = float(total)
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("shape has no finite positive fission source")
        return phi / total, source / total, total

    forward_flux, initial_source_shape, _ = normalized_neutron_shape(
        anchor_solver, forward_flux)
    # The initial fission-normalized shape defines P/P0 = amplitude = 1.  At
    # later shape corrections ``power_shape_factor`` converts the continuously
    # marched adjoint-weighted amplitude back to fission power.  Updating this
    # factor by the old/new time-importance ratio is the integrated
    # <phi*, V^-1 dpsi/dt> term.
    power_shape_factor = 1.0
    _, initial_beta_fields = _mapped_kinetics(anchor_solver, ctx.kinetics)
    spatial_precursor_shape = xp.stack([
        initial_beta_fields[i] / ctx.kinetics.decay[i] * initial_source_shape
        for i in range(ctx.kinetics.n_families)])

    def time_dependent_corrector(start_spec, start_hook, end_spec, end_hook,
                                 width, start_flux, start_precursors,
                                 substeps=1, start_time=None):
        """One IQS macro solve, or a fine-step full-diffusion fallback."""
        from .transient import TransientSolver

        substeps = int(substeps)
        if substeps < 1:
            raise ValueError("corrector substeps must be positive")
        start_result = copy.copy(steady.neutronics)
        start_result.k_eff = steady.k_eff
        start_result.flux = xp.asarray(start_flux, dtype=ctx.dtype)

        def local_problem(t):
            if t <= 0.0:
                return start_spec
            if start_time is None or substeps == 1:
                return end_spec
            return _unpack_problem(
                get_problem(float(start_time) + float(t)),
                ctx.mix_material, ctx.mix_weight)

        def local_feedback(t):
            return start_hook if t <= 0.0 else end_hook

        corrector = TransientSolver(
            ctx.grid, local_problem, ctx.kinetics, bc=ctx.bc,
            active=ctx.active, mask_bc=ctx.mask_bc,
            mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
            group_operator=ctx.group_operator, eig_solver=ctx.solver_cls,
            device=ctx.device, dtype=ctx.dtype,
            precond_degree=iqs_precond_degree,
            xs_update_at=local_feedback)
        groups = ctx.materials[0].n_groups
        options = ({"rebalance": True, "anderson_depth": 1,
                    "max_sweeps": 4000, "scatter_subsweeps": 6}
                   if groups > 2 else {"rebalance": False})
        options["linsolve_kwargs"] = {"check_every": 4}
        options.update(iqs_kwargs or {})
        result = corrector.solve(
            t_end=width, dt=width / substeps,
            initial_steady=start_result,
            initial_precursors=start_precursors, **options)
        predicted = float(result.power[-1])
        if not np.isfinite(predicted) or predicted <= 0.0:
            raise RuntimeError("IQS corrector produced invalid amplitude")
        return (result, result.flux / predicted,
                result.precursors / predicted, predicted)

    def thermal_solver_for(width):
        return ConductionSolver(
            ctx.grid, ctx.thermal_materials, ctx.material_map,
            bc=ctx.thermal_bc, active=ctx.active,
            mask_bc=ctx.thermal_mask_bc,
            ambient_temperature=ctx.ambient_temperature,
            mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
            device=ctx.device, dtype=ctx.dtype, time_step=width,
            precond_degree=thermal_precond_degree)

    thermal = thermal_solver_for(dt_thermal)
    volume = thermal.cell_volume
    fuel = _fuel_mask(ctx)
    fuel_dev = xp.asarray(fuel)
    active_dev = (None if ctx.active is None
                  else xp.asarray(ctx.active).astype(bool))

    def normalized_power_shape(spec, flux):
        from .blend import MaterialBlend
        from .power import fission_energy_xs

        mats, mmap, mix_m, mix_w = spec
        table, _ = fission_energy_xs(mats)
        blend = MaterialBlend(
            xp, ctx.grid.shape, mmap, len(mats), dtype=thermal.dtype,
            mix_material=mix_m, mix_weight=mix_w)
        kappa_xs = xp.stack(
            [blend.linear(table[:, g]) for g in range(table.shape[1])])
        flux_stack = flux if hasattr(flux, "shape") else xp.stack(flux)
        raw = xp.sum(kappa_xs * flux_stack, axis=0)
        if active_dev is not None:
            raw = xp.where(active_dev, raw, 0.0)
        integral = float(xp.sum(raw * volume))
        if not np.isfinite(integral) or integral <= 0.0:
            raise RuntimeError("no finite positive fission power in the anchor flux")
        return raw / integral

    with profiler.region("power_shape"):
        unit_power_shape = normalized_power_shape(spec0, forward_flux)

    steady_temperature = np.asarray(steady.temperature)
    cached_peak = float(steady_temperature[fuel].max())
    cached_mean = float(steady_temperature[fuel].mean())
    times = np.arange(steps + 1, dtype=float) * dt
    powers = [amplitude]
    amplitudes = [amplitude]
    power_shape_factors = [power_shape_factor]
    rhos = [0.0]
    peaks = [cached_peak]
    means = [cached_mean]
    power_accum = None
    window_steps = 0
    shape_times = []
    shape_reasons = []
    shape_ks = [float(steady.k_eff)]
    shape_state = getattr(ctx, "_state", None)
    adjoint_state = anchor_solver.state
    shape_updates = 0
    residual_times = []
    residual_values = []
    adjoint_residual_times = []
    adjoint_residual_values = []
    predictor_disagreement_times = []
    predictor_disagreement_values = []
    fallback_times = []
    spatial_precursors = spatial_precursor_shape.copy()
    anchor_spec = spec0
    anchor_hook = feedback_hook
    anchor_flux = forward_flux
    anchor_precursors = spatial_precursor_shape
    anchor_time = 0.0
    anchor_amplitude = amplitude
    anchor_power_shape_factor = power_shape_factor
    adaptive_shape_every = shape_every
    next_shape_step = shape_every
    current_solver = anchor_solver
    start_parameters = parameters
    previous_spec = spec0
    previous_hook = feedback_hook
    # Object identity is the cache contract used by TransientSolver too.
    current_key = tuple(id(item) for item in spec0) + (id(feedback_hook),)
    # IQS predictor-corrector state. ``checkpoint`` is the accepted state at
    # the current anchor. After each end-shape solve the interval is
    # re-marched from it with the interpolated shape correction (``replay``),
    # or restarted from it with half the width when the iteration fails.
    # ``interval_start_parameters`` holds the start-shape kinetics of every
    # fine step since the anchor for the inexpensive amplitude re-march.
    interval_start_parameters = []
    replay = None

    def snapshot(anchor_step):
        return {"n": anchor_step, "amplitude": amplitude,
                "precursors": precursors.copy(),
                "start_parameters": start_parameters,
                "power_shape_factor": power_shape_factor,
                "spatial_precursors": spatial_precursors.copy(),
                "temperature": temperature, "feedback_hook": feedback_hook,
                "power_accum": power_accum, "window_steps": window_steps,
                "cached_peak": cached_peak, "cached_mean": cached_mean,
                "current_solver": current_solver, "current_key": current_key,
                "previous_spec": previous_spec,
                "previous_hook": previous_hook,
                "history": len(powers), "residuals": len(residual_times)}

    def iqs_end_shape(shape_solver, spec, hook, width):
        """Corrector shape, precursor shape and power shape at the end."""
        with profiler.region("iqs_shape_solve"):
            corrected, flux, precursor_shape, predicted = \
                time_dependent_corrector(
                    anchor_spec, anchor_hook, spec, hook, width, anchor_flux,
                    anchor_precursors / anchor_power_shape_factor,
                    substeps=iqs_substeps, start_time=anchor_time)
        counters["iqs_shape_solves"] += 1
        counters["iqs_corrector_substeps"] += iqs_substeps
        counters["iqs_inner_iterations"] += corrected.total_inner_iterations
        counters["iqs_fixed_point_sweeps"] += sum(corrected.step_iterations)
        # Feedback can change at the thermal exchange immediately before
        # this solve. Normalize flux and precursor shape together under that
        # exact end operator.
        flux, _, scale = normalized_neutron_shape(shape_solver, flux)
        with profiler.region("power_shape"):
            power_shape = normalized_power_shape(spec, flux)
        return {"flux": flux, "precursors": precursor_shape / scale,
                "predicted": predicted, "power_shape": power_shape,
                "key": tuple(id(item) for item in spec) + (id(hook),)}

    def iqs_correction(solver, shape, start, anchor_importance):
        """End-shape minus start-shape kinetics on the interval-end operator.

        Both projections use the operator that advanced the last fine step, so
        the correction isolates the shape change; control and feedback timing
        stay exactly those of the predictor march.
        """
        with profiler.region("reactivity_projection"):
            raw_end = project_effective_kinetics(
                solver, shape["flux"], adjoint.flux, ctx.kinetics,
                k0=steady.k_eff)
        counters["reactivity_projections"] += 1
        delta = (raw_end.rho - rho_reference - start.rho,
                 raw_end.generation_time - start.generation_time,
                 raw_end.beta - start.beta)
        return {"delta": delta, "T0": anchor_importance,
                "T1": raw_end.time_importance}

    def remarch_amplitude(correction, first_step, width):
        """Re-march the anchor amplitude with a shape correction over this
        march's start-shape kinetics. Host-only: no spatial work."""
        value = checkpoint["amplitude"]
        effective = checkpoint["precursors"]
        t0 = float(times[first_step - 1])
        history = {"amplitude": [], "power": [], "factor": [], "rho": []}
        for j, start in enumerate(interval_start_parameters):
            weight = (float(times[first_step + j]) - t0) / width
            kinetics = _interpolated_kinetics(
                start, correction["delta"], weight)
            value, effective = advance_point_kinetics(
                kinetics, dt, value, effective)
            factor = (anchor_power_shape_factor * correction["T0"]
                      / ((1.0 - weight) * correction["T0"]
                         + weight * correction["T1"]))
            history["amplitude"].append(value)
            history["power"].append(value * factor)
            history["factor"].append(factor)
            history["rho"].append(kinetics.rho)
        history["precursors"] = effective
        return history

    checkpoint = snapshot(0)

    synchronize(xp)
    initialization_seconds = time.perf_counter() - initialization_start
    march_start = time.perf_counter()
    march_context = (profiler.region("quasistatic_total")
                     if profile else nullcontext())
    with march_context:
        n = 0
        while n < steps:
            n += 1
            t = float(times[n])
            force_shape = False
            fallback_step = False
            closed_window = None
            spec = _unpack_problem(get_problem(t), ctx.mix_material,
                                   ctx.mix_weight)
            key = tuple(id(item) for item in spec) + (id(feedback_hook),)
            step_start_solver = current_solver
            step_start_key = current_key
            if key != current_key:
                with profiler.region("operator_rebuild"):
                    current_solver = build_solver(spec, feedback_hook)
                counters["operator_rebuilds"] += 1
                with profiler.region("reactivity_projection"):
                    raw = project_effective_kinetics(
                        current_solver, forward_flux, adjoint.flux,
                        ctx.kinetics, k0=steady.k_eff)
                counters["reactivity_projections"] += 1
                start_parameters = replace(raw, rho=raw.rho - rho_reference)
                # A re-march ends at the interval already chosen; its safety
                # triggers were evaluated by the predictor march.
                if replay is None and (residual_tol is not None
                                       or fallback_residual is not None):
                    with profiler.region("shape_residual"):
                        residual = projected_shape_residual(
                            current_solver, forward_flux, adjoint.flux)
                    residual_times.append(t)
                    residual_values.append(residual)
                    counters["residual_evaluations"] += 1
                    force_shape = (residual_tol is not None
                                   and residual >= residual_tol)
                    fallback_step = (fallback_residual is not None
                                     and residual >= fallback_residual)
                if (start_parameters.beta.shape != precursors.shape
                        or not np.array_equal(start_parameters.decay,
                                              anchor_raw.decay)):
                    raise ValueError("delayed family count/decay constants "
                                     "changed in time")
                current_key = key
            interval_start_parameters.append(start_parameters)
            if replay is None:
                parameters = start_parameters
                edit_shape = unit_power_shape
            else:
                # Corrector pass: interpolate the end-shape correction and
                # the matching time-importance/power-shape coordinates.
                weight = (t - replay["t0"]) / replay["width"]
                parameters = _interpolated_kinetics(
                    start_parameters, replay["delta"], weight)
                power_shape_factor = (
                    replay["q0"] * replay["T0"]
                    / ((1.0 - weight) * replay["T0"]
                       + weight * replay["T1"]))
                edit_shape = (replay["U1"] if n == replay["n_end"]
                              else (1.0 - weight) * unit_power_shape
                              + weight * replay["U1"])

            forward_flux_before_step = forward_flux
            amplitude_before = amplitude
            physical_power_before = amplitude_before * power_shape_factor
            if fallback_step:
                # The residual says the frozen shape is outside the trusted
                # QS neighborhood. Advance this fine interval with the full
                # spatial diffusion equations and carry their precursor field.
                # TransientSolver renormalizes the start flux to P(0)=1 under
                # its start operator; scale the precursors identically.
                start_key = (tuple(id(item) for item in previous_spec)
                             + (id(previous_hook),))
                if start_key == step_start_key:
                    start_solver = step_start_solver
                else:
                    with profiler.region("operator_rebuild"):
                        start_solver = build_solver(previous_spec,
                                                    previous_hook)
                    counters["operator_rebuilds"] += 1
                _, _, start_scale = normalized_neutron_shape(
                    start_solver, forward_flux)
                with profiler.region("full_diffusion_fallback"):
                    fallback, new_flux, new_precursor_shape, predicted = \
                        time_dependent_corrector(
                            previous_spec, previous_hook, spec, feedback_hook,
                            dt, forward_flux,
                            spatial_precursors
                            / (physical_power_before * start_scale),
                            substeps=1, start_time=t - dt)
                _, precursors = advance_point_kinetics(
                    parameters, dt, amplitude_before, precursors)
                fallback_physical_power = physical_power_before * predicted
                # Keep the old coordinate until the new shape's adjoint
                # importance is available below.  The shape correction then
                # changes the coordinate factor, not this physical fallback
                # result.
                amplitude = fallback_physical_power / power_shape_factor
                forward_flux, _, precursor_scale = normalized_neutron_shape(
                    current_solver, new_flux)
                spatial_precursor_shape = new_precursor_shape / precursor_scale
                spatial_precursors = (fallback_physical_power
                                      * spatial_precursor_shape)
                with profiler.region("power_shape"):
                    unit_power_shape = normalized_power_shape(spec, forward_flux)
                edit_shape = unit_power_shape
                fallback_times.append(t)
                counters["full_diffusion_fallbacks"] += 1
                counters["fallback_inner_iterations"] += \
                    fallback.total_inner_iterations
                counters["fallback_fixed_point_sweeps"] += sum(
                    fallback.step_iterations)
            else:
                with profiler.region("amplitude_solve"):
                    amplitude, precursors = advance_point_kinetics(
                        parameters, dt, amplitude, precursors)
                physical_power = amplitude * power_shape_factor
                # Maintain an actual spatial precursor field between IQS
                # corrections. The macro corrector replaces this approximation
                # with its implicit spatial history at the next shape event.
                source_shape = current_solver.fields.fission_source(
                    [forward_flux[g] for g in range(current_solver.n_groups)])
                source_shape = source_shape / steady.k_eff
                _, beta_fields = _mapped_kinetics(current_solver, ctx.kinetics)
                for i in range(ctx.kinetics.n_families):
                    spatial_precursors[i] = (
                        spatial_precursors[i]
                        + dt * beta_fields[i] * physical_power * source_shape
                    ) / (1.0 + ctx.kinetics.decay[i] * dt)
            counters["amplitude_steps"] += 1
            physical_power = amplitude * power_shape_factor
            amplitudes.append(amplitude)
            powers.append(physical_power)
            power_shape_factors.append(power_shape_factor)
            rhos.append(parameters.rho)
            edit_power = physical_power
            with profiler.region("power_edit"):
                weighted_shape = physical_power * edit_shape
                power_accum = (weighted_shape if power_accum is None
                               else power_accum + weighted_shape)
            window_steps += 1

            final_window = n == steps
            if window_steps == every or final_window:
                width = window_steps * dt
                stepper = (thermal if window_steps == every
                           else thermal_solver_for(width))
                # Kept so a shape correction of this step's power can redo
                # the exchange it has just been consumed by.
                closed_window = {"temperature": temperature,
                                 "accum": power_accum,
                                 "steps": window_steps, "stepper": stepper}
                power_density = ctx.total_power * power_accum / window_steps
                next_thermal = counters["thermal_steps"] + 1
                diagnostics = (thermal_diagnostics_every > 0 and
                               next_thermal % thermal_diagnostics_every == 0)
                with profiler.region("thermal_solve"):
                    thermal_result = stepper.step(
                        power_density, temperature, rtol=thermal_rtol,
                        maxiter=thermal_maxiter,
                        check_every=thermal_check_every,
                        diagnostics=diagnostics,
                        synchronize_timing=False)
                temperature = thermal_result.temperature
                counters["thermal_steps"] += 1
                counters["thermal_iterations"] += thermal_result.iterations
                counters["thermal_diagnostics"] += int(diagnostics)
                with profiler.region("feedback_update"):
                    feedback_hook = ctx.feedback.hook(temperature)
                counters["feedback_updates"] += 1
                with profiler.region("telemetry_transfer"):
                    packet = asnumpy(xp.stack((
                        xp.max(temperature[fuel_dev]),
                        xp.mean(temperature[fuel_dev]))))
                counters["telemetry_transfers"] += 1
                cached_peak, cached_mean = map(float, packet)
                power_accum = None
                window_steps = 0

            if replay is not None:
                due_interval = replay["due_interval"]
                due_final = replay["due_final"]
                due_residual = replay["due_residual"]
                shape_event = n == replay["n_end"]
            else:
                due_interval = (
                    shape_every is not None
                    and ((n % shape_every == 0)
                         if (iqs_predictor_tol is None
                             and shape_method != "iqs")
                         else n >= next_shape_step))
                due_final = (shape_every is not None and final_window
                             and shape_on_final and not due_interval)
                due_residual = force_shape and not fallback_step
                shape_event = (due_interval or due_final or due_residual
                               or fallback_step)
            if shape_event:
                # Re-anchor against the end-of-step control and temperature
                # state. The adjoint-weighted population amplitude remains
                # continuous; the fission-power factor is corrected for the
                # new shape's time importance.
                with profiler.region("shape_operator_rebuild"):
                    shape_solver = build_solver(spec, feedback_hook)
                counters["operator_rebuilds"] += 1
                shape_hook = feedback_hook
                shape_key = tuple(id(item) for item in spec) + (id(shape_hook),)
                old_forward_flux = (forward_flux_before_step if fallback_step
                                    else forward_flux)
                # Measure both sides of the shape replacement with the same
                # adjoint.  If the adjoint itself is refreshed below it is
                # rescaled so this old-shape functional remains continuous.
                old_time_importance = _shape_time_importance(
                    shape_solver, old_forward_flux, adjoint.flux,
                    ctx.kinetics)
                predictor_disagreement = None
                precursors_follow_flux = True
                accepted_power = None
                new_unit_power_shape = None
                iteration = 0
                iqs_converged = True
                if fallback_step:
                    # The fine-step fallback already supplied the new shape and
                    # spatial precursor history. Only re-anchor its projection
                    # against feedback updated by the thermal half-step.
                    reason = "full_diffusion_fallback"
                    shape_state = [forward_flux[g].copy()
                                   for g in range(shape_solver.n_groups)]
                    new_unit_power_shape = unit_power_shape
                elif shape_method == "iqs":
                    reason = ("residual" if due_residual else
                              "maximum_interval" if due_interval else
                              "final_state")
                    iteration = 0 if replay is None else replay["iteration"]
                    end_shape = None if replay is None else replay["shape"]
                    first_step = checkpoint["n"] + 1
                    interval_steps = n - checkpoint["n"]
                    width = t - anchor_time
                    if end_shape is None:
                        end_shape = iqs_end_shape(
                            shape_solver, spec, shape_hook, width)
                    anchor_importance = _shape_time_importance(
                        current_solver, anchor_flux, adjoint.flux,
                        ctx.kinetics)
                    correction = iqs_correction(
                        current_solver, end_shape, start_parameters,
                        anchor_importance)
                    trial = remarch_amplitude(correction, first_step, width)
                    trial_power = trial["power"][-1]
                    change = abs(trial_power - powers[-1]) / trial_power
                    iqs_converged = change <= iqs_iteration_tol
                    if (not iqs_converged and iteration > 0
                            and end_shape["key"] != shape_key):
                        # Feedback re-marched with the corrected power moved
                        # the end operator: solve the end shape again.
                        end_shape = iqs_end_shape(
                            shape_solver, spec, shape_hook, width)
                        correction = iqs_correction(
                            current_solver, end_shape, start_parameters,
                            anchor_importance)
                        trial = remarch_amplitude(
                            correction, first_step, width)
                        trial_power = trial["power"][-1]
                    corrector_power = (anchor_amplitude
                                       * anchor_power_shape_factor
                                       * end_shape["predicted"])
                    disagreement = (abs(corrector_power - trial_power)
                                    / trial_power)
                    counters["iqs_max_iteration_change_ppm"] = max(
                        counters["iqs_max_iteration_change_ppm"],
                        int(round(1e6 * change)) if iteration > 0 else 0)
                    predictor_rejects = (iqs_predictor_tol is not None
                                         and disagreement > iqs_predictor_tol)
                    exhausted = (not iqs_converged
                                 and iteration >= iqs_max_iterations)
                    rewind = None
                    if interval_steps > 1 and (predictor_rejects or exhausted):
                        # Reject the whole interval and retry it with half
                        # the shape step from the accepted anchor.
                        rewind = "reject"
                        counters["iqs_interval_rejections"] += 1
                        if predictor_rejects:
                            counters["iqs_predictor_interval_reductions"] += 1
                        else:
                            counters["iqs_unconverged_rejections"] += 1
                        adaptive_shape_every = max(1, interval_steps // 2)
                        next_shape_step = checkpoint["n"] + adaptive_shape_every
                        replay = None
                    elif not iqs_converged and not exhausted:
                        rewind = "replay"
                        counters["iqs_corrector_iterations"] += 1
                        replay = dict(
                            correction, iteration=iteration + 1,
                            shape=end_shape, U1=end_shape["power_shape"],
                            q0=anchor_power_shape_factor, t0=anchor_time,
                            width=width, n_end=n, due_interval=due_interval,
                            due_final=due_final, due_residual=due_residual)
                    if rewind is not None:
                        cp = checkpoint
                        amplitude = cp["amplitude"]
                        precursors = cp["precursors"].copy()
                        start_parameters = parameters = cp["start_parameters"]
                        power_shape_factor = cp["power_shape_factor"]
                        spatial_precursors = cp["spatial_precursors"].copy()
                        temperature = cp["temperature"]
                        feedback_hook = cp["feedback_hook"]
                        power_accum = cp["power_accum"]
                        window_steps = cp["window_steps"]
                        cached_peak = cp["cached_peak"]
                        cached_mean = cp["cached_mean"]
                        current_solver = cp["current_solver"]
                        current_key = cp["current_key"]
                        previous_spec = cp["previous_spec"]
                        previous_hook = cp["previous_hook"]
                        for history in (amplitudes, powers,
                                        power_shape_factors, rhos, peaks,
                                        means):
                            del history[cp["history"]:]
                        if rewind == "reject":
                            del residual_times[cp["residuals"]:]
                            del residual_values[cp["residuals"]:]
                        interval_start_parameters = []
                        n = cp["n"]
                        continue
                    if not iqs_converged:
                        counters["iqs_unconverged_accepts"] += 1
                    if predictor_rejects:
                        counters["iqs_predictor_tol_accepts"] += 1
                    # Accept the re-projected amplitude history. The thermal
                    # windows of this march consumed powers within the
                    # iteration tolerance of it; the corrector supplies the
                    # end shape and the spatial distribution of the accepted
                    # effective precursors.
                    start = first_step
                    amplitudes[start:] = trial["amplitude"]
                    powers[start:] = trial["power"]
                    power_shape_factors[start:] = trial["factor"]
                    rhos[start:] = trial["rho"]
                    amplitude = trial["amplitude"][-1]
                    precursors = trial["precursors"]
                    accepted_power = trial_power
                    forward_flux = end_shape["flux"]
                    new_unit_power_shape = end_shape["power_shape"]
                    shape_state = [forward_flux[g].copy()
                                   for g in range(shape_solver.n_groups)]
                    precursors_follow_flux = False
                    previous_spatial_precursors = spatial_precursors
                    spatial_precursors, _ = _match_spatial_precursor_importance(
                        shape_solver, adjoint.flux, end_shape["precursors"],
                        precursors, ctx.kinetics,
                        anchor_power_shape_factor * old_time_importance)
                    delta_c = spatial_precursors - previous_spatial_precursors
                    base_c = xp.sum(previous_spatial_precursors**2)
                    relative_c = float(xp.sqrt(
                        xp.sum(delta_c**2) / xp.maximum(base_c, 1e-300)))
                    counters["iqs_precursor_shape_corrections"] += 1
                    counters["iqs_max_precursor_shape_change_ppm"] = max(
                        counters["iqs_max_precursor_shape_change_ppm"],
                        int(round(1e6 * relative_c)))
                    predictor_disagreement = disagreement
                    predictor_disagreement_times.append(t)
                    predictor_disagreement_values.append(disagreement)
                    counters["iqs_predictor_checks"] += 1
                    counters["iqs_max_amplitude_error_ppm"] = max(
                        counters["iqs_max_amplitude_error_ppm"],
                        int(round(1e6 * disagreement)))
                else:
                    precursors_follow_flux = False
                    forward_options = dict(ctx.eigen_kwargs)
                    forward_options.update(shape_kwargs or {})
                    with profiler.region("forward_shape_solve"):
                        forward = shape_solver.solve(
                            state0=shape_state, **forward_options)
                    counters["forward_shape_solves"] += 1
                    counters["forward_shape_outer_iterations"] += \
                        forward.outer_iterations
                    counters["forward_shape_inner_iterations"] += \
                        forward.inner_iterations
                    if not forward.converged:
                        raise RuntimeError(
                            f"quasi-static forward shape did not converge: {forward}")
                    shape_state = shape_solver.state
                    forward_flux = forward.flux
                    reason = ("residual" if due_residual else
                              "maximum_interval" if due_interval else
                              "final_state")
                replay = None
                # Feedback can change at the thermal exchange immediately
                # before this re-anchor. Normalize flux and precursor shape
                # together under that exact end operator.
                forward_flux, _, precursor_scale = normalized_neutron_shape(
                    shape_solver, forward_flux)
                if precursors_follow_flux:
                    spatial_precursors = spatial_precursors / precursor_scale
                shape_updates += 1

                adjoint_defect = None
                if adjoint_residual_tol is not None and not fallback_step:
                    with profiler.region("adjoint_residual"):
                        adjoint_defect = projected_adjoint_residual(
                            shape_solver, forward_flux, adjoint.flux)
                    adjoint_residual_times.append(t)
                    adjoint_residual_values.append(adjoint_defect)
                    counters["adjoint_residual_evaluations"] += 1
                    counters["max_adjoint_residual_ppm"] = max(
                        counters["max_adjoint_residual_ppm"],
                        int(round(1e6 * adjoint_defect)))
                cadence_refresh = shape_updates % adjoint_every == 0
                residual_refresh = (adjoint_defect is not None
                                    and adjoint_defect >= adjoint_residual_tol)
                if fallback_step or cadence_refresh or residual_refresh:
                    if residual_refresh and not cadence_refresh:
                        counters["adjoint_residual_refreshes"] += 1
                    with profiler.region("adjoint_shape_solve"):
                        adjoint = shape_solver.solve(
                            adjoint=True, state0=adjoint_state,
                            **adjoint_options)
                    counters["adjoint_eigen_solves"] += 1
                    counters["adjoint_shape_outer_iterations"] += \
                        adjoint.outer_iterations
                    counters["adjoint_shape_inner_iterations"] += \
                        adjoint.inner_iterations
                    if not adjoint.converged:
                        raise RuntimeError(
                            f"quasi-static adjoint shape did not converge: {adjoint}")
                    adjoint_state = shape_solver.state
                    # Eigenvectors have arbitrary scale. Preserve the previous
                    # amplitude coordinate by matching the old shape's time
                    # importance before using the refreshed adjoint.
                    refreshed_old_importance = _shape_time_importance(
                        shape_solver, old_forward_flux, adjoint.flux,
                        ctx.kinetics)
                    adjoint.flux = (adjoint.flux
                                    * (old_time_importance
                                       / refreshed_old_importance))

                with profiler.region("reactivity_projection"):
                    raw = project_effective_kinetics(
                        shape_solver, forward_flux, adjoint.flux,
                        ctx.kinetics, k0=steady.k_eff)
                counters["reactivity_projections"] += 1
                parameters = replace(raw, rho=raw.rho - rho_reference)
                start_parameters = parameters
                # Integrated adjoint-weighted shape derivative.  With
                # phi=A*psi and fission-normalized psi, continuity of
                # <phi*,V^-1 phi> gives
                #   P = A * q,  q_new/q_old = T_old/T_new.
                # A remains the fine point-kinetics coordinate; q converts it
                # to physical fission power. An accepted IQS march already
                # carries this factor, so its physical power is kept and only
                # the amplitude coordinate follows a refreshed adjoint.
                previous_factor = anchor_power_shape_factor
                power_shape_factor = previous_factor * (
                    old_time_importance / raw.time_importance)
                if (not np.isfinite(power_shape_factor)
                        or power_shape_factor <= 0.0):
                    raise RuntimeError("quasi-static shape normalization is invalid")
                if fallback_step:
                    amplitude = fallback_physical_power / power_shape_factor
                    amplitudes[-1] = amplitude
                elif accepted_power is not None:
                    amplitude = accepted_power / power_shape_factor
                    amplitudes[-1] = amplitude
                correction = power_shape_factor / previous_factor
                counters["shape_derivative_corrections"] += 1
                counters["max_shape_derivative_correction_ppm"] = max(
                    counters["max_shape_derivative_correction_ppm"],
                    int(round(1e6 * abs(correction - 1.0))))
                precursors = _project_spatial_precursors(
                    shape_solver, adjoint.flux, spatial_precursors,
                    ctx.kinetics,
                    power_shape_factor * raw.time_importance)
                if new_unit_power_shape is None:
                    with profiler.region("power_shape"):
                        new_unit_power_shape = normalized_power_shape(
                            spec, forward_flux)
                unit_power_shape = new_unit_power_shape
                corrected_power = amplitude * power_shape_factor
                powers[-1] = corrected_power
                power_shape_factors[-1] = power_shape_factor
                edit_changed = (
                    edit_shape is not unit_power_shape
                    or abs(corrected_power - edit_power)
                    > 1e-14 * abs(corrected_power))
                if power_accum is not None and edit_changed:
                    # Replace this fine step's provisional edit if the thermal
                    # window has not already been consumed.
                    power_accum = (power_accum
                                   - edit_power * edit_shape
                                   + corrected_power * unit_power_shape)
                elif closed_window is not None and edit_changed:
                    # The window closed on this step's provisional edit: redo
                    # that exchange with the corrected power and shape.
                    accum = (closed_window["accum"]
                             - edit_power * edit_shape
                             + corrected_power * unit_power_shape)
                    with profiler.region("thermal_solve"):
                        thermal_result = closed_window["stepper"].step(
                            ctx.total_power * accum / closed_window["steps"],
                            closed_window["temperature"], rtol=thermal_rtol,
                            maxiter=thermal_maxiter,
                            check_every=thermal_check_every,
                            diagnostics=False, synchronize_timing=False)
                    temperature = thermal_result.temperature
                    counters["thermal_steps"] += 1
                    counters["thermal_redos"] += 1
                    counters["thermal_iterations"] += thermal_result.iterations
                    with profiler.region("feedback_update"):
                        feedback_hook = ctx.feedback.hook(temperature)
                    counters["feedback_updates"] += 1
                    with profiler.region("telemetry_transfer"):
                        packet = asnumpy(xp.stack((
                            xp.max(temperature[fuel_dev]),
                            xp.mean(temperature[fuel_dev]))))
                    counters["telemetry_transfers"] += 1
                    cached_peak, cached_mean = map(float, packet)
                spatial_precursor_shape = spatial_precursors / amplitude
                # The next fine step continues from the end operator; a redone
                # thermal exchange changes the hook and so forces a rebuild.
                current_solver = shape_solver
                current_key = shape_key
                start_parameters = parameters
                shape_times.append(t)
                shape_reasons.append(reason)
                denominator = 1.0 - raw.rho
                shape_ks.append(steady.k_eff / denominator
                                if denominator > 0.0 else float("nan"))
                anchor_spec = spec
                anchor_hook = shape_hook
                anchor_flux = forward_flux
                anchor_precursors = spatial_precursor_shape
                anchor_time = t
                anchor_amplitude = amplitude
                anchor_power_shape_factor = power_shape_factor
                if ((shape_method == "iqs" or iqs_predictor_tol is not None)
                        and not final_window):
                    if (predictor_disagreement is not None
                            and adaptive_shape_every < shape_every
                            and iqs_converged and iteration <= 1
                            and (iqs_predictor_tol is None
                                 or predictor_disagreement
                                 < 0.25 * iqs_predictor_tol)):
                        # A calm accepted interval restores the step doubled.
                        restored = min(shape_every, 2 * adaptive_shape_every)
                        if restored > adaptive_shape_every:
                            counters["iqs_predictor_interval_recoveries"] += 1
                        adaptive_shape_every = restored
                    proposed_shape_step = n + adaptive_shape_every
                    if due_interval:
                        next_shape_step = proposed_shape_step
                    else:
                        # A residual/fallback event is an extra safety update;
                        # it must not postpone an already scheduled maximum-
                        # interval correction.
                        next_shape_step = min(next_shape_step,
                                              proposed_shape_step)
            peaks.append(cached_peak)
            means.append(cached_mean)
            if verbose and (n % max(1, steps // 20) == 0 or n == steps):
                print(f"  t = {t:8.3f} s   P/P0 = {powers[-1]:.5f}   "
                      f"T_fuel = {cached_mean:.2f} K   "
                      f"shape updates = {shape_updates}", flush=True)
            previous_spec = spec
            previous_hook = feedback_hook
            if shape_event:
                checkpoint = snapshot(n)
                interval_start_parameters = []

    synchronize(xp)
    seconds = time.perf_counter() - march_start
    with profiler.region("result_transfer"):
        final_temperature = asnumpy(temperature)
    counters["result_transfers"] += 1
    counters["shape_updates"] = shape_updates
    for name in ("forward_shape_solves", "iqs_shape_solves",
                 "full_diffusion_fallbacks", "residual_evaluations",
                 "iqs_predictor_checks", "shape_derivative_corrections",
                 "max_shape_derivative_correction_ppm",
                 "iqs_precursor_shape_corrections",
                 "iqs_max_precursor_shape_change_ppm",
                 "iqs_corrector_substeps",
                 "iqs_predictor_interval_reductions",
                 "iqs_predictor_interval_recoveries",
                 "iqs_corrector_iterations", "iqs_max_iteration_change_ppm",
                 "iqs_interval_rejections", "iqs_unconverged_rejections",
                 "iqs_unconverged_accepts", "iqs_predictor_tol_accepts",
                 "thermal_redos",
                 "adjoint_residual_evaluations",
                 "adjoint_residual_refreshes",
                 "max_adjoint_residual_ppm"):
        counters[name] += 0
    phase_seconds = profiler.seconds()
    phase_seconds.pop("neutronics_solve", None)
    return QuasiStaticResult(
        times=times, power=np.asarray(powers), rho=np.asarray(rhos),
        peak_temperature=np.asarray(peaks),
        mean_temperature=np.asarray(means), temperature=final_temperature,
        flux=forward_flux, adjoint_flux=adjoint.flux, k0=steady.k_eff,
        steady=steady, seconds=seconds,
        initialization_seconds=initialization_seconds,
        device=ctx.device, steps=steps,
        amplitude=np.asarray(amplitudes),
        power_shape_factor=np.asarray(power_shape_factors),
        effective_precursors=np.asarray(precursors),
        spatial_precursors=spatial_precursors,
        shape_update_times=np.asarray(shape_times),
        shape_update_reasons=shape_reasons,
        shape_k_eff=np.asarray(shape_ks),
        residual_times=np.asarray(residual_times),
        shape_residual=np.asarray(residual_values),
        adjoint_residual_times=np.asarray(adjoint_residual_times),
        adjoint_residual=np.asarray(adjoint_residual_values),
        predictor_disagreement_times=np.asarray(
            predictor_disagreement_times),
        predictor_disagreement=np.asarray(predictor_disagreement_values),
        fallback_times=np.asarray(fallback_times),
        phase_seconds=phase_seconds, counters=dict(counters))


def fixed_shape_coupled_transient(ctx, t_end, dt, **kwargs):
    """Coupled quasi-static prototype with the time-zero shape frozen.

    Control and temperature changes are represented through adjoint-projected
    effective kinetics, but no forward shape correction is made. This is exact
    for shape-preserving perturbations and useful as the cheapest comparison
    mode. Use :func:`quasistatic_coupled_transient` for control-drum motion.
    """
    forbidden = {"shape_dt", "adjoint_every", "shape_on_final", "shape_method",
                 "residual_tol", "fallback_residual", "adjoint_residual_tol",
                 "iqs_kwargs",
                 "iqs_precond_degree", "iqs_substeps",
                 "iqs_predictor_tol", "iqs_iteration_tol",
                 "iqs_max_iterations"} & kwargs.keys()
    if forbidden:
        raise TypeError("fixed_shape_coupled_transient does not accept "
                        + ", ".join(sorted(forbidden)))
    return _coupled_quasistatic(
        ctx, t_end, dt, shape_dt=None, adjoint_every=1,
        shape_on_final=False, **kwargs)


def quasistatic_coupled_transient(ctx, t_end, dt, *, shape_dt=2.0,
                                  adjoint_every=1, shape_on_final=True,
                                  shape_method="iqs", residual_tol=None,
                                  fallback_residual=None,
                                  adjoint_residual_tol=None,
                                  **kwargs):
    """Run an improved or adiabatic quasi-static coupled transient.

    The adjoint-weighted population amplitude and effective delayed precursors
    advance every ``dt`` while ``shape_method='iqs'`` advances a
    time-dependent spatial shape and its precursor fields every ``shape_dt``.
    Physical fission power includes the old/new adjoint time-importance ratio,
    i.e. the integrated IQS shape-derivative term. ``'adiabatic'`` selects the
    cheaper instantaneous eigen shape. The adjoint is refreshed at least every
    ``adjoint_every`` corrections. ``adjoint_residual_tol`` can refresh it
    sooner when the reused importance shape no longer satisfies the current
    transposed eigenproblem; hard fallback always refreshes it.
    ``iqs_substeps`` optionally resolves cached
    control states inside each macro corrector instead of applying only the
    endpoint state. Each IQS interval is re-marched with kinetics
    interpolated between its start and end shapes until its end power changes
    by at most ``iqs_iteration_tol``; after ``iqs_max_iterations`` replays,
    or when the corrector's own end power disagrees with the accepted power
    by more than ``iqs_predictor_tol``, the interval is rejected and retried
    with half the width. The width is restored after the transient settles.

    ``residual_tol`` forces an early shape correction when the amplitude-free
    spatial defect grows too large. ``fallback_residual`` is the hard safety
    threshold: that fine interval is advanced by the full spatial diffusion
    equations instead of point kinetics.
    """
    return _coupled_quasistatic(
        ctx, t_end, dt, shape_dt=shape_dt, adjoint_every=adjoint_every,
        shape_on_final=shape_on_final, shape_method=shape_method,
        residual_tol=residual_tol, fallback_residual=fallback_residual,
        adjoint_residual_tol=adjoint_residual_tol,
        **kwargs)
