"""Coupled neutronics/thermal steady state.

The two physics are solved separately and reconciled by a fixed-point
iteration on the temperature field:

    T  ->  cross sections (feedback)  ->  k-eigenvalue solve  ->  phi
       ->  q'''  (normalized to rated power)  ->  conduction  ->  T'

**What is being computed, and what is not.** A k-eigenvalue solve renormalizes
its flux, so the power *level* is imposed by the operator -- rated thermal
power -- and not computed. Feedback therefore cannot change how much heat the
core makes; it changes where the heat is made, and it changes `k`. That is the
standard steady-state coupled formulation (the same one Griffin/BISON and
MPACT/CTF use), and it is what makes the iteration strongly contractive: with
the level pinned, Doppler can only redistribute power by a fraction of a
percent, so the map is a near-constant and Picard converges in a handful of
steps without needing acceleration to be viable.

The reportable result is `k` at hot full power versus `k` on the cold
unfed core -- the **temperature defect** -- and, through
:func:`criticality_search`, the control-drum angle that holds the core critical
once that defect is paid for.

**Why the map lives in free functions.** :func:`neutronics_step` and
:func:`thermal_step` are called both by :class:`CoupledSolver` here and by the
preCICE participants in ``examples/precice/``. Cross-verifying an external
coupling against an internal one only means something if the two are running
identical physics; the way to guarantee that is to have one implementation of
each half and let the couplers differ.
"""

from __future__ import annotations

import inspect
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import numpy as np

from . import kernels
from .backend import asnumpy, synchronize
from .linalg import AndersonAccelerator
from .power import power_density
from .thermal import ConductionSolver
from .tri import TriDiffusionEigenSolver


class _CoupledPhaseProfiler:
    """Low-intrusion coupled phase timer.

    CPU regions use ``perf_counter``. GPU regions use CUDA events and are
    resolved after one final synchronization, so profiling does not turn every
    phase boundary into the device/host stall it is intended to diagnose.
    """

    def __init__(self, xp, enabled=False):
        self.xp = xp
        self.enabled = bool(enabled)
        self.gpu = self.enabled and xp is not np
        self._pairs = defaultdict(list)
        self._neutronics_start = None
        self.counters = defaultdict(int)

    def _stamp(self):
        if self.gpu:
            event = self.xp.cuda.Event()
            event.record()
            return event
        return time.perf_counter()

    def _add(self, name, start, stop):
        if self.enabled:
            self._pairs[name].append((start, stop))

    @contextmanager
    def region(self, name):
        if not self.enabled:
            yield
            return
        start = self._stamp()
        if self.gpu:
            from .profiling import nvtx_range
            marker = nvtx_range(f"garnet.coupled.{name}")
        else:
            marker = nullcontext()
        with marker:
            yield
        self._add(name, start, self._stamp())
        if name == "operator_rebuild":
            self.counters["operator_rebuilds"] += 1

    def resume_neutronics(self):
        if self.enabled:
            self._neutronics_start = self._stamp()

    def pause_neutronics(self):
        if self.enabled and self._neutronics_start is not None:
            self._add("neutronics_total", self._neutronics_start, self._stamp())
            self._neutronics_start = None

    def seconds(self):
        if not self.enabled:
            return {}
        synchronize(self.xp)
        out = {}
        for name, pairs in self._pairs.items():
            if self.gpu:
                out[name] = sum(
                    self.xp.cuda.get_elapsed_time(a, b) for a, b in pairs) / 1e3
            else:
                out[name] = sum(b - a for a, b in pairs)
        # Initial criticality and rebuild work are nested within the interval
        # from transient entry to coupling callbacks. Report an exclusive
        # time-march neutron solve so the table adds up rather than attributing
        # startup to time-step work or double-counting rebuilds.
        neutron_total = out.pop("neutronics_total", 0.0)
        out["neutronics_solve"] = max(
            0.0, neutron_total - out.get("initial_eigen_solve", 0.0)
            - out.get("operator_rebuild", 0.0))
        return out


@dataclass
class CouplingContext:
    """Everything both halves need, assembled once.

    Built from an ``HpmrProblem`` by :func:`context_from_problem`, but nothing
    here is HP-MR specific -- any geometry the eigen solvers accept works.
    """

    grid: object
    materials: list
    material_map: np.ndarray
    thermal_materials: list
    feedback: object                       # ThermalFeedback
    total_power: float                     # watts
    active: np.ndarray | None = None
    mask_bc: object = "vacuum"
    bc: object = "reflective"
    mix_material: np.ndarray | None = None
    mix_weight: np.ndarray | None = None
    thermal_bc: object = "adiabatic"
    thermal_mask_bc: object = 1e-3         # W/(cm^2 K) to the vessel
    ambient_temperature: float = 400.0
    solver_cls: object = TriDiffusionEigenSolver
    #: Within-group operator matching solver_cls; only the transient needs it
    #: explicitly (the eigen solvers pick their own). None => infer from grid.
    group_operator: object = None
    #: Delayed-neutron data. Only the transient reads it.
    kinetics: object = None
    device: str = "cpu"
    dtype: object = np.float64
    eigen_kwargs: dict = field(default_factory=lambda: {"tol_k": 1e-10,
                                                        "tol_source": 1e-9})
    #: Reuse the previous flux and k as the next solve's starting point. A
    #: 3-4x speed-up in production -- but it makes the coupled map depend on
    #: its own history, so a lockstep comparison against another coupling tool
    #: must switch it OFF to make the map a pure function of T.
    warm_start: bool = True

    # -- mutable warm-start cache ------------------------------------------
    _state: object = None
    _k: float = 1.0
    _thermal: object = None

    def __post_init__(self):
        if self.group_operator is None:
            from .stencil import GroupOperator
            from .tri import TriGrid, TriGroupOperator
            self.group_operator = (TriGroupOperator
                                   if isinstance(self.grid, TriGrid)
                                   else GroupOperator)

    def reset_state(self):
        """Forget the warm start, making the coupled map deterministic."""
        self._state = None
        self._k = 1.0

    @property
    def shape(self):
        return tuple(self.grid.shape)

    def thermal_solver(self) -> ConductionSolver:
        """The conduction solver -- built once and reused: unlike the
        neutronics, its coefficients do not depend on the coupled state."""
        if self._thermal is None:
            self._thermal = ConductionSolver(
                self.grid, self.thermal_materials, self.material_map,
                bc=self.thermal_bc, active=self.active,
                mask_bc=self.thermal_mask_bc,
                ambient_temperature=self.ambient_temperature,
                mix_material=self.mix_material, mix_weight=self.mix_weight,
                device=self.device, dtype=self.dtype)
        return self._thermal

    def initial_temperature(self) -> np.ndarray:
        """A physically-scaled starting guess: the temperature the fuel would
        reach if all the power left through the heat pipes with no conduction
        at all, T_sink + q/h. Both participants must start from this same
        field for a lockstep comparison to be meaningful."""
        th = self.thermal_solver()
        xp = th.xp
        h, t_sink = th.h, th.t_sink
        sinked_volume = float(xp.sum(th.cell_volume * (h > 0)))
        if sinked_volume <= 0.0:
            return np.full(self.shape, self.ambient_temperature)
        q_mean = self.total_power / sinked_volume
        safe_h = xp.where(h > 0, h, 1.0)
        return asnumpy(xp.where(h > 0, t_sink + q_mean / safe_h,
                                self.ambient_temperature))


@dataclass
class CoupledResult:
    k_eff: float
    temperature: np.ndarray
    power_density: np.ndarray
    flux: object
    iterations: int
    converged: bool
    seconds: float
    device: str
    k_history: list
    residual_history: list
    thermal: object                 # the last ThermalResult
    neutronics: object              # the last eigen Result

    @property
    def peak_temperature(self) -> float:
        return float(np.max(self.temperature))

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return (f"CoupledResult(k_eff = {self.k_eff:.6f}, "
                f"T = {np.min(self.temperature):.1f} .. "
                f"{self.peak_temperature:.1f} K, "
                f"{status} in {self.iterations} coupling iterations, "
                f"{self.seconds:.1f} s)")


def neutronics_step(temperature, ctx: CouplingContext):
    """T (K) -> volumetric fission power q''' (W/cm^3), plus k and the flux.

    Rebuilds the eigen solver, because the operators copy their coefficients at
    construction and so cannot see a mutated ``Fields``. Measured, that rebuild
    is 12 ms against a 530 ms solve on the 2-group HP-MR at refine 6 (2.3%),
    and a smaller share again on the 11-group core where the solve is ~15x
    longer. Not worth a second, incremental code path to keep bit-consistent
    with this one.
    """
    solver = ctx.solver_cls(
        ctx.grid, ctx.materials, ctx.material_map, bc=ctx.bc,
        active=ctx.active, mask_bc=ctx.mask_bc,
        mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
        device=ctx.device, dtype=ctx.dtype,
        xs_update=_feedback_hook(ctx, temperature))

    kw = dict(ctx.eigen_kwargs)
    if ctx.warm_start and ctx._state is not None:
        # The flux only -- deliberately NOT k_guess. The power iteration stops
        # on the CHANGE between successive outers, so seeding it with the
        # previous k as well makes the first |dk| vanishingly small and the
        # solve exits before the flux has responded to the new cross sections:
        # measured, that froze the coupling after one iteration and reported a
        # k 0.9 pcm off the converged value, with a residual of exactly zero
        # that looked like convergence. Warm-starting the shape alone leaves
        # k_guess at its default, so the first outer must do real work.
        kw["state0"] = ctx._state
    res = solver.solve(**kw)
    if not res.converged:
        raise RuntimeError(f"eigenvalue solve did not converge: {res}")
    if ctx.warm_start:
        ctx._state = solver.state
        ctx._k = res.k_eff

    # The thermal solver owns the volume convention -- on an r-z grid the true
    # cell volume is the annulus, not the operator's radius weight -- and the
    # rated power must be divided by the SAME volume the sink term integrates
    # over, or the two halves disagree about how big the core is.
    q = power_density(res.flux, ctx.materials, ctx.material_map,
                      total_power=ctx.total_power,
                      cell_volume=ctx.thermal_solver().cell_volume,
                      mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
                      active=ctx.active,
                      fission_scale=_fission_scale(ctx, solver.fields))
    return q, res


def _feedback_hook(ctx, temperature):
    """``ctx.feedback``'s xs_update at ``temperature``, told the material list
    when it can use it (a volume-mixed cell weights each component's own
    change by that component's own cross sections)."""
    hook = ctx.feedback.hook
    if "materials" in inspect.signature(hook).parameters:
        return hook(temperature, materials=ctx.materials)
    return hook(temperature)


def _fission_scale(ctx, fields):
    """(G, *shape) factor by which the feedback moved nu*Sigma_f in each cell.

    The power edit weights by kappa*Sigma_f from the static material tables,
    which no feedback touches. But a density law or tabulated branches move
    Sigma_f -- the HP-MR fuel's nu*Sigma_f changes 1.7-2.5% between 600 and
    1200 K -- and the energy release moves with it: kappa/nu is a property of
    the fissioning nuclide, ~constant in temperature. So kappa*Sigma_f takes
    the same per-cell, per-group ratio the feedback applied to nu*Sigma_f.
    Cells with no production keep a factor of 1 (they make no power anyway).
    """
    from .blend import MaterialBlend

    xp = fields.blend.xp
    dtype = fields.blend.dtype
    blend = MaterialBlend(xp, ctx.grid.shape, ctx.material_map,
                          len(ctx.materials), dtype=dtype,
                          mix_material=ctx.mix_material,
                          mix_weight=ctx.mix_weight)
    table = np.array([np.atleast_1d(m.nu_sigma_f) for m in ctx.materials],
                     dtype=float)
    scale = []
    for g in range(table.shape[1]):
        base = blend.linear(table[:, g])
        moved = xp.asarray(fields.nu_sigma_f[g], dtype=dtype)
        nonzero = base != 0.0
        scale.append(xp.where(nonzero, moved / xp.where(nonzero, base, 1.0), 1.0))
    return xp.stack(scale)


def thermal_step(power, ctx: CouplingContext, t0=None):
    """q''' (W/cm^3) -> T (K)."""
    res = ctx.thermal_solver().solve(power, t0=t0)
    return asnumpy(res.temperature), res


def coupled_map(temperature, ctx: CouplingContext):
    """One Gauss-Seidel application G(T), neutronics first.

    Neutronics-first matches a preCICE ``serial-implicit`` scheme declared with
    ``first="Neutronics"``; running them the other way round is a different
    fixed-point iteration with the same fixed point, and the two would not
    agree iterate by iterate.
    """
    q, nres = neutronics_step(temperature, ctx)
    t_new, tres = thermal_step(q, ctx)
    return t_new, {"power": q, "neutronics": nres, "thermal": tres}


class CoupledSolver:
    """Picard / Anderson fixed point on the temperature field."""

    def __init__(self, ctx: CouplingContext):
        self.ctx = ctx

    def solve(self, t0=None, tol=1e-8, max_iter=50, relaxation=1.0,
              anderson_depth=0, verbose=False) -> CoupledResult:
        """Iterate to a self-consistent (T, phi, k).

        tol        : convergence on the FIXED-POINT residual
                     max|G(T) - T| relative to the temperature span, so it
                     means the same thing whatever the absolute level. Not
                     the step |T_{n+1} - T_n|: under relaxation omega that is
                     omega times smaller than the residual (loosening the
                     tolerance by 1/omega), and an Anderson step can be small
                     while the residual is not.

                     There is a FLOOR, and asking below it hangs rather than
                     errors. The inner eigen solve converges only to
                     ``tol_source``, and that noise reappears as jitter in the
                     temperature: on the 11-group HP-MR at tol_source = 1e-9
                     the coupled residual settles around 1e-6 K on a ~430 K
                     span, i.e. ~2e-9 relative. Measured: tol = 1e-8 converges
                     in 4 iterations and 23 s, while tol = 1e-10 chases noise
                     for 31 iterations and 164 s and lands on the same k and
                     the same peak temperature to 7 digits. Keep tol at least
                     ~50x above tol_source / span.
        relaxation : under-relaxation factor for plain Picard. Applied as
                     ``omega*G(T) + (1-omega)*T`` -- the literal form an
                     external coupling tool's constant-relaxation scheme uses,
                     so the two produce identical iterates rather than merely
                     equivalent ones.
        anderson_depth : > 1 switches to Anderson acceleration over that many
                     iterates. The converged answer is unchanged.

        The result is ONE evaluated state: ``temperature`` is the field the
        cross sections were evaluated at, and ``k_eff``, ``flux`` and
        ``power_density`` are the neutronics at exactly that field. The
        conduction response to that power, G(T), is ``thermal.temperature``;
        at convergence the two differ by less than ``tol`` times the span.
        """
        ctx = self.ctx
        T = np.asarray(ctx.initial_temperature() if t0 is None else t0, dtype=float)
        acc = AndersonAccelerator(depth=anderson_depth, beta=relaxation)

        k_hist, res_hist = [], []
        info = None
        converged = False
        t_start = time.perf_counter()
        evaluated = T
        for n in range(1, max_iter + 1):
            T_raw, info = coupled_map(T, ctx)
            evaluated = T
            residual = float(np.max(np.abs(T_raw - T)))
            span = max(float(np.ptp(T_raw)), 1.0)
            k_hist.append(info["neutronics"].k_eff)
            res_hist.append(residual)
            if verbose:
                print(f"  coupling {n:3d}  k = {k_hist[-1]:.7f}  "
                      f"max|G(T)-T| = {residual:.3e} K  "
                      f"T_peak = {T_raw.max():.1f} K", flush=True)
            if residual / span < tol:
                converged = True
                break
            T = acc.step(T, T_raw)
        seconds = time.perf_counter() - t_start

        return CoupledResult(
            k_eff=k_hist[-1], temperature=evaluated, power_density=info["power"],
            flux=info["neutronics"].flux, iterations=len(k_hist),
            converged=converged, seconds=seconds, device=ctx.device,
            k_history=k_hist, residual_history=res_hist,
            thermal=info["thermal"], neutronics=info["neutronics"])


@dataclass
class CoupledTransientResult:
    times: np.ndarray
    power: np.ndarray                # P(t)/P(0), neutronics
    peak_temperature: np.ndarray     # max fuel T at each step, K
    mean_temperature: np.ndarray
    k0: float
    temperature: np.ndarray          # final field
    flux: object
    steady: object                   # the coupled steady state it started from
    seconds: float
    device: str
    steps: int
    time_scheme: str = "backward-euler"
    time_orders: list = field(default_factory=list)
    step_widths: list = field(default_factory=list)
    local_errors: list = field(default_factory=list)
    rejected_steps: int = 0
    step_source_weights: list = field(default_factory=list)
    step_frequencies: list = field(default_factory=list)
    ftm_fallback_reasons: list = field(default_factory=list)
    phase_seconds: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"CoupledTransientResult(P/P0 {self.power.min():.4f} .. "
                f"{self.power.max():.4f}, peak T {self.peak_temperature.max():.1f} K, "
                f"{self.steps} steps in {self.seconds:.1f} s)")


@dataclass
class _CoupledTransientState:
    """Thermal state and source integral over the current exchange window."""

    temperature: object
    peak_temperatures: list = field(default_factory=list)
    mean_temperatures: list = field(default_factory=list)
    power_integral: object | None = None
    neutron_steps_in_window: int = 0
    window_width: float = 0.0
    last_accepted_time: float = 0.0
    feedback_hook: object | None = None
    min_power_integral: object | None = None
    kappa_xs: object | None = None
    cached_peak: float = 0.0
    cached_mean: float = 0.0


def coupled_transient(ctx: CouplingContext, t_end, dt, *, problem_at=None,
                      initial_coupled_steady=None,
                      dt_thermal=None, verbose=False, transient_kwargs=None,
                      solver_cls=None, precond_degree=1, check_every=4,
                      precond_dtype=None,
                      thermal_rtol=1e-8, thermal_maxiter=20000,
                      thermal_check_every=4, thermal_precond_degree=0,
                      thermal_diagnostics_every=0, profile=False, on_step=None,
                      power_quadrature="scheme"):
    """March the neutronics and the thermal solution together.

    Operator splitting advances neutronics with cross sections evaluated at
    the last accepted thermal state. The resulting power advances conduction
    at the next thermal exchange. The exchange interval controls the splitting
    error; neutron and thermal equations are solved separately. This is
    explicit Lie splitting, so the coupled result is FIRST order in
    ``dt_thermal`` even when the neutronics runs BDF2/BDF3 or FTM.

    The neutronics step is set by prompt kinetics (milliseconds); the thermal
    response is seconds. Backward Euler on the conduction side is
    unconditionally stable, so ``dt_thermal`` can be a multiple of a fixed
    ``dt``. With adaptive BDF, thermal exchange times are inserted as exact
    non-restarting step endpoints. In both cases the temperature advances with
    the time-weighted mean power over its interval.

    problem_at : as ``TransientSolver``'s, for the driving perturbation (a
                 drum rotation, a rod). Defaults to a stationary core, i.e. the
                 transient is driven by the feedback alone.
    initial_coupled_steady : compatible converged :class:`CoupledResult` to
                 reuse. This separates one-time hot-equilibrium startup from
                 repeated method/tolerance benchmark legs.
    precond_degree, check_every, precond_dtype : GPU-oriented controls for the neutron
                 Neumann-PCG solves. Explicit values in
                 ``transient_kwargs['linsolve_kwargs']`` take precedence over
                 ``check_every``. ``precond_dtype="float32"`` retains FP64
                 state/residual checks while evaluating only the polynomial
                 preconditioner in FP32, with automatic FP64 fallback.
    thermal_rtol, thermal_maxiter, thermal_check_every,
    thermal_precond_degree : controls for the backward-Euler conduction solve.
                 The coupled tolerance is deliberately looser than the
                 conduction solver's verification-oriented standalone default.
    thermal_diagnostics_every : compute the exact thermal energy balance every
                 N thermal advances. Zero disables its four global reductions.
    profile : collect phase times without intermediate GPU synchronization,
                 plus step/iteration/rebuild/transfer counters. GPU phases use
                 CUDA events and emit NVTX ranges.
    on_step : optional read-only observer ``(t, flux, power, temperature)``
                 after an accepted neutron step and any due thermal exchange.
                 Device arrays are passed without copying or synchronizing.
    power_quadrature : ``"scheme"`` uses FTM's exponential endpoint-source
                 integral; other schemes retain their existing endpoint rule.
                 ``"endpoint"`` forces the old rectangle rule for controlled
                 quadrature comparisons. Thermal exchange cadence is unchanged.
    """
    from .transient import TransientSolver, _step_schedule

    if (transient_kwargs or {}).get("adaptive_ftm") is not None:
        raise ValueError("adaptive_ftm currently supports neutron/precursor integration only")
    if on_step is not None and not callable(on_step):
        raise TypeError("on_step must be callable")
    xp = ctx.thermal_solver().xp
    if power_quadrature not in ("scheme", "endpoint"):
        raise ValueError("power_quadrature must be 'scheme' or 'endpoint'")
    profiler = _CoupledPhaseProfiler(xp, enabled=profile)
    adaptive_neutronics = ((transient_kwargs or {}).get("adaptive_bdf")
                           is not None)
    explicit_schedule = np.ndim(dt) != 0
    if explicit_schedule:
        if adaptive_neutronics:
            raise ValueError("an explicit coupled schedule cannot use adaptive_bdf")
        target_times, dt = _step_schedule(t_end, dt)
        initial_width = float(dt[0])
    else:
        dt = initial_width = float(dt)
    dt_thermal = float(initial_width if dt_thermal is None else dt_thermal)
    if (not np.isfinite(initial_width) or not np.isfinite(dt_thermal)
            or initial_width <= 0.0 or dt_thermal <= 0.0):
        raise ValueError("dt and dt_thermal must be finite and positive")
    if explicit_schedule:
        for exchange in np.arange(dt_thermal, t_end, dt_thermal):
            if not np.any(np.isclose(target_times, exchange, rtol=1e-12, atol=1e-14)):
                raise ValueError("explicit neutron schedule must end at every thermal exchange")
    elif not adaptive_neutronics:
        every = int(round(dt_thermal / dt))
        if every < 1 or not np.isclose(
                every * dt, dt_thermal, rtol=1e-12,
                atol=1e-14 * max(1.0, dt_thermal)):
            raise ValueError("dt_thermal must be an integer multiple of dt")
    thermal_diagnostics_every = int(thermal_diagnostics_every)
    if thermal_diagnostics_every < 0:
        raise ValueError("thermal_diagnostics_every must be non-negative")
    precond_degree = int(precond_degree)
    check_every = int(check_every)
    thermal_precond_degree = int(thermal_precond_degree)
    thermal_check_every = int(thermal_check_every)
    thermal_maxiter = int(thermal_maxiter)
    if precond_degree < 0 or thermal_precond_degree < 0:
        raise ValueError("preconditioner degrees must be non-negative")
    if check_every < 1 or thermal_check_every < 1:
        raise ValueError("residual check cadences must be positive")
    if not np.isfinite(thermal_rtol) or thermal_rtol <= 0.0:
        raise ValueError("thermal_rtol must be finite and positive")
    if thermal_maxiter < 1:
        raise ValueError("thermal_maxiter must be positive")

    # Start from the converged coupled steady state, so t=0 is an equilibrium
    # of BOTH physics -- otherwise the run opens with a transient that is
    # nothing but the initial condition relaxing.
    steady = (CoupledSolver(ctx).solve(tol=1e-8, anderson_depth=5)
              if initial_coupled_steady is None else initial_coupled_steady)
    if not steady.converged:
        raise RuntimeError(f"coupled steady state did not converge: {steady}")
    if np.asarray(steady.temperature).shape != tuple(ctx.shape):
        raise ValueError("initial_coupled_steady temperature shape is "
                         "incompatible with the coupling context")

    def thermal_solver_for(step):
        return ConductionSolver(
            ctx.grid, ctx.thermal_materials, ctx.material_map,
            bc=ctx.thermal_bc, active=ctx.active, mask_bc=ctx.thermal_mask_bc,
            ambient_temperature=ctx.ambient_temperature,
            mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
            device=ctx.device, dtype=ctx.dtype, time_step=step,
            precond_degree=thermal_precond_degree)

    thermal = thermal_solver_for(dt_thermal)

    # Everything the loop touches lives on the solve device from here on.
    steady_temperature = np.asarray(steady.temperature)
    state = _CoupledTransientState(
        temperature=xp.asarray(steady_temperature, dtype=thermal.dtype))
    state.feedback_hook = _feedback_hook(ctx, state.temperature)
    used_source_weights = []
    fuel = _fuel_mask(ctx)
    fuel_mask_device = xp.asarray(fuel)

    def xs_update_at(_t):
        # The SAME object until the temperature actually moves. The transient
        # rebuilds its operators on identity change, so with a thermal step
        # coarser than the neutronics step this skips the rebuild on every
        # sub-step -- the cross sections genuinely have not changed.
        return state.feedback_hook

    # Per-group kappa*Sigma_f on the DEVICE, blended by the same rules the
    # cross sections use. Keeping this on device is what makes the GPU path
    # worth having: the host-side power_density would pull the whole flux back
    # every step (11 groups x 12k cells is ~1 MB a step, thousands of steps),
    # and that transfer would sit in series with every solve.
    kappa_static = xp.stack(_device_fission_energy(ctx, xp, thermal.dtype))
    active_cells_device = None if ctx.active is None else xp.asarray(ctx.active).astype(bool)
    if active_cells_device is not None:
        kappa_static = xp.where(active_cells_device[None, ...], kappa_static, 0.0)

    def feedback_kappa(hook):
        # The same feedback-moved kappa the steady power edit uses, refreshed
        # whenever the temperature (hence the feedback) moves -- so t = 0 is
        # the steady state's own power shape, not a frozen-kappa variant.
        from .solver import Fields
        fields = Fields(xp, ctx.grid, ctx.materials, ctx.material_map,
                        thermal.dtype, mix_material=ctx.mix_material,
                        mix_weight=ctx.mix_weight, xs_update=hook)
        return kappa_static * _fission_scale(ctx, fields)

    state.kappa_xs = feedback_kappa(state.feedback_hook)
    cell_volumes = thermal.cell_volume
    raw_power = xp.zeros(ctx.shape, dtype=thermal.dtype)
    initial_peak = float(steady_temperature[fuel].max())
    initial_mean = float(steady_temperature[fuel].mean())
    state.cached_peak, state.cached_mean = initial_peak, initial_mean

    def advance_step(t, phi, power_ratio):
        profiler.pause_neutronics()
        try:
            step_width = float(t) - state.last_accepted_time
            if not np.isfinite(step_width) or step_width <= 0.0:
                raise RuntimeError("coupled transient received non-increasing "
                                   "accepted times")
            state.last_accepted_time = float(t)
            with profiler.region("power_edit"):
                raw_power.fill(0)
                # One contraction kernel when the multigroup transient hands
                # us its existing contiguous flux stack. If batching was not
                # available, stack only on the fused GPU path; the CPU fallback
                # consumes the legacy group list directly.
                phi_power = (phi if not kernels.use_fused(xp, "groups") or
                             hasattr(phi, "shape") else xp.stack(phi))
                kernels.group_accumulate(xp, raw_power, state.kappa_xs, phi_power)
                integral = xp.sum(raw_power * cell_volumes)       # device scalar
                scale = ctx.total_power * power_ratio / integral
                power_density_field = raw_power * scale
                state.min_power_integral = (
                    integral if state.min_power_integral is None else
                    xp.minimum(state.min_power_integral, integral))
                # Accumulate power over the sub-steps so a coarser thermal step
                # sees the mean source over its own interval, not a snapshot.
                source_weight = (getattr(neutron_solver, "step_source_weight", step_width)
                                 if power_quadrature == "scheme" else step_width)
                if not np.isfinite(source_weight) or source_weight <= 0.0:
                    raise RuntimeError("neutron source integral weight must be finite and positive")
                used_source_weights.append(float(source_weight))
                weighted_power_density = power_density_field * source_weight
                state.power_integral = (weighted_power_density if state.power_integral is None
                                  else state.power_integral + weighted_power_density)
                state.neutron_steps_in_window += 1
                state.window_width += step_width
            profiler.counters["neutronics_steps"] += 1

            # Flush a trailing partial interval too. A constant-width thermal
            # solver is reused for full windows; the final short window gets a
            # correctly shifted backward-Euler operator.
            final_window = np.isclose(t, t_end, rtol=1e-12,
                                      atol=1e-14 * max(1.0, t_end))
            full_window = np.isclose(
                state.window_width, dt_thermal, rtol=1e-12,
                atol=1e-14 * max(1.0, dt_thermal))
            if state.window_width > dt_thermal and not full_window:
                raise RuntimeError(
                    "adaptive neutron step crossed a thermal exchange time")
            if full_window or final_window:
                width = state.window_width
                stepper = (thermal if full_window
                           else thermal_solver_for(width))
                next_thermal = profiler.counters["thermal_steps"] + 1
                diagnostics = (thermal_diagnostics_every > 0 and
                               next_thermal % thermal_diagnostics_every == 0)
                with profiler.region("thermal_solve"):
                    thermal_result = stepper.step(
                        state.power_integral / width, state.temperature,
                        rtol=thermal_rtol, maxiter=thermal_maxiter,
                        check_every=thermal_check_every,
                        diagnostics=diagnostics, synchronize_timing=False)
                state.temperature = thermal_result.temperature
                profiler.counters["thermal_steps"] += 1
                profiler.counters["thermal_iterations"] += thermal_result.iterations
                profiler.counters["thermal_diagnostics"] += int(diagnostics)

                with profiler.region("feedback_update"):
                    state.feedback_hook = _feedback_hook(ctx, state.temperature)
                    state.kappa_xs = feedback_kappa(state.feedback_hook)
                profiler.counters["feedback_updates"] += 1

                # One small transfer per thermal window supplies both
                # telemetry values and validates every device-side power
                # normalization since the previous exchange.
                with profiler.region("telemetry_transfer"):
                    packet = asnumpy(xp.stack((
                        state.min_power_integral,
                        xp.max(state.temperature[fuel_mask_device]),
                        xp.mean(state.temperature[fuel_mask_device]))))
                profiler.counters["telemetry_transfers"] += 1
                if not np.isfinite(packet[0]) or packet[0] <= 0.0:
                    raise RuntimeError("no finite positive fission power in "
                                       "the transient flux")
                state.cached_peak = float(packet[1])
                state.cached_mean = float(packet[2])
                state.power_integral, state.neutron_steps_in_window = None, 0
                state.window_width = 0.0
                state.min_power_integral = None

            # Temperature is constant between thermal exchanges; reuse the
            # last two host scalars instead of synchronizing twice per neutron
            # step merely to rediscover the same values.
            state.peak_temperatures.append(state.cached_peak)
            state.mean_temperatures.append(state.cached_mean)
            if on_step is not None:
                on_step(float(t), phi, power_ratio, state.temperature)
        finally:
            profiler.resume_neutronics()

    materials = ctx.materials
    stationary = (lambda _t: (materials, ctx.material_map,
                              ctx.mix_material, ctx.mix_weight))
    neutron_solver = TransientSolver(
        ctx.grid, problem_at or stationary, ctx.kinetics, bc=ctx.bc,
        active=ctx.active, mask_bc=ctx.mask_bc,
        mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
        group_operator=ctx.group_operator, eig_solver=ctx.solver_cls,
        device=ctx.device, dtype=ctx.dtype,
        precond_degree=precond_degree,
        precond_dtype=precond_dtype,
        xs_update_at=xs_update_at, on_step=advance_step,
        phase_context=(profiler.region if profile else None))

    # Upscattering multigroup problems use rebalanced Picard; two-group
    # problems retain Anderson. Both solve the same endpoint equations.
    # Callers may override this workload-dependent choice in transient_kwargs.
    n_groups = ctx.materials[0].n_groups
    if n_groups > 2:
        # Extra group sweeps settle the spectrum before amplitude rebalance.
        step_kwargs = dict(rebalance=True, anderson_depth=1, max_sweeps=4000,
                           scatter_subsweeps=6)
    else:
        step_kwargs = dict(rebalance=False)
    step_kwargs.update(transient_kwargs or {})
    if adaptive_neutronics and dt_thermal < t_end:
        thermal_times = np.arange(dt_thermal, t_end, dt_thermal)
        thermal_times = thermal_times[
            thermal_times < t_end - 1e-14 * max(1.0, t_end)]
        supplied_value = step_kwargs.get("step_alignment_times")
        supplied = np.asarray(
            () if supplied_value is None else supplied_value, dtype=float)
        step_kwargs["step_alignment_times"] = np.unique(np.concatenate(
            (supplied, thermal_times)))
    # The coupled fixed point has already paid for a converged hot eigenpair.
    # Hand it to the transient initializer instead of immediately solving the
    # same time-zero problem again. Callers can still override this explicitly
    # through transient_kwargs for a compatibility/convergence experiment.
    step_kwargs.setdefault("initial_steady", steady.neutronics)
    if str(step_kwargs.get("step_solver", "fixed-point")).lower() != "monolithic":
        linear_options = dict(step_kwargs.get("linsolve_kwargs") or {})
        linear_options.setdefault("check_every", check_every)
        step_kwargs["linsolve_kwargs"] = linear_options
    t0 = time.perf_counter()
    profiler.resume_neutronics()
    with (profiler.region("transient_total") if profile else nullcontext()):
        transient_result = neutron_solver.solve(t_end=t_end, dt=dt, verbose=verbose, **step_kwargs)
    profiler.pause_neutronics()
    seconds = time.perf_counter() - t0

    with profiler.region("result_transfer"):
        final_temperature = asnumpy(state.temperature)
    profiler.counters["result_transfers"] += 1
    profiler.counters["initial_eigen_outer_iterations"] = \
        transient_result.steady.outer_iterations
    profiler.counters["initial_eigen_inner_iterations"] = \
        transient_result.steady.inner_iterations
    profiler.counters["initial_state_reuses"] = int(transient_result.initial_state_reused)
    profiler.counters["coupled_steady_reuses"] = int(
        initial_coupled_steady is not None)
    profiler.counters["neutron_inner_iterations"] = transient_result.total_inner_iterations
    profiler.counters["neutron_operator_rebuilds"] = transient_result.operator_rebuilds
    profiler.counters["neutron_ftm_fallbacks"] = sum(
        bool(reason) for reason in transient_result.ftm_fallback_reasons)
    profiler.counters["neutron_outer_iterations"] = sum(transient_result.step_iterations)
    profiler.counters["neutron_rejected_outer_iterations"] = sum(
        transient_result.rejected_step_iterations)
    profiler.counters["neutron_total_outer_iterations"] = (
        sum(transient_result.step_iterations) + sum(transient_result.rejected_step_iterations))
    profiler.counters["neutron_rejected_steps"] = transient_result.rejected_steps
    profiler.counters["neutron_monolithic_steps"] = int(
        transient_result.step_method == "monolithic") * (len(transient_result.times) - 1)
    # Backward-compatible name. For a monolithic run these are FGMRES applies,
    # not fission-source fixed-point sweeps; prefer neutron_outer_iterations.
    profiler.counters["neutron_fixed_point_sweeps"] = sum(transient_result.step_iterations)
    profiler.counters["mixed_precision_fallbacks"] = \
        transient_result.mixed_precision_fallbacks
    profiler.counters["cuda_graph_captures"] = transient_result.cuda_graph_captures
    profiler.counters["cuda_graph_replays"] = transient_result.cuda_graph_replays
    profiler.counters["cuda_graph_errors"] = len(transient_result.cuda_graph_errors)
    profiler.counters["adjoint_coarse_setups"] = transient_result.adjoint_coarse_setups
    profiler.counters["group_batch_active"] = int(transient_result.group_batch_active)
    profiler.counters["fgmres_workspace_bytes"] = transient_result.fgmres_workspace_bytes
    profiler.counters["neutron_bdf_max_order"] = max(transient_result.time_orders, default=1)
    phase_seconds = profiler.seconds()

    return CoupledTransientResult(
        times=transient_result.times, power=transient_result.power,
        peak_temperature=np.array([steady.peak_temperature] + state.peak_temperatures),
        mean_temperature=np.array(
            [float(np.asarray(steady.temperature)[fuel].mean())] + state.mean_temperatures),
        k0=transient_result.k0, temperature=final_temperature, flux=transient_result.flux, steady=steady,
        seconds=seconds, device=ctx.device, steps=len(transient_result.times) - 1,
        time_scheme=transient_result.time_scheme, time_orders=list(transient_result.time_orders),
        step_widths=list(transient_result.step_widths), local_errors=list(transient_result.local_errors),
        rejected_steps=transient_result.rejected_steps,
        step_source_weights=used_source_weights,
        step_frequencies=list(transient_result.step_frequencies),
        ftm_fallback_reasons=list(transient_result.ftm_fallback_reasons),
        phase_seconds=phase_seconds, counters=dict(profiler.counters))



def _device_fission_energy(ctx, xp, dtype):
    """Per-group kappa*Sigma_f as device fields, blended like the cross
    sections so a drum cell is the same fraction of absorber to the power edit
    as it is to the neutronics."""
    from .blend import MaterialBlend
    from .power import fission_energy_xs

    table, _ = fission_energy_xs(ctx.materials)
    blend = MaterialBlend(xp, ctx.grid.shape, ctx.material_map,
                          len(ctx.materials), dtype=dtype,
                          mix_material=ctx.mix_material,
                          mix_weight=ctx.mix_weight)
    return [blend.linear(table[:, g]) for g in range(table.shape[1])]


def _fuel_mask(ctx):
    """Cells that make power -- where 'peak temperature' actually means
    something. Falls back to the active mask if no material is fissile."""
    fissile = [i for i, m in enumerate(ctx.materials) if m.is_fissile]
    mm = np.asarray(ctx.material_map)
    if not fissile:
        return (np.ones(mm.shape, bool) if ctx.active is None
                else np.asarray(ctx.active).astype(bool))
    mask = np.zeros(mm.shape, bool)
    for i in fissile:
        mask |= mm == i
    return mask


def cell_centroids(grid, raster=None) -> np.ndarray:
    """(n_cells, dim) cell centroids, C-ordered to match ``grid.shape.ravel()``.

    The mesh's own physical frame -- for the triangular lattice that means the
    rasterizer's frame, which is why ``raster`` is required there: a generic
    tri-lattice convention differs by a rotation and an offset, and would place
    every vertex on the wrong cell without erroring.
    """
    from .grid import Grid
    from .tri import TriGrid

    if isinstance(grid, TriGrid):
        if raster is None:
            raise ValueError(
                "triangular centroids need the problem's raster (its lattice "
                "origin and side); pass HpmrProblem.raster")
        from .pin_power import tri_cell_centroids
        xy = tri_cell_centroids(raster)                       # (nr, nc, 2, 2)
        if len(grid.shape) == 3:
            return np.ascontiguousarray(xy.reshape(-1, 2))
        nz = grid.shape[3]
        z = (np.arange(nz) + 0.5) * grid.dz
        out = np.empty((*grid.shape, 3))
        out[..., :2] = xy[:, :, :, None, :]
        out[..., 2] = z
        return np.ascontiguousarray(out.reshape(-1, 3))

    if isinstance(grid, Grid):
        cx, cy, cz = (grid.cell_centers(a) for a in range(3))
        if grid.geometry == "cylindrical":
            r, zz = np.meshgrid(cx, cz, indexing="ij")        # ny == 1
            return np.ascontiguousarray(np.stack([r.reshape(-1),
                                                  zz.reshape(-1)], axis=1))
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        return np.ascontiguousarray(np.stack([gx.reshape(-1), gy.reshape(-1),
                                              gz.reshape(-1)], axis=1))

    raise TypeError(f"no centroid rule for {type(grid).__name__}")


def coupling_vertices(problem):
    """(coords, flat_indices) for the ACTIVE cells, in one place -- every cell
    when the problem has no ``active`` mask.

    Both preCICE participants must call this on identically-built problems.
    The nearest-neighbour mapping is exact -- the identity permutation,
    contributing precisely zero interpolation error -- only because the two
    coordinate arrays are bit-identical. If the two sides drift (a different
    ``refine``, a different drum angle), preCICE will not complain: it will
    quietly map each vertex to a nearby wrong cell, and the coupling becomes a
    smoother. That is what the echo test in the preCICE test-suite is for.
    """
    coords = cell_centroids(problem.grid, getattr(problem, "raster", None))
    active = getattr(problem, "active", None)
    idx = (np.arange(len(coords)) if active is None
           else np.flatnonzero(np.asarray(active).ravel()))
    return np.ascontiguousarray(coords[idx]), idx


def uncoupled_k(ctx: CouplingContext, temperature=None) -> float:
    """k with the feedback evaluated at a frozen temperature -- the reference
    the temperature defect is measured against. ``None`` means each material
    at its OWN reference temperature, i.e. the cross sections exactly as
    tabulated -- so no feedback hook at all. (One uniform temperature, such
    as the largest t_ref, would move every material whose reference differs.)
    """
    xs_update = (None if temperature is None
                 else _feedback_hook(ctx, temperature))
    solver = ctx.solver_cls(
        ctx.grid, ctx.materials, ctx.material_map, bc=ctx.bc,
        active=ctx.active, mask_bc=ctx.mask_bc,
        mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
        device=ctx.device, dtype=ctx.dtype,
        xs_update=xs_update)
    return solver.solve(**ctx.eigen_kwargs).k_eff


def temperature_defect_pcm(k_cold: float, k_hot: float) -> float:
    """Reactivity lost between the cold, unfed core and the hot one, in pcm.
    Negative: heating a reactor costs reactivity."""
    return 1e5 * (1.0 / k_cold - 1.0 / k_hot)


def criticality_search(rebuild, ctx_kwargs, bracket=(0.0, 180.0), target_k=1.0,
                       tol=1e-6, max_iter=20, solve_kwargs=None, verbose=False):
    """Find the control-drum angle that holds the core critical at rated power.

    This is the question the coupling exists to answer. `k(T)` on its own is a
    number; the operationally meaningful statement is how far the drums have to
    rotate to pay for the temperature defect, which needs the thermal fixed
    point re-converged at every trial angle.

    rebuild    : callable ``angle_deg -> CouplingContext``, rebuilding the
                 problem (the drum rotation changes the material map and the
                 absorber volume fractions, so it is a new problem, not a new
                 parameter).
    bracket    : (angle_lo, angle_hi) in degrees, 0 = arcs inserted.

    Returns (angle_deg, CoupledResult). Secant rather than bisection: k is
    smooth and monotone in the angle once the absorber is volume-mixed
    ("polar"), so secant lands in ~4 evaluations instead of ~20 -- and each
    evaluation is a full coupled solve.

    Raises RuntimeError unless |k - target_k| < tol at the returned angle:
    running out of iterations, or a secant step that stalls on a flat stretch
    of the worth curve (a coarse-mesh staircase), is not an answer.
    """
    solve_kwargs = dict(solve_kwargs or {})
    cache = {}

    def k_at(angle):
        if angle not in cache:
            ctx = rebuild(angle, **ctx_kwargs)
            res = CoupledSolver(ctx).solve(**solve_kwargs)
            if not res.converged:
                raise RuntimeError(f"coupled solve did not converge at "
                                   f"{angle:.3f} deg")
            cache[angle] = res
            if verbose:
                print(f"  drum {angle:7.3f} deg -> k = {res.k_eff:.7f}  "
                      f"T_peak = {res.peak_temperature:.1f} K", flush=True)
        return cache[angle]

    a, b = float(bracket[0]), float(bracket[1])
    fa = k_at(a).k_eff - target_k
    fb = k_at(b).k_eff - target_k
    if fa * fb > 0:
        raise ValueError(
            f"k - {target_k} does not change sign across the bracket "
            f"({fa + target_k:.6f} at {a} deg, {fb + target_k:.6f} at {b} deg): "
            f"the core cannot be made critical by drum rotation alone here")

    for _ in range(max_iter):
        if abs(fb) < tol or abs(fb - fa) < 1e-14:
            break
        c = b - fb * (b - a) / (fb - fa)
        c = min(max(c, min(bracket)), max(bracket))
        fc = k_at(c).k_eff - target_k
        a, fa, b, fb = b, fb, c, fc
    if not abs(fb) < tol:
        raise RuntimeError(
            f"criticality search did not converge: k = {fb + target_k:.7f} at "
            f"{b:.4f} deg, |k - {target_k}| = {abs(fb):.2e} >= tol = {tol:.1e} "
            f"after {len(cache)} coupled solves (the secant step stalled or "
            f"max_iter = {max_iter} ran out)")
    return b, cache[b]
