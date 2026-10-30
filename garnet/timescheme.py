"""Time-integration schemes for garnet's transient solvers.

The BE, theta and BDF schemes can be written in one shape -- the *backward-Euler
shape* the engines already solve:

    (A + a0 theta) u^{n+1} = S^{n+1} + a0 theta Psi^n,     theta = 1/(v_g dt)

so an engine never learns which scheme is running: it is handed a collision
shift ``a0 theta`` and a carried field ``Psi``. A scheme supplies only

  * ``a0(step)``  -- the shift multiplier (may differ on the first step, where
    a multistep formula has no history yet), and
  * the rule advancing ``Psi``.

Concretely, with (1/v) du/dt = F(u) = -A u + S:

  backward Euler (BDF1)   a0 = 1      Psi = u^n
  theta-method            a0 = 1/w    Psi^{n+1} = u + ((1-w)/w)(u - Psi^n)
  BDFq                    a0 = a[q,0] Psi = H/a0, q = min(step, max_order)

For example, BDF2 has ``a = (3/2, -2, 1/2)``.  Its history term is
``H = 2 u^n - u^{n-1}/2``, hence ``Psi = H/a0 =
(4 u^n - u^{n-1})/3``.  Higher orders use the same representation, so the
spatial engines remain unaware of the time order.

Precursors
----------
Writing any BDF as ``(a0 C^{n+1} - H)/dt = beta S^{n+1} - lambda C^{n+1}`` with
the history combination ``H = sum_{j>=1} (-a_j) C^{n+1-j}`` gives

    C^{n+1} = (dt beta S^{n+1} + H) / (a0 + lambda dt)

so the analytic substitution into the flux equation carries

    bcoef_i   = beta_i a0 / (a0 + lambda_i dt)      (end-of-step fission weight)
    decayed_i = lambda_i H_i / (a0 + lambda_i dt)   (delayed source from history)

which reduces to the familiar ``beta/(1 + lambda dt)`` at a0 = 1. Consistency
of a BDF forces ``sum_j a_j = 0``, hence ``H = a0 C`` at equilibrium, hence
``C^{n+1} = C`` exactly: the unperturbed t=0 state stays an exact equilibrium
for *any* a0, which is what keeps power evolution driven purely by the applied
perturbation. The theta-method is not of this form (it carries F^n, not past
states), so it has no matching precursor treatment and is prompt-only.

Reactor-transient reference: Cherezov, Vasiliev and Ferroukhi, "Application of
Backward Differential Formula and Anderson's method for multigroup diffusion
transient equation", Ann. Nucl. Energy 210 (2025), 110837,
doi:10.1016/j.anucene.2024.110837. This module supplies the time formulas;
the spatial discretization and error-controller choices are GARNET's own.

Experimental FTM separates the diagonal multiplier from the history-source
multiplier: 1+omega*dt versus exp(omega*dt). The diffusion engine consumes
``history_factor`` independently, so no carried-state division is needed.
Its precursors and source quadrature integrate the frozen exponential model.
The second-order ``ftm2`` variant carries the physical neutron derivative and
integrates linear transformed precursor production, with no heat transform.
Its opt-in ``precursor_production`` control instead integrates linear or
PARCS-style quadratic **physical** production, independent of frequency.
``ftm-bdf2`` instead uses a damped BDF2 recurrence in a fresh exponential frame
for both neutrons and precursors, with BDF1 startup and endpoint heat weights.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import kernels
from .backend import asnumpy
from ._exponential import (ExponentialBDF2Step, ExponentialStep,
                           ExponentialThetaStep, PhysicalProductionStep,
                           population_frequency)


def _lin(xs, coeffs):
    """Linear combination of per-group array lists."""
    out = []
    for parts in zip(*xs):
        xp = kernels.module_of(parts[0])
        acc = xp.empty_like(parts[0])
        xp.multiply(parts[0], coeffs[0], out=acc)
        for c, p in zip(coeffs[1:], parts[1:]):
            kernels.axpy_inplace(xp, acc, p, c)
        out.append(acc)
    return out


def _error_norm_sums(corrected, predicted, rtol, atol):
    """Return per-field defect/scale squared norms with one host transfer.

    Reductions remain as zero-dimensional device arrays while all state fields
    are visited.  Besides requiring only one GPU synchronization for the whole
    multigroup/precursor state, streaming the fields avoids retaining two more
    complete copies of that state as residual and scale lists.
    """
    xp = kernels.module_of(corrected[0])
    sums = []
    for c, p in zip(corrected, predicted):
        residual = c - p
        sums.append(kernels.dot(xp, residual, residual))
        if atol == 0.0:
            sums.append((rtol * rtol) * kernels.dot(xp, c, c))
        else:
            scale = atol + rtol * xp.abs(c)
            sums.append(kernels.dot(xp, scale, scale))
    # A single copy of 2 scalars per field is the only synchronization here.
    pairs = asnumpy(xp.stack(sums)).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def physical_state_errors(fine, coarse, previous, *, rtol, atol=0.0, weights=None):
    """Per-field weighted RMS defects in physical flux/precursor variables.

    Each field has its own scale, so large precursor concentrations cannot
    hide a flux defect. The scale uses the accepted starting state and the
    candidate fine state. Zero fields contribute zero only for zero defect.
    ``weights`` includes active-cell masks and geometric volume factors.
    No Richardson divisor is assumed for changing frequencies/BDF histories.
    """
    if not fine or len(fine) != len(coarse) or len(fine) != len(previous):
        raise ValueError("physical error states must have matching nonempty fields")
    xp = kernels.module_of(fine[0])
    sums = []
    for new, full, old in zip(fine, coarse, previous):
        scale = atol + rtol * xp.maximum(xp.abs(old), xp.abs(new))
        defect = xp.abs(new - full)
        ratio = defect / xp.where(scale > 0, scale, 1.0)
        ratio = xp.where(scale > 0, ratio, xp.where(defect == 0, 0.0, xp.inf))
        if weights is None:
            sums.append(xp.mean(ratio * ratio))
        else:
            ratio = xp.where(weights > 0, ratio, 0.0)
            sums.append(xp.sum(weights * ratio * ratio) / xp.sum(weights))
    errors = np.sqrt(asnumpy(xp.stack(sums)))
    return np.where(np.isfinite(errors), errors, np.inf).tolist()


def physical_history_prediction(previous, start, midpoint, *, half_width, previous_half_width):
    """Quadratic physical-state extrapolation to the second-half endpoint.

    Nodes relative to the macro start are ``(-d, 0, k)`` and the target is
    ``2*k``, with actual accepted previous half width d and current k. The
    weights are ``2*r**2/(1+r), -(1+2*r), 2*(1+2*r)/(1+r)``, r=k/d.
    Difference form preserves constant fields. No exponential-frame history
    or presumed relation to a Richardson error constant is used. The raw
    prediction/correction difference is an O(k**3) resolution indicator on
    smooth second-order trajectories, not a guaranteed local-error bound.
    Callers must restart after events and guard ill-conditioned width ratios.
    """
    if not start or len(previous) != len(start) or len(midpoint) != len(start):
        raise ValueError("physical history states must have matching nonempty fields")
    k, d = float(half_width), float(previous_half_width)
    if not np.isfinite(k) or not np.isfinite(d) or min(k, d) <= 0:
        raise ValueError("physical history widths must be finite and positive")
    ratio = k / d
    if not 0.25 <= ratio <= 2.0:
        raise ValueError("physical history width ratio must lie in [0.25, 2]")
    forward = (3 * ratio + 1) / (ratio + 1)
    backward = 2 * ratio**2 / (ratio + 1)
    return [half + forward * (half - old) + backward * (prior - old)
            for prior, old, half in zip(previous, start, midpoint)]


class TimeScheme:
    """Base: backward Euler (BDF1). Stateless -- Psi is just the last solution."""

    name = "backward-euler"
    order = 1
    is_bdf = True                 # has a matching analytic precursor treatment

    def __init__(self):
        self._u = None

    def start(self, u0):
        """Seed from the (critically adjusted) steady state."""
        self._u = list(u0)
        return list(u0)

    def a0(self, step):
        return 1.0

    def history_factor(self, step):
        """Multiplier of carried flux in the physical history source."""
        return self.a0(step)

    def source_integral_weight(self, dt):
        """Endpoint-source quadrature, matching the existing thermal coupler."""
        return float(dt)

    def prepare_step(self, step, dt):
        """Prepare a possibly variable-width step (one-step schemes: no-op)."""

    def carried(self, step):
        return list(self._u)

    def push(self, u_new):
        self._u = list(u_new)

    # ---- precursors: BDF form (see module docstring) ----------------------
    def _history(self, C_hist):
        """H = sum_{j>=1} (-a_j) C^{n+1-j}; C^n for BDF1."""
        return list(C_hist[0])

    def precursor_bcoef(self, step, beta, lam, dt):
        """beta_i a0/(a0 + lam_i dt) -- the end-of-step fission weight."""
        a0 = self.a0(step)
        return [b * a0 / (a0 + l * dt) for b, l in zip(beta, lam)]

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        """lam_i H_i/(a0 + lam_i dt) -- the delayed source from history."""
        a0 = self.a0(step)
        H = self._history(C_hist)
        return [(l / (a0 + l * dt)) * h for l, h in zip(lam, H)]

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        """C^{n+1} = (dt beta S^{n+1} + H)/(a0 + lam_i dt)."""
        a0 = self.a0(step)
        H = self._history(C_hist)
        return [(h + (dt * b) * S_new) / (a0 + l * dt)
                for h, b, l in zip(H, beta, lam)]


class FrequencyTransform(TimeScheme):
    """Experimental first-order shared-frequency diffusion integration.

    Flux is backward differenced after removing exp(omega*(t-t_old)).
    Precursors integrate a constant transformed end-of-step source exactly.
    With ``frequency=None``, omega is estimated from the last two accepted
    neutron populations and the actual previous width. A prescribed frequency
    is useful for verification and zero-frequency control experiments.

    Exponents outside ``max_exponent`` use the actual BE coefficients for that
    step, rather than clipping frequency or dividing by 1+omega*dt. The bound
    keeps the time diagonal positive; it is not a full-system stability proof.
    Adaptive BDF and transport engines do not support this prototype.
    """

    name = "ftm"
    order = 1
    _coefficient_type = ExponentialStep

    def __init__(self, *, frequency=None, max_exponent=0.5,
                 frequency_iterations=0, frequency_tolerance=1e-6,
                 frequency_relaxation=1.0):
        super().__init__()
        self.prescribed_frequency = None if frequency is None else float(frequency)
        if (self.prescribed_frequency is not None
                and not np.isfinite(self.prescribed_frequency)):
            raise ValueError("FTM frequency must be finite")
        self.max_exponent = float(max_exponent)
        if not np.isfinite(self.max_exponent) or not 0.0 < self.max_exponent < 1.0:
            raise ValueError("FTM max_exponent must lie strictly between zero and one")
        if (isinstance(frequency_iterations, (bool, np.bool_))
                or not np.isfinite(frequency_iterations)
                or int(frequency_iterations) != frequency_iterations
                or frequency_iterations < 0):
            raise ValueError("FTM frequency_iterations must be a non-negative integer")
        self.frequency_iterations = int(frequency_iterations)
        if self.frequency_iterations and frequency is not None:
            raise ValueError("frequency refinement cannot use a prescribed frequency")
        self.frequency_tolerance = float(frequency_tolerance)
        self.frequency_relaxation = float(frequency_relaxation)
        if not np.isfinite(self.frequency_tolerance) or self.frequency_tolerance <= 0.0:
            raise ValueError("FTM frequency_tolerance must be finite and positive")
        if (not np.isfinite(self.frequency_relaxation)
                or not 0.0 < self.frequency_relaxation <= 1.0):
            raise ValueError("FTM frequency_relaxation must lie in (0, 1]")
        self.frequency = 0.0
        self.fallback_reason = ""
        self._coefficients = None
        self._population = self._previous_population = None
        self._accepted_width = None
        self.frequency_refinements = 0
        self.frequency_log_defect = None

    def start(self, u0):
        self._population = self._previous_population = None
        self._accepted_width = None
        return super().start(u0)

    def observe_population(self, population):
        """Commit an accepted population; never call for provisional solves."""
        self._previous_population = self._population
        self._population = float(population)

    @property
    def coefficient_key(self):
        return self.frequency, bool(self.fallback_reason)

    def prepare_step(self, step, dt):
        self.frequency_refinements = 0
        self.frequency_log_defect = None
        frequency = self.prescribed_frequency
        self.fallback_reason = ""
        if frequency is None:
            frequency = 0.0
            if self._previous_population is not None:
                try:
                    frequency = population_frequency(
                        self._population, self._previous_population,
                        self._accepted_width)
                except ValueError:
                    self.fallback_reason = "invalid-population-history"
        if not np.isfinite(frequency * dt) or abs(frequency * dt) > self.max_exponent:
            self.fallback_reason = "frequency-bound"
        self.frequency = 0.0 if self.fallback_reason else frequency
        self._coefficients = self._coefficient_type(float(dt), self.frequency)

    def refine_frequency(self, population):
        """Fit the provisional endpoint without committing accepted history.

        This is a bounded research corrector inspired by recursive frequency
        updates, not a reconstruction of SIMULATE's unpublished update rules.
        A true return requests another solve at this same endpoint. Reaching
        the iteration cap is reported through the remaining log defect.
        """
        if not self.frequency_iterations or self.fallback_reason:
            return False
        width = self._coefficients.width
        observed = population_frequency(float(population), self._population, width)
        self.frequency_log_defect = width * abs(observed-self.frequency)
        if (self.frequency_log_defect <= self.frequency_tolerance
                or self.frequency_refinements >= self.frequency_iterations):
            return False
        frequency = self.frequency + self.frequency_relaxation*(observed-self.frequency)
        if abs(frequency*width) > self.max_exponent:
            self.fallback_reason = "frequency-bound"
            self.frequency_log_defect = None
            frequency = 0.0
        self.frequency = frequency
        self._coefficients = self._coefficient_type(width, frequency)
        self.frequency_refinements += 1
        return True

    def a0(self, step):
        return self._coefficients.diagonal_factor

    def history_factor(self, step):
        return self._coefficients.history_factor

    def source_integral_weight(self, dt):
        return float(dt) if self.fallback_reason else self._coefficients.source_weight

    def push(self, u_new):
        super().push(u_new)
        self._accepted_width = self._coefficients.width

    def precursor_bcoef(self, step, beta, lam, dt):
        if self.fallback_reason:
            return super().precursor_bcoef(step, beta, lam, dt)
        return [b * self._coefficients.precursor(float(l))[2]
                for b, l in zip(beta, lam)]

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        if self.fallback_reason:
            return super().precursor_decayed(step, C_hist, S_prev, beta, lam, dt)
        return [float(l) * self._coefficients.precursor(float(l))[0] * c
                for l, c in zip(lam, C_hist[0])]

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        if self.fallback_reason:
            return super().precursor_update(step, C_hist, S_new, S_prev, beta, lam, dt)
        out = []
        for c, b, l in zip(C_hist[0], beta, lam):
            retention, source, _ = self._coefficients.precursor(float(l))
            out.append(retention * c + (b * source) * S_new)
        return out


class ExponentialTheta(FrequencyTransform):
    """Experimental shared-frequency transformed trapezoidal neutron scheme.

    Carries the accepted physical neutron derivative, so changing frequency
    or width does not reuse a derivative in the previous transformed frame.
    By default precursors integrate linearly interpolated transformed
    production. ``precursor_production="linear-physical"`` or
    ``"quadratic-physical"`` instead interpolates physical production; the
    quadratic history spans the previous accepted width and starts or
    restarts as linear-physical. The thermal driver receives its usual
    endpoint source weight, not an exponential heat quadrature or a
    transformed heat equation.
    """

    name = "ftm2"
    order = 2
    _coefficient_type = ExponentialThetaStep
    precursor_productions = ("linear-transformed", "linear-physical",
                             "quadratic-physical")

    def __init__(self, *, startup_be_steps=0,
                 precursor_production="linear-transformed", **kwargs):
        super().__init__(**kwargs)
        if (isinstance(startup_be_steps, (bool, np.bool_))
                or not np.isfinite(startup_be_steps)
                or int(startup_be_steps) != startup_be_steps or startup_be_steps < 0):
            raise ValueError("FTM2 startup_be_steps must be a non-negative integer")
        if precursor_production not in self.precursor_productions:
            raise ValueError("FTM2 precursor_production must be one of "
                             f"{self.precursor_productions}")
        self.startup_be_steps = int(startup_be_steps)
        self._startup_remaining = self.startup_be_steps
        self.precursor_production = precursor_production

    def start(self, u0):
        result = super().start(u0)
        self._derivative = [0.0 * u for u in u0]
        self._beta_old = self._pending_beta = None
        self.step_production = None
        self._clear_production_history()
        self._prepared_carried = None
        self._startup_remaining = self.startup_be_steps
        return result

    def seed_derivative(self, derivative, beta):
        """Seed the actual physical endpoint RHS at initialization/restart."""
        self._derivative = [d.copy() for d in derivative]
        self._beta_old = [b.copy() if hasattr(b, "copy") else b for b in beta]
        self._clear_production_history()
        self._prepared_carried = None

    def _clear_production_history(self):
        # Quadratic physical production must not span a restart or event.
        self._older_production = self._pending_older = None

    @property
    def production_interpolation(self):
        """Precursor production model of the prepared step, for diagnostics."""
        if self.fallback_reason:
            return "backward-euler"
        if self.precursor_production == "quadratic-physical":
            return ("quadratic-physical" if self._older_production is not None
                    else "linear-physical")
        return self.precursor_production

    def _production_weights(self, decay):
        """Return retention, older, old and new weights and removed beta."""
        if self.precursor_production == "linear-transformed":
            retention, old, new, removed = self._coefficients.linear_precursor(decay)
            return retention, 0.0, old, new, removed
        older = self._older_production
        previous = (older[2] if older is not None
                    and self.precursor_production == "quadratic-physical" else None)
        return PhysicalProductionStep(self._coefficients.width, previous).weights(decay)

    def prepare_step(self, step, dt):
        super().prepare_step(step, dt)
        if self._startup_remaining:
            self.frequency = 0.0
            self.fallback_reason = "startup-backward-euler"
            self._coefficients = self._coefficient_type(float(dt), 0.0)
        self._prepared_carried = None

    def refine_frequency(self, population):
        changed = super().refine_frequency(population)
        if changed:
            self._prepared_carried = None
        return changed

    def a0(self, step):
        return 1.0 if self.fallback_reason else self._coefficients.diagonal_factor

    def order_at(self, step):
        return 1 if self.fallback_reason else 2

    def carried(self, step):
        if self.fallback_reason:
            return list(self._u)
        if self._prepared_carried is None:
            c = self._coefficients
            self._prepared_carried = [c.old_state_factor*u + c.width*d
                                      for u, d in zip(self._u, self._derivative)]
        return list(self._prepared_carried)

    def push(self, u_new):
        width = self._coefficients.width
        diagonal, history = self.a0(0), self.history_factor(0)
        self._derivative = [(diagonal*u-history*p)/width
                            for u, p in zip(u_new, self.carried(0))]
        if self._pending_beta is not None:
            self._beta_old = [b.copy() if hasattr(b, "copy") else b
                              for b in self._pending_beta]
        if self._pending_older is not None:
            self._older_production = self._pending_older + (width,)
            self._pending_older = None
        super().push(u_new)
        self._prepared_carried = None
        self._startup_remaining = max(0, self._startup_remaining-1)

    def source_integral_weight(self, dt):
        return float(dt)

    def precursor_bcoef(self, step, beta, lam, dt):
        if self.fallback_reason:
            return TimeScheme.precursor_bcoef(self, step, beta, lam, dt)
        return [b * self._production_weights(float(l))[4]
                for b, l in zip(beta, lam)]

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        if self.fallback_reason:
            return TimeScheme.precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt)
        if self._beta_old is None:
            raise RuntimeError("FTM2 needs the accepted endpoint production history")
        S_older, beta_older, _ = self._older_production or (None, None, None)
        out = []
        for i, (c, b, l) in enumerate(zip(C_hist[0], self._beta_old, lam)):
            retention, older, old, _, _ = self._production_weights(float(l))
            history = retention*c + old*b*S_prev
            if older:
                history = history + older*beta_older[i]*S_older
            out.append(float(l) * history)
        return out

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        self._pending_beta = list(beta)
        self.step_production = self.production_interpolation
        if self.precursor_production == "quadratic-physical":
            # The previous endpoint becomes the older quadratic node on
            # acceptance, whichever formula this step uses: BE fallback and
            # startup steps still record physical production.
            self._pending_older = (
                S_prev.copy() if hasattr(S_prev, "copy") else S_prev,
                [b.copy() if hasattr(b, "copy") else b for b in self._beta_old])
        if self.fallback_reason:
            return TimeScheme.precursor_update(
                self, step, C_hist, S_new, S_prev, beta, lam, dt)
        S_older, beta_older, _ = self._older_production or (None, None, None)
        out = []
        for i, (c, b_old, b_new, l) in enumerate(zip(C_hist[0], self._beta_old, beta, lam)):
            retention, older, old, new, _ = self._production_weights(float(l))
            value = retention*c + old*b_old*S_prev + new*b_new*S_new
            if older:
                value = value + older*beta_older[i]*S_older
            out.append(value)
        return out


class ThetaMethod(TimeScheme):
    """theta-method with weight w on the end-of-step operator (w=1/2 is CN).

    A-stable for w >= 1/2 but *not* L-stable at w = 1/2, so a prompt jump rings
    rather than decays; opt-in. Precursors use the matching theta form below.

    Crank--Nicolson is first order for a perturbation applied at t=0: the
    first step's explicit half uses ``F^0 = 0``, the steady state of
    ``problem_at(0)`` (the pre-step material), not the perturbed operator.
    Second order holds for data continuous at t=0. A mid-transient restart
    must re-seed ``F^n`` through :meth:`start`.
    """

    is_bdf = True          # has its own analytic precursor treatment (below)

    def __init__(self, w):
        super().__init__()
        if not 0.0 < w <= 1.0:
            raise ValueError(f"time_weight must be in (0, 1], got {w}")
        self.w = float(w)
        self.order = 2 if self.w == 0.5 else 1
        self.name = "crank-nicolson" if self.w == 0.5 else f"theta({self.w})"
        self._carry = (1.0 - self.w) / self.w
        self._psi = None

    def start(self, u0, increment=None):
        """Seed ``Psi^n = u^n + (1 - w) v dt F^n``.

        ``increment`` is the explicit-Euler increment ``v dt F^n`` at ``u0``
        for the coming step width. The critically adjusted t=0 steady state
        has ``F^0 = 0`` (evaluated with ``problem_at(0)``), so it may be
        omitted there. A mid-transient restart must supply it: ``Psi = u``
        alone drops the ``(1 - w) F^n`` half of the trapezoid and makes the
        restarted step first order.
        """
        if increment is None:
            self._psi = list(u0)
        else:
            self._psi = [u + (1.0 - self.w) * d for u, d in zip(u0, increment)]
        return list(u0)

    def a0(self, step):
        return 1.0 / self.w

    def carried(self, step):
        return list(self._psi)

    def push(self, u_new):
        if self._carry == 0.0:
            self._psi = list(u_new)
        else:
            self._psi = [u + self._carry * (u - p)
                         for u, p in zip(u_new, self._psi)]

    # ---- precursors: theta-method form ------------------------------------
    # Applying the same weight to dC/dt = beta S - lambda C gives
    #     C^{n+1}(1 + w lam dt) = C^n(1 - (1-w) lam dt)
    #                             + dt beta [w S^{n+1} + (1-w) S^n]
    # so the end-of-step fission weight carries beta/(1 + w lam dt) -- the
    # backward-Euler coefficient with dt -> w dt -- and the history source
    # needs S^n as well as C^n (the BDF form needs only past C).
    # Equilibrium survives: with C = beta S0/lam the numerator collapses to
    # (beta S0/lam)(1 + w lam dt), so C^{n+1} = C exactly, and the flux-side
    # cancellation is identical to the BDF case.

    def precursor_bcoef(self, step, beta, lam, dt):
        return [b / (1.0 + self.w * l * dt) for b, l in zip(beta, lam)]

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        u = 1.0 - self.w
        return [(l / (1.0 + self.w * l * dt))
                * (c * (1.0 - u * l * dt) + (u * dt * b) * S_prev)
                for l, c, b in zip(lam, C_hist[0], beta)]

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        u = 1.0 - self.w
        return [(c * (1.0 - u * l * dt)
                 + (dt * b) * (self.w * S_new + u * S_prev))
                / (1.0 + self.w * l * dt)
                for c, l, b in zip(C_hist[0], lam, beta)]


# Constant-step BDF coefficients, normalized so that
#     sum_j a_j u^{n+1-j} = dt f(u^{n+1}).
# Fractions keep the defining table exact; conversion to float happens only
# when a coefficient is used with a numerical field.
_BDF_COEFFICIENTS = {
    1: (Fraction(1), Fraction(-1)),
    2: (Fraction(3, 2), Fraction(-2), Fraction(1, 2)),
    3: (Fraction(11, 6), Fraction(-3), Fraction(3, 2), Fraction(-1, 3)),
    4: (Fraction(25, 12), Fraction(-4), Fraction(3),
        Fraction(-4, 3), Fraction(1, 4)),
    5: (Fraction(137, 60), Fraction(-5), Fraction(5),
        Fraction(-10, 3), Fraction(5, 4), Fraction(-1, 5)),
    6: (Fraction(49, 20), Fraction(-6), Fraction(15, 2),
        Fraction(-20, 3), Fraction(15, 4), Fraction(-6, 5),
        Fraction(1, 6)),
}


class BDF(TimeScheme):
    """Fixed- or variable-step BDF, starting at order one up to ``max_order``.

    Orders one and two are A-stable.  Orders three through six are only
    A(alpha)-stable, with a shrinking stability wedge, so they remain opt-in
    for reactor kinetics.  All orders are strongly damping on the negative
    real axis, but rapid control motion can excite modes outside that wedge.

    The first accepted step uses BDF1, the next BDF2, and so on until the
    requested maximum order is reached.  This avoids inventing unavailable
    pre-history and preserves an initially critical equilibrium exactly.

    Order cap: at a constant width those startup steps carry O(h**2) and
    O(h**3) local errors that the later steps cannot remove, so fixed-step
    BDF3--BDF6 converge globally at second order (measured 2.09 for BDF3 on
    a smooth ramp), with a smaller constant than BDF2. Restarts repeat the
    startup. Third order needs startup widths that shrink faster than h: the
    adaptive controller selects them, or an explicit nonuniform schedule
    graded from ~h*sqrt(h/T) by a ratio <= 1.5 (measured 3.25 for BDF3).
    """

    is_bdf = True

    def __init__(self, max_order=2):
        super().__init__()
        max_order = int(max_order)
        if max_order not in _BDF_COEFFICIENTS:
            raise ValueError("BDF max_order must be between 1 and 6")
        self.max_order = max_order
        self.order = max_order
        self.name = ("backward-euler" if max_order == 1
                     else f"bdf{max_order}")
        self._history_u = []       # [u^n, u^{n-1}, ...]
        self._dt_history = []      # [dt_n, dt_{n-1}, ...], accepted widths
        self._prepared = None
        self._automatic_order = False
        self._selected_order = max_order

    def start(self, u0):
        self._history_u = [list(u0)]
        self._dt_history = []
        self._prepared = None
        if self._automatic_order:
            self._selected_order = 1
        self._u = self._history_u[0]
        return list(u0)

    def order_at(self, step):
        # History availability, rather than the absolute step number, also
        # makes an explicit discontinuity restart drop immediately to BDF1.
        available = (len(self._history_u) if self._history_u
                     else max(int(step), 1))
        cap = self._selected_order if self._automatic_order else self.max_order
        return min(max(int(step), 1), cap, available)

    def enable_order_selection(self):
        """Let a controller choose the active order up to ``max_order``."""
        self._automatic_order = True
        self._selected_order = 1
        self._prepared = None

    def select_order(self, order):
        """Select the order used by the next prepared step."""
        order = int(order)
        if not self._automatic_order:
            raise RuntimeError("BDF automatic order selection is not enabled")
        if order < 1 or order > self.max_order:
            raise ValueError("selected BDF order must be between one and max_order")
        self._selected_order = order
        self._prepared = None

    @property
    def history_length(self):
        """Number of accepted states available to recurrence/predictors."""
        return len(self._history_u)

    def coefficients(self, step):
        if self._prepared is not None and self._prepared[0] == int(step):
            return self._prepared[2]
        return tuple(float(a) for a in _BDF_COEFFICIENTS[self.order_at(step)])

    def prepare_step(self, step, dt):
        """Construct variable-step coefficients from accepted step widths.

        With nodes measured backwards from the new time and scaled by the
        proposed width, the coefficients satisfy
        ``sum_j a_j x_j**m = delta[m, 1]`` through degree q.  This is the
        non-uniform-grid BDF formula; equal widths reproduce the exact table.
        """
        step, dt = int(step), float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("BDF step width must be finite and positive")
        q = self.order_at(step)
        if len(self._dt_history) < q - 1:
            raise ValueError("insufficient accepted step-width history")
        nodes = [0.0]
        distance = dt
        for j in range(1, q + 1):
            nodes.append(-distance / dt)
            if j < q:
                distance += self._dt_history[j - 1]
        vandermonde = np.asarray(
            [[x**m for x in nodes] for m in range(q + 1)], dtype=float)
        rhs = np.zeros(q + 1)
        rhs[1] = 1.0
        coeffs = tuple(float(a) for a in np.linalg.solve(vandermonde, rhs))
        self._prepared = (step, dt, coeffs)

    def a0(self, step):
        return self.coefficients(step)[0]

    @staticmethod
    def _combine_history(history, coeffs):
        if len(history) < len(coeffs):
            raise ValueError("insufficient BDF history for requested order")
        return _lin(history[:len(coeffs)], coeffs)

    def carried(self, step):
        a = self.coefficients(step)
        # H = sum_{j>=1} -a_j u^{n+1-j}; the engine expects Psi = H/a0.
        return self._combine_history(
            self._history_u, [-a_j / a[0] for a_j in a[1:]])

    def predict_history(self, history, dt, *, max_degree=None):
        """Extrapolate an accepted state history one step forward.

        ``history`` is ordered newest first and may contain any state with the
        same list-of-arrays representation as the flux.  This lets the
        adaptive controller apply the same polynomial predictor to precursor
        and coupled state without maintaining a second, subtly different set
        of time nodes.

        ``max_degree`` is primarily useful for candidate-order diagnostics.
        The production predictor uses the scheme's configured maximum order.
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("BDF prediction width must be finite and positive")
        history = list(history)
        degree = self.max_order if max_degree is None else int(max_degree)
        if degree < 0 or degree > self.max_order:
            raise ValueError("BDF prediction degree must be between zero and "
                             "max_order")
        n_points = min(len(history), degree + 1)
        if n_points == 0:
            raise RuntimeError("BDF predictor needs non-empty accepted history")
        if len(self._dt_history) < n_points - 1:
            raise ValueError("insufficient accepted step-width history")
        nodes = [0.0]
        distance = 0.0
        for j in range(1, n_points):
            distance += self._dt_history[j - 1]
            nodes.append(-distance)
        target = dt
        weights = []
        for j, xj in enumerate(nodes):
            w = 1.0
            for m, xm in enumerate(nodes):
                if m != j:
                    w *= (target - xm) / (xj - xm)
            weights.append(w)
        return _lin(history[:n_points], weights)

    def predict(self, dt, *, max_degree=None):
        """Polynomially extrapolate accepted flux history one step forward.

        This is the multilevel BDF predictor used to initialize nonlinear
        feedback iterations and estimate local error. Up to ``max_order + 1``
        accepted states are retained, giving a degree-``max_order`` predictor
        once startup is complete; earlier steps use every available state.
        Actual nonuniform time nodes are respected. Startup with one state is
        a constant predictor. No state is mutated, so a rejected/probed step
        is free to call it repeatedly.
        """
        if not self._history_u:
            raise RuntimeError("BDF predictor must be seeded with start()")
        return self.predict_history(
            self._history_u, dt, max_degree=max_degree)

    error_norms = ("concatenated", "blockwise")

    @staticmethod
    def error_norm(corrected, predicted, *, rtol, atol=0.0, norm="concatenated"):
        """Normalized predictor--corrector defect used for step acceptance.

        This follows Eq. (44) of Cherezov et al.: the Euclidean norm of the
        state correction divided by the norm of ``atol + rtol*corrected``.
        Both arguments are iterables of fields (group fluxes, precursor
        families, thermal or other coupled state). ``"concatenated"`` applies
        Eq. (44) to all fields joined, so the field with the largest values in
        its own units dominates both norms. ``"blockwise"`` applies it to each
        field and returns the largest, so every field meets the tolerance
        relative to its own scale. A value no greater than one passes.
        """
        if norm not in BDF.error_norms:
            raise ValueError(f"BDF error norm must be one of {BDF.error_norms}")
        rtol, atol = float(rtol), float(atol)
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("BDF error rtol must be finite and positive")
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError("BDF error atol must be finite and non-negative")
        corrected, predicted = list(corrected), list(predicted)
        if len(corrected) != len(predicted) or not corrected:
            raise ValueError("corrected and predicted states must have equal, "
                             "non-zero lengths")
        numerator, denominator = _error_norm_sums(
            corrected, predicted, rtol, atol)
        if norm == "concatenated":
            # Plain left-to-right sums, as the fields were always accumulated
            # (Python's sum() of floats is compensated since 3.12).
            numerator = np.add.accumulate(numerator)[-1:]
            denominator = np.add.accumulate(denominator)[-1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denominator > 0.0, numerator / denominator,
                             np.where(numerator == 0.0, 0.0, np.inf))
        return float(np.sqrt(ratio.max()))

    def push(self, u_new):
        self._history_u.insert(0, list(u_new))
        # The BDFq recurrence needs q accepted states, while a degree-q LTE
        # predictor needs q+1. Retain that one extra state solely for error
        # estimation; coefficients/carried() still consume at most q.
        del self._history_u[self.max_order + 1:]
        if self._prepared is not None:
            self._dt_history.insert(0, self._prepared[1])
            del self._dt_history[self.max_order:]
        self._u = self._history_u[0]

    def _history(self, C_hist, step=None):
        # Precursor callers know the step; legacy callers do not.  The number
        # of populated entries identifies the same startup order in the latter
        # case because histories are advanced exactly once per accepted step.
        q = (min(len(C_hist), self.max_order) if step is None
             else self.order_at(step))
        a = (self.coefficients(step) if step is not None else
             tuple(float(x) for x in _BDF_COEFFICIENTS[q]))
        return self._combine_history(C_hist, [-a_j for a_j in a[1:]])

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        a0 = self.a0(step)
        H = self._history(C_hist, step)
        return [(l / (a0 + l * dt)) * h for l, h in zip(lam, H)]

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        a0 = self.a0(step)
        H = self._history(C_hist, step)
        return [(h + (dt * b) * S_new) / (a0 + l * dt)
                for h, b, l in zip(H, beta, lam)]


class BDF2(BDF):
    """Backward-compatible spelling for the second-order BDF scheme."""

    def __init__(self):
        super().__init__(2)


class ExponentialBDF2(FrequencyTransform):
    """Experimental damped frequency transform for neutrons and precursors.

    BDF2 differentiates all state fields in one frozen frame per step. The
    physical time diagonal is a0/h + omega; history j is multiplied by
    exp(omega * age_j). Changing width or frequency therefore never reuses
    transformed values from a different frame. Precursors follow this same
    rational recurrence, not the analytic source convolution of ``ftm2``.

    Native startup/restarts use BDF1. Unsafe frequency, history exponent or
    width-ratio growth uses actual BE. Fixed and explicit nonuniform stepping
    are supported; this class deliberately is not an adaptive BDF controller.
    Constant-step BDF2 damps stiff modes, but arbitrary-step positivity and
    stability of the history-fitted frequency do not follow from that fact.
    """

    name = "ftm-bdf2"
    order = 2
    max_order = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history_u = []
        self._previous_width = None

    def start(self, u0):
        result = super().start(u0)
        self._history_u = [list(u0)]
        self._previous_width = None
        return result

    def _prepare_coefficients(self, width):
        previous = self._previous_width if len(self._history_u) > 1 else None
        if previous is not None and not self.fallback_reason:
            if width / previous > 1.0 + np.sqrt(2.0):
                self.fallback_reason = "step-ratio-bound"
            elif abs(self.frequency * (width + previous)) > 2.0 * self.max_exponent:
                self.fallback_reason = "history-frequency-bound"
        if self.fallback_reason:
            self.frequency, previous = 0.0, None
        self._coefficients = ExponentialBDF2Step(width, self.frequency, previous)

    def prepare_step(self, step, dt):
        super().prepare_step(step, dt)
        self._prepare_coefficients(float(dt))

    def refine_frequency(self, population):
        changed = super().refine_frequency(population)
        if changed:
            self._prepare_coefficients(self._coefficients.width)
        return changed

    def order_at(self, step):
        return len(self._coefficients.coefficients) - 1

    def history_factor(self, step):
        return 1.0  # carried() already contains each BDF/exponential weight.

    def carried(self, step):
        return _lin(self._history_u, self._coefficients.history_weights)

    def _history(self, C_hist):
        return _lin(C_hist[:len(self._coefficients.history_weights)],
                    self._coefficients.history_weights)

    def precursor_bcoef(self, step, beta, lam, dt):
        return TimeScheme.precursor_bcoef(self, step, beta, lam, dt)

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        return TimeScheme.precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt)

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        return TimeScheme.precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt)

    def push(self, u_new):
        super().push(u_new)
        self._history_u.insert(0, list(u_new))
        del self._history_u[2:]
        self._previous_width = self._coefficients.width

    def source_integral_weight(self, dt):
        return float(dt)  # Keep the existing endpoint thermal exchange rule.


class BDFStepController:
    """Bounded predictor--corrector step-width controller.

    The accepted-step proposal is the order-``q`` form of Cherezov et al.
    Eq. (45), with conservative growth/shrink limits. A failed error test uses
    the paper's explicit factor-two reduction. Candidate order selection
    compares the ``q-1``, ``q``, and ``q+1`` width proposals without another
    spatial solve.
    """

    def __init__(self, *, safety=1.25, floor=1e-6,
                 min_factor=0.2, max_factor=5.0,
                 rejection_strategy="half", reject_max_factor=0.5):
        safety = float(safety)
        floor = float(floor)
        min_factor, max_factor = float(min_factor), float(max_factor)
        if not np.isfinite(safety) or safety <= 1.0:
            raise ValueError("BDF controller safety must be finite and > 1")
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("BDF controller floor must be finite and >= 0")
        if not (0.0 < min_factor <= 1.0 <= max_factor):
            raise ValueError("BDF controller factors must bracket one")
        if rejection_strategy not in ("half", "error"):
            raise ValueError("BDF rejection_strategy must be 'half' or 'error'")
        reject_max_factor = float(reject_max_factor)
        if (not np.isfinite(reject_max_factor)
                or not min_factor <= reject_max_factor < 1.0):
            raise ValueError(
                "BDF reject_max_factor must be finite, >= min_factor, and < 1")
        self.safety = safety
        self.floor = floor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.rejection_strategy = rejection_strategy
        self.reject_max_factor = reject_max_factor

    def factor(self, error, order, *, accepted=True):
        """Return the multiplicative proposal ``h_new / h``."""
        error, order = float(error), int(order)
        if order < 1 or order > 6:
            raise ValueError("BDF controller order must be between 1 and 6")
        if not np.isfinite(error) or error < 0.0:
            if np.isinf(error) and error > 0.0:
                return (0.5 if self.rejection_strategy == "half"
                        else self.min_factor)
            raise ValueError("BDF controller error must be non-negative")
        if not accepted or error > 1.0:
            if self.rejection_strategy == "half":
                return 0.5
            raw = 1.0 / (self.safety
                         * (error + self.floor) ** (1.0 / (order + 1)))
            return min(max(raw, self.min_factor), self.reject_max_factor)
        raw = 1.0 / (self.safety
                     * (error + self.floor) ** (1.0 / (order + 1)))
        return min(max(raw, self.min_factor), self.max_factor)

    def propose(self, dt, error, order, *, accepted=True):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("BDF controller step width must be positive")
        return dt * self.factor(error, order, accepted=accepted)

    def choose_order(self, dt, errors, current_order, *, hysteresis=1.05):
        """Choose the candidate with the largest safe next-step proposal.

        ``errors`` maps candidate BDF orders to normalized defects. An order
        change must improve the proposed width by ``hysteresis`` over retaining
        the current order, which suppresses one-step order chatter.
        """
        dt, current_order = float(dt), int(current_order)
        hysteresis = float(hysteresis)
        if current_order not in errors:
            raise ValueError("candidate errors must include current_order")
        if not np.isfinite(hysteresis) or hysteresis < 1.0:
            raise ValueError("BDF order hysteresis must be finite and >= 1")
        proposals = {
            int(order): self.propose(
                dt, error, int(order), accepted=float(error) <= 1.0)
            for order, error in errors.items()
        }
        best = max(proposals, key=lambda order: (proposals[order], order))
        if (best != current_order
                and proposals[best] < hysteresis * proposals[current_order]):
            best = current_order
        return best, proposals[best]


def make_time_scheme(time_scheme="backward-euler", time_weight=None, *, ftm_kwargs=None):
    """Resolve the ``time_scheme`` / ``time_weight`` arguments to an object."""
    name = str(time_scheme).lower()
    if ftm_kwargs is not None and name not in ("ftm", "exponential", "ftm2", "ftm-bdf2"):
        raise ValueError("ftm_kwargs requires time_scheme='ftm', 'ftm2' or 'ftm-bdf2'")
    if time_weight is not None and time_weight != 1.0:
        if time_scheme not in ("backward-euler", "theta"):
            raise ValueError("time_weight applies to the theta-method only; "
                             f"got time_scheme={time_scheme!r}")
        return ThetaMethod(time_weight)
    if name in ("ftm", "exponential"):
        return FrequencyTransform(**(ftm_kwargs or {}))
    if name == "ftm2":
        return ExponentialTheta(**(ftm_kwargs or {}))
    if name == "ftm-bdf2":
        return ExponentialBDF2(**(ftm_kwargs or {}))
    if name in ("backward-euler", "be"):
        return TimeScheme()
    if name == "bdf1":
        return BDF(1)
    if name.startswith("bdf") and name[3:].isdigit():
        return BDF(int(name[3:]))
    if name in ("crank-nicolson", "cn"):
        return ThetaMethod(0.5)
    raise ValueError(f"unknown time_scheme {time_scheme!r} (backward-euler, "
                     f"bdf2..bdf6, ftm, ftm2, ftm-bdf2, crank-nicolson)")
