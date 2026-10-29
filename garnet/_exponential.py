"""Scalar coefficients for the experimental diffusion frequency transform.

This module intentionally has no array/backend dependencies. A shared frozen
frequency applies to all neutron groups during one step. The trapezoidal
variant integrates production at both endpoints; transformed BDF applies one
recurrence to neutrons and precursors. Spatial operators still act on physical
fluxes; heat transfer is not transformed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


def phi1(z: float) -> float:
    """Return (exp(z)-1)/z with its continuous value at zero."""
    if abs(z) < 1e-5:
        return 1.0 + z * (0.5 + z * (1.0/6 + z * (1.0/24 + z/120)))
    return math.expm1(z) / z


def phi2(z: float) -> float:
    """Return (exp(z)-1-z)/z**2, including its value 1/2 at zero."""
    if abs(z) < 0.5:
        term = total = 0.5
        for denominator in range(3, 18):
            term *= z / denominator
            total += term
        return total
    return (phi1(z) - 1.0) / z


def decay_moments(x: float) -> tuple[float, float, float]:
    """Return K_m = int_0^1 exp(-x*u)*u**m du for m = 0, 1, 2 and x >= 0.

    u is the fraction of the step remaining before its end. All three moments
    are positive, so weights written in them avoid the cancellation of the
    equivalent phi-function combinations for stiff families. The forward
    recurrence K_m = (m*K_(m-1) - exp(-x))/x is stable for x > 2.
    """
    if not math.isfinite(x) or x < 0.0:
        raise ValueError("decay moment argument must be finite and non-negative")
    if x > 2.0:
        decay = math.exp(-x)
        k0 = -math.expm1(-x) / x
        k1 = (k0 - decay) / x
        return k0, k1, (2.0 * k1 - decay) / x
    moments = []
    for m in range(3):
        term, total = 1.0, 1.0 / (m + 1)
        for j in range(1, 40):
            term *= -x / j
            contribution = term / (m + j + 1)
            total += contribution
            if abs(contribution) <= 1e-17 * total:
                break
        moments.append(total)
    return tuple(moments)


def population_frequency(current: float, previous: float, width: float) -> float:
    """Logarithmic growth over an accepted interval, without ratio overflow."""
    if not all(math.isfinite(v) and v > 0.0
               for v in (current, previous, width)):
        raise ValueError("frequency history needs finite positive populations and width")
    change = (current - previous) / previous
    logarithm = (math.log1p(change) if abs(change) < 0.5
                 else math.log(current) - math.log(previous))
    return logarithm / width


@dataclass(frozen=True)
class ExponentialStep:
    """Frozen-frequency temporal coefficients in physical endpoint variables."""

    width: float
    frequency: float

    def __post_init__(self):
        if not math.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("exponential step width must be finite and positive")
        if not math.isfinite(self.frequency):
            raise ValueError("exponential frequency must be finite")
        if abs(self.frequency * self.width) >= 1.0:
            raise ValueError("prototype requires abs(frequency * width) < 1")

    @property
    def diagonal_factor(self):
        return 1.0 + self.frequency * self.width

    @property
    def history_factor(self):
        return math.exp(self.frequency * self.width)

    @property
    def source_weight(self):
        """Integral weight for an exponentially varying endpoint source."""
        return self.width * phi1(-self.frequency * self.width)

    def precursor(self, decay):
        """Return old-state retention, source weight and removed-beta factor.

        C_new = retention*C_old + beta*source_weight*S_new.
        The fission emission weight is chi_total - sum chi_d*beta*removed.
        The last factor is evaluated without subtracting nearly equal numbers
        for large decay*width; at zero frequency it is exactly retention.
        """
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("precursor decay must be finite and non-negative")
        decay_times_width = decay * self.width
        frequency_times_width = self.frequency * self.width
        z = decay_times_width + frequency_times_width
        source_weight = self.width * phi1(-z)
        removed = (1.0 - decay * source_weight if abs(z) < 1e-5 else
                   (frequency_times_width + decay_times_width * math.exp(-z)) / z)
        return math.exp(-decay_times_width), source_weight, removed


@dataclass(frozen=True)
class ExponentialThetaStep(ExponentialStep):
    """Transformed trapezoidal neutron step with linear production history.

    For M*dphi/dt=R and frozen omega, the physical endpoint equation is

      M*(2/h+omega)*phi_new - R_new
          = exp(omega*h)*(M*(2/h-omega)*phi_old + R_old).

    These coefficients apply to neutrons and delayed-neutron production.
    No heat-transfer equation is transformed.
    """

    @property
    def diagonal_factor(self):
        return 2.0 + self.frequency * self.width

    @property
    def old_state_factor(self):
        return 2.0 - self.frequency * self.width

    def linear_precursor(self, decay):
        """Return retention, old/new production weights and removed beta.

        C_new = retention*C_old + old_weight*(beta*S)_old
                                  + new_weight*(beta*S)_new.
        Interpolation is linear after removing the frozen exponential.
        The removed-beta coefficient is 1-lambda*new_weight.
        """
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("precursor decay must be finite and non-negative")
        decay_times_width, frequency_times_width = decay * self.width, self.frequency * self.width
        z = -(decay_times_width + frequency_times_width)
        new_weight = self.width * phi2(z)
        # The old moment decays as 1/z**2. Do not subtract the nearly equal
        # phi_1 and phi_2 when a precursor family is very stiff.
        old_moment = (phi1(z) - phi2(z) if abs(z) < 0.5 else
                      (((z - 1.0) * math.exp(z) + 1.0) / z) / z)
        old_weight = math.exp(frequency_times_width) * self.width * old_moment
        removed = (1.0 - decay * new_weight if abs(z) < 0.5 else
                   (frequency_times_width + decay_times_width * phi1(z)) / (-z))
        return math.exp(-decay_times_width), old_weight, new_weight, removed


@dataclass(frozen=True)
class PhysicalProductionStep:
    """Precursor weights for polynomial **physical** production histories.

    Production beta*S is interpolated in physical time through the new and
    previous endpoints and, when ``previous_width`` is given, the endpoint
    before that (PARCS v2.6 Theory Manual, Eqs. 2.23-2.27, generalized to
    mapped beta products). No frequency enters: these weights are a control
    that isolates production interpolation from the transformed neutron
    equation. The older weight is negative, so a quadratic history does not
    guarantee nonnegative precursors.

      C_new = retention*C_old + older*(beta*S)_older
              + old*(beta*S)_old + new*(beta*S)_new.

    The removed-beta factor is 1-lambda*new, evaluated without cancellation.
    """

    width: float
    previous_width: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("production step width must be finite and positive")
        if (self.previous_width is not None and
                (not math.isfinite(self.previous_width) or self.previous_width <= 0.0)):
            raise ValueError("previous production width must be finite and positive")

    def weights(self, decay):
        """Return retention, older/old/new production weights and removed beta."""
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("precursor decay must be finite and non-negative")
        h = self.width
        k0, k1, k2 = decay_moments(decay * h)
        retention = math.exp(-decay * h)
        if self.previous_width is None:
            return retention, 0.0, h * k1, h * (k0 - k1), k0
        r = self.previous_width / h
        new = h * ((1.0 + r) * k0 - (2.0 + r) * k1 + k2) / (1.0 + r)
        old = h * ((1.0 + r) * k1 - k2) / r
        older = h * (k2 - k1) / (r * (1.0 + r))
        removed = ((2.0 + r) * k0 - 2.0 * k1) / (1.0 + r)
        return retention, older, old, new, removed


@dataclass(frozen=True)
class ExponentialSDIRK2Step:
    """Research coefficients for a damped, two-stage transformed kinetic step.

    Apply Alexander's SDIRK2 method to ``z = exp(-omega*(t-t_n))*y``,
    transforming neutrons AND precursors. Both physical stage equations are
    ``(diagonal_rate*I - A(t_stage))*y_stage = history/stage_width``.
    A diagonal mass matrix can replace I with the same history weighting.

    The underlying method is second order and L-stable, but does not guarantee
    positive states for arbitrarily large steps. Each macro step costs two
    implicit stage solves. Precursors use the same rational stage equations;
    this is distinct from FTM2's analytic production convolution. These scalar
    coefficients are not yet wired into TransientSolver or thermal coupling.

    R. Alexander (1977), SIAM J. Numer. Anal. 14, 1006--1021,
    doi:10.1137/0714068. See tools/assess_exponential_damping.py for the
    independent kinetics gate before a finite-volume implementation.
    """

    width: float
    frequency: float
    gamma: ClassVar[float] = 1.0 - 1.0 / math.sqrt(2.0)

    def __post_init__(self):
        # Reuse only the finite-width/frequency guard, not BE's coefficients.
        ExponentialStep(self.width, self.frequency)

    @property
    def stage_width(self):
        return self.gamma * self.width

    @property
    def diagonal_rate(self):
        return 1.0 / self.stage_width + self.frequency

    def stage_fraction(self, stage):
        if stage not in (1, 2):
            raise ValueError("SDIRK2 stage must be 1 or 2")
        return self.gamma if stage == 1 else 1.0

    def history(self, stage, initial, first_stage=None):
        """Physical stage history for a scalar or an array, without mutation."""
        fraction = self.stage_fraction(stage)
        if stage == 1:
            return math.exp(self.frequency * self.width * fraction) * initial
        if first_stage is None:
            raise ValueError("SDIRK2 stage 2 needs the first-stage state")
        a = (1.0 - self.gamma) / self.gamma
        return ((1.0 - a) * math.exp(self.frequency * self.width) * initial
                + a * math.exp(self.frequency * self.width * (1.0 - self.gamma))
                * first_stage)


@dataclass(frozen=True)
class ExponentialBDF2Step(ExponentialStep):
    """Variable-width BDF2 in one frozen exponential frame for the full state.

    Each physical history value gets its own exp(omega * age) factor. This
    avoids carrying a value in yesterday's frame when frequency changes.
    Precursors obey the same transformed recurrence as neutrons, rather than
    FTM2's analytic production convolution. One endpoint solve is sufficient.
    With no previous width this is transformed BDF1 for consistent startup.
    """

    previous_width: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if (self.previous_width is not None and
                (not math.isfinite(self.previous_width) or self.previous_width <= 0)):
            raise ValueError("previous BDF width must be finite and positive")

    @property
    def coefficients(self):
        if self.previous_width is None:
            return (1.0, -1.0)
        ratio = self.width / self.previous_width
        return ((1.0 + 2.0 * ratio) / (1.0 + ratio),
                -(1.0 + ratio), ratio * ratio / (1.0 + ratio))

    @property
    def diagonal_factor(self):
        return self.coefficients[0] + self.frequency * self.width

    @property
    def history_weights(self):
        values = [-self.coefficients[1] * math.exp(self.frequency * self.width)]
        if self.previous_width is not None:
            values.append(-self.coefficients[2] * math.exp(
                self.frequency * (self.width + self.previous_width)))
        return tuple(values)

    def history(self, initial, older=None):
        weights = self.history_weights
        if len(weights) == 1:
            return weights[0] * initial
        if older is None:
            raise ValueError("BDF2 needs the previous two accepted states")
        return weights[0] * initial + weights[1] * older
