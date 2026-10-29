"""Independent scalar gates; runnable without NumPy/SciPy or package import.

    python3 tests/verification/test_exponential_coefficients.py

Load only the production scalar coefficient module. Array solver integration
is covered separately in test_exponential_transient.py.
"""

from decimal import Decimal, localcontext
import importlib.util
import math
from pathlib import Path
import sys
import unittest


_path = Path(__file__).resolve().parents[2] / "garnet" / "_exponential.py"
_spec = importlib.util.spec_from_file_location("_garnet_ftm_coefficients", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
ExponentialStep = _module.ExponentialStep
ExponentialThetaStep = _module.ExponentialThetaStep
PhysicalProductionStep = _module.PhysicalProductionStep
decay_moments = _module.decay_moments
population_frequency = _module.population_frequency
phi1 = _module.phi1
phi2 = _module.phi2


class TestExponentialCoefficients(unittest.TestCase):
    def test_phi1_against_high_precision_including_zero(self):
        with localcontext() as context:
            context.prec = 70
            for z in (0.0, 1e-12, -1e-12, 1e-6, -1e-4, 0.4, -20.0, -1000.0):
                x = Decimal(str(z))
                reference = float((x.exp() - 1) / x) if z else 1.0
                self.assertTrue(math.isclose(phi1(z), reference, rel_tol=3e-15), z)

    def test_precursor_convolution_and_removable_singularity(self):
        with localcontext() as context:
            context.prec = 70
            for width, frequency, decay in ((0.1, 0.3, 0.08),
                                             (2.0, -0.08, 0.08),
                                             (2.0, -0.080000000001, 0.08),
                                             (0.1, -2.0, 0.0),
                                             (1.0, 0.0, 1000.0)):
                step = ExponentialStep(width, frequency)
                retention, source, removed = step.precursor(decay)
                h, w, lam = map(lambda v: Decimal(str(v)), (width, frequency, decay))
                z = (lam + w) * h
                integral = h if not z else h * (1 - (-z).exp()) / z
                self.assertTrue(math.isclose(source, float(integral), rel_tol=3e-15))
                self.assertTrue(math.isclose(retention, float((-lam*h).exp()),
                                             rel_tol=3e-15, abs_tol=1e-300))
                self.assertTrue(math.isclose(removed, float(1 - lam*integral),
                                             rel_tol=3e-15, abs_tol=1e-16))

    def test_zero_frequency_retains_equilibrium_without_cancellation(self):
        for decay in (0.08, 4.0, 100.0):
            retention, source, removed = ExponentialStep(0.1, 0.0).precursor(decay)
            self.assertAlmostEqual(removed, retention, delta=3e-16)
            beta, fission = 0.0065, 2.0
            old = beta * fission / decay
            self.assertAlmostEqual(retention*old + beta*source*fission, old,
                                   delta=1e-15)

    def test_positive_and_negative_complete_kinetics_eigenmodes(self):
        beta, decay, generation = 0.0065, 0.08, 1e-4
        for reactivity in (0.002, -0.002):
            a, b, c = (reactivity-beta)/generation, decay, beta/generation
            gap = math.sqrt((a+decay)**2 + 4*b*c)
            omega = (a-decay+gap)/2
            state = (1.0, c/(decay+omega))
            width = 0.1
            coefficients = ExponentialStep(width, omega)
            retention, source, _ = coefficients.precursor(decay)
            n, precursor = state
            n_new = ((coefficients.history_factor*n/width + b*retention*precursor)
                     / (coefficients.diagonal_factor/width-a-b*c*source))
            C_new = retention*precursor + c*source*n_new
            for initial, new in zip(state, (n_new, C_new)):
                self.assertTrue(math.isclose(new, initial*math.exp(omega*width),
                                             rel_tol=3e-14))

    def test_heat_source_weight_integrates_physical_power(self):
        for frequency in (-2.0, 0.0, 2.0):
            width, power0 = 0.1, 10.0
            power1 = power0 * math.exp(frequency*width)
            measured = ExponentialStep(width, frequency).source_weight * power1
            expected = power0 * width * phi1(frequency*width)
            self.assertTrue(math.isclose(measured, expected, rel_tol=3e-15))

    def test_frequency_uses_actual_previous_width_and_large_populations(self):
        self.assertAlmostEqual(population_frequency(math.exp(0.03), 1.0, 0.01), 3.0)
        self.assertTrue(math.isfinite(population_frequency(1e300, 1e-300, 0.1)))
        for args in ((0.0, 1.0, 0.1), (1.0, -1.0, 0.1), (1.0, 1.0, 0.0)):
            with self.assertRaises(ValueError):
                population_frequency(*args)

    def test_unsafe_step_coefficients_are_rejected(self):
        for width, frequency in ((0.0, 0.0), (0.1, math.inf), (1.0, -1.0), (1.0, 1.0)):
            with self.assertRaises(ValueError):
                ExponentialStep(width, frequency)


class TestExponentialThetaCoefficients(unittest.TestCase):
    def test_phi2_against_high_precision(self):
        with localcontext() as context:
            context.prec = 90
            for z in (0.0, 1e-12, -1e-12, 0.1, -0.1, 0.499, 0.5,
                      -0.5, -20.0, -1e12):
                x = Decimal(str(z))
                exact = float((x.exp()-1-x)/(x*x)) if z else 0.5
                self.assertTrue(math.isclose(phi2(z), exact, rel_tol=3e-15), z)

    def test_linear_production_weights_against_high_precision(self):
        with localcontext() as context:
            context.prec = 90
            for width, frequency, decay in ((0.1, 0.3, 0.08),
                                             (2.0, -0.08, 0.08),
                                             (2.0, -0.080000000001, 0.08),
                                             (0.1, -2.0, 0.0),
                                             (1.0, 0.0, 1e12)):
                step = ExponentialThetaStep(width, frequency)
                retention, old, new, removed = step.linear_precursor(decay)
                h, w, lam = [Decimal(str(v)) for v in (width, frequency, decay)]
                z = -(lam+w)*h
                f1 = (z.exp()-1)/z if z else Decimal(1)
                f2 = (z.exp()-1-z)/(z*z) if z else Decimal('0.5')
                expected = ((-lam*h).exp(), (w*h).exp()*h*(f1-f2),
                            h*f2, 1-lam*h*f2)
                for value, exact in zip((retention, old, new, removed), expected):
                    self.assertTrue(math.isclose(value, float(exact), rel_tol=4e-15,
                                                 abs_tol=1e-300), (width, frequency, decay))

    def test_exact_complete_kinetics_eigenmodes(self):
        beta, decay, generation = 0.0065, 0.08, 1e-4
        for rho in (-0.002, 0.002):
            a, production = (rho-beta)/generation, beta/generation
            gap = math.sqrt((a+decay)**2 + 4*decay*production)
            frequency = (a-decay+gap)/2
            n, precursor, width = 1.0, production/(decay+frequency), 0.1
            step = ExponentialThetaStep(width, frequency)
            retention, old, new, _ = step.linear_precursor(decay)
            history = step.history_factor*(step.old_state_factor*n/width + frequency*n)
            new_n = ((history + decay*(retention*precursor + old*production*n))
                     / (step.diagonal_factor/width-a-decay*new*production))
            new_C = retention*precursor + old*production*n + new*production*new_n
            for initial, final in ((n, new_n), (precursor, new_C)):
                self.assertTrue(math.isclose(final, initial*math.exp(frequency*width),
                                             rel_tol=4e-14))

    def test_mixed_modes_converge_at_second_order(self):
        beta, decay, generation, rho = 0.0065, 0.08, 1/6600, 0.002
        a, production = (rho-beta)/generation, beta/generation
        initial_C = beta/(generation*decay)
        gap = math.sqrt((a+decay)**2 + 4*decay*production)
        slow, fast = (a-decay+gap)/2, (a-decay-gap)/2
        fraction = (a+decay*initial_C-fast)/(slow-fast)

        def exact(t):
            return fraction*math.exp(slow*t) + (1-fraction)*math.exp(fast*t)

        for prescribed in (0.0, 0.5, -0.5, None):
            errors = []
            for width in (0.005, 0.0025, 0.00125):
                n, precursor, derivative, previous = 1.0, initial_C, rho/generation, None
                error = 0.0
                for index in range(round(0.2/width)):
                    frequency = (prescribed if prescribed is not None else
                                 0.0 if previous is None else
                                 population_frequency(n, previous, width))
                    step = ExponentialThetaStep(width, frequency)
                    retention, old, new, _ = step.linear_precursor(decay)
                    history = step.history_factor*(step.old_state_factor*n/width+derivative)
                    new_n = ((history+decay*(retention*precursor+old*production*n))
                             / (step.diagonal_factor/width-a-decay*new*production))
                    precursor = retention*precursor+old*production*n+new*production*new_n
                    derivative = step.diagonal_factor*new_n/width-history
                    previous, n = n, new_n
                    error = max(error, abs(n/exact((index+1)*width)-1))
                errors.append(error)
            self.assertGreater(errors[0]/errors[1], 3.8, (prescribed, errors))
            self.assertGreater(errors[1]/errors[2], 3.8, (prescribed, errors))


def _decimal_lagrange_weights(width, previous, decay):
    """Independent reference: s-monomial Lagrange basis, integrated by parts.

    Returns the exact older/old/new weights of int_0^h exp(-lam*(h-s)) P(s) ds
    for P interpolated through s=-previous, 0, h (linear without previous).
    """
    h, lam = Decimal(str(width)), Decimal(str(decay))
    if lam:
        moments = [(1 - (-lam*h).exp())/lam]
        for k in (1, 2):
            moments.append((h**k - k*moments[-1])/lam)
    else:
        moments = [h, h*h/2, h**3/3]
    if previous is None:
        # P = P0*(1-s/h) + P1*s/h
        return (Decimal(0), moments[0] - moments[1]/h, moments[1]/h)
    p = Decimal(str(previous))
    # Monomial coefficients (1, s, s**2) of each Lagrange basis polynomial.
    older = (Decimal(0), -h/(p*(p+h)), 1/(p*(p+h)))
    old = (Decimal(1), (h-p)/(p*h), -1/(p*h))
    new = (Decimal(0), p/(h*(p+h)), 1/(h*(p+h)))
    return tuple(sum(c*m for c, m in zip(basis, moments)) for basis in (older, old, new))


class TestPhysicalProductionCoefficients(unittest.TestCase):
    def test_decay_moments_against_high_precision(self):
        with localcontext() as context:
            context.prec = 90
            for x in (0.0, 1e-12, 1e-6, 0.3, 1.999, 2.0, 2.001, 7.0, 40.0, 1e12):
                X = Decimal(str(x))
                if x:
                    k0 = (1 - (-X).exp())/X
                    k1 = (k0 - (-X).exp())/X
                    exact = (k0, k1, (2*k1 - (-X).exp())/X)
                else:
                    exact = (Decimal(1), Decimal(1)/2, Decimal(1)/3)
                for value, reference in zip(decay_moments(x), exact):
                    self.assertTrue(math.isclose(value, float(reference), rel_tol=4e-15), x)

    def test_physical_production_weights_against_high_precision(self):
        with localcontext() as context:
            context.prec = 90
            for width, previous, decay in ((0.1, None, 0.08), (0.1, 0.1, 0.08),
                                           (0.02, 0.05, 3.01), (0.3, 0.07, 0.0),
                                           (1.0, 2.0, 30.0), (0.01, 0.0025, 1e4),
                                           (0.5, 0.4, 4.0), (1.0, 1.0, 1e12),
                                           (0.004, 0.001, 0.0124)):
                retention, older, old, new, removed = PhysicalProductionStep(
                    width, previous).weights(decay)
                exact = _decimal_lagrange_weights(width, previous, decay)
                lam_h = Decimal(str(decay))*Decimal(str(width))
                case = (width, previous, decay)
                self.assertTrue(math.isclose(retention, float((-lam_h).exp()),
                                             rel_tol=3e-15, abs_tol=1e-300), case)
                scale = float(sum(abs(w) for w in exact))
                for value, reference in zip((older, old, new), exact):
                    self.assertTrue(math.isclose(value, float(reference), rel_tol=2e-14,
                                                 abs_tol=1e-15*scale), case)
                # The removed fraction is formed without cancellation, even
                # when it is a tiny difference for a very stiff family.
                expected = 1 - Decimal(str(decay))*exact[2]
                self.assertTrue(math.isclose(removed, float(expected), rel_tol=2e-14), case)

    def test_linear_physical_is_zero_frequency_linear_transformed(self):
        for width, decay in ((0.1, 0.08), (2.0, 0.4), (0.01, 1e4), (0.3, 0.0)):
            retention, older, old, new, removed = PhysicalProductionStep(width).weights(decay)
            expected = ExponentialThetaStep(width, 0.0).linear_precursor(decay)
            self.assertEqual(older, 0.0)
            for value, reference in zip((retention, old, new, removed), expected):
                self.assertTrue(math.isclose(value, reference, rel_tol=3e-15,
                                             abs_tol=1e-300), (width, decay))

    def test_quadratic_weights_are_exact_for_constant_and_linear_production(self):
        # Constant production in equilibrium, and a long previous interval
        # approaching the two-point linear interpolant.
        for decay in (0.08, 3.0, 300.0):
            retention, older, old, new, removed = PhysicalProductionStep(0.2, 0.07).weights(decay)
            self.assertAlmostEqual(retention + decay*(older + old + new), 1.0, delta=4e-16)
            self.assertAlmostEqual(removed, 1 - decay*new, delta=4e-16)
            self.assertLess(older, 0.0)
            far = PhysicalProductionStep(0.2, 2e8).weights(decay)
            linear = PhysicalProductionStep(0.2).weights(decay)
            for value, reference in zip(far, linear):
                self.assertAlmostEqual(value, reference, delta=1e-9)

    def test_invalid_physical_production_inputs_are_rejected(self):
        for args in ((0.0,), (math.inf,), (0.1, 0.0), (0.1, -1.0), (0.1, math.nan)):
            with self.assertRaises(ValueError):
                PhysicalProductionStep(*args)
        for decay in (-1.0, math.inf):
            with self.assertRaises(ValueError):
                PhysicalProductionStep(0.1, 0.1).weights(decay)
        with self.assertRaises(ValueError):
            decay_moments(-1e-3)


if __name__ == "__main__":
    unittest.main()
