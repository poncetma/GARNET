"""Independent matrix, boundary, conservation, and analytic column checks."""

import numpy as np
import pytest

from garnet.axial_diffusion import AxialDiffusionOperator
from garnet.grid import Grid
from garnet.stencil import GroupOperator


def column_matrix(diffusion, removal, widths, bc):
    """Assemble from interface resistances, without production stencil helpers."""
    size = len(widths)
    matrix = np.diag(removal.copy())
    for lower in range(size - 1):
        upper = lower + 1
        conductance = 2.0 / (widths[lower] / diffusion[lower]
                             + widths[upper] / diffusion[upper])
        matrix[lower, lower] += conductance / widths[lower]
        matrix[lower, upper] -= conductance / widths[lower]
        matrix[upper, upper] += conductance / widths[upper]
        matrix[upper, lower] -= conductance / widths[upper]
    for end, spec in zip((0, -1), bc):
        resistance = {"vacuum": 2.0, "reflective": np.inf, "zero-flux": 0.0}[spec]
        matrix[end, end] += 1.0 / (resistance + widths[end] / (2 * diffusion[end])) / widths[end]
    return matrix


@pytest.mark.parametrize("bc", [("vacuum", "vacuum"), ("reflective", "vacuum"),
                                ("vacuum", "reflective"), ("reflective", "reflective"),
                                ("zero-flux", "vacuum")])
@pytest.mark.parametrize("widths", [[0.4], [0.2, 0.7, 1.1], [0.5, 0.5, 0.5]])
def test_columns_match_independent_dense_solve(bc, widths):
    widths = np.asarray(widths)
    diffusion = np.stack([np.linspace(0.2, 1.3, len(widths)),
                          np.linspace(1.1, 0.3, len(widths))])
    removal = np.full(diffusion.shape, 0.12)
    source = np.arange(diffusion.size).reshape(diffusion.shape) - 1.5
    operator = AxialDiffusionOperator(diffusion, removal, widths, bc)
    for column in range(2):
        matrix = column_matrix(diffusion[column], removal[column], widths, bc)
        expected = np.linalg.solve(matrix, source[column])
        np.testing.assert_allclose(operator.solve(source)[column], expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(operator.apply(operator.solve(source)), source, atol=1e-13)
    currents = operator.currents(operator.solve(source))
    integrated_leakage = np.sum(operator.leakage(operator.solve(source)) * widths, axis=-1)
    np.testing.assert_allclose(integrated_leakage, currents[..., -1] - currents[..., 0], atol=1e-14)


@pytest.mark.parametrize("bc", [("vacuum", "vacuum"), ("reflective", "vacuum"),
                                ("reflective", "reflective")])
def test_uniform_columns_reproduce_existing_diffusion_operator(bc):
    diffusion = np.array([0.8, 0.2, 1.5, 0.7])
    removal = np.array([0.01, 0.2, 0.07, 0.15])
    operator = AxialDiffusionOperator(diffusion, removal, 0.6, bc)
    grid = Grid((1, 1, 4), (1.0, 1.0, 2.4))
    existing = GroupOperator(np, grid, diffusion[None, None, :], removal[None, None, :],
                             bc=("reflective", "reflective", bc))
    flux = np.array([1.0, -0.2, 0.7, 1.1])
    np.testing.assert_allclose(operator.apply(flux), existing.apply(flux[None, None, :]).ravel(), atol=2e-15)


def test_marshak_slab_has_second_order_axial_convergence():
    diffusion, removal, height = 0.8, 0.12, 12.0
    inverse_length = np.sqrt(removal / diffusion)
    denominator = (np.cosh(inverse_length * height / 2)
                   + 2 * diffusion * inverse_length * np.sinh(inverse_length * height / 2))
    errors = []
    for layers in (32, 64, 128):
        width = height / layers
        centers = (np.arange(layers) + 0.5) * width
        # Exact CELL AVERAGES of the constant-source Robin slab solution.
        average_cosh = (np.cosh(inverse_length * (centers - height / 2))
                        * np.sinh(inverse_length * width / 2) / (inverse_length * width / 2))
        exact = (1.0 - average_cosh / denominator) / removal
        numerical = AxialDiffusionOperator(np.full(layers, diffusion), removal, width).solve(np.ones(layers))
        errors.append(np.max(np.abs(numerical - exact)))
    assert errors[0] / errors[1] > 3.7
    assert errors[1] / errors[2] > 3.8


def test_reflective_flat_flux_and_singular_column():
    operator = AxialDiffusionOperator(np.array([0.8, 0.2, 1.5]), 0.2, [0.1, 1, 0.4], "reflective")
    np.testing.assert_allclose(operator.solve(np.full(3, 0.6)), 3.0, atol=2e-14)
    np.testing.assert_allclose(operator.currents(np.full(3, 3.0)), 0.0)
    singular = AxialDiffusionOperator(np.ones(3), 0.0, 1.0, "reflective")
    np.testing.assert_array_equal(singular.leakage(np.ones(3)), 0.0)
    with pytest.raises(ValueError, match="singular"):
        singular.solve(np.ones(3))


@pytest.mark.parametrize("diffusion,removal,widths", [([0], .1, 1), ([np.nan], .1, 1),
                                                    ([1], -.1, 1), ([1], .1, 0),
                                                    ([1], .1, np.inf)])
def test_invalid_coefficients_rejected(diffusion, removal, widths):
    with pytest.raises(ValueError):
        AxialDiffusionOperator(diffusion, removal, widths)
