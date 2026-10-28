"""Coupling verification independent of full-3D transport approximation error."""

import numpy as np
import pytest

from garnet import Material, TriGrid
from garnet._sn_plane import _PlanarFixedSource
from garnet.axial_diffusion import AxialDiffusionOperator
from garnet.sn_2d1d import SN2D1DSolver
from garnet.tri_sn import TriSNTransportSolver


def materials():
    scatter = np.array([[.18, .04], [.007, .27]])
    absorption = np.array([.08, .11])
    total = absorption + scatter.sum(axis=1)
    medium = Material(diffusion=1 / (3 * total), sigma_a=absorption,
                      nu_sigma_f=[.02, .04], sigma_s=scatter, total=total, chi=[1., 0.])
    scatter = np.array([[.20, .05], [.015, .30]])
    absorption = np.array([.95, 1.285])
    total = absorption + scatter.sum(axis=1)
    absorber = Material(diffusion=1 / (3 * total), sigma_a=absorption,
                        nu_sigma_f=[0., 0.], sigma_s=scatter, total=total)
    return [medium, absorber]


def dense_step_reference(grid, mats, material_map, active, source, mu, eta, weights,
                         axial_bc=("vacuum", "vacuum"), *, eigenvalue=False):
    """Assemble angular equations directly, without production spatial operators.

    Every angular equation includes the isotropic axial diffusion of its scalar
    moment. Independently spelling out the triangular normals, face streaming,
    scattering transfer, and axial boundary resistances checks their coupling.
    ``eigenvalue=True`` solves the independently assembled fission eigenproblem
    and returns ``(k, flux, currents)`` with unit integrated production.
    """
    cells = list(zip(*np.where(active)))
    cell_id = {cell: i for i, cell in enumerate(cells)}
    groups, angles, radial_cells, layers = len(mats[0].diffusion), len(weights), len(cells), grid.nz
    scalar_cells = radial_cells * layers
    size = groups * angles * scalar_cells
    matrix = np.zeros((size, size))
    fission = np.zeros_like(matrix) if eigenvalue else None
    rhs = np.zeros(size)
    edge_normals = (((0, 0, 1, np.sqrt(3)/2, .5), (-1, 0, 1, 0., -1.),
                     (0, -1, 1, -np.sqrt(3)/2, .5)),
                    ((0, 0, 0, -np.sqrt(3)/2, -.5), (1, 0, 0, 0., 1.),
                     (0, 1, 0, np.sqrt(3)/2, -.5)))
    edge_to_area = 4 / (np.sqrt(3) * grid.side)

    def index(group, angle, radial, layer):
        return (group * angles + angle) * scalar_cells + radial * layers + layer

    for group in range(groups):
        for radial, (row, col, kind) in enumerate(cells):
            for layer in range(layers):
                material = mats[material_map[row, col, kind, layer]]
                diffusion = 1 / (3 * material.total[group])
                for angle in range(angles):
                    equation = index(group, angle, radial, layer)
                    matrix[equation, equation] += material.total[group]
                    rhs[equation] = source[group, row, col, kind, layer]
                    for dr, dc, neighbor_kind, nx, ny in edge_normals[kind]:
                        streaming = (mu[angle] * nx + eta[angle] * ny) * edge_to_area
                        neighbor = cell_id.get((row + dr, col + dc, neighbor_kind))
                        if streaming > 0:
                            matrix[equation, equation] += streaming
                        elif neighbor is not None:
                            matrix[equation, index(group, angle, neighbor, layer)] += streaming
                    for from_group in range(groups):
                        for moment_angle, weight in enumerate(weights):
                            matrix[equation, index(from_group, moment_angle, radial, layer)] -= (
                                material.sigma_s[from_group, group] * weight)
                            if eigenvalue:
                                fission[equation, index(from_group, moment_angle, radial, layer)] = (
                                    material.chi[group] * material.nu_sigma_f[from_group] * weight)
                    for neighbor_layer in (layer - 1, layer + 1):
                        if 0 <= neighbor_layer < layers:
                            neighbor_material = mats[material_map[row, col, kind, neighbor_layer]]
                            neighbor_diffusion = 1 / (3 * neighbor_material.total[group])
                            coupling = 2 / (grid.dz**2 * (1/diffusion + 1/neighbor_diffusion))
                            for moment_angle, weight in enumerate(weights):
                                matrix[equation, index(group, moment_angle, radial, layer)] += coupling * weight
                                matrix[equation, index(group, moment_angle, radial, neighbor_layer)] -= coupling * weight
                        else:
                            spec = axial_bc[0 if neighbor_layer < 0 else 1]
                            boundary = 0.0 if spec == "reflective" else 1 / (grid.dz * (2 + grid.dz/(2*diffusion)))
                            for moment_angle, weight in enumerate(weights):
                                matrix[equation, index(group, moment_angle, radial, layer)] += boundary * weight
    if eigenvalue:
        values, vectors = np.linalg.eig(np.linalg.solve(matrix, fission))
        fundamental = np.argmax(values.real)
        assert abs(values[fundamental].imag) < 1e-12
        k_eff = values[fundamental].real
        angular = vectors[:, fundamental].real.reshape(groups, angles, radial_cells, layers)
        scalar = np.einsum("m,gmrk->grk", weights, angular)
        production = sum(
            mats[material_map[row, col, kind, layer]].nu_sigma_f @ scalar[:, radial, layer]
            for radial, (row, col, kind) in enumerate(cells) for layer in range(layers))
        angular /= production * grid.cell_volume
    else:
        angular = np.linalg.solve(matrix, rhs).reshape(groups, angles, radial_cells, layers)
    scalar = np.einsum("m,gmrk->grk", weights, angular)
    flux = np.zeros((groups,) + grid.shape)
    currents = np.zeros((groups, 3) + grid.shape)
    for radial, (row, col, kind) in enumerate(cells):
        flux[:, row, col, kind, :] = scalar[:, radial, :]
        for face, (dr, dc, neighbor_kind, nx, ny) in enumerate(edge_normals[kind]):
            neighbor = cell_id.get((row+dr, col+dc, neighbor_kind))
            for angle, weight in enumerate(weights):
                cosine = mu[angle] * nx + eta[angle] * ny
                if cosine > 0:
                    face_flux = angular[:, angle, radial, :]
                else:
                    face_flux = 0.0 if neighbor is None else angular[:, angle, neighbor, :]
                currents[:, face, row, col, kind, :] += weight * cosine * face_flux
    return (k_eff, flux, currents) if eigenvalue else (flux, currents)


@pytest.mark.parametrize("scheme", ["step", "scb"])
@pytest.mark.parametrize("engine", ["lu", "levels"])
def test_plane_contract_signed_sources_balance_and_buffer_ownership(scheme, engine):
    grid = TriGrid((3, 3, 2), 1.7)
    active = np.ones(grid.shape, bool)
    active[0, 0, :] = False
    material_map = np.zeros(grid.shape, int)
    material_map[1, 1, :] = 1
    transport = TriSNTransportSolver(grid, materials(), material_map, active=active,
                                     scheme=scheme, engine=engine, n_polar=1, n_azi=4,
                                     acceleration="gmres")
    plane = _PlanarFixedSource(transport)
    source = np.arange(grid.n_cells).reshape(grid.shape) / 5 - 1
    first = plane.sweep(0, source)
    saved_flux, saved_currents = first.flux.copy(), first.currents.copy()
    second = plane.sweep(0, -2 * source)
    np.testing.assert_allclose(second.flux, -2 * first.flux, atol=2e-14)
    assert first.residual < 1e-14
    np.testing.assert_array_equal(first.flux, saved_flux)
    np.testing.assert_array_equal(first.currents, saved_currents)
    for row, col, kind in zip(*np.where(active)):
        neighbors = (((0, 0, 1), (-1, 0, 1), (0, -1, 1)) if kind == 0
                     else ((0, 0, 0), (1, 0, 0), (0, 1, 0)))
        for face, (dr, dc, neighbor_kind) in enumerate(neighbors):
            neighbor_row, neighbor_col = row + dr, col + dc
            if (0 <= neighbor_row < 3 and 0 <= neighbor_col < 3
                    and active[neighbor_row, neighbor_col, neighbor_kind]):
                assert first.currents[face, row, col, kind] == pytest.approx(
                    -first.currents[face, neighbor_row, neighbor_col, neighbor_kind], abs=2e-14)
    result = plane.solve_group(0, source, tol=1e-10)
    assert result.converged
    removal = transport.st[0] - transport.ss_self[0]
    np.testing.assert_allclose(result.leakage + removal * result.flux,
                               np.where(active, source, 0), atol=1e-10)


@pytest.mark.parametrize("engine", ["lu", "levels"])
@pytest.mark.parametrize("axial_bc", [("vacuum", "vacuum"), ("reflective", "vacuum")])
def test_coupled_multigroup_matches_independent_angular_block_system(engine, axial_bc):
    grid = TriGrid((2, 2, 2, 3), 2.1, height=2.7)
    material_map = np.zeros(grid.shape, int)
    material_map[1, 1, :, 1:] = 1
    active = np.ones(grid.shape[:-1], bool)
    active[0, 0, 0] = False
    source = np.linspace(.1, 1.0, 2 * grid.n_cells).reshape((2,) + grid.shape)
    solver = SN2D1DSolver(grid, materials(), material_map, active=active,
                          engine=engine, n_polar=1, n_azi=4, axial_bc=axial_bc)
    result = solver.solve(source, tol=1e-10)
    assert result.converged, result.reason
    transport = solver.planes[0].transport
    expected_flux, expected_currents = dense_step_reference(
        grid, materials(), material_map, active, source,
        transport.mu, transport.eta, transport.w, axial_bc)
    np.testing.assert_allclose(result.flux, expected_flux, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(result.radial_currents, expected_currents, rtol=1e-9, atol=1e-11)
    assert result.balance_residual < 1e-10
    assert result.flux_disagreement < 1e-10
    np.testing.assert_allclose(result.balance_defect.sum() * grid.cell_volume, 0.0, atol=1e-10)
    np.testing.assert_array_equal(result.flux[:, ~active, :], 0.0)


@pytest.mark.parametrize("scheme", ["step", "scb"])
@pytest.mark.parametrize("coupling", ["krylov", "picard"])
def test_z_invariant_reflective_ends_recover_original_plane(scheme, coupling):
    grid = TriGrid((3, 3, 2, 5), 2.0, height=4.0)
    mat = materials()[0].replace(nu_sigma_f=[0, 0])
    source = np.zeros((2,) + grid.shape)
    source[0] = .6
    source[1] = .2
    solver = SN2D1DSolver(grid, mat, scheme=scheme, axial_bc="reflective",
                          n_polar=1, n_azi=4, acceleration="gmres")
    result = solver.solve(source, tol=1e-10, coupling=coupling)
    assert result.converged, result.reason
    # Fully independent energy iteration using the original plane solver.
    transport = TriSNTransportSolver(TriGrid(grid.shape[:-1], grid.side), mat,
                                     scheme=scheme, n_polar=1, n_azi=4,
                                     require_fissile=False, acceleration="gmres")
    flux = np.zeros((2,) + grid.shape[:-1])
    for _ in range(100):
        previous = flux.copy()
        for group in range(2):
            other = 1 - group
            rhs = source[group, ..., 0] + mat.sigma_s[other, group] * flux[other]
            flux[group] = transport._solve_group(group, rhs.ravel(), flux[group].ravel(), 1e-12).reshape(grid.shape[:-1])
        if np.max(np.abs(flux - previous)) < 1e-13:
            break
    np.testing.assert_allclose(result.flux, np.repeat(flux[..., None], grid.nz, axis=-1), rtol=1e-9)
    np.testing.assert_allclose(result.axial_currents, 0.0, atol=1e-12)


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_radially_uniform_periodic_case_reduces_to_axial_diffusion(scheme):
    mat = Material(diffusion=[.8], sigma_a=[.12], nu_sigma_f=[0.])
    grid = TriGrid((2, 2, 2, 8), 1.7, height=6.0)
    source = np.broadcast_to(np.linspace(.2, 1., grid.nz), (1,) + grid.shape)
    solver = SN2D1DSolver(grid, mat, radial_bc="periodic", scheme=scheme,
                          n_polar=1, n_azi=4)
    result = solver.solve(source, tol=1e-10)
    expected = AxialDiffusionOperator(np.full(grid.nz, .8), .12, grid.dz).solve(source[0, 0, 0, 0])
    assert result.converged, result.reason
    np.testing.assert_allclose(result.flux, np.broadcast_to(expected, result.flux.shape), rtol=1e-9)
    np.testing.assert_allclose(result.radial_leakage, 0.0, atol=1e-12)


@pytest.mark.parametrize("layers", [1, 2, 4, 8, 16, 32, 64])
def test_thin_layers_manufactured_hybrid_solution(layers):
    grid = TriGrid((2, 2, 2, layers), 1.2, height=2.0)
    mat = Material(diffusion=[.8], sigma_a=[.12], nu_sigma_f=[0.])
    solver = SN2D1DSolver(grid, mat, radial_bc="periodic", axial_bc="reflective",
                          n_polar=1, n_azi=4)
    cosine = np.cos(np.pi * (np.arange(layers)+.5)/layers)
    target = np.broadcast_to(1 + .2 * cosine, (1,) + grid.shape).copy()
    # A cosine is an eigenvector of the cell-centred reflecting stencil.
    # Construct its source algebraically, without applying the production op.
    eigenvalue = 4 * mat.diffusion[0] * np.sin(np.pi/(2*layers))**2 / grid.dz**2
    source = mat.sigma_a[0] * target + .2 * eigenvalue * cosine
    result = solver.solve(source, tol=1e-9, maxiter=200)
    assert result.converged, result.reason
    np.testing.assert_allclose(result.flux, target, rtol=1e-8, atol=1e-10)
    assert result.balance_residual < 1e-9
    assert result.flux_disagreement < 1e-9
    assert result.iterations < 80


@pytest.mark.parametrize("layers", [1, 2, 4, 8, 16, 32, 64])
def test_thin_vacuum_layers_converge_on_levels_engine(layers):
    grid = TriGrid((2, 2, 2, layers), 1.2, height=2.0)
    material_map = np.zeros(grid.shape, int)
    material_map[1, 1, :, layers//2:] = 1
    solver = SN2D1DSolver(grid, materials(), material_map, engine="levels", n_polar=1, n_azi=4)
    source = np.linspace(.2, 1., np.prod(solver.shape)).reshape(solver.shape)
    result = solver.solve(source, tol=1e-9, maxiter=200)
    assert result.converged, result.reason
    assert result.balance_residual < 1e-9
    assert result.flux_disagreement < 1e-9
    assert result.iterations < 80


def test_manufactured_continuous_hybrid_solution_refines_at_second_order():
    height, diffusion, absorption = 4., .8, .12
    mat = Material(diffusion=[diffusion], sigma_a=[absorption], nu_sigma_f=[0.])
    errors = []
    for layers in (8, 16, 32):
        grid = TriGrid((2, 2, 2, layers), 1., height=height)
        solver = SN2D1DSolver(grid, mat, radial_bc="periodic", axial_bc="reflective",
                              n_polar=1, n_azi=4)
        phase = np.pi * (np.arange(layers) + .5) / layers
        average_cosine = np.cos(phase) * np.sinc(.5/layers)
        exact = 1 + .2 * average_cosine
        source = absorption * exact + .2 * diffusion * (np.pi/height)**2 * average_cosine
        result = solver.solve(source, tol=1e-10)
        assert result.converged, result.reason
        errors.append(np.max(np.abs(result.flux - exact)))
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.9


def test_picard_and_krylov_have_the_same_fixed_point_and_report_exhaustion():
    grid = TriGrid((2, 2, 2, 3), 2.0, height=6.0)
    mat = Material(diffusion=[.7], sigma_a=[.15], nu_sigma_f=[0.])
    solver = SN2D1DSolver(grid, mat, n_polar=1, n_azi=4, acceleration="gmres")
    source = np.linspace(.5, 1.5, grid.n_cells).reshape((1,) + grid.shape)
    krylov = solver.solve(source, tol=1e-8)
    picard = solver.solve(source, coupling="picard", tol=1e-8, maxiter=1200)
    assert krylov.converged and picard.converged, picard.reason
    np.testing.assert_allclose(picard.flux, krylov.flux, rtol=1e-7)
    failed = solver.solve(source, coupling="picard", maxiter=1, tol=1e-12)
    assert not failed.converged
    assert "limit" in failed.reason
    assert len(picard.leakage_change_history) == picard.iterations
    failed = solver.solve(source, maxiter=1, tol=1e-12)
    assert not failed.converged
    assert failed.iterations == 1


def test_signed_source_zero_source_repeatability_and_plane_reuse():
    solver = SN2D1DSolver(TriGrid((2, 2, 2, 3), 1.0, height=1.5), materials(), n_polar=1, n_azi=4)
    source = np.linspace(-1, 1, np.prod(solver.shape)).reshape(solver.shape)
    first = solver.solve(source, tol=1e-10)
    saved = first.flux.copy()
    second = solver.solve(-2 * source, initial_flux=first.flux, tol=1e-10)
    assert first.converged and second.converged
    assert first.negative_effective_sources > 0
    np.testing.assert_allclose(second.flux, -2 * first.flux, rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(first.flux, saved)
    assert len({id(plane) for plane in solver.planes}) == 1
    zero = solver.solve(0.0)
    assert zero.converged
    np.testing.assert_array_equal(zero.flux, 0.0)


def test_c5g7_material_tables_with_upscatter_and_axial_absorber():
    from garnet.benchmarks.c5g7 import c5g7_materials

    tables = c5g7_materials()
    mats = [tables[name] for name in ("UO2", "Water", "Control Rod")]
    grid = TriGrid((2, 2, 2, 4), 1.0, height=3.0)
    material_map = np.zeros(grid.shape, int)
    material_map[:, 0, :, :] = 1
    material_map[1, 1, :, 2:] = 2
    solver = SN2D1DSolver(grid, mats, material_map, n_polar=1, n_azi=4)
    source = np.zeros(solver.shape)
    source[0] = 1.0
    result = solver.solve(source, tol=1e-9, maxiter=500)
    assert result.converged, result.reason
    assert np.all(result.flux >= -1e-11)
    assert result.negative_effective_sources > 0
    assert result.balance_residual < 1e-9


def test_existing_material_blending_is_preserved_in_planes_and_columns():
    mats = materials()
    weight = .4
    total = (1-weight) * mats[0].total + weight * mats[1].total
    blended = Material(diffusion=1/(3*total), total=total,
                       sigma_a=(1-weight)*mats[0].sigma_a + weight*mats[1].sigma_a,
                       sigma_s=(1-weight)*mats[0].sigma_s + weight*mats[1].sigma_s,
                       nu_sigma_f=[0., 0.])
    grid = TriGrid((2, 2, 2, 3), 1.3, height=2.)
    mixed = SN2D1DSolver(grid, mats, mix_material=np.ones(grid.shape, int),
                         mix_weight=np.full(grid.shape, weight), n_polar=1, n_azi=4)
    explicit = SN2D1DSolver(grid, blended, n_polar=1, n_azi=4)
    result = mixed.solve(1., tol=1e-10)
    expected = explicit.solve(1., tol=1e-10)
    assert result.converged and expected.converged
    np.testing.assert_allclose(result.flux, expected.flux, rtol=1e-10, atol=1e-12)


def test_plane_failure_is_not_reported_as_coupled_convergence():
    grid = TriGrid((2, 2, 2, 2), 2., height=3.)
    solver = SN2D1DSolver(grid, materials()[0], n_polar=1, n_azi=4,
                          acceleration="si", max_inner=1)
    result = solver.solve(1., coupling="picard", tol=1e-12)
    assert not result.converged
    assert "planar" in result.reason


def test_existing_hpmr_builder_mask_and_polar_blend_integration():
    from garnet.benchmarks import build_hpmr3d

    problem = build_hpmr3d(refine=1, nz=10, absorber="polar")
    solver = SN2D1DSolver(
        problem.grid, problem.materials, problem.material_map, active=problem.active,
        mix_material=problem.mix_material, mix_weight=problem.mix_weight,
        n_polar=1, n_azi=4, engine="levels")
    source = np.zeros(solver.shape)
    source[0] = problem.material_map == 1
    result = solver.solve(source, tol=1e-8, maxiter=500)
    assert result.converged, result.reason
    assert result.balance_residual < 1e-8
    assert result.flux_disagreement < 1e-8
    np.testing.assert_array_equal(result.flux[:, ~problem.active], 0.0)


def test_geometry_and_source_guards():
    mat = materials()[0]
    with pytest.raises(ValueError, match="extruded"):
        SN2D1DSolver(TriGrid((2, 2, 2), 1.), mat)
    grid = TriGrid((2, 2, 2, 2), 1.)
    active = np.ones(grid.shape, bool)
    active[0, 0, 0, 1] = False
    with pytest.raises(ValueError, match="constant in z"):
        SN2D1DSolver(grid, mat, active=active)
    with pytest.raises(ValueError, match="at least one"):
        SN2D1DSolver(grid, mat, active=np.zeros(grid.shape, bool))
    with pytest.raises(ValueError, match="vacuum"):
        SN2D1DSolver(grid, mat, radial_bc="reflective")
    solver = SN2D1DSolver(grid, mat, n_polar=1, n_azi=4)
    for kwargs in (dict(tol=0), dict(maxiter=0), dict(relaxation=0), dict(coupling="unknown")):
        with pytest.raises(ValueError):
            solver.solve(1., **kwargs)
    with pytest.raises(ValueError, match="finite"):
        solver.solve(np.nan)


@pytest.mark.parametrize("engine", ["lu", "levels"])
@pytest.mark.parametrize("axial_bc", [("vacuum", "vacuum"), ("reflective", "vacuum")])
def test_eigenvalue_matches_independent_angular_eigenproblem(engine, axial_bc):
    mats = [materials()[0].replace(chi=[.7, .3]),
            materials()[1].replace(nu_sigma_f=[.012, .02], chi=[.2, .8])]
    grid = TriGrid((2, 2, 2, 2), 2.1, height=2.7)
    material_map = np.zeros(grid.shape, int)
    material_map[1, 1, :, 1] = 1
    active = np.ones(grid.shape[:-1], bool)
    active[0, 0, 0] = False
    solver = SN2D1DSolver(grid, mats, material_map, active=active, engine=engine,
                          n_polar=1, n_azi=4, axial_bc=axial_bc)
    result = solver.solve_eigenvalue(tol_k=1e-10, tol_source=1e-10)
    assert result.converged, result.reason
    transport = solver.planes[0].transport
    expected_k, expected_flux, expected_currents = dense_step_reference(
        grid, mats, material_map, active, np.zeros(solver.shape),
        transport.mu, transport.eta, transport.w, axial_bc, eigenvalue=True)
    assert result.k_eff == pytest.approx(expected_k, abs=1e-10)
    np.testing.assert_allclose(result.flux, expected_flux, rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(result.radial_currents, expected_currents, rtol=2e-8, atol=1e-10)
    assert result.fission_source.sum() * grid.cell_volume == pytest.approx(1., abs=1e-13)
    assert result.transport_residual < 1e-9
    assert result.balance_residual < 1e-9
    assert result.flux_disagreement < 1e-9
    assert len(result.k_history) == result.outer_iterations
    assert sum(result.inner_iteration_history) == result.iterations
    np.testing.assert_array_equal(result.flux[:, ~active, :], 0.)
    # Every returned moment/current uses the normalized eigen-source. This
    # catches scaling only the scalar flux or retaining the lagged fission RHS.
    fixed = solver.solve(solver.chi * result.fission_source / result.k_eff,
                         initial_flux=result.flux, tol=1e-10)
    assert fixed.converged, fixed.reason
    np.testing.assert_allclose(fixed.flux, result.flux, rtol=2e-8)
    np.testing.assert_allclose(fixed.axial_currents, result.axial_currents, rtol=2e-8, atol=1e-10)


@pytest.mark.parametrize("scheme", ["step", "scb"])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("coupling", ["krylov", "picard"])
def test_eigenvalue_infinite_medium_limit(scheme, groups, coupling):
    from garnet import k_infinite

    mat = (Material(diffusion=[.8], sigma_a=[.12], nu_sigma_f=[.15]) if groups == 1
           else materials()[0].replace(chi=[.3, .7]))
    grid = TriGrid((2, 2, 2, 3), 2., height=4.)
    solver = SN2D1DSolver(grid, mat, radial_bc="periodic", axial_bc="reflective",
                          scheme=scheme, n_polar=1, n_azi=4)
    result = solver.solve_eigenvalue(tol_k=1e-9, tol_source=1e-9, initial_k=.4, coupling=coupling)
    assert result.converged, result.reason
    assert result.k_eff == pytest.approx(k_infinite(mat), abs=1e-10)
    for group_flux in result.flux:
        np.testing.assert_allclose(group_flux, group_flux.flat[0], rtol=1e-10)
    np.testing.assert_allclose(result.axial_currents, 0., atol=1e-12)
    np.testing.assert_allclose(result.radial_leakage, 0., atol=1e-12)


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_eigenvalue_z_invariant_limit_matches_existing_sn_solver(scheme):
    mat = materials()[0].replace(chi=[.6, .4])
    grid = TriGrid((3, 3, 2, 3), 2., height=4.)
    solver = SN2D1DSolver(grid, mat, axial_bc="reflective", scheme=scheme, n_polar=1, n_azi=4)
    result = solver.solve_eigenvalue(tol_k=1e-9, tol_source=1e-9)
    original = TriSNTransportSolver(TriGrid(grid.shape[:-1], grid.side), mat,
                                    scheme=scheme, n_polar=1, n_azi=4)
    reference = original.solve(tol_k=1e-10, tol_source=1e-10)
    assert result.converged and reference.converged
    assert result.k_eff == pytest.approx(reference.k_eff, abs=1e-9)
    expected = np.repeat(reference.flux[..., None], grid.nz, axis=-1)
    expected /= np.sum(mat.nu_sigma_f[:, None, None, None, None] * expected) * grid.cell_volume
    np.testing.assert_allclose(result.flux, expected, rtol=2e-8)


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_eigenvalue_axial_diffusion_limit(scheme):
    from garnet import k_from_buckling

    mat = materials()[0].replace(chi=[.6, .4])
    grid = TriGrid((2, 2, 2, 8), 2., height=8.)
    solver = SN2D1DSolver(grid, mat, radial_bc="periodic", axial_bc="zero-flux",
                          scheme=scheme, n_polar=1, n_azi=4)
    result = solver.solve_eigenvalue(tol_k=1e-9, tol_source=1e-9)
    # The centred FV Dirichlet slab has an exact discrete sine eigenmode.
    buckling = 4*np.sin(np.pi/(2*grid.nz))**2 / grid.dz**2
    assert result.converged, result.reason
    assert result.k_eff == pytest.approx(k_from_buckling(mat, buckling), abs=1e-9)
    shape = np.sin(np.pi * (np.arange(grid.nz)+.5) / grid.nz)
    numerical_shape = result.flux[0, 0, 0, 0]
    np.testing.assert_allclose(numerical_shape / numerical_shape.max(), shape / shape.max(), rtol=2e-8)


def test_eigenvalue_blended_spectra_and_initial_normalization_invariance():
    mats = [materials()[0].replace(chi=[.8, .2]),
            materials()[1].replace(nu_sigma_f=[.01, .06], chi=[.1, .9])]
    weight = .4
    total = (1-weight)*mats[0].total + weight*mats[1].total
    production = (1-weight)*mats[0].nu_sigma_f + weight*mats[1].nu_sigma_f
    spectrum = ((1-weight)*mats[0].nu_sigma_f.sum()*mats[0].chi
                + weight*mats[1].nu_sigma_f.sum()*mats[1].chi) / production.sum()
    blended = Material(diffusion=1/(3*total), total=total,
                       sigma_a=(1-weight)*mats[0].sigma_a + weight*mats[1].sigma_a,
                       sigma_s=(1-weight)*mats[0].sigma_s + weight*mats[1].sigma_s,
                       nu_sigma_f=production, chi=spectrum)
    grid = TriGrid((2, 2, 2, 3), 1.3, height=2.)
    mixed = SN2D1DSolver(grid, mats, mix_material=np.ones(grid.shape, int),
                         mix_weight=np.full(grid.shape, weight), n_polar=1, n_azi=4)
    explicit = SN2D1DSolver(grid, blended, n_polar=1, n_azi=4)
    result = mixed.solve_eigenvalue(tol_k=1e-9, tol_source=1e-9)
    expected = explicit.solve_eigenvalue(tol_k=1e-9, tol_source=1e-9)
    repeated = mixed.solve_eigenvalue(initial_flux=result.flux * 1e12, initial_k=.2,
                                      tol_k=1e-9, tol_source=1e-9)
    assert result.converged and expected.converged and repeated.converged
    assert result.k_eff == pytest.approx(expected.k_eff, abs=1e-10)
    assert result.k_eff == pytest.approx(repeated.k_eff, abs=1e-10)
    for field in ("flux", "radial_flux", "axial_flux", "radial_currents", "axial_currents",
                  "radial_leakage", "axial_leakage", "fission_source"):
        np.testing.assert_allclose(getattr(result, field), getattr(expected, field), rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(getattr(result, field), getattr(repeated, field), rtol=1e-7, atol=1e-10)


def test_eigenvalue_failure_reporting_and_input_guards():
    grid = TriGrid((2, 2, 2, 3), 2., height=3.)
    solver = SN2D1DSolver(grid, materials()[0], n_polar=1, n_azi=4)
    for kwargs in (dict(tol_k=0), dict(tol_source=np.nan), dict(tol_inner=0),
                   dict(initial_k=0), dict(max_outer=0), dict(max_inner=0),
                   dict(initial_flux=0.), dict(initial_flux=-1.)):
        with pytest.raises(ValueError):
            solver.solve_eigenvalue(**kwargs)
    outer_failed = solver.solve_eigenvalue(max_outer=1, tol_k=1e-10, tol_source=1e-10)
    assert not outer_failed.converged
    assert outer_failed.outer_iterations == 1
    assert "outer" in outer_failed.reason
    inner_failed = solver.solve_eigenvalue(max_inner=1, tol_inner=1e-12)
    assert not inner_failed.converged
    assert inner_failed.outer_iterations == 0
    assert "fixed-source solve failed" in inner_failed.reason
    assert inner_failed.iterations == 1
    nonfissile = SN2D1DSolver(grid, materials()[1], n_polar=1, n_azi=4)
    with pytest.raises(ValueError, match="no active fissile"):
        nonfissile.solve_eigenvalue()
    active = np.ones(grid.shape[:-1], bool)
    active[0, 0, :] = False
    material_map = np.ones(grid.shape, int)
    material_map[0, 0, :, :] = 0
    masked_fuel = SN2D1DSolver(grid, materials(), material_map, active=active, n_polar=1, n_azi=4)
    with pytest.raises(ValueError, match="no active fissile"):
        masked_fuel.solve_eigenvalue()


def test_eigenvalue_stagnant_k_does_not_hide_an_inaccurate_inner_solve():
    solver = SN2D1DSolver(TriGrid((2, 2, 2, 3), 2., height=3.), materials()[0],
                          n_polar=1, n_azi=4)
    result = solver.solve_eigenvalue(tol_k=1e-9, tol_source=1e-9, tol_inner=.01, max_outer=80)
    assert not result.converged
    assert result.k_change_history[-1] < 1e-9
    assert result.source_change_history[-1] < 1e-9
    assert "balance or flux consistency" in result.reason
    assert max(result.transport_residual, result.balance_residual, result.flux_disagreement) > 1e-9
