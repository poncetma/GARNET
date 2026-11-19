"""Diffusion transients on unstructured meshes (garnet.mesh.MeshGrid) and
time-dependent boundary coefficients (bcf) in garnet.transient.TransientSolver.

The geometry is the six-drum HexLattice of test_equivalence: the drum-band
mesh with levels=0 (the triangular raster as an unstructured mesh, orthogonal,
equal areas) and levels=1 (drum band refined, hanging nodes, so non-orthogonal
and cells of two sizes). Checks:

* the mesh eigen solver of the structured framework (blends through Fields,
  implicit non-orthogonal correction, GMRES) equals UnstructuredDiffusionSolver
  (per-cell blended materials, lagged correction, CG) to round-off, with and
  without boundary coefficients;
* the levels=0 mesh transient equals the triangular-raster transient
  (TriGroupOperator) through an absorber step;
* a static bcf keeps the eigenstate steady, on the mesh and on the raster;
* bcf = 1 reproduces the transient without bcf exactly;
* the reported power is the volume-weighted fission integral of the flux.

Broken on purpose (scratch scripts, 2026-11-19): without the non-orthogonal
correction the eigenvalue misses by 279 pcm; with bcf dropped from the step
operators (but kept by the initial eigen solve) the steady power drifts by
110 %; without the volume weights the reported power is 17 % off the flux
integral. The cheap ones are kept below as broken cases.
"""

import numpy as np
import pytest

from garnet import Kinetics, Material, TransientSolver
from garnet.blend import per_cell_blended_materials
import garnet.transient as transient
from garnet.mesh import (MeshDiffusionEigenSolver, MeshGrid, MeshGroupOperator,
                         UnstructuredDiffusionSolver)
from garnet.tri import TriDiffusionEigenSolver, TriGroupOperator

from test_equivalence import TIGHT, lattice

KIN = Kinetics(velocities=[1.25e7, 2.5e5], beta=[0.0065], decay=[0.0784])
ABS2 = Material(name="b4c+", diffusion=[1.0, 0.3], sigma_a=[0.06, 0.6],
                nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
STEADY = dict(tol_k=1e-11, tol_source=1e-10)
G = 2


@pytest.fixture(scope="module")
def meshes():
    lat = lattice(4)
    return lat, {0: lat.build_mesh(levels=0), 1: lat.build_mesh(levels=1)}


def _bcf(n):
    return np.random.default_rng(1).uniform(0.3, 2.5, (G, n))


def _absorber_step(mats):
    i = [m.name for m in mats].index("b4c")
    return mats[:i] + [ABS2] + mats[i + 1:]


def _mesh_transient(q, bcf=None, perturb=True, t_end=0.1, dt=0.02, on_step=None):
    grid = MeshGrid(q.mesh, q.alpha_boundary)
    mats = list(q.materials)
    pert = _absorber_step(mats) if perturb else mats

    def problem_at(t):
        return (mats if t <= 0 else pert, q.cell_material, q.mix_material, q.mix_weight, bcf)

    s = TransientSolver(grid, problem_at, KIN, group_operator=MeshGroupOperator,
                        eig_solver=MeshDiffusionEigenSolver, device="cpu",
                        linear_solver="gmres", on_step=on_step)
    return s.solve(t_end, dt, tol_step=1e-10, steady_kwargs=STEADY)


@pytest.mark.parametrize("levels", [0, 1])
@pytest.mark.parametrize("with_bcf", [False, True])
def test_mesh_eigen_solver_equals_the_unstructured_solver(meshes, levels, with_bcf):
    _, qs = meshes
    q = qs[levels]
    bcf = _bcf(q.mesh.n_cells) if with_bcf else None
    mats, cells = q.solver_materials()
    ref = UnstructuredDiffusionSolver(q.mesh, mats, cells, device="cpu", bcf=bcf).solve(**TIGHT)
    grid = MeshGrid(q.mesh, q.alpha_boundary)
    assert grid.nonorthogonal == (levels == 1)
    r = MeshDiffusionEigenSolver(grid, q.materials, q.cell_material, mix_material=q.mix_material,
                                 mix_weight=q.mix_weight, device="cpu", linear_solver="gmres",
                                 bcf=bcf).solve(**TIGHT)
    assert r.k_eff == pytest.approx(ref.k_eff, abs=1e-9)
    f, fr = r.flux_numpy, ref.flux
    np.testing.assert_allclose(f / f.max(), fr / fr.max(), atol=1e-8)


def test_broken_mesh_operator_without_the_correction_misses_k(meshes):
    _, qs = meshes
    q = qs[1]
    mats, cells = q.solver_materials()
    ref = UnstructuredDiffusionSolver(q.mesh, mats, cells, device="cpu").solve(**TIGHT)
    grid = MeshGrid(q.mesh, q.alpha_boundary, nonorthogonal_correction=False)
    r = MeshDiffusionEigenSolver(grid, q.materials, q.cell_material, mix_material=q.mix_material,
                                 mix_weight=q.mix_weight, device="cpu").solve(**TIGHT)
    assert abs(r.k_eff - ref.k_eff) > 1e-3                    # 279 pcm


def test_non_orthogonal_mesh_rejects_cg_and_monolithic_steps(meshes):
    _, qs = meshes
    q = qs[1]
    grid = MeshGrid(q.mesh, q.alpha_boundary)
    with pytest.raises(ValueError, match="gmres"):
        MeshDiffusionEigenSolver(grid, q.materials, q.cell_material, mix_material=q.mix_material,
                                 mix_weight=q.mix_weight, device="cpu")
    s = TransientSolver(grid, lambda t: (q.materials, q.cell_material, q.mix_material,
                                         q.mix_weight), KIN,
                        group_operator=MeshGroupOperator, eig_solver=MeshDiffusionEigenSolver,
                        device="cpu", linear_solver="gmres")
    with pytest.raises(ValueError, match="symmetric"):
        s.solve(0.02, 0.02, step_solver="monolithic", steady_kwargs=STEADY)


def test_mesh_transient_equals_the_raster_transient(meshes):
    lat, qs = meshes
    pm = _mesh_transient(qs[0])
    rx = lat.build()
    mats = list(rx.materials)
    pert = _absorber_step(mats)
    s = TransientSolver(rx.grid, lambda t: (mats if t <= 0 else pert, rx.material_map,
                                            rx.mix_material, rx.mix_weight),
                        KIN, group_operator=TriGroupOperator, eig_solver=TriDiffusionEigenSolver,
                        device="cpu", active=rx.active, mask_bc=rx.mask_bc)
    pr = s.solve(0.1, 0.02, tol_step=1e-10, steady_kwargs=STEADY)
    assert pm.power[-1] < 0.75                                # the step does something
    np.testing.assert_allclose(pm.power, pr.power, atol=1e-7)


def _raster_steady_with_bcf(lat):
    rx = lat.build()
    bcf = np.random.default_rng(2).uniform(0.3, 2.5, (G,) + rx.grid.shape)
    s = TransientSolver(rx.grid, lambda t: (rx.materials, rx.material_map, rx.mix_material,
                                            rx.mix_weight, bcf),
                        KIN, group_operator=TriGroupOperator, eig_solver=TriDiffusionEigenSolver,
                        device="cpu", active=rx.active, mask_bc=rx.mask_bc)
    return s.solve(0.1, 0.02, tol_step=1e-10, steady_kwargs=STEADY)


@pytest.mark.parametrize("geometry", ["raster", "mesh"])
def test_static_boundary_coefficients_keep_the_eigenstate(meshes, geometry):
    lat, qs = meshes
    if geometry == "raster":
        r = _raster_steady_with_bcf(lat)
    else:
        r = _mesh_transient(qs[1], bcf=_bcf(qs[1].mesh.n_cells), perturb=False)
    np.testing.assert_allclose(r.power, 1.0, atol=1e-9)


def test_broken_step_operators_without_bcf_drift(meshes, monkeypatch):
    """Negative control: the eigen solve sees bcf, the steps do not."""
    lat, _ = meshes
    original = transient.TransientSolver._step_operators

    def drop(self, fields, inv_vdt, groups, bcf=transient._CURRENT):
        return original(self, fields, inv_vdt, groups, bcf=None)

    monkeypatch.setattr(transient.TransientSolver, "_step_operators", drop)
    r = _raster_steady_with_bcf(lat)
    assert np.abs(r.power - 1.0).max() > 1e-2


def _flux_integral_ratio(q, record, k0):
    A = np.asarray(q.mesh.area)
    pert = _absorber_step(list(q.materials))
    ms, cm = per_cell_blended_materials(pert, q.cell_material, q.mix_material, q.mix_weight)
    nsf = np.array([[ms[m].nu_sigma_f[g] for m in cm] for g in range(G)])
    return np.array([(nsf * f * A).sum() / k0 / p for _, f, p in record])


def test_unit_bcf_reproduces_the_plain_mesh_transient_and_power_is_volume_weighted(meshes):
    _, qs = meshes
    q = qs[1]
    record = []
    plain = _mesh_transient(q, on_step=lambda t, f, p: record.append((t, np.array(f), p)))
    unit = _mesh_transient(q, bcf=np.ones((G, q.mesh.n_cells)))
    assert plain.power[-1] < 0.75
    np.testing.assert_array_equal(unit.power, plain.power)
    np.testing.assert_allclose(_flux_integral_ratio(q, record, plain.k0), 1.0, atol=1e-12)


def test_broken_power_without_volume_weights_misses_the_flux_integral(meshes, monkeypatch):
    _, qs = meshes
    q = qs[1]
    monkeypatch.setattr(transient, "_volume_weights", lambda grid, xp, dtype: None)
    record = []
    r = _mesh_transient(q, t_end=0.04, on_step=lambda t, f, p: record.append((t, np.array(f), p)))
    assert np.abs(_flux_integral_ratio(q, record, r.k0) - 1.0).max() > 0.05      # 17 %
