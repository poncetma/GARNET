"""Outer-iteration robustness of the k-eigenvalue power iteration.

* Default SDP3 on a vacuum-bounded core (non-symmetric block) must converge:
  BiCGStab's irregular inner error used to poison the Anderson history
  (BIBLIS: NOT converged after 2000 outers); the default is now GMRES and an
  explicit BiCGStab gets a tighter inner tolerance.
* Plain power iteration (and the adjoint, which always runs it) stops on an
  error bound, not on the per-outer step: the step understates the error by
  1/(1 - dominance ratio).
* Excised (inactive) cells carry no fission or scattering source, whatever
  material the map puts there.
"""

import numpy as np
import pytest

from garnet import Grid, Material, SDP3EigenSolver
from garnet.benchmarks.biblis import build_biblis
from garnet.materials import PWR_TWO_GROUP
from garnet.solver import DiffusionEigenSolver, SP3EigenSolver


@pytest.fixture(scope="module")
def biblis():
    return build_biblis(2)


def _sdp3(p, **kw):
    return SDP3EigenSolver(p.grid, p.materials, p.material_map, bc=p.bc,
                           active=p.active, mask_bc=p.mask_bc, device="cpu", **kw)


def test_default_sdp3_converges_on_vacuum_bounded_core(biblis):
    s = _sdp3(biblis)
    assert not all(op.symmetric for op in s.ops)       # really the GMRES path
    r = s.solve(max_outer=200)
    assert r.converged and r.outer_iterations < 60, r
    ref = _sdp3(biblis, linear_solver="gmres").solve(tol_k=1e-9, tol_source=1e-8)
    assert r.k_eff == pytest.approx(ref.k_eff, abs=1e-6)


def test_explicit_bicgstab_sdp3_keeps_anderson_working(biblis):
    r = _sdp3(biblis, linear_solver="bicgstab").solve(max_outer=200)
    assert r.converged and r.outer_iterations < 60, r


def test_plain_power_iteration_stops_on_error_not_step():
    grid = Grid((60, 60, 1), (400.0, 400.0, 1.0))
    s = DiffusionEigenSolver(grid, PWR_TWO_GROUP,
                             bc=("vacuum", "vacuum", "reflective"), device="cpu")
    ref = s.solve(tol_k=1e-12, tol_source=1e-10, inner_rtol_floor=1e-13).flux_numpy
    ref /= np.linalg.norm(ref)
    r = s.solve(anderson_depth=1)                       # default tol_source=1e-6
    assert r.converged
    phi = r.flux_numpy / np.linalg.norm(r.flux_numpy)
    # Before: stopped at a 1e-6 step with a 5e-5 shape error.
    assert np.linalg.norm(phi - ref) < 5e-6


def _masked_box():
    grid = Grid((20, 20, 1), (100.0, 100.0, 1.0))
    x = (np.arange(20) + 0.5) * 5.0
    X, Y = np.meshgrid(x, x, indexing="ij")
    active = (X ** 2 + Y ** 2 < 90.0 ** 2)[:, :, None]
    bc = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    return grid, active, bc


@pytest.mark.parametrize("cls", [DiffusionEigenSolver, SP3EigenSolver])
def test_excised_cells_carry_no_source(cls):
    grid, active, bc = _masked_box()
    core = Material(diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    void = Material(diffusion=[1.0], sigma_a=[0.0], nu_sigma_f=[0.0])
    hot = Material(diffusion=[1.0], sigma_a=[0.1], nu_sigma_f=[1.5])
    mm = np.where(active, 0, 1)
    k_void = cls(grid, [core, void], mm, bc=bc, active=active,
                 device="cpu").solve().k_eff
    s = cls(grid, [core, hot], mm, bc=bc, active=active, device="cpu")
    r = s.solve()
    # Before: the isolated excised cells were a k = nu_sigma_f = 1.5 problem.
    assert r.k_eff == pytest.approx(k_void, abs=1e-6)
    phi = r.flux_numpy
    assert float(np.abs(phi[:, ~active]).max()) < 1e-12 * float(phi.max())


def test_fully_excised_fuel_is_not_fissile():
    grid, active, bc = _masked_box()
    fuel = Material(diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    refl = Material(diffusion=[1.0], sigma_a=[0.01], nu_sigma_f=[0.0])
    with pytest.raises(ValueError, match="fissile"):
        DiffusionEigenSolver(grid, [refl, fuel], np.where(active, 0, 1),
                             bc=bc, active=active, device="cpu")
