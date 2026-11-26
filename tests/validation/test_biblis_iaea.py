"""BIBLIS (2D vacuum) and IAEA (3D albedo) diffusion benchmarks.

These exercise the Robin/albedo boundary term and the active-cell mask for
non-rectangular cores, and check k_eff against the published references and
against FEMFFUSION's own diffusion result.
"""

import numpy as np
import pytest

from garnet import DiffusionEigenSolver
from garnet.benchmarks import build_biblis, build_iaea
from garnet.benchmarks.biblis import K_REFERENCE as BIBLIS_K_REFERENCE

TOL = dict(tol_k=1e-8, tol_source=1e-7)


def _solve(p):
    return DiffusionEigenSolver(p.grid, p.materials, p.material_map, bc=p.bc,
                                active=p.active, mask_bc=p.mask_bc,
                                device="cpu").solve(**TOL)


def test_biblis_geometry():
    p = build_biblis(cells_per_assembly=2)
    assert p.grid.shape == (34, 34, 1)
    assert int(p.active.sum()) == 257 * 4        # matches FEMFFUSION active cells
    # Fully symmetric octagonal core.
    m = p.material_map[:, :, 0]
    assert np.array_equal(m, m[:, ::-1]) and np.array_equal(m, m.T)


def test_biblis_keff_converges_to_femffusion_reference():
    # K_REFERENCE is the extrapolated FEMFFUSION FE3 diffusion limit,
    # 1.025258 +- 0.5 pcm. The cell-centred FV error is +341 / +72 / +14 pcm at
    # 2 / 4 / 8 cells per assembly (second order from above), so the 8-cell
    # mesh must sit within 30 pcm and closer than the 4-cell one. Shown to fail
    # with the old 1.0287 constant and with a 0.3 % thermal-absorption error in
    # one fuel. It cannot see boundary or face-mean defects: the reflector screens
    # the boundary (vacuum vs reflective moves k by 3 pcm) and all compositions
    # share nearly the same D; tests/verification covers those.
    k = {c: _solve(build_biblis(cells_per_assembly=c)) for c in (4, 8)}
    assert all(r.converged for r in k.values())
    err = {c: k[c].k_eff - BIBLIS_K_REFERENCE for c in k}
    assert 0.0 < err[8] < 3e-4
    assert err[8] < err[4] / 3.0


def test_iaea_geometry_and_keff():
    p = build_iaea(cells_per_node=2)
    assert p.grid.shape == (34, 34, 38)
    assert int(p.active.sum()) == 241 * 19 * 8   # 241 active/plane, 19 planes, r^3
    res = _solve(p)
    assert res.converged
    # Classic IAEA-3D reference k_eff = 1.02903.
    assert res.k_eff == pytest.approx(1.02903, abs=1e-3)


def test_boundary_barely_matters_for_reflected_iaea():
    # The thick reflector screens the outer boundary: k is nearly BC-independent.
    p = build_iaea(cells_per_node=1)
    kw = dict(active=p.active, device="cpu")
    k_alb = DiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                 bc=p.bc, mask_bc=p.mask_bc, **kw).solve(**TOL).k_eff
    k_ref = DiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                 bc="reflective", mask_bc="reflective", **kw).solve(**TOL).k_eff
    assert abs(k_alb - k_ref) < 5e-3
