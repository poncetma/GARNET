"""Unstructured finite volume on non-orthogonal (skewed) triangulations.

The two-point flux in garnet.mesh couples neighbours through the
centroid-to-centroid distance alone, which is consistent only when that line
is orthogonal to the shared face. Splitting every square of a uniform grid
along the same diagonal gives triangles whose centroid lines are skewed to
every face, and there the bare TPFA k did not converge: +177, -197, -292,
-316 pcm at n = 8, 16, 32, 64 (the same squares uncut converge at second
order: +863, +215, +54 pcm at n = 8, 16, 32). With the deferred
non-orthogonal correction (normal-distance two-point flux plus lagged
least-squares tangential gradients) the cut squares converge at second order
too: -179, -60, -17, -4 pcm.
"""

import math

import numpy as np
import pytest

from garnet.materials import Material
from garnet.mesh import UnstructuredDiffusionSolver, assemble_mesh

L = 10.0
MAT = Material(name="m", diffusion=[1.0], sigma_a=[0.1], nu_sigma_f=[0.3])
_ZERO_FLUX = 1e10   # Robin coefficient in the Dirichlet limit


def _bare_square_k():
    return 0.3 / (0.1 + 1.0 * 2.0 * (math.pi / L) ** 2)


def _diagonal_cut_square(n, perturb=0.0, seed=0):
    """An n x n grid of squares, each cut along the same diagonal; ``perturb``
    jitters the interior nodes by up to that fraction of the spacing."""
    xs = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    if perturb:
        rng = np.random.default_rng(seed)
        inner = (X > 0) & (X < L) & (Y > 0) & (Y < L)
        h = L / n
        X = X + np.where(inner, rng.uniform(-perturb, perturb, X.shape) * h, 0.0)
        Y = Y + np.where(inner, rng.uniform(-perturb, perturb, Y.shape) * h, 0.0)
    coords = np.c_[X.ravel(), Y.ravel()]
    idx = lambda i, j: i * (n + 1) + j
    cells = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            cells += [(a, b, c), (a, c, d)]
    return assemble_mesh(coords, cells, [0] * len(cells))


def _solve(mesh, **kw):
    return UnstructuredDiffusionSolver(mesh, [MAT], np.zeros(mesh.n_cells, int),
                                       alpha_boundary=_ZERO_FLUX, device="cpu",
                                       **kw).solve(tol_k=1e-10, tol_source=1e-9)


def test_diagonal_cut_triangles_converge_to_bare_square_k():
    k_exact = _bare_square_k()
    err = []
    for n in (8, 16, 32):
        mesh = _diagonal_cut_square(n)
        assert mesh.area.sum() == pytest.approx(L * L, rel=1e-12)
        res = _solve(mesh)
        assert res.converged
        err.append(abs(res.k_eff - k_exact))
    assert err[0] > err[1] > err[2]   # converging with refinement
    assert err[2] < 1e-3              # < 100 pcm (orthogonal quads: 54 pcm at n = 32)
    assert err[0] / err[1] > 2.5 and err[1] / err[2] > 2.5   # ~second order


def test_uncorrected_tpfa_is_inconsistent_on_skewed_triangles():
    # The control: the same mesh with the correction switched off leaves an
    # O(1) error (normal-distance TPFA alone is not consistent).
    res = _solve(_diagonal_cut_square(32), nonorthogonal_correction=False)
    assert abs(res.k_eff - _bare_square_k()) > 1e-2


def test_randomly_perturbed_triangulation_converges():
    # Jittered nodes (up to 25 % of the spacing): every face skewed by a
    # different amount, triangles neither right nor equilateral.
    k_exact = _bare_square_k()
    err = []
    for n in (16, 32, 64):
        mesh = _diagonal_cut_square(n, perturb=0.25, seed=n)
        res = _solve(mesh)
        assert res.converged
        err.append(abs(res.k_eff - k_exact))
    assert err[0] > err[1] > err[2]
    assert err[2] < 1e-4                # < 10 pcm at n = 64


def test_orthogonal_meshes_have_no_correction():
    # Rectangles: centroid lines are face normals, so the correction is
    # identically zero and is not even set up.
    xs = np.linspace(0.0, L, 5)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    idx = lambda i, j: i * 5 + j
    cells = [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
             for i in range(4) for j in range(4)]
    mesh = assemble_mesh(np.c_[X.ravel(), Y.ravel()], cells, [0] * 16)
    s = UnstructuredDiffusionSolver(mesh, [MAT], np.zeros(16, int), device="cpu")
    assert s._noc is None
