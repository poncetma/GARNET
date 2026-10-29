"""Triangular / hex lattice boundaries and the discontinuity-factor adjoint.

* With no active mask the lattice's outer in-plane edges must take the bc spec's
  in-plane axes (they used to be silently reflective: k = k_inf whatever bc
  said); the result must equal the same core padded with a void border.
* An axial face between an active and an inactive cell is a mask boundary
  (mask_bc), so masking off the top layers equals the shorter core.
* An active mask touching the array edge is accepted (that edge is exposed).
* With discontinuity factors the tri operator is non-symmetric; the adjoint
  must solve with its transpose: adjoint k = forward k and the adjoint flux is
  the dense left eigenvector.
"""

import numpy as np
import pytest

from garnet import k_infinite
from garnet.hex import HexDiffusionEigenSolver, HexGrid
from garnet.materials import Material
from garnet.tri import TriDiffusionEigenSolver, TriGrid

TOL = dict(tol_k=1e-11, tol_source=1e-10)
FUEL = Material(name="f", diffusion=[1.3], sigma_a=[0.1], nu_sigma_f=[0.3])


def _tri_k(shape, active=None, **kw):
    g = TriGrid(shape=shape, side=1.0, height=float(shape[3]) if len(shape) == 4 else 1.0)
    return TriDiffusionEigenSolver(g, [FUEL], np.zeros(shape, int), active=active,
                                   device="cpu", **kw).solve(**TOL).k_eff


def _padded(shape):
    act = np.zeros((shape[0] + 2, shape[1] + 2) + tuple(shape[2:]), bool)
    act[1:-1, 1:-1] = True
    return act


@pytest.mark.parametrize("bc", ["vacuum", "zero-flux", 0.3])
def test_tri_unmasked_outer_edges_take_bc(bc):
    k_nomask = _tri_k((8, 8, 2), bc=bc)
    k_mask = _tri_k((10, 10, 2), active=_padded((8, 8, 2)), mask_bc=bc)
    assert k_nomask < k_infinite(FUEL) - 0.1
    assert k_nomask == pytest.approx(k_mask, abs=1e-8)


def test_tri_unmasked_reflective_in_plane_is_kinf():
    assert _tri_k((4, 4, 2), bc="reflective") == pytest.approx(k_infinite(FUEL), abs=1e-9)
    # per-axis: rows reflective, columns vacuum differs from both extremes
    k_mix = _tri_k((6, 6, 2), bc=("reflective", "vacuum", "reflective"))
    assert _tri_k((6, 6, 2), bc="vacuum") < k_mix < k_infinite(FUEL)


def test_tri_mask_touching_array_edge_is_exposed():
    # An all-ones mask used to raise "void border"; its array edges are now
    # mask boundaries, identical to padding.
    k_full = _tri_k((6, 6, 2), active=np.ones((6, 6, 2), bool), mask_bc="vacuum")
    k_pad = _tri_k((8, 8, 2), active=_padded((6, 6, 2)), mask_bc="vacuum")
    assert k_full == pytest.approx(k_pad, abs=1e-8)


def test_tri_axial_mask_face_is_robin():
    def k(nz, nact):
        a = _padded((6, 6, 2, nz))
        a[..., nact:] = False
        return _tri_k(a.shape, active=a, mask_bc="vacuum",
                      bc=("reflective", "reflective", "vacuum"))
    k5 = k(5, 5)
    assert k(8, 5) == pytest.approx(k5, abs=1e-8)       # was 0.699 vs 0.580


def _hex_k(shape, active=None, **kw):
    g = HexGrid(shape=shape, pitch=1.0, height=float(shape[2]))
    return HexDiffusionEigenSolver(g, [FUEL], np.zeros(shape, int), active=active,
                                   device="cpu", **kw).solve(**TOL).k_eff


def test_hex_unmasked_outer_faces_take_bc():
    k_nomask = _hex_k((7, 7, 1), bc=("vacuum", "vacuum", "reflective"))
    act = np.zeros((9, 9, 1), bool)
    act[1:-1, 1:-1] = True
    k_mask = _hex_k((9, 9, 1), active=act, mask_bc="vacuum",
                    bc=("vacuum", "vacuum", "reflective"))
    assert k_nomask < k_infinite(FUEL) - 0.1
    assert k_nomask == pytest.approx(k_mask, abs=1e-8)
    assert _hex_k((4, 4, 1), bc="reflective") == pytest.approx(k_infinite(FUEL), abs=1e-9)


def test_hex_axial_mask_face_is_robin():
    def k(nz, nact):
        a = np.zeros((7, 7, nz), bool)
        a[1:-1, 1:-1, :nact] = True
        return _hex_k(a.shape, active=a, mask_bc="vacuum",
                      bc=("reflective", "reflective", "vacuum"))
    assert k(8, 5) == pytest.approx(k(5, 5), abs=1e-8)


TWO_GROUP = [
    Material(name="a", diffusion=[1.4, 0.4], sigma_a=[0.010, 0.08],
             nu_sigma_f=[0.007, 0.14], sigma_s=[[0.0, 0.02], [0.0, 0.0]], chi=[1, 0]),
    Material(name="b", diffusion=[1.1, 0.3], sigma_a=[0.012, 0.12],
             nu_sigma_f=[0.004, 0.10], sigma_s=[[0.0, 0.017], [0.001, 0.0]], chi=[1, 0]),
]


def test_tri_df_adjoint_is_dense_left_eigenvector():
    shape = (3, 3, 2)
    grid = TriGrid(shape=shape, side=2.0)
    rng = np.random.default_rng(3)
    mmap = rng.integers(0, 2, shape)
    act = np.ones(shape, bool)
    df = 0.7 + 0.6 * rng.random((2,) + shape)
    kw = dict(active=act, mask_bc="vacuum", df=df, linear_solver="gmres", device="cpu")
    s = TriDiffusionEigenSolver(grid, TWO_GROUP, mmap, **kw)
    fwd = s.solve(tol_k=1e-12, tol_source=1e-11)
    adj = TriDiffusionEigenSolver(grid, TWO_GROUP, mmap, **kw).solve(
        tol_k=1e-12, tol_source=1e-11, adjoint=True, max_outer=4000)

    # Dense M phi = (1/k) F phi from the operators' own assembled matrices.
    n = int(np.prod(shape))
    flat = mmap.ravel()
    A = [s.ops[g].assemble().toarray() for g in range(2)]
    assert not np.allclose(A[0], A[0].T)               # genuinely non-symmetric
    M = np.zeros((2 * n, 2 * n))
    F = np.zeros((2 * n, 2 * n))
    for g in range(2):
        M[g * n:(g + 1) * n, g * n:(g + 1) * n] = A[g]
        for gp in range(2):
            if gp != g:
                ss = np.array([TWO_GROUP[m].sigma_s[gp][g] for m in flat])
                M[g * n:(g + 1) * n, gp * n:(gp + 1) * n] -= np.diag(ss)
            chi = np.array([TWO_GROUP[m].chi[g] for m in flat])
            nsf = np.array([TWO_GROUP[m].nu_sigma_f[gp] for m in flat])
            F[g * n:(g + 1) * n, gp * n:(gp + 1) * n] = np.diag(chi * nsf)
    ev, vec = np.linalg.eig(np.linalg.solve(M, F))
    i = np.argmax(ev.real)
    k_dense = ev[i].real
    evl, vecl = np.linalg.eig(np.linalg.solve(M.T, F.T))
    left = vecl[:, np.argmax(evl.real)].real
    left /= left.sum()

    assert fwd.k_eff == pytest.approx(k_dense, rel=1e-8)
    assert adj.k_eff == pytest.approx(k_dense, rel=1e-8)
    got = np.asarray(adj.flux).reshape(2 * n)
    got = got / got.sum()
    np.testing.assert_allclose(got, left, rtol=1e-6, atol=1e-9)
