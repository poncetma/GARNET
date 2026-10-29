"""Adjoint eigen-solve of the moment-block (SP3/SPN/SDPN) solvers against an
independently assembled dense transpose.

The discrete forward problem is L U = (1/k) F U with L = blockdiag(A_g) - S.
Every piece is assembled column by column from the *forward* code paths only
(``ops[g].apply``, ``_rhs``, ``_phi``, ``_src_weight``), so the reference cannot
inherit a mistake from the adjoint path. The left eigenvector U* of (L, F)
gives the scalar importance R_g^T U*_g, where R_g is the forward map from an
isotropic scalar source to the block right-hand side (without the r-z metric
weight, which belongs to the operator, exactly as for diffusion) -- the
quantity ``solve(adjoint=True).flux`` must reproduce.

SDPN is the case that bites: its block coupling and Marshak boundary matrices
are non-symmetric, and its source/flux weight vectors differ (SDP2: src =
c1[:, 0], phi0 = c1[0, :]), so reusing the forward spatial block or the forward
weights on the adjoint path gives a wrong importance and eigenvalue.
"""

import numpy as np
import pytest

from garnet import (Grid, Material, SDP1EigenSolver, SDP2EigenSolver,
                    SDP3EigenSolver, SP3EigenSolver, SP5EigenSolver)
from garnet.solver import SDPNEigenSolver

TIGHT = dict(tol_k=1e-12, tol_source=1e-11, max_outer=20000,
             inner_rtol_floor=1e-13)

FUEL = Material(name="f", diffusion=[1.4730, 0.3294], sigma_a=[0.0096, 0.0764],
                nu_sigma_f=[0.0067, 0.1241], sigma_s=[[0.0, 0.0161], [0.0, 0.0]],
                chi=[1.0, 0.0])
WATER = Material(name="w", diffusion=[1.7639, 0.2278], sigma_a=[0.0003, 0.0097],
                 nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.0380], [0.0, 0.0]],
                 chi=[1.0, 0.0])
VAC = (("vacuum", "vacuum"), "reflective", "reflective")
FIXED = (("zero-flux", "reflective"), "reflective", "reflective")

N = 16
GRID = Grid(shape=(N, 1, 1), size=(8.0, 1.0, 1.0))
MAP = np.where((np.arange(N) >= 4) & (np.arange(N) < 12), 1, 0).reshape(N, 1, 1)
HYBRID = np.zeros((N, 1, 1), dtype=bool)
HYBRID[3:7] = True

SDPN1 = type("SDPN1", (SDPNEigenSolver,), {"_order": 1})

CASES = [
    ("sp3-vacuum", SP3EigenSolver, VAC, {}),
    ("sdp1-vacuum", SDP1EigenSolver, VAC, {}),
    ("sdpn1-marshak", SDPN1, VAC, {"marshak_vacuum": True}),
    ("sdp2-vacuum-symmetrized", SDP2EigenSolver, VAC, {}),
    ("sdp2-vacuum-plain", SDP2EigenSolver, VAC, {"symmetrize": False}),
    ("sdp2-marshak", SDP2EigenSolver, VAC, {"marshak_vacuum": True}),
    ("sdp2-marshak-spn-precond", SDP2EigenSolver, VAC,
     {"marshak_vacuum": True, "spn_precondition": 2}),
    ("sdp3-vacuum", SDP3EigenSolver, VAC, {}),
    ("sdp3-marshak", SDP3EigenSolver, VAC, {"marshak_vacuum": True}),
    ("sdp3-congruence", SDP3EigenSolver, FIXED, {}),
    ("sp5-marshak", SP5EigenSolver, VAC, {"marshak_vacuum": True}),
    ("sp3-hybrid", SP3EigenSolver, VAC, {"hybrid_mask": HYBRID}),
    ("sdp2-hybrid", SDP2EigenSolver, VAC,
     {"hybrid_mask": HYBRID, "symmetrize": False}),
    ("sdp3-hybrid-confined", SDP3EigenSolver, VAC,
     {"hybrid_mask": HYBRID, "hybrid_confine": True}),
]


def _dense_reference(s):
    """(k, scalar importance (G, N)) from the dense forward L, F and their
    transposes, built only from forward code paths."""
    G = s.n_groups
    shape = s.grid.shape
    n = int(np.prod(shape))
    M = np.asarray(s._initial_state()[0]).size // n      # moments per group
    size = M * n
    weight = 1.0 if s._src_weight is None else np.asarray(s._src_weight)

    def unit_state(c):
        e = np.zeros(size)
        e[c] = 1.0
        return e.reshape(s._initial_state()[0].shape)

    # R_g: scalar source -> block rhs; P: block state -> scalar flux.
    R = np.empty((G, size, n))
    for g in range(G):
        for c in range(n):
            q = np.zeros(n)
            q[c] = 1.0
            R[g][:, c] = np.asarray(s._rhs(g, q.reshape(shape))).ravel()
    Rw = R * np.broadcast_to(weight, shape).ravel()      # metric-weighted source
    P = np.empty((n, size))
    for c in range(size):
        P[:, c] = np.asarray(s._phi(unit_state(c))).ravel()

    L = np.zeros((G * size, G * size))
    F = np.zeros_like(L)
    chi = [np.asarray(x).ravel() * np.ones(n) for x in s.chi]
    nsf = [np.asarray(x).ravel() * np.ones(n) for x in s.nu_sigma_f]
    for g in range(G):
        blk = slice(g * size, (g + 1) * size)
        for c in range(size):
            L[blk, g * size + c] = np.asarray(s.ops[g].apply(unit_state(c))).ravel()
        for gp in range(G):
            cols = slice(gp * size, (gp + 1) * size)
            F[blk, cols] = Rw[g] @ (chi[g][:, None] * nsf[gp][:, None] * P)
            sig = s.sigma_s[gp][g]
            if gp != g and sig is not None:
                sig = np.asarray(sig).ravel() * np.ones(n)
                L[blk, cols] -= Rw[g] @ (sig[:, None] * P)

    def dominant(Lm, Fm):
        w, v = np.linalg.eig(np.linalg.solve(Lm, Fm))
        i = np.argmax(w.real)
        return w[i].real, v[:, i].real

    k_fwd, _ = dominant(L, F)
    k_adj, u_star = dominant(L.T, F.T)
    assert k_adj == pytest.approx(k_fwd, rel=1e-10)
    importance = np.stack([R[g].T @ u_star[g * size:(g + 1) * size]
                           for g in range(G)])
    return k_fwd, importance


def _normalized(a):
    a = np.asarray(a).reshape(a.shape[0], -1)
    a = a / np.abs(a).max()
    return -a if a.sum() < 0 else a


@pytest.mark.parametrize("name,cls,bc,kw", CASES, ids=[c[0] for c in CASES])
def test_adjoint_matches_dense_transpose(name, cls, bc, kw):
    s = cls(GRID, [WATER, FUEL], material_map=MAP, bc=bc, device="cpu", **kw)
    k_ref, imp_ref = _dense_reference(s)
    fwd = s.solve(**TIGHT)
    adj = s.solve(adjoint=True, **TIGHT)
    assert fwd.converged and adj.converged, (fwd, adj)
    assert fwd.k_eff == pytest.approx(k_ref, rel=1e-9)
    assert adj.k_eff == pytest.approx(k_ref, rel=1e-9)
    err = np.abs(_normalized(adj.flux_numpy) - _normalized(imp_ref)).max()
    assert err < 1e-7, err


def test_adjoint_matches_dense_transpose_cylindrical():
    # r-z: the volume-weighted moment stencils and the rhs metric weight.
    grid = Grid(shape=(N, 1, 1), size=(8.0, 1.0, 1.0), geometry="cylindrical")
    s = SDP2EigenSolver(grid, [WATER, FUEL], material_map=MAP,
                        bc=(("reflective", "vacuum"), "reflective", "reflective"),
                        device="cpu", symmetrize=False)
    k_ref, imp_ref = _dense_reference(s)
    adj = s.solve(adjoint=True, **TIGHT)
    assert adj.converged, adj
    assert adj.k_eff == pytest.approx(k_ref, rel=1e-9)
    err = np.abs(_normalized(adj.flux_numpy) - _normalized(imp_ref)).max()
    assert err < 1e-7, err
