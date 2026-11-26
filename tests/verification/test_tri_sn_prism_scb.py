"""Corner balance in z on extruded prisms (``axial_corners=2``, the default).

Each prism is split into six corner sub-prisms: the three in-plane SCB
corners times a lower and an upper half, the halves coupled through an
internal mid-height face carrying their average (SCB in z). With three corners
spanning the full height (``axial_corners=1``) the caps are upwinded, which is
first order in z.

Checks:

* a problem flat in z (periodic axially) is solved identically by both: k
  equal to 4e-16;
* LA-13511 slab 2 laid along z (S64, SCB): the error falls by 3.4-3.6 per
  halving of dz (second order), to -10.0 pcm at 200 layers against the S64
  angular floor of about -9 pcm; three corners give +546 / +262 / +125 / +58
  pcm at 25-200 layers (first order). P2 slab 35 behaves the same (-4.9 pcm at
  200 layers, ratios 3.1-3.4);
* the levels and LU engines agree with six corners (P2 block in
  test_tri_sn_3d_anisotropic.py) and with three, and the levels engine's
  one-group table cache (``level_cache="group"``, refilled in place on every
  group switch) gives the same k as keeping every group (below; the CUDA-graph
  replay after a refill is checked in test_gpu_paths.py).

Broken on purpose: the mid-height face value upwinded instead of averaged (a
step inside the prism) brings back first-order behaviour: +55.8 pcm instead of
-13.0 pcm at 100 layers.
"""

import numpy as np
import pytest

from garnet.benchmarks.sood import PROBLEMS, sood_material
from garnet.tri import TriGrid
from garnet.tri_sn import TriSNTransportSolver

TOL = dict(tol_k=1e-11, tol_source=1e-10)


def _slab(nz, axial_corners=2, number=2):
    p = PROBLEMS[number]
    grid = TriGrid(shape=(2, 2, 2, nz), side=1.0, height=2.0 * p.rc)
    return TriSNTransportSolver(grid, [sood_material(p.material, p.order)], n_polar=64,
                                n_azi=8 if p.order > 1 else 4, bc=("periodic", "vacuum"),
                                scheme="scb", engine="lu", device="cpu",
                                scattering_order=p.order,
                                axial_corners=axial_corners).solve(**TOL).k_eff


def test_flat_in_z_is_identical_with_three_or_six_corners():
    m = sood_material("Ua", 0)
    grid = TriGrid(shape=(6, 6, 2, 3), side=1.0, height=3.0)
    act = np.ones(grid.shape, bool)
    act[0, :] = act[:, 0] = False
    ks = [TriSNTransportSolver(grid, [m], active=act, n_polar=4, n_azi=8,
                               bc=("periodic", "periodic"), scheme="scb", engine="lu",
                               device="cpu", axial_corners=a).solve(**TOL).k_eff
          for a in (1, 2)]
    assert ks[1] == pytest.approx(ks[0], abs=1e-13)


def test_slab_along_z_converges_at_second_order():
    k = [_slab(nz) for nz in (25, 50, 100)]
    assert (k[0] - k[1]) / (k[1] - k[2]) > 3.0              # 3.4: second order
    ext = k[2] + (k[2] - k[1]) / 3.0
    assert abs(ext - 1.0) < 1.5e-4                          # S64 floor ~ -9 pcm


def test_three_corners_are_first_order_in_z():
    k = [_slab(nz, axial_corners=1) for nz in (25, 50, 100)]
    assert 1.8 < (k[0] - k[1]) / (k[1] - k[2]) < 2.3       # 2.1: first order
    assert k[2] - 1.0 > 1e-3                                # +125 pcm at 100 layers


def test_broken_upwinded_mid_height_face_is_first_order(monkeypatch):
    blocks = TriSNTransportSolver._scb3d_blocks

    def upwinded(self, g, m):
        B = blocks(self, g, m)
        d = self._scb
        h2, hi, Acap, Vc = self._scb3d_geometry()
        c = d["axi_sign"] * self.xi[m] * Acap                 # outward z flow
        for cc in range(6):
            B[:, cc, cc] -= 0.5 * c[cc]                       # remove the average
            B[:, cc, (cc + 3) % 6] -= 0.5 * c[cc]
            if c[cc] > 0:
                B[:, cc, cc] += c[cc]                         # outflow: own value
            else:
                B[:, cc, (cc + 3) % 6] += c[cc]               # inflow: upwind value
        return B

    good = _slab(100)
    monkeypatch.setattr(TriSNTransportSolver, "_scb3d_blocks", upwinded)
    bad = _slab(100)
    assert abs(bad - good) > 2e-4


@pytest.mark.parametrize("axial_corners,cache", [(1, "all"), (2, "all"), (2, "group")])
def test_levels_engine_matches_lu(axial_corners, cache):
    """Two groups, so the group cache refills its one table on every switch."""
    m = sood_material("URRa", 0)
    grid = TriGrid(shape=(5, 5, 2, 4), side=1.5, height=5.0)
    act = np.ones(grid.shape, bool)
    act[0, 0] = False
    ks = [TriSNTransportSolver(grid, [m], active=act, n_polar=4, n_azi=8, bc="vacuum",
                               scheme="scb", engine=e, device="cpu",
                               axial_corners=axial_corners, level_cache=cache
                               ).solve(**TOL).k_eff
          for e in ("lu", "levels")]
    assert ks[1] == pytest.approx(ks[0], abs=1e-10)


def _cmfd_reuse_run(reuse, max_outer=500):
    m = sood_material("URRa", 0)
    grid = TriGrid(shape=(8, 8, 2, 6), side=2.0, height=12.0)
    act = np.ones(grid.shape, bool)
    act[0, :] = False
    s = TriSNTransportSolver(grid, [m], active=act, n_polar=4, n_azi=8, bc="vacuum",
                             scheme="scb", engine="levels", device="cpu", cmfd_reuse=reuse)
    return s, s.solve(tol_k=1e-9, tol_source=1e-8, max_outer=max_outer)


def test_cmfd_lagged_factorization_matches_refactorizing():
    """cmfd_reuse: the last LU preconditions GMRES on the new drift matrix
    (k equal to 1e-12, same outers, 3 factorizations instead of 8)."""
    _, ref = _cmfd_reuse_run(0)
    s, r = _cmfd_reuse_run(4)
    assert r.converged and r.k_eff == pytest.approx(ref.k_eff, abs=1e-12)
    assert r.outer_iterations == ref.outer_iterations
    assert s.cmfd_reuse_stats["factorizations"] < ref.outer_iterations


def test_broken_stale_factorization_without_gmres_correction(monkeypatch):
    """The stale LU alone solves last outer's drift matrix: k = 0.26298
    instead of 0.25931, not converged in 60 outers."""
    import scipy.sparse.linalg as spla
    _, ref = _cmfd_reuse_run(0)
    monkeypatch.setattr(spla, "gmres", lambda A, b, x0=None, **k: (x0, 0))
    _, r = _cmfd_reuse_run(4, max_outer=60)
    assert not r.converged or abs(r.k_eff - ref.k_eff) > 1e-4
