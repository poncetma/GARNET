"""Coupled Marshak vacuum boundary: dedicated SP3 block and active masks.

* The dedicated symmetric SP3 block (SP3EigenSolver) applies the coupled
  Brantley-Larsen Marshak condition on vacuum faces by default, so it must
  reproduce the U-form SP3 (SPNEigenSolver order 1, marshak_vacuum=True) --
  the two are the same discretization up to the congruence diag(1, 3). The
  old per-moment alpha=1/2 Robin term missed it by 120-250 pcm.
* With an active mask, the coupled term must also sit on the core/void faces
  (mask_bc="vacuum") and never on excised cells: a box embedded in excised
  cells must give the same eigenvalue as the bare box.
"""

import numpy as np
import pytest

from garnet import Grid, Material, SP3EigenSolver
from garnet.solver import SDP2EigenSolver, SP5EigenSolver, SPNEigenSolver

FUEL = Material(name="f", diffusion=[1.4730, 0.3294], sigma_a=[0.0096, 0.0764],
                nu_sigma_f=[0.0067, 0.1241], sigma_s=[[0.0, 0.0161], [0, 0]],
                chi=[1.0, 0.0])
WATER = Material(name="w", diffusion=[1.7639, 0.2278], sigma_a=[0.0003, 0.0097],
                 nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.0380], [0, 0]],
                 chi=[1.0, 0.0])
TIGHT = dict(tol_k=1e-10, tol_source=1e-9)


class _SP3U(SPNEigenSolver):
    _order = 1


def _slab(L=20.0, refl=5.0, h=0.25):
    n = int(round((L + 2 * refl) / h))
    grid = Grid(shape=(n, 1, 1), size=(n * h, 1.0, 1.0))
    x = (np.arange(n) + 0.5) * h
    mm = np.where((x > refl) & (x < refl + L), 1, 0).reshape(n, 1, 1)
    bc = (("vacuum", "vacuum"), "reflective", "reflective")
    return grid, mm, bc


def test_dedicated_sp3_vacuum_is_coupled_marshak():
    grid, mm, bc = _slab()
    mats = [WATER, FUEL]
    k_ded = SP3EigenSolver(grid, mats, material_map=mm, bc=bc,
                           device="cpu").solve(**TIGHT).k_eff
    k_u = _SP3U(grid, mats, material_map=mm, bc=bc, device="cpu",
                marshak_vacuum=True).solve(**TIGHT).k_eff
    assert k_ded == pytest.approx(k_u, abs=1e-6)
    # The per-moment Robin form stays available (marshak_vacuum=False); it is
    # a distinctly different, cruder boundary (~100+ pcm here).
    k_pm = SP3EigenSolver(grid, mats, material_map=mm, bc=bc, device="cpu",
                          marshak_vacuum=False).solve(**TIGHT).k_eff
    assert abs(k_pm - k_ded) > 5e-4


def test_dedicated_sp3_marshak_block_stays_symmetric():
    grid, mm, bc = _slab(L=4.0, refl=1.0, h=0.5)
    s = SP3EigenSolver(grid, [WATER, FUEL], material_map=mm, bc=bc, device="cpu")
    op = s.ops[1]
    assert op.marshak_vacuum
    n = op.inv_diag.size
    A = np.column_stack([op.apply(e.reshape(op.inv_diag.shape)).ravel()
                         for e in np.eye(n)])
    assert np.allclose(A, A.T, atol=1e-13)
    assert np.all(np.linalg.eigvalsh(A) > 0)
    assert np.allclose(1.0 / np.diag(A), op.inv_diag.ravel())


def test_marshak_unavailable_for_sdp1_or_hybrid_falls_back():
    grid, mm, bc = _slab(L=4.0, refl=1.0, h=0.5)
    from garnet import SDP1EigenSolver
    s = SDP1EigenSolver(grid, [WATER, FUEL], material_map=mm, bc=bc, device="cpu")
    assert not s.ops[0].marshak_vacuum
    with pytest.raises(ValueError):
        SDP1EigenSolver(grid, [WATER, FUEL], material_map=mm, bc=bc,
                        device="cpu", marshak_vacuum=True)


def _embedded(n=10, border=3, h=2.0, outer_bc="reflective"):
    small = Grid(shape=(n, n, 1), size=(n * h, n * h, 1.0))
    mm_s = np.zeros((n, n, 1), int)
    mm_s[:4, :4] = 1
    m = n + border
    big = Grid(shape=(m, m, 1), size=(m * h, m * h, 1.0))
    mm_b = np.zeros((m, m, 1), int)
    mm_b[:n, :n] = mm_s
    active = np.zeros((m, m, 1), bool)
    active[:n, :n] = True
    bc_s = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    bc_b = (("reflective", outer_bc), ("reflective", outer_bc), "reflective")
    return small, mm_s, bc_s, big, mm_b, bc_b, active


@pytest.mark.parametrize("outer_bc", ["reflective", "vacuum"])
@pytest.mark.parametrize("cls, kw", [
    (SP5EigenSolver, {"marshak_vacuum": True}),
    (SDP2EigenSolver, {"marshak_vacuum": True}),
    (SP3EigenSolver, {}),
], ids=["sp5", "sdp2", "sp3"])
def test_marshak_on_mask_faces_matches_bare_box(cls, kw, outer_bc):
    # The excised border is decoupled; its outer box face (vacuum in one case)
    # must add nothing, and the core/void faces must carry the coupled term.
    small, mm_s, bc_s, big, mm_b, bc_b, active = _embedded(outer_bc=outer_bc)
    mats = [WATER, FUEL]
    k_box = cls(small, mats, mm_s, bc=bc_s, device="cpu", **kw).solve(**TIGHT).k_eff
    k_mask = cls(big, mats, mm_b, bc=bc_b, active=active, mask_bc="vacuum",
                 device="cpu", **kw).solve(**TIGHT).k_eff
    assert k_mask == pytest.approx(k_box, abs=1e-8)
