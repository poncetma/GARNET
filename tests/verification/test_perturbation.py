"""First-order perturbation theory: garnet.first_order_reactivity.

Two regimes, both physically important:

* Weak perturbations (feedback coefficients, cross-section sensitivities) -- the
  first-order estimate matches a direct re-solve to fractions of a percent, and
  the error shrinks quadratically as the perturbation does. This is the regime
  the tool is for.
* A strong localized black absorber (the HP-MR B4C drum arc) -- first-order PT
  over-predicts the worth several-fold, because it neglects the order-one flux
  self-shielding the absorber creates. We assert that breakdown so the
  limitation stays documented (and nobody mistakes PT for a drum-worth shortcut).
"""

import numpy as np
import pytest

from garnet import (DiffusionEigenSolver, Grid, Material, PWR_TWO_GROUP,
                   SP3EigenSolver, first_order_reactivity)
from garnet.benchmarks.hpmr import build_hpmr2d, _placeholder_materials
from garnet.tri import TriDiffusionEigenSolver

TIGHT = dict(tol_k=1e-10, tol_source=1e-9)
GRID = Grid(shape=(20, 20, 20), size=(80.0, 80.0, 80.0))


def _poison(material, dsigma_a):
    return Material(name="perturbed", diffusion=material.diffusion,
                    sigma_a=material.sigma_a + np.asarray(dsigma_a),
                    nu_sigma_f=material.nu_sigma_f, sigma_s=material.sigma_s,
                    chi=material.chi)


def test_first_order_matches_resolve_and_converges_quadratically():
    ref = DiffusionEigenSolver(GRID, PWR_TWO_GROUP, device="cpu")
    fwd = ref.solve(**TIGHT)
    adj = ref.solve(adjoint=True, **TIGHT)

    def errors(eps):
        pert = DiffusionEigenSolver(GRID, _poison(PWR_TWO_GROUP, [0.0, eps]), device="cpu")
        drho_exact = 1.0 / fwd.k_eff - 1.0 / pert.solve(**TIGHT).k_eff
        drho_pt = first_order_reactivity(ref, fwd, adj, pert)
        return abs(drho_pt - drho_exact), abs(drho_exact)

    e_small, rho_small = errors(1e-4)
    e_big, rho_big = errors(1e-3)
    assert e_small / rho_small < 1e-3               # accurate for a weak perturbation
    # error is second order in the perturbation: 10x eps -> ~100x error
    assert 40 < e_big / e_small < 250, e_big / e_small


def test_first_order_on_rz_grid_is_volume_weighted():
    # r-z: the loss operator is volume-weighted (rhs weight = cell radius), so
    # in-scatter and fission must carry the same weight. Without it the zero
    # perturbation returned O(10) instead of 0. Perturb only the near-axis
    # region, where the radius weight differs most from uniform.
    grid = Grid(shape=(20, 1, 20), size=(80.0, 80.0, 80.0), geometry="cylindrical")
    inner = np.zeros(grid.shape, dtype=int)
    inner[:6] = 1
    ref = DiffusionEigenSolver(grid, [PWR_TWO_GROUP, PWR_TWO_GROUP],
                               material_map=inner, device="cpu")
    fwd = ref.solve(**TIGHT)
    adj = ref.solve(adjoint=True, **TIGHT)
    assert abs(first_order_reactivity(ref, fwd, adj, ref)) < 1e-9

    def errors(eps):
        pert = DiffusionEigenSolver(
            grid, [PWR_TWO_GROUP, _poison(PWR_TWO_GROUP, [0.0, eps])],
            material_map=inner, device="cpu")
        drho_exact = 1.0 / fwd.k_eff - 1.0 / pert.solve(**TIGHT).k_eff
        drho_pt = first_order_reactivity(ref, fwd, adj, pert)
        return abs(drho_pt - drho_exact), abs(drho_exact)

    e_small, rho_small = errors(1e-4)
    e_big, rho_big = errors(1e-3)
    assert e_small / rho_small < 1e-2, e_small / rho_small
    assert 40 < e_big / e_small < 250, e_big / e_small


def test_block_solvers_are_rejected_clearly():
    grid = Grid(shape=(8, 8, 1), size=(40.0, 40.0, 1.0))
    sp3 = SP3EigenSolver(grid, PWR_TWO_GROUP, device="cpu")
    fwd = sp3.solve()
    with pytest.raises(TypeError, match="diffusion-family"):
        first_order_reactivity(sp3, fwd, fwd, sp3)


def _tri_hpmr(angle, materials=None):
    p = build_hpmr2d(refine=3, drum_angle_deg=angle, absorber="polar", materials=materials)
    return TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                                   mask_bc=p.mask_bc, mix_material=p.mix_material,
                                   mix_weight=p.mix_weight)


def test_first_order_reactivity_on_tri_weak_perturbation():
    # A weak absorption bump in the fuel: PT on the real HP-MR tri solver agrees
    # with a direct re-solve.
    base = _placeholder_materials()
    ref = _tri_hpmr(90.0)
    fwd = ref.solve(**TIGHT)
    adj = ref.solve(adjoint=True, **TIGHT)

    perturbed_mats = [base[0], _poison(base[1], [0.0, 1e-3])] + base[2:]
    pert = _tri_hpmr(90.0, materials=perturbed_mats)
    drho_exact = 1.0 / fwd.k_eff - 1.0 / pert.solve(**TIGHT).k_eff
    drho_pt = first_order_reactivity(ref, fwd, adj, pert)
    assert drho_exact < 0
    assert abs(drho_pt - drho_exact) / abs(drho_exact) < 0.05


def test_first_order_over_predicts_black_drum_worth():
    # Full drum swing (arc out -> arc into core). First-order PT badly
    # over-predicts because the near-black B4C arc self-shields: the neglected
    # flux depression is an order-one effect, not a small correction.
    ref = _tri_hpmr(180.0)                        # 180 = arcs out (withdrawn)
    fwd = ref.solve(**TIGHT)
    adj = ref.solve(adjoint=True, **TIGHT)

    pert = _tri_hpmr(0.0)                          # 0 = arcs into the core (inserted)
    worth_exact = 1.0 / fwd.k_eff - 1.0 / pert.solve(**TIGHT).k_eff
    worth_pt = first_order_reactivity(ref, fwd, adj, pert)
    assert worth_exact < 0 and worth_pt < 0
    # PT over-predicts the magnitude by more than 3x (self-shielding it ignores)
    assert worth_pt / worth_exact > 3.0
