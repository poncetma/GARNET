"""Triangular S_N against the LA-13511 analytic criticality set (Sood et al. 1999).

Every problem is exactly critical (k = 1) at its tabulated dimension, so k - 1
is the solver's error. The anisotropic problems -- one-group slabs with P1 and
P2 scattering (34, 35), one-group cylinders with P1 scattering (36, 37; 37 is so
forward-peaked that its kernel is negative near mu = -1) and a two-group slab
with P1 within-group and down-scatter (71) -- run on coarse meshes next to
their isotropic controls (12, 13, 54). Two things are checked: the anisotropic
problems are as close to k = 1 as the isotropic ones on the same relative mesh,
and the same meshes without the anisotropy are thousands of pcm off, so a broken
anisotropic path cannot pass.

Converged results (tools/verify_sn_sood.py, mesh and quadrature extrapolation;
docs/qa/sph_audit_evidence/verify_sn_sood.out): cylinders 13 / 36 / 37 at
-5.8 / -5.5 / -3.9 pcm; slab controls 2 / 12 at -35 / -19 and anisotropic slabs
34 / 35 at -29 / -26 pcm (product quadrature on optically thin slabs); two-group
54 at -6 and 71 at -35 pcm.
"""

import numpy as np
import pytest

from garnet import transport_corrected
from garnet.benchmarks.sood import PROBLEMS, build_sood_problem, slab_flux_edits
from garnet.tri_sn import TriSNTransportSolver

TOL = dict(tol_k=1e-10, tol_source=1e-9)


def _k(case, fuel=None, order=None, quad=(4, 16), flux=False):
    mats = list(case.materials)
    if fuel is not None:
        mats[0] = fuel
    r = TriSNTransportSolver(case.grid, mats, case.material_map, active=case.active,
                             mix_material=case.mix_material, mix_weight=case.mix_weight,
                             n_polar=quad[0], n_azi=quad[1], bc=case.bc, scheme="scb",
                             engine=case.engine, device="cpu",
                             scattering_order=case.scattering_order if order is None else order
                             ).solve(**TOL)
    assert r.converged
    return (r.k_eff, r.flux) if flux else r.k_eff


def _isotropic(fuel):
    return fuel.replace(sigma_s_moments=None)


def test_problem_data_reproduce_the_tabulated_infinite_medium_values():
    from garnet.benchmarks.sood import sood_material
    from garnet.tri import TriGrid
    for name, k_inf in (("Ua", 2.25), ("Ua-aniso", 2.25), ("PUb-aniso", 2.5),
                        ("URRa", 1.631452), ("URRa-aniso", 1.631452)):
        k = TriSNTransportSolver(TriGrid(shape=(2, 2, 2), side=1.0), [sood_material(name)],
                                 n_polar=1, n_azi=4, bc="periodic", engine="lu",
                                 device="cpu").solve(**TOL).k_eff
        assert k == pytest.approx(k_inf, abs=2e-6)


@pytest.mark.parametrize("number", [34, 35])
def test_anisotropic_one_group_slabs_are_critical(number):
    case = build_sood_problem(number, 80)
    control = abs(_k(build_sood_problem(12, 80)) - 1.0)     # isotropic, same mesh
    k = _k(case)
    assert abs(k - 1.0) < 120e-5 and abs(k - 1.0) < control + 80e-5
    # without the anisotropy the slab is far from critical
    assert abs(_k(case, fuel=_isotropic(case.materials[0]), order=0) - 1.0) > 4000e-5
    if number == 35:          # and P2 matters: P1 alone is 900 pcm off
        fuel = case.materials[0]
        p1 = fuel.replace(sigma_s_moments=fuel.sigma_s_moments[:2])
        assert abs(_k(case, fuel=p1, order=1) - 1.0) > 500e-5


def test_anisotropic_cylinders_track_their_isotropic_control():
    err = {n: _k(build_sood_problem(n, 32)) - 1.0 for n in (13, 36, 37)}
    # SCB at 32 cells across the diameter: -388 / -358 / -231 pcm (all three
    # converge to within 6 pcm of k = 1)
    assert all(-450e-5 < e < 0.0 for e in err.values())
    assert abs(err[36]) <= abs(err[13]) + 20e-5
    assert abs(err[37]) <= abs(err[13]) + 20e-5
    case = build_sood_problem(37, 32)
    fuel = case.materials[0]
    assert _k(case, fuel=_isotropic(fuel), order=0) - 1.0 > 15000e-5
    assert _k(case, fuel=transport_corrected(fuel, "diagonal"), order=0) - 1.0 > 1500e-5


def test_two_group_anisotropic_slab_is_critical_with_the_reference_flux_shape():
    case = build_sood_problem(71, 160)
    k, flux = _k(case, flux=True)
    control = _k(build_sood_problem(54, 160)) - 1.0       # isotropic twin: -294 pcm
    assert abs(k - 1.0) < 350e-5 and abs(k - 1.0) < abs(control) + 60e-5
    vals, _ = slab_flux_edits(case, flux)
    ref = np.array(PROBLEMS[71].flux)
    assert np.abs(vals[:, :-1] / ref[:, :-1] - 1.0).max() < 0.005    # interior points
    # the fast -> slow P1 transfer alone is worth +215 pcm; ignoring all of the
    # anisotropy is worth +15000
    fuel = case.materials[0]
    assert _k(case, fuel=_isotropic(fuel), order=0) - 1.0 > 10000e-5
