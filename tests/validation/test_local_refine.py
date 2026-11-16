"""Locally-refined HP-MR mesh: nonconforming FV coupling and drum resolution.

Local refinement splits only the coarse cells in each drum's absorber band into
four, meeting the surrounding coarse cells at 2:1 hanging nodes. The key
correctness check is conservation: a homogeneous medium with reflective
boundaries must give exactly k_inf whether or not the drums are refined -- if
the nonconforming coarse-to-fine coupling lost or double-counted current, it
would not. On top of that, refining the absorber band resolves the arc that a
coarse raster misses, so the drum worth appears.
"""

import numpy as np
import pytest

from garnet import k_infinite
from garnet.benchmarks.hpmr import _placeholder_materials, hpmr_locally_refined_mesh
from garnet.mesh import UnstructuredDiffusionSolver


def test_nonconforming_reflective_is_kinf():
    # Homogeneous medium (all six slots the same material) + reflective BC:
    # k must equal k_inf exactly, refined or not -- the conservation test.
    fuel = _placeholder_materials()[1]
    homog = [fuel] * 6
    kinf = k_infinite(fuel)
    for refine_drums in (False, True):
        mesh, cm, _, _ = hpmr_locally_refined_mesh(
            refine=3, drum_angle_deg=90.0, refine_drums=refine_drums, materials=homog)
        res = UnstructuredDiffusionSolver(mesh, homog, cm, alpha_boundary=0.0).solve(tol_k=1e-9)
        assert res.k_eff == pytest.approx(kinf, abs=1e-7), (refine_drums, res.k_eff)


def test_local_refinement_adds_fine_cells_in_the_band():
    m0, c0, _, _ = hpmr_locally_refined_mesh(refine=3, drum_angle_deg=120.0, refine_drums=False)
    m1, c1, _, _ = hpmr_locally_refined_mesh(refine=3, drum_angle_deg=120.0, refine_drums=True)
    assert m1.n_cells > m0.n_cells                 # the drum bands were refined
    # refinement is local: it stays a small fraction of the whole mesh
    assert m1.n_cells < 1.5 * m0.n_cells
    assert int((c1 == 5).sum()) >= int((c0 == 5).sum())   # >= absorber cells resolved


def test_hanging_node_refinement_is_consistent():
    # Consistency of the 2:1 coupling: in a homogeneous medium with vacuum
    # boundaries (so the flux has real curvature and leakage), the locally
    # refined k must converge toward an independent globally fine reference as
    # the base resolution rises. A broken hanging-node coefficient would either
    # not converge or converge to the wrong limit (an O(1) inconsistency);
    # instead the error shrinks monotonically and stays small. (The band
    # refinement itself buys no accuracy here -- there is no feature to resolve
    # -- so this isolates the interface treatment, not the refinement's payoff.)
    fuel = _placeholder_materials()[1]
    homog = [fuel] * 6

    def k(refine, refine_drums):
        mesh, cm, _, alpha = hpmr_locally_refined_mesh(
            refine=refine, drum_angle_deg=90.0, refine_drums=refine_drums, materials=homog)
        return UnstructuredDiffusionSolver(mesh, homog, cm, alpha).solve(tol_k=1e-9).k_eff

    k_ref = k(8, False)                                    # fine uniform reference
    err = [abs(k(r, True) - k_ref) * 1e5 for r in (2, 3, 4)]   # pcm
    assert err[0] < 40.0                                   # no O(1) inconsistency
    assert err[2] < err[1] < err[0]                        # converging, monotone
    assert err[2] < 15.0                                   # approaching the true k


def test_local_refine_gives_negative_drum_worth():
    def k(angle):
        mesh, cm, mats, alpha = hpmr_locally_refined_mesh(
            refine=3, drum_angle_deg=angle, refine_drums=True)
        r = UnstructuredDiffusionSolver(mesh, mats, cm, alpha).solve(tol_k=1e-7)
        assert r.converged
        return r.k_eff
    # Convention (hpmr._drum_geometry): 0 = the B4C arc faces the core centre
    # (inserted), 180 = outward (withdrawn). Verified geometrically -- at 0 deg
    # the absorber cells sit ~11 cm core-side of their drum centre, at 180 deg
    # ~12 cm outward.
    k_in, k_out = k(0.0), k(180.0)
    assert k_out > k_in                             # arcs toward core remove reactivity
    assert (1 / k_in - 1 / k_out) * 1e5 > 500       # worth, positive by insertion


# --- volume-mixed arc, deeper refinement, and SPH zones on the refined mesh ---

def test_polar_refined_mesh_carries_the_exact_absorber_area():
    import math
    from garnet.benchmarks.hpmr import (DRUM_ABSORBER_INNER, DRUM_ARC_HALF_DEG,
                                        DRUM_RADIUS, hpmr_locally_refined_problem)
    sector = math.radians(DRUM_ARC_HALF_DEG) * (DRUM_RADIUS ** 2 - DRUM_ABSORBER_INNER ** 2)
    for refine, levels, angle in ((3, 1, 0.0), (4, 2, 37.0)):
        p = hpmr_locally_refined_problem(refine=refine, drum_angle_deg=angle,
                                         levels=levels, absorber="polar")
        area = float((p.mix_weight * p.mesh.area).sum())
        assert area == pytest.approx(12 * sector, rel=1e-12)
        assert not np.any(p.cell_material == 5)            # no painted absorber


def test_polar_refined_homogeneous_medium_is_kinf():
    # Conservation again, now through the per-cell blended materials: a mix of
    # a material with itself must leave k_inf unchanged on the 4:1 mesh.
    from garnet.benchmarks.hpmr import hpmr_locally_refined_problem
    fuel = _placeholder_materials()[1]
    homog = [fuel] * 6
    p = hpmr_locally_refined_problem(refine=3, drum_angle_deg=60.0, levels=2,
                                     absorber="polar", materials=homog)
    mats, cells = p.solver_materials()
    assert len(mats) > 6                                    # blends were built
    res = UnstructuredDiffusionSolver(p.mesh, mats, cells, alpha_boundary=0.0).solve(tol_k=1e-9)
    assert res.k_eff == pytest.approx(k_infinite(fuel), abs=1e-7)


def test_blended_material_matches_the_raster_blend_rules():
    from garnet.blend import MaterialBlend, blend_material
    be, b4c = _placeholder_materials()[4], _placeholder_materials()[5]
    w = 0.37
    m = blend_material(be, b4c, w)
    blend = MaterialBlend(np, (1,), np.array([0]), 2, mix_material=np.array([1]),
                          mix_weight=np.array([w]))
    for g in range(2):
        for attr in ("sigma_a", "sigma_t", "nu_sigma_f"):
            table = np.array([getattr(be, attr)[g], getattr(b4c, attr)[g]])
            assert getattr(m, attr)[g] == pytest.approx(float(blend.linear(table)[0]), rel=1e-14)
        table = np.array([be.diffusion[g], b4c.diffusion[g]])
        assert m.diffusion[g] == pytest.approx(float(blend.harmonic(table)[0]), rel=1e-14)


def test_mesh_zones_and_footprints_agree_with_the_raster():
    from garnet.benchmarks.hpmr import (build_hpmr2d, hpmr_assembly_zones,
                                        hpmr_locally_refined_problem, hpmr_mesh_zones)
    from garnet.benchmarks.hpmr_core_pins import R_FUEL, hpmr_pin_layout
    from garnet.pin_power import mesh_pin_footprints
    p = hpmr_locally_refined_problem(refine=3, drum_angle_deg=20.0, levels=1)
    z = hpmr_mesh_zones(p)
    r = build_hpmr2d(refine=3, drum_angle_deg=20.0, absorber="polar")
    zr = hpmr_assembly_zones(r)
    assert z.names == zr.names
    vol = np.bincount(z.region_map, weights=p.mesh.area, minlength=len(z.names))
    vol_r = np.bincount(zr.region_map.ravel(), minlength=len(zr.names)) * r.grid.cell_volume
    np.testing.assert_allclose(vol[1:], vol_r[1:], rtol=1e-12)    # void is not meshed
    merged = hpmr_mesh_zones(p, merge_drum=True)
    assert len(merged.names) == len(z.names) - 1 and "drum" in merged.names
    pins = hpmr_pin_layout()
    W = mesh_pin_footprints(p.mesh, pins.xy[pins.fuel], R_FUEL)
    np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel(), np.pi * R_FUEL ** 2,
                               rtol=1e-12)
