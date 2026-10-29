"""FEMFFUSION cross-section file readers.

The .xsec reader is checked against the hand-transcribed VVER-440 table in
garnet.benchmarks.vver440 (both originate from FEMFFUSION's
examples/2D_VVER440/2D_VVER440.xsec, so they must agree exactly), and the XML
reader against values read straight off the C5G7 file, including the
SigmaS[to, from] -> sigma_s[from, to] transpose and upscatter.
"""

import os

import numpy as np
import pytest

from garnet.benchmarks.vver440 import _XS
from garnet.femffusion import read_material_xml, read_xsec

FEMFFUSION = os.environ.get(
    "FEMFFUSION_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "FEMFFUSION"))
VVER_XSEC = os.path.join(FEMFFUSION, "examples", "2D_VVER440", "2D_VVER440.xsec")
C5G7_XML = os.path.join(FEMFFUSION, "examples", "2D_C5G7", "input.mat.xml")

needs_femffusion = pytest.mark.skipif(
    not os.path.isdir(FEMFFUSION), reason="FEMFFUSION checkout not available")


@needs_femffusion
def test_xsec_matches_vver440_transcription():
    xs = read_xsec(VVER_XSEC)
    assert len(xs.materials) == 8
    for mid, (tr1, a1, nsf1, s12, tr2, a2, nsf2) in _XS.items():
        m = xs.materials[mid - 1]
        assert m.n_groups == 2
        np.testing.assert_allclose(m.diffusion, [1 / (3 * tr1), 1 / (3 * tr2)])
        np.testing.assert_allclose(m.sigma_a, [a1, a2])
        np.testing.assert_allclose(m.nu_sigma_f, [nsf1, nsf2])
        np.testing.assert_allclose(m.sigma_s, [[0, s12], [0, 0]])
        np.testing.assert_allclose(m.chi, [1, 0])


@needs_femffusion
def test_xsec_core_map_and_kinetics():
    xs = read_xsec(VVER_XSEC)
    # 25 hex rows; 349 assemblies + reflector ring = 421 entries, ids 1..8
    assert len(xs.core_map) == 25
    flat = [v for row in xs.core_map for v in row]
    assert len(flat) == 421
    assert set(flat) <= set(range(1, 9))
    assert xs.kinetics is not None
    np.testing.assert_allclose(xs.kinetics.beta, [0.0065])
    np.testing.assert_allclose(xs.kinetics.decay, [0.07841])
    np.testing.assert_allclose(xs.kinetics.velocities, [1.25e7, 2.5e5])


@needs_femffusion
def test_xsec_solves_vver440():
    # End-to-end: file -> materials -> tri solver reproduces the benchmark k.
    from garnet.benchmarks import build_vver440
    from garnet.tri import TriDiffusionEigenSolver

    xs = read_xsec(VVER_XSEC)
    p = build_vver440(refine=1)
    materials = [p.materials[0]] + xs.materials      # keep the void at index 0
    res = TriDiffusionEigenSolver(p.grid, materials, p.material_map, active=p.active,
                                  mask_bc=p.mask_bc, device="cpu").solve(
        tol_k=1e-8, tol_source=1e-7)
    assert res.converged
    assert res.k_eff == pytest.approx(1.0035, abs=6e-3)


@needs_femffusion
def test_xml_c5g7_values_and_transpose():
    mats = read_material_xml(C5G7_XML)
    assert all(m.n_groups == 7 for m in mats)
    mod = mats[0]                                    # mix 0 = moderator
    assert not mod.is_fissile
    # Straight off the file: SigmaT row, SigmaA row.
    np.testing.assert_allclose(mod.sigma_t[0], 1.59206e-1)
    np.testing.assert_allclose(mod.sigma_a[-1], 3.72390e-2)
    # SigmaS file row 2, col 1 (to=2, from=1) = 1.13400e-1 -> sigma_s[0, 1].
    np.testing.assert_allclose(mod.sigma_s[0, 1], 1.13400e-1)
    # Upscatter: file row 4, col 5 (to=4, from=5) = 7.14370e-5 -> sigma_s[4, 3].
    np.testing.assert_allclose(mod.sigma_s[4, 3], 7.14370e-5)
    fuel = mats[1]
    assert fuel.is_fissile
    np.testing.assert_allclose(fuel.chi.sum(), 1.0)


@needs_femffusion
def test_xml_c5g7_cross_transcription():
    # The FEMFFUSION XML and GARNET's OpenMOC-derived table are independent
    # transcriptions of the same NEA C5G7 set; after the SigmaS transpose they
    # must agree entry-for-entry.
    from garnet.benchmarks._c5g7_data import C5G7_XS
    mats = {m.name: m for m in read_material_xml(C5G7_XML)}
    for xml_name, ref_name in [("fuelUO2", "UO2"), ("fuelMOX4.3", "MOX-4.3%"),
                               ("moderator", "Water")]:
        m, ref = mats[xml_name], C5G7_XS[ref_name]
        np.testing.assert_allclose(m.sigma_t, ref["total"], rtol=1e-5)
        np.testing.assert_allclose(m.nu_sigma_f, ref["nu_fission"], rtol=1e-5)
        np.testing.assert_allclose(m.sigma_s, ref["scatter"], rtol=1e-5)
        if m.is_fissile:
            np.testing.assert_allclose(m.chi, ref["chi"], rtol=1e-5)


_XSEC_3D = """\
# two planes, hex-staggered rows (ragged), as in FEMFFUSION's 3D_Langenbuch
Materials
PLANE 1
1     1 1
2   1 1 1
3     1 1
PLANE 2
1     1 2
2   2 3 1
3     1 1

XSecs 3 # Number of materials defined
#    Sigma_tr1    Sigma_a1   nuSigma_f1   Sigma_f1    Sigma_12
#    Sigma_tr2    Sigma_a2   nuSigma_f2   Sigma_f2
1   0.20   0.0026   0.0     0.0      0.027
    1.26   0.049    0.0     0.0
2   0.23   0.011    0.0075  0.0030   0.017
    0.95   0.099    0.1378  0.0551
3   0.23   0.010    0.0065  0.0026   0.0175
    0.93   0.088    0.1127  0.0451

Velocity
1.25e7   2.5e5
"""


def _write(tmp_path, text):
    path = tmp_path / "case.xsec"
    path.write_text(text)
    return str(path)


def test_xsec_plane_headers_give_per_plane_maps(tmp_path):
    """PLANE n lines are headers, not map rows: reading them as rows put a stray
    [n] between planes and merged every plane into one corrupted map."""
    xs = read_xsec(_write(tmp_path, _XSEC_3D))
    assert xs.n_planes == 2
    assert xs.core_map == [[[1, 1], [1, 1, 1], [1, 1]],
                           [[1, 2], [2, 3, 1], [1, 1]]]
    assert len(xs.materials) == 3


def test_xsec_2d_map_is_unchanged_by_plane_support(tmp_path):
    lines = _XSEC_3D.splitlines(keepends=True)
    text = "".join(lines[:2] + lines[3:6] + lines[10:])  # plane 1, no headers
    xs = read_xsec(_write(tmp_path, text))
    assert xs.n_planes == 0
    assert xs.core_map == [[1, 1], [1, 1, 1], [1, 1]]


def test_xsec_plane_out_of_order_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="PLANE 2"):
        read_xsec(_write(tmp_path, _XSEC_3D.replace("PLANE 2", "PLANE 3")))


def test_xsec_keeps_the_power_cross_section_as_kappa(tmp_path):
    """The Sigma_f column becomes kappa*Sigma_f in energy units (200 MeV/fission
    in J by default); dropping it sent the power edit back to nu-weighting."""
    from garnet.femffusion import ENERGY_PER_FISSION_J
    xs = read_xsec(_write(tmp_path, _XSEC_3D))
    assert xs.materials[0].kappa_fission is None     # reflector: not fissile
    np.testing.assert_allclose(xs.materials[1].kappa_fission,
                               ENERGY_PER_FISSION_J * np.array([0.0030, 0.0551]))
    np.testing.assert_allclose(xs.materials[2].kappa_fission,
                               ENERGY_PER_FISSION_J * np.array([0.0026, 0.0451]))
    custom = read_xsec(_write(tmp_path, _XSEC_3D), energy_per_fission=1.0)
    np.testing.assert_allclose(custom.materials[1].kappa_fission, [0.0030, 0.0551])
