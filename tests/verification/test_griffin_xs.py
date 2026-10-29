"""Griffin / YakXs cross-section reader.

Exercised on a tiny inline two-group YakXs fixture with a known scattering
matrix (including upscatter), so the test is self-contained -- it does not
need the multi-MB VTB library on disk. The crux is the scattering profile:
each row is a *sink* group listing the *source*-group range, so the parser
must place value k of sink row g at sigma_s[source, g] (a transpose relative
to the file's row order). The balance Total = Absorption + scatter-out and the
volume-homogenization reaction-rate conservation are checked too.
"""

import numpy as np
import pytest

from garnet.analytic import k_infinite
from garnet.griffin_xs import read_library, read_material, volume_homogenize

# Two-group fixture. Scattering (source->sink): 1->1=0.44, 1->2=0.05,
# 2->1=0.001 (upscatter), 2->2=0.899. Profile rows are per SINK group; each
# lists the source range "first last" then the values for those sources.
#   Total = Absorption + scatter_out  =>  g1: 0.01 + (0.44+0.05) = 0.50,
#                                          g2: 0.10 + (0.001+0.899) = 1.00.
_FIXTURE = """<YakXs>
  <Multigroup_Cross_Section_Libraries Name="fix" NGroup="2">
    <Multigroup_Cross_Section_Library ID="1">
      <Tfuel>800</Tfuel><Tmod>800</Tmod>
      <Table gridIndex="1 1">
        <Isotope Name="pseudo">
          <Total>0.5 1.0</Total>
          <Absorption>0.01 0.10</Absorption>
          <nuFission>0.005 0.15</nuFission>
          <FissionSpectrum>1.0 0.0</FissionSpectrum>
          <Scattering profile="1">
            <Profile>
              1 2
              1 2
            </Profile>
            <Value>
              0.44 0.001
              0.05 0.899
            </Value>
          </Scattering>
        </Isotope>
      </Table>
    </Multigroup_Cross_Section_Library>
    <Multigroup_Cross_Section_Library ID="2">
      <Tfuel>800</Tfuel><Tmod>800</Tmod>
      <Table gridIndex="1 1">
        <Isotope Name="pseudo">
          <Total>0.6 1.2</Total>
          <Absorption>0.02 0.20</Absorption>
          <nuFission>0.0 0.0</nuFission>
          <FissionSpectrum>0.0 0.0</FissionSpectrum>
          <Scattering profile="1">
            <Profile>
              1 1
              1 2
            </Profile>
            <Value>
              0.58
              0.02 1.0
            </Value>
          </Scattering>
        </Isotope>
      </Table>
    </Multigroup_Cross_Section_Library>
  </Multigroup_Cross_Section_Libraries>
</YakXs>
"""


@pytest.fixture
def lib_path(tmp_path):
    p = tmp_path / "fix.xml"
    p.write_text(_FIXTURE)
    return str(p)


def test_scattering_profile_transpose(lib_path):
    m = read_material(lib_path, 1, grid_index="1 1")
    # sigma_s[source, sink]
    np.testing.assert_allclose(m.sigma_s[0, 0], 0.44)
    np.testing.assert_allclose(m.sigma_s[0, 1], 0.05)
    np.testing.assert_allclose(m.sigma_s[1, 0], 0.001)   # upscatter preserved
    np.testing.assert_allclose(m.sigma_s[1, 1], 0.899)


def test_reactions_and_balance(lib_path):
    m = read_material(lib_path, 1, grid_index="1 1")
    np.testing.assert_allclose(m.sigma_t, [0.5, 1.0])
    np.testing.assert_allclose(m.diffusion, [1 / (3 * 0.5), 1 / (3 * 1.0)])
    np.testing.assert_allclose(m.nu_sigma_f, [0.005, 0.15])
    np.testing.assert_allclose(m.chi.sum(), 1.0)
    # Total = Absorption + scatter-out, group by group.
    scatter_out = m.sigma_s.sum(axis=1)
    np.testing.assert_allclose(m.sigma_a + scatter_out, m.sigma_t, atol=1e-12)


def test_nonfissile_material_is_not_fissile(lib_path):
    m = read_material(lib_path, 2, grid_index="1 1")
    assert not m.is_fissile  # chi is moot for a non-fissile material


def test_volume_homogenize_conserves_reaction_rates(lib_path):
    lib = read_library(lib_path, (1, 2), grid_index="1 1")
    frac = {1: 0.7, 2: 0.3}
    hom = volume_homogenize(lib, frac, chi_from=1)
    # each macroscopic XS is the volume-weighted average
    for g in range(2):
        exp = 0.7 * lib[1].sigma_a[g] + 0.3 * lib[2].sigma_a[g]
        np.testing.assert_allclose(hom.sigma_a[g], exp)
    np.testing.assert_allclose(hom.sigma_t, 0.7 * lib[1].sigma_t + 0.3 * lib[2].sigma_t)
    assert hom.is_fissile and hom.chi[0] == 1.0


def test_sph_factor_scales_cross_sections(lib_path):
    lib = read_library(lib_path, (1, 2), grid_index="1 1")
    base = volume_homogenize(lib, {1: 0.7, 2: 0.3}, chi_from=1)
    mu = np.array([1.2, 0.7])            # non-uniform per-group SPH factors
    scaled = volume_homogenize(lib, {1: 0.7, 2: 0.3}, chi_from=1, sph_factors=mu)
    np.testing.assert_allclose(scaled.nu_sigma_f, mu * base.nu_sigma_f)
    np.testing.assert_allclose(scaled.sigma_t, mu * base.sigma_t)
    np.testing.assert_allclose(scaled.sigma_s, mu[:, None] * base.sigma_s)
    # A uniform SPH factor scales every reaction equally and cancels in the
    # eigenvalue ratio, leaving k_inf invariant -- the property that makes SPH
    # a pure spectral reshaping, not a criticality knob.
    uniform = volume_homogenize(lib, {1: 0.7, 2: 0.3}, chi_from=1,
                                sph_factors=np.array([0.9, 0.9]))
    assert k_infinite(uniform) == pytest.approx(k_infinite(base), rel=1e-12)


def _write_without(tmp_path, *tags, material="1"):
    """The fixture with ``tags`` removed from one material's table."""
    lines, current = [], None
    for line in _FIXTURE.splitlines():
        if "Multigroup_Cross_Section_Library ID=" in line:
            current = line.split('ID="')[1].split('"')[0]
        if current == material and any(f"<{t}>" in line for t in tags):
            continue
        lines.append(line)
    p = tmp_path / "missing.xml"
    p.write_text("\n".join(lines))
    return str(p)


@pytest.mark.parametrize("tag", ["Total", "Absorption"])
def test_missing_required_tag_raises_instead_of_zero_filling(tmp_path, tag):
    # Zero-filled, Total gave D = inf and Absorption a non-absorbing material.
    with pytest.raises(KeyError, match=tag):
        read_material(_write_without(tmp_path, tag), 1, grid_index="1 1")


def test_non_fissile_material_may_omit_all_fission_tags(tmp_path):
    # The VTB library omits nuFission/FissionSpectrum/kappaFission on every
    # non-fissile table; that is the legitimately optional case.
    path = _write_without(tmp_path, "nuFission", "FissionSpectrum", material="2")
    m = read_material(path, 2, grid_index="1 1")
    assert not m.is_fissile
    np.testing.assert_array_equal(m.nu_sigma_f, 0.0)


def test_incomplete_fission_data_is_rejected(tmp_path):
    with pytest.raises(KeyError, match="nuFission"):
        read_material(_write_without(tmp_path, "nuFission"), 1, grid_index="1 1")
    with pytest.raises(KeyError, match="FissionSpectrum"):
        read_material(_write_without(tmp_path, "FissionSpectrum"), 1, grid_index="1 1")
