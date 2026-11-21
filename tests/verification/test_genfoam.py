"""GeN-Foam nuclearData reader (garnet.genfoam).

A synthetic two-group, two-family dictionary in GeN-Foam's format (SI units,
OpenFOAM syntax with comments, two states, two zones) is read and checked
against hand conversions:

* units: 1/m -> 1/cm, D m -> cm, IV s/m -> v cm/s, sigmaPow J/m -> J/cm;
* conventions of GeN-Foam's S_N: scatteringMatrixP<l>[from][to] without the
  (2l+1) factor, total = sigmaRemoval + P0 self-scatter, absorption = removal
  - out-scatter, equilibrium chi = (1 - beta) chiPrompt + beta chiDelayed;
* the infinite-medium k of the returned Material equals the two-group formula
  evaluated on the raw file values, k = (nsf1 + nsf2 S12 / R2) / R1;
* a combined spectrum not summing to 1 is folded into nu_sigma_f with k kept.

Broken on purpose: the scattering matrix read transposed (to, from) gives an
upscatter material whose k_inf is 0.29 times the formula; the (2l+1) factor
applied to P1 breaks the D / transport consistency of the real Godiva data.

The real GeN-Foam Godiva file is checked only when GARNET_GENFOAM_NUCLEARDATA
points to it (its data are not copied into this repository).
"""

import os

import numpy as np
import pytest

from garnet.genfoam import parse_foam_dict, read_nuclear_data

_FILE = r"""/*--------------------------------*- C++ -*----------------------------------*\
|  synthetic GeN-Foam nuclearData for the reader tests                         |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      nuclearData;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

energyGroups    2;
precGroups      2;
legendreMoments 2;   // P0-P1 used by the solver
isLowMemory     true;
xsVariables
{}

states
(
    reference
    {
        zones
        (
            fuel
            {
                fuelFraction    1.000000e+00;
                IV              nonuniform List<scalar> 2 (5.0e-08 2.5e-04);
                D               nonuniform List<scalar> 2 (1.5e-02 4.0e-03);
                nuSigmaEff      nonuniform List<scalar> 2 (0.8e+00 1.2e+01);
                sigmaPow        nonuniform List<scalar> 2 (1.0e-11 1.5e-10);
                scatteringMatrixP0  2  2 (
                    ( 2.00e+01  1.60e+00 )
                    ( 0.00e+00  5.00e+01 )
                );
                scatteringMatrixP1  2  2 (
                    ( 4.00e+00  0.30e+00 )
                    ( 0.00e+00  2.00e+00 )
                );
                sigmaRemoval    nonuniform List<scalar> 2 (2.6e+00 1.0e+01);
                chiPrompt       nonuniform List<scalar> 2 (1.0e+00 0.0e+00);
                chiDelayed      nonuniform List<scalar> 2 (1.0e+00 0.0e+00);
                Beta            nonuniform List<scalar> 2 (2.0e-03 4.0e-03);
                lambda          nonuniform List<scalar> 2 (1.2e-02 3.0e-01);
                discFactor      nonuniform List<scalar> 2 (1 1);
                integralFlux    nonuniform List<scalar> 2 (1 1);
            }
            reflector
            {
                IV              nonuniform List<scalar> 2 (5.0e-08 2.5e-04);
                D               nonuniform List<scalar> 2 (1.2e-02 3.0e-03);
                nuSigmaEff      nonuniform List<scalar> 2 (0 0);
                scatteringMatrixP0  2  2 (
                    ( 3.00e+01  2.00e+00 )
                    ( 0.00e+00  6.00e+01 )
                );
                sigmaRemoval    nonuniform List<scalar> 2 (2.1e+00 0.5e+00);
                chiPrompt       nonuniform List<scalar> 2 (1 0);
                chiDelayed      nonuniform List<scalar> 2 (1 0);
                Beta            nonuniform List<scalar> 2 (0 0);
                lambda          nonuniform List<scalar> 2 (1.2e-02 3.0e-01);
            }
        );
    }
    fuelTemp
    {
        zones
        (
            fuel
            {
                IV              nonuniform List<scalar> 2 (5.0e-08 2.5e-04);
                D               nonuniform List<scalar> 2 (1.5e-02 4.0e-03);
                nuSigmaEff      nonuniform List<scalar> 2 (0.79e+00 1.19e+01);
                scatteringMatrixP0  2  2 (
                    ( 2.00e+01  1.60e+00 )
                    ( 0.00e+00  5.00e+01 )
                );
                sigmaRemoval    nonuniform List<scalar> 2 (2.6e+00 1.0e+01);
                chiPrompt       nonuniform List<scalar> 2 (1 0);
                chiDelayed      nonuniform List<scalar> 2 (1 0);
                Beta            nonuniform List<scalar> 2 (2.0e-03 4.0e-03);
                lambda          nonuniform List<scalar> 2 (1.2e-02 3.0e-01);
            }
        );
    }
);

// ************************************************************************* //
"""


def _k_inf(m):
    """Infinite-medium k of a Material: (diag(removal) - in-scatter) phi =
    chi nsf^T phi / k, in-scatter[to, from] = sigma_s[from, to] off-diagonal."""
    S = m.sigma_s - np.diag(np.diag(m.sigma_s))
    A = np.diag(m.removal) - S.T
    F = np.outer(m.chi, m.nu_sigma_f)
    return float(np.max(np.abs(np.linalg.eigvals(np.linalg.solve(A, F)))))


def _k_formula():
    # raw file values: removal R, downscatter S12, chi all fast
    R1, R2, S12, n1, n2 = 2.6, 10.0, 1.6, 0.8, 12.0
    return (n1 + n2 * S12 / R2) / R1


@pytest.fixture
def zones(tmp_path):
    p = tmp_path / "nuclearData"
    p.write_text(_FILE)
    return read_nuclear_data(p)


def test_states_and_zones_are_read(zones):
    assert set(zones) == {("reference", "fuel"), ("reference", "reflector"),
                          ("fuelTemp", "fuel")}
    z = zones[("reference", "fuel")]
    assert z.energy_groups == 2 and z.prec_groups == 2
    assert z.available_order == 1 and z.solver_order == 1
    assert zones[("fuelTemp", "fuel")].data["nuSigmaEff"][1] == pytest.approx(11.9)


def test_units_and_conventions(zones):
    z = zones[("reference", "fuel")]
    m, kin = z.material(), z.kinetics()
    np.testing.assert_allclose(m.diffusion, [1.5, 0.4])
    np.testing.assert_allclose(m.total, [(2.6 + 20.0) / 100, (10.0 + 50.0) / 100])
    np.testing.assert_allclose(m.sigma_a, [(2.6 - 1.6) / 100, 10.0 / 100])
    np.testing.assert_allclose(m.removal, [0.026, 0.10])
    np.testing.assert_allclose(m.sigma_s, [[0.20, 0.016], [0.0, 0.50]])     # [from, to]
    np.testing.assert_allclose(m.sigma_s_moments[1], [[0.04, 0.003], [0.0, 0.02]])
    np.testing.assert_allclose(m.nu_sigma_f, [0.008, 0.12])
    np.testing.assert_allclose(m.kappa_fission, [1e-13, 1.5e-12])
    np.testing.assert_allclose(m.chi, [1.0, 0.0])
    np.testing.assert_allclose(kin.velocities, [100 / 5e-8, 100 / 2.5e-4])
    np.testing.assert_allclose(kin.beta, [2e-3, 4e-3])
    np.testing.assert_allclose(kin.decay, [1.2e-2, 0.3])
    assert z.material(0).sigma_s_moments is None


def test_infinite_medium_k_matches_the_two_group_formula(zones):
    m = zones[("reference", "fuel")].material()
    assert _k_inf(m) == pytest.approx(_k_formula(), rel=1e-12)


def test_broken_transposed_matrix_misses_the_formula(zones):
    m = zones[("reference", "fuel")].material()
    wrong = m.replace(sigma_s=m.sigma_s.T.copy(),
                      sigma_a=m.removal - (m.sigma_s.T.sum(axis=1) - np.diag(m.sigma_s)))
    assert abs(_k_inf(wrong) / _k_formula() - 1.0) > 0.1


def test_chi_sum_is_folded_into_production(tmp_path):
    text = _FILE.replace("chiPrompt       nonuniform List<scalar> 2 (1.0e+00 0.0e+00);",
                         "chiPrompt       nonuniform List<scalar> 2 (0.9e+00 0.2e+00);")
    p = tmp_path / "nuclearData"
    p.write_text(text)
    z = read_nuclear_data(p)[("reference", "fuel")]
    b = 6e-3
    chi_raw = (1 - b) * np.array([0.9, 0.2]) + b * np.array([1.0, 0.0])
    assert z.chi_sum == pytest.approx(chi_raw.sum())
    m = z.material()
    assert m.chi.sum() == pytest.approx(1.0)
    # k of the raw (unnormalized) operator
    S = np.array([[0.0, 0.016], [0.0, 0.0]])
    A = np.diag([0.026, 0.10]) - S.T
    k_raw = float(np.max(np.abs(np.linalg.eigvals(
        np.linalg.solve(A, np.outer(chi_raw, [0.008, 0.12]))))))
    assert _k_inf(m) == pytest.approx(k_raw, rel=1e-12)


def test_equilibrium_spectrum_mixes_prompt_and_delayed(tmp_path):
    text = _FILE.replace("chiDelayed      nonuniform List<scalar> 2 (1.0e+00 0.0e+00);",
                         "chiDelayed      nonuniform List<scalar> 2 (0.0e+00 1.0e+00);")
    p = tmp_path / "nuclearData"
    p.write_text(text)
    m = read_nuclear_data(p)[("reference", "fuel")].material()
    np.testing.assert_allclose(m.chi, [1 - 6e-3, 6e-3])


def test_old_format_and_errors():
    d = parse_foam_dict("energyGroups 1; precGroups 1; zones ( a { x 1; } );")
    assert d["zones"] == {"a": {"x": 1.0}}
    with pytest.raises(ValueError):
        parse_foam_dict("a ( 1 2 ;")
    with pytest.raises(ValueError):
        parse_foam_dict("m 2 2 ( (1 2) (3) );")


def test_missing_key_is_reported(tmp_path):
    p = tmp_path / "nuclearData"
    p.write_text(_FILE.replace("sigmaRemoval    nonuniform List<scalar> 2 (2.1e+00 0.5e+00);", ""))
    with pytest.raises(KeyError, match="sigmaRemoval"):
        read_nuclear_data(p)


_GODIVA = os.environ.get("GARNET_GENFOAM_NUCLEARDATA")


@pytest.mark.skipif(not _GODIVA, reason="GARNET_GENFOAM_NUCLEARDATA not set")
def test_godiva_data_d_matches_the_p1_transport_cross_section():
    """Independent check of the P_l convention on the real data: the file's D
    equals 1 / (3 (Sigma_t - sum_g' Sigma_s1[g -> g'])) to 0.4 % in groups
    2-18, which holds only without the (2l+1) factor (with it: 13-39 % off)."""
    z = read_nuclear_data(_GODIVA)[("reference", "uranium")]
    m = z.material(1)
    inv = 1.0 / (3.0 * m.diffusion)
    s1 = m.sigma_s_moments[1]
    good = m.total - s1.sum(axis=1)
    broken = m.total - s1.sum(axis=1) / 3.0
    sl = slice(2, 18)
    assert np.max(np.abs(inv[sl] / good[sl] - 1.0)) < 5e-3
    assert np.min(np.abs(inv[sl] / broken[sl] - 1.0)) > 1e-2
