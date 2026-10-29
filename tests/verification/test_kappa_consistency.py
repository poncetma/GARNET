"""kappa*Sigma_f consistency across the power edit, SPH and the .xsec reader.

kappaSigma_f carries energy units and nuSigma_f does not, so the power edit must
never weight one fissile material by one and its neighbour by the other: two
identical fuels, one with a library kappa in J/cm and one without, used to split
the power ~1e-11 : 1. And every step that rebuilds a Material (SPH
homogenization, file readers) must carry kappa through, or the power shape
silently reverts to nu-weighting downstream.
"""

import numpy as np
import pytest

from garnet import Material, flux_weighted_homogenize
from garnet.power import fission_energy_xs, power_density


def _fuel(**kw):
    return Material(name=kw.pop("name", "fuel"), diffusion=[1.4, 0.4],
                    sigma_a=[0.01, 0.1], nu_sigma_f=[0.007, 0.2],
                    sigma_s=[[0.0, 0.02], [0.0, 0.0]], chi=[1.0, 0.0], **kw)


REFL = Material(name="refl", diffusion=[1.2, 0.9], sigma_a=[2e-4, 5e-3],
                nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.04], [0.0, 0.0]])


def test_fissile_material_without_kappa_next_to_one_with_it_is_rejected():
    with_kappa = _fuel(name="libfuel", kappa_fission=[0.003 * 3.2e-11, 0.08 * 3.2e-11])
    without = _fuel(name="plainfuel")
    with pytest.raises(ValueError, match="plainfuel"):
        fission_energy_xs([REFL, with_kappa, without])
    mmap = np.array([0, 1]).reshape(2, 1, 1)
    with pytest.raises(ValueError, match="kappa_fission"):
        power_density(np.ones((2, 2, 1, 1)), [with_kappa, without], mmap,
                      total_power=1.0)


def test_non_fissile_material_without_kappa_contributes_zero():
    fuel = _fuel(kappa_fission=[0.003, 0.08])
    table, key = fission_energy_xs([REFL, fuel])
    assert key == "kappa_fission"
    np.testing.assert_array_equal(table[0], 0.0)
    np.testing.assert_allclose(table[1], fuel.kappa_fission)


def test_identical_fuels_split_power_equally():
    kappa = [0.003 * 3.2e-11, 0.08 * 3.2e-11]
    a, b = _fuel(name="a", kappa_fission=kappa), _fuel(name="b", kappa_fission=kappa)
    q = power_density(np.ones((2, 2, 1, 1)), [a, b],
                      np.array([0, 1]).reshape(2, 1, 1), total_power=1.0)
    np.testing.assert_allclose(q.ravel(), [0.5, 0.5], rtol=1e-14)


def _fine_problem(seed=0):
    rng = np.random.default_rng(seed)
    shape = (6, 4, 1)
    fuel = _fuel(kappa_fission=[0.004, 0.05])      # NOT proportional to nu*Sigma_f
    mats = [REFL, fuel]
    mmap = np.zeros(shape, dtype=int)
    mmap[1:5, 1:3] = 1
    region = np.zeros(shape, dtype=int)
    region[3:] = 1
    region[0] = 2                                   # all reflector: non-fissile
    region[1:3, 1:3] = 0
    flux = rng.uniform(0.5, 2.0, size=(2, *shape))
    vol = rng.uniform(0.5, 1.5, size=shape)         # per-cell volumes
    return flux, mats, mmap, region, vol


def test_flux_weighted_homogenize_preserves_the_kappa_reaction_rate():
    flux, mats, mmap, region, vol = _fine_problem()
    hmats, rflux, rvol = flux_weighted_homogenize(flux, mats, mmap, region,
                                                  cell_volume=vol)
    kappa = np.array([np.zeros(2), mats[1].kappa_fission])
    for i, m in enumerate(hmats):
        cells = region == i
        fine = (kappa[mmap[cells]].T * flux[:, cells] * vol[cells]).sum(axis=1)
        if not m.is_fissile:
            assert m.kappa_fission is None
            assert np.all(fine == 0.0)
            continue
        np.testing.assert_allclose(m.kappa_fission * rflux[i] * rvol[i], fine,
                                   rtol=1e-12)
    # and the homogenized model's power edit is still kappa-weighted
    assert fission_energy_xs(hmats)[1] == "kappa_fission"


def test_flux_weighted_homogenize_without_kappa_stays_without():
    flux, mats, mmap, region, vol = _fine_problem()
    plain = [REFL, _fuel()]
    hmats, _, _ = flux_weighted_homogenize(flux, plain, mmap, region, cell_volume=vol)
    assert all(m.kappa_fission is None for m in hmats)


def test_flux_weighted_homogenize_rejects_inconsistent_kappa():
    flux, mats, mmap, region, vol = _fine_problem()
    mats = mats + [_fuel(name="plainfuel")]
    mmap = mmap.copy()
    mmap[4, 1] = 2
    with pytest.raises(ValueError, match="plainfuel"):
        flux_weighted_homogenize(flux, mats, mmap, region, cell_volume=vol)


def test_mixed_cells_carry_kappa_at_their_volume_fraction():
    """A cell blended 30% fuel / 70% reflector is homogenized from fractional
    entries; kappa must follow the same split as nu*Sigma_f."""
    fuel = _fuel(kappa_fission=[0.004, 0.05])
    shape = (2, 1, 1)
    mmap = np.zeros(shape, dtype=int)               # all reflector base
    mix_m = np.ones(shape, dtype=int)
    mix_w = np.full(shape, 0.3)
    flux = np.ones((2, *shape))
    hmats, _, _ = flux_weighted_homogenize(
        flux, [REFL, fuel], mmap, np.zeros(shape, dtype=int), cell_volume=1.0,
        mix_material=mix_m, mix_weight=mix_w, mix_region_map=np.zeros(shape, dtype=int))
    np.testing.assert_allclose(hmats[0].kappa_fission, 0.3 * fuel.kappa_fission,
                               rtol=1e-14)
    np.testing.assert_allclose(hmats[0].nu_sigma_f, 0.3 * fuel.nu_sigma_f,
                               rtol=1e-14)
