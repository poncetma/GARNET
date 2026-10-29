"""Temperature feedback in a volume-mixed cell: per material, then blended.

A mixed cell is two materials sharing a volume, so its feedback is each
material's own change -- under its own coefficient, its own reference
temperature and its own cross sections -- blended by the cross sections' rules.
Blending the coefficients first and evaluating once put the fuel's Doppler on
its partner's absorption (50.5x the right answer in a 50/50 fuel/absorber
cell) and averaged reference temperatures (a cell whose fuel sat at ITS
reference still moved). Every expected value here is computed by hand from the
material tables.
"""

import numpy as np
import pytest

from garnet import DiffusionEigenSolver, Grid, Material
from garnet.feedback import TabulatedFeedback, ThermalFeedback
from garnet.solver import Fields

FUEL = Material(name="fuel", diffusion=[1.2, 0.4], sigma_a=[0.01, 0.10],
                nu_sigma_f=[0.006, 0.15], sigma_s=[[0.0, 0.02], [0.0, 0.0]],
                chi=[1.0, 0.0])
ABSORBER = Material(name="b4c", diffusion=[0.8, 0.3], sigma_a=[0.5, 2.0],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
MATS = [FUEL, ABSORBER]
GRID = Grid(shape=(3, 1, 1), size=(3.0, 1.0, 1.0))
# cell 0 pure fuel | cell 1 pure absorber | cell 2 half fuel, half absorber
MMAP = np.array([0, 1, 0]).reshape(GRID.shape)
MIX = np.array([-1, -1, 1]).reshape(GRID.shape)
W = np.array([0.0, 0.0, 0.5]).reshape(GRID.shape)


def _fields(feedback=None, temperature=None, materials=MATS, mix=MIX, weight=W,
            mmap=MMAP):
    hook = (None if feedback is None
            else feedback.hook(np.full(GRID.shape, float(temperature)),
                               materials=materials))
    return Fields(np, GRID, MATS, mmap, np.float64, mix_material=mix,
                  mix_weight=weight, xs_update=hook)


def _cell(field, i=2):
    return float(np.asarray(field).ravel()[i])


def test_doppler_lands_on_the_fuel_share_only():
    # c_D = 0.01 on the fuel alone; sqrt(1600) - sqrt(900) = 10.
    fb = ThermalFeedback(t_ref=[900.0, 900.0], doppler=[0.01, 0.0])
    cold, hot = _fields(), _fields(fb, 1600.0)
    for g in range(2):
        d_sa = 0.5 * 0.01 * FUEL.sigma_a[g] * 10.0          # = 0.05 Sa_fuel
        assert _cell(hot.sigma_a[g]) - _cell(cold.sigma_a[g]) == pytest.approx(
            d_sa, rel=1e-12)
        assert _cell(hot.removal[g]) - _cell(cold.removal[g]) == pytest.approx(
            d_sa, rel=1e-12)
        assert _cell(hot.sigma_t[g]) - _cell(cold.sigma_t[g]) == pytest.approx(
            d_sa, rel=1e-12)
        # Harmonic blend: 1/D = sum_i w_i (1/D_i + 3 dSa_i).
        inv_d = 0.5 / FUEL.diffusion[g] + 0.5 / ABSORBER.diffusion[g] + 3.0 * d_sa
        assert _cell(hot.diffusion[g]) == pytest.approx(1.0 / inv_d, rel=1e-12)
        # Pure cells keep the plain law.
        assert _cell(hot.sigma_a[g], 0) - FUEL.sigma_a[g] == pytest.approx(
            0.1 * FUEL.sigma_a[g], rel=1e-12)
        assert _cell(hot.sigma_a[g], 1) == ABSORBER.sigma_a[g]


def test_each_material_is_measured_from_its_own_reference_temperature():
    # Fuel sits exactly at its 900 K reference; the absorber (400 K reference,
    # c_D = 0.02) is 900 K hot: sqrt(900) - sqrt(400) = 10. Blending t_ref to
    # 650 K would have moved both.
    fb = ThermalFeedback(t_ref=[900.0, 400.0], doppler=[0.01, 0.02])
    cold, hot = _fields(), _fields(fb, 900.0)
    for g in range(2):
        expected = 0.5 * 0.02 * ABSORBER.sigma_a[g] * 10.0
        assert _cell(hot.sigma_a[g]) - _cell(cold.sigma_a[g]) == pytest.approx(
            expected, rel=1e-12)
        assert _cell(hot.sigma_a[g], 0) == pytest.approx(FUEL.sigma_a[g], rel=1e-15)


def test_a_material_at_its_reference_contributes_nothing_whatever_its_partner():
    # The model-level mapping gives unlisted materials t_ref = ambient and no
    # coefficient. A 900/300 K pair at T = 900 K must be an exact no-op (it
    # was dSa = 0.0139 when the reference temperatures were averaged). Both
    # laws vanish, so no material list is even needed.
    fb = ThermalFeedback(t_ref=[900.0, 300.0], doppler=[0.01, 0.0])
    cold, hot = _fields(), _fields(fb, 900.0, materials=None)
    for g in range(2):
        np.testing.assert_array_equal(hot.sigma_a[g], cold.sigma_a[g])


def test_expansion_scales_each_material_by_its_own_density_factor():
    # beta = 1e-3 on the fuel only, 100 K hot: f_fuel = 0.9, f_absorber = 1.
    fb = ThermalFeedback(t_ref=[800.0, 800.0], doppler=[0.0, 0.0],
                         expansion=[1e-3, 0.0])
    hot = _fields(fb, 900.0)
    for g in range(2):
        for attr in ("sigma_a", "removal", "sigma_t", "nu_sigma_f"):
            expected = (0.5 * 0.9 * getattr(FUEL, attr)[g]
                        + 0.5 * getattr(ABSORBER, attr)[g])
            assert _cell(getattr(hot, attr)[g]) == pytest.approx(expected, rel=1e-12), attr
        inv_d = 0.5 * 0.9 / FUEL.diffusion[g] + 0.5 / ABSORBER.diffusion[g]
        assert _cell(hot.diffusion[g]) == pytest.approx(1.0 / inv_d, rel=1e-12)
    expected_ss = 0.5 * 0.9 * 0.02 + 0.5 * 0.01
    assert _cell(hot.sigma_s[0][1]) == pytest.approx(expected_ss, rel=1e-12)


def test_doppler_and_expansion_compose_per_material():
    # Fuel: c_D = 0.01, beta = 1e-4, both references 900 K, T = 1600 K:
    # k = 0.1 and f = 1 - 0.07 = 0.93. Absorber inert.
    fb = ThermalFeedback(t_ref=[900.0, 900.0], doppler=[0.01, 0.0],
                         expansion=[1e-4, 0.0])
    hot = _fields(fb, 1600.0)
    k, f = 0.1, 0.93
    for g in range(2):
        sa_f, sa_a = FUEL.sigma_a[g], ABSORBER.sigma_a[g]
        assert _cell(hot.sigma_a[g]) == pytest.approx(
            0.5 * f * sa_f * (1 + k) + 0.5 * sa_a, rel=1e-12)
        assert _cell(hot.removal[g]) == pytest.approx(
            0.5 * f * (FUEL.removal[g] + k * sa_f) + 0.5 * ABSORBER.removal[g],
            rel=1e-12)
        inv_d = (0.5 * f * (1.0 / FUEL.diffusion[g] + 3.0 * k * sa_f)
                 + 0.5 / ABSORBER.diffusion[g])
        assert _cell(hot.diffusion[g]) == pytest.approx(1.0 / inv_d, rel=1e-12)
        assert _cell(hot.nu_sigma_f[g]) == pytest.approx(
            0.5 * f * FUEL.nu_sigma_f[g], rel=1e-12)


def test_a_fully_converted_cell_takes_its_partners_law():
    # Base absorber, partner fuel at w = 1: the cell IS fuel.
    fb = ThermalFeedback(t_ref=[900.0, 900.0], doppler=[0.01, 0.0])
    mmap = np.array([0, 1, 1]).reshape(GRID.shape)
    mix = np.array([-1, -1, 0]).reshape(GRID.shape)
    weight = np.array([0.0, 0.0, 1.0]).reshape(GRID.shape)
    hot = _fields(fb, 1600.0, materials=None, mix=mix, weight=weight, mmap=mmap)
    for g in range(2):
        assert _cell(hot.sigma_a[g]) == pytest.approx(1.1 * FUEL.sigma_a[g], rel=1e-12)


def test_differing_laws_without_material_tables_are_refused():
    """A fields object that does not carry its materials (solver.Fields always
    does) cannot evaluate a mixed cell's per-material law without them."""
    class BareFields(Fields):
        materials = property(lambda self: None, lambda self, value: None)

    fb = ThermalFeedback(t_ref=[900.0, 900.0], doppler=[0.01, 0.0])
    with pytest.raises(ValueError, match="materials="):
        BareFields(np, GRID, MATS, MMAP, np.float64, mix_material=MIX,
                   mix_weight=W, xs_update=fb.hook(np.full(GRID.shape, 1600.0)))


def test_solver_fields_carry_their_materials_so_no_list_is_needed():
    fb = ThermalFeedback(t_ref=[900.0, 900.0], doppler=[0.01, 0.0])
    hot = Fields(np, GRID, MATS, MMAP, np.float64, mix_material=MIX,
                 mix_weight=W, xs_update=fb.hook(np.full(GRID.shape, 1600.0)))
    ref = _fields(fb, 1600.0)
    for g in range(2):
        np.testing.assert_allclose(hot.sigma_a[g], ref.sigma_a[g], rtol=1e-15)


def test_mixed_hook_equals_mixing_the_heated_materials():
    """Integral check: under a uniform T the mixed cell must be exactly the
    blend of the two materials each heated under its own law, so the
    eigenvalue with the hook equals the one with pre-heated materials."""
    grid = Grid(shape=(12, 12, 1), size=(60.0, 60.0, 1.0))
    mmap = np.zeros(grid.shape, dtype=int)
    mix = np.full(grid.shape, -1)
    weight = np.zeros(grid.shape)
    mix[3:9, 3:9] = 1
    weight[3:9, 3:9] = 0.02                      # a trace of absorber in the fuel
    fb = ThermalFeedback(t_ref=[900.0, 600.0], doppler=[0.01, 0.003],
                         expansion=[2e-5, 1e-5])
    T = 1225.0

    def heated(m, c, tr, beta):
        k = c * (np.sqrt(T) - np.sqrt(tr))
        f = 1.0 - beta * (T - tr)
        d_sa = k * m.sigma_a
        return m.replace(sigma_a=f * (m.sigma_a + d_sa), nu_sigma_f=f * m.nu_sigma_f,
                         sigma_s=f * m.sigma_s,
                         diffusion=1.0 / (f * (1.0 / m.diffusion + 3.0 * d_sa)))

    pre = [heated(FUEL, 0.01, 900.0, 2e-5), heated(ABSORBER, 0.003, 600.0, 1e-5)]
    common = dict(bc="vacuum", mix_material=mix, mix_weight=weight, device="cpu")
    k_hook = DiffusionEigenSolver(
        grid, MATS, mmap, xs_update=fb.hook(np.full(grid.shape, T), materials=MATS),
        **common).solve(tol_k=1e-12, tol_source=1e-11).k_eff
    k_pre = DiffusionEigenSolver(grid, pre, mmap, **common).solve(
        tol_k=1e-12, tol_source=1e-11).k_eff
    assert k_hook == pytest.approx(k_pre, rel=1e-10)


# -- TabulatedFeedback ---------------------------------------------------------

def _tabulated(reference=1):
    temps = np.array([600.0, 800.0, 1200.0])
    tables = {"sigma_a": np.array([FUEL.sigma_a * s for s in (0.98, 1.0, 1.03)])}
    return TabulatedFeedback(material=0, temperatures=temps, tables=tables,
                             reference=reference)


def test_tabulated_reference_temperature_is_the_reference_node():
    # Three nodes: the old "middle node" was 800 K whatever `reference` said.
    np.testing.assert_array_equal(_tabulated(reference=0).t_ref, [600.0])
    np.testing.assert_array_equal(_tabulated(reference=2).t_ref, [1200.0])


def test_tabulated_material_as_the_blend_partner_is_refused():
    # Base absorber, fuel as the PARTNER at 40%: the old check looked at the
    # base material only, so this cell silently got no feedback.
    mmap = np.array([0, 1, 1]).reshape(GRID.shape)
    mix = np.array([-1, -1, 0]).reshape(GRID.shape)
    weight = np.array([0.0, 0.0, 0.4]).reshape(GRID.shape)
    with pytest.raises(ValueError, match="volume-mixed"):
        Fields(np, GRID, MATS, mmap, np.float64, mix_material=mix,
               mix_weight=weight,
               xs_update=_tabulated().hook(np.full(GRID.shape, 1000.0)))
    # Unmixed, it applies.
    f = Fields(np, GRID, MATS, MMAP, np.float64,
               xs_update=_tabulated().hook(np.full(GRID.shape, 1200.0)))
    assert _cell(f.sigma_a[0], 0) == pytest.approx(1.03 * FUEL.sigma_a[0], rel=1e-12)
