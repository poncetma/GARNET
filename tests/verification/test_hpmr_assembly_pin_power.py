"""Pin-resolved HP-MR assembly geometry and pin-power reconstruction.

The heterogeneous assembly (:mod:`garnet.benchmarks.hpmr_assembly`) is decoded
from the VTB Serpent deck, so the geometry has an independent check available:
the pin counts and radii must reproduce the assembly volume fractions recorded
separately in :mod:`garnet.benchmarks.hpmr`. Two sources agreeing is the test --
neither is fitted to the other.

The reconstruction side (:mod:`garnet.pin_power`) is pinned on the two things
that silently give wrong answers rather than errors: sampling the core flux in
the wrong coordinate frame, and pin positions that miss their own assembly.
"""

import numpy as np
import pytest

from garnet.benchmarks.hpmr import FUEL as CORE_FUEL
from garnet.benchmarks.hpmr import (PITCH, _FUEL_SITES, build_hpmr2d,
                                   hpmr_materials_builtin)
from garnet.benchmarks.hpmr_assembly import (FUEL, HEATPIPE, HP_SHELL, MOD_SHELL,
                                            MODERATOR, N_FUEL, N_HP, N_MOD,
                                            PIN_MATERIAL_NAMES, PIN_PITCH,
                                            R_FUEL, R_HP, R_HP_SHELL, R_MOD,
                                            R_MOD_SHELL, build_hpmr_assembly2d,
                                            pin_fluxes, pin_layer_fractions,
                                            pin_materials_builtin, pin_powers,
                                            vtb_pin_lattice)
from garnet.hexraster import hex_site_xy
from garnet.pin_power import reconstruct_pin_powers, tri_cell_centroids

# Volume fractions recorded in hpmr.py, decoded independently of the pin map.
DOC_FRACTIONS = {"fuel_compact": 0.3193, "moderator": 0.0931,
                 "mod_shell": 0.0227, "heatpipe": 0.1765,
                 "hp_shell": 0.0383, "graphite": 0.3501}
ASSEMBLY_AREA = np.sqrt(3.0) / 2.0 * PITCH ** 2


def test_pin_counts_match_the_deck():
    _, kinds = vtb_pin_lattice()
    kinds = np.asarray(kinds)
    assert len(kinds) == N_FUEL + N_MOD + N_HP == 127
    assert (kinds == "fuel").sum() == N_FUEL
    assert (kinds == "mod").sum() == N_MOD
    assert (kinds == "hp").sum() == N_HP


def test_counts_and_radii_reproduce_the_documented_volume_fractions():
    """Six documented numbers from five geometric parameters -- not a fit."""
    pi = np.pi
    got = {
        "fuel_compact": N_FUEL * pi * R_FUEL ** 2,
        "moderator": N_MOD * pi * R_MOD ** 2,
        "mod_shell": N_MOD * pi * (R_MOD_SHELL ** 2 - R_MOD ** 2),
        "heatpipe": N_HP * pi * R_HP ** 2,
        "hp_shell": N_HP * pi * (R_HP_SHELL ** 2 - R_HP ** 2),
    }
    total = 0.0
    for name, area in got.items():
        frac = area / ASSEMBLY_AREA
        total += frac
        assert frac == pytest.approx(DOC_FRACTIONS[name], abs=5e-5)
    assert 1.0 - total == pytest.approx(DOC_FRACTIONS["graphite"], abs=5e-5)


def test_pins_fit_the_assembly_and_do_not_overlap():
    """The check that caught a wrong reconstructed pitch.

    Volume fractions depend only on counts and radii, so they cannot detect a
    wrong lattice spacing; packing can. An earlier reconstruction put the pins
    at PITCH/13 = 2.0578 cm, which overlaps a heat pipe (1.07) against a fuel
    compact (1.00) needing 2.070 cm between centres.
    """
    xy, kinds = vtb_pin_lattice()
    rad = np.array([{"fuel": R_FUEL, "mod": R_MOD_SHELL,
                     "hp": R_HP_SHELL}[k] for k in kinds])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() == pytest.approx(PIN_PITCH, abs=1e-9)
    assert d.min() >= R_HP_SHELL + R_FUEL

    # Every pin inside the assembly hexagon (apothem PITCH/2). vtb_pin_lattice
    # applies the deck's 30 deg rotation, so the cluster sits in the FLAT-X
    # hexagon -- face normals at 0, 60, 120 deg. Using the other orientation
    # puts the corner-direction extent (13.80 cm) against the apothem and the
    # check fails spuriously.
    apo = PITCH / 2.0
    ang = np.radians(60.0 * np.arange(6))
    n = np.column_stack([np.cos(ang), np.sin(ang)])
    assert (xy @ n.T).max(axis=1).max() + rad.max() < apo

    # and no overlap with the pins of a NEIGHBOURING assembly (the check that
    # catches a missing lattice rotation)
    A1 = np.array([PITCH, 0.0])
    A2 = PITCH * np.array([0.5, np.sqrt(3.0) / 2.0])
    worst = min(np.linalg.norm(xy[:, None, :] - (xy + di * A1 + dj * A2)[None],
                               axis=2).min()
                for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0))
    assert worst >= 2.0 * R_HP_SHELL


@pytest.mark.parametrize("refine, tol", [(4, 8e-3), (8, 4e-3), (16, 1e-4)])
def test_meshed_volume_fractions_converge(refine, tol):
    """Volume mixing must converge; centroid assignment does not.

    Without mixing the heat-pipe fraction sat at a refinement-INDEPENDENT
    -0.0123, a structural bias rather than a resolution one. What remains is
    the two-constituent truncation of cells that hold three (max 0.0074 at
    refine 4, 0.0030 at 8, 5e-5 at 16).
    """
    p = build_hpmr_assembly2d(refine=refine)
    for name, ref in DOC_FRACTIONS.items():
        assert p.volume_fractions[name] == pytest.approx(ref, abs=tol)


def _analytic_fractions():
    pi = np.pi
    area = {"fuel_compact": N_FUEL * pi * R_FUEL ** 2,
            "moderator": N_MOD * pi * R_MOD ** 2,
            "mod_shell": N_MOD * pi * (R_MOD_SHELL ** 2 - R_MOD ** 2),
            "heatpipe": N_HP * pi * R_HP ** 2,
            "hp_shell": N_HP * pi * (R_HP_SHELL ** 2 - R_HP ** 2)}
    frac = {k: v / ASSEMBLY_AREA for k, v in area.items()}
    frac["graphite"] = 1.0 - sum(frac.values())
    return frac


@pytest.mark.parametrize("refine", [2, 4])
def test_cell_fractions_are_exact_at_any_refinement(refine):
    """Per-cell constituent areas are closed-form, so they sum to the analytic
    pin areas on any mesh. The former barycentric sampler used only the upright
    sub-triangle centroids: at refine 4 it put mod_shell at 0.0183 against
    0.0227 and hp_shell at 0.0278 against 0.0383."""
    p = build_hpmr_assembly2d(refine=refine)
    cf = p.cell_fractions.reshape(-1, len(PIN_MATERIAL_NAMES))
    np.testing.assert_allclose(cf.sum(axis=1), 1.0, atol=1e-12)
    assert cf.min() >= 0.0
    exact = _analytic_fractions()
    for i, name in enumerate(PIN_MATERIAL_NAMES):
        assert cf[:, i].mean() == pytest.approx(exact[name], rel=1e-12, abs=1e-14)
    # Every pin owns cells, and only covered cells are owned.
    owned = p.pin_index.reshape(-1) >= 0
    assert len(np.unique(p.pin_index.reshape(-1)[owned])) == N_FUEL + N_MOD + N_HP
    np.testing.assert_array_equal(owned, cf[:, 1:].sum(axis=1) > 0.0)


def test_pin_layer_fractions_closed_form():
    s = 0.2
    tri = np.array([[0.0, 0.0], [s, 0.0], [0.5 * s, 0.5 * np.sqrt(3.0) * s]])
    # wholly inside the fuel compact, wholly outside it
    assert pin_layer_fractions(tri, np.zeros(2), "fuel") == [(FUEL, pytest.approx(1.0))]
    assert pin_layer_fractions(tri, np.array([5.0, 0.0]), "fuel") == [(FUEL, 0.0)]
    # a triangle that contains a whole moderator pin sees disk and annulus areas
    big = 10.0 * np.array([[-1.0, -1.0], [2.0, -1.0], [-1.0, 2.0]])
    big_area = 0.5 * 30.0 * 30.0
    (m0, f0), (m1, f1) = pin_layer_fractions(big, np.zeros(2), "mod")
    assert (m0, m1) == (MODERATOR, MOD_SHELL)
    assert f0 * big_area == pytest.approx(np.pi * R_MOD ** 2, rel=1e-13)
    assert f1 * big_area == pytest.approx(np.pi * (R_MOD_SHELL ** 2 - R_MOD ** 2), rel=1e-12)
    # a triangle straddling the heat-pipe wall holds pipe, wall and graphite
    parts = dict(pin_layer_fractions(tri + np.array([-0.5 * s, R_HP - 0.05]),
                                     np.zeros(2), "hp"))
    assert 0.0 < parts[HEATPIPE] < 1.0 and 0.0 < parts[HP_SHELL] < 1.0
    assert sum(parts.values()) < 1.0


def test_builtin_materials_are_the_real_library():
    mats = pin_materials_builtin()
    assert len(mats) == 6
    assert mats[0].n_groups == 11
    fuel = mats[1]
    assert np.any(np.asarray(fuel.nu_sigma_f) > 0)
    assert fuel.kappa_fission is not None          # power needs kappa*Sigma_f
    # only the fuel compact is fissile / heat-producing
    for other in [mats[0]] + list(mats[2:]):
        assert not np.any(np.asarray(other.nu_sigma_f) > 0)
        assert other.kappa_fission is None


def test_form_functions_are_normalized_and_power_is_fuel_only():
    p = build_hpmr_assembly2d(refine=3)
    rng = np.random.default_rng(0)
    G = p.materials[0].n_groups
    flux = rng.uniform(0.5, 1.5, (G,) + tuple(p.grid.shape))
    power, form = pin_powers(p, flux)
    phi, fform = pin_fluxes(p, flux)
    fuel = np.array([k == "fuel" for k in p.pin_kind])

    assert form[fuel].mean() == pytest.approx(1.0)
    assert np.allclose(fform[fuel].mean(axis=0), 1.0)
    # no power outside the fuel, whatever the flux
    assert np.all(power[~fuel] == 0.0)
    assert np.all(power[fuel] > 0.0)


def test_cell_centroids_match_the_rasters_own_frame():
    """A generic tri-lattice convention is rotated relative to the raster's;
    using it would silently sample the core flux from the wrong cells."""
    p = build_hpmr2d(refine=4, drum_angle_deg=180.0, absorber="polar")
    cen = tri_cell_centroids(p.raster)
    for (a, b, t) in [(3, 5, 0), (10, 12, 1), (20, 7, 0), (30, 30, 1)]:
        assert cen[a, b, t] == pytest.approx(p.raster.cell_centroid(a, b, t))


def test_every_pin_lands_in_its_own_fuel_assembly():
    p = build_hpmr2d(refine=6, drum_angle_deg=180.0, absorber="polar")
    cen = tri_cell_centroids(p.raster).reshape(-1, 2)
    mm = np.asarray(p.material_map).reshape(-1)
    sites = np.array([hex_site_xy(R, C, PITCH) for R, C in _FUEL_SITES])
    centres, _ = vtb_pin_lattice()
    pts = (sites[:, None, :] + centres[None, :, :]).reshape(-1, 2)
    d = np.linalg.norm(pts[:, None, :] - cen[None, :, :], axis=2)
    assert np.all(mm[d.argmin(axis=1)] == CORE_FUEL)


def test_reconstruction_is_shape_times_form():
    """P = S x f, with S sampled per pin -- not one scalar per assembly."""
    from garnet import TriDiffusionEigenSolver
    from garnet.benchmarks.hpmr_assembly import vtb_pin_lattice

    asm = build_hpmr_assembly2d(refine=3)
    fuel = np.array([k == "fuel" for k in asm.pin_kind])
    core = build_hpmr2d(refine=4, drum_angle_deg=180.0, absorber="polar",
                        materials=hpmr_materials_builtin(asm.materials[1]))
    r = TriDiffusionEigenSolver(core.grid, core.materials, core.material_map,
                                active=core.active, mask_bc=core.mask_bc,
                                mix_material=core.mix_material,
                                mix_weight=core.mix_weight).solve(tol_k=1e-8)
    sites = np.array([hex_site_xy(R, C, PITCH) for R, C in _FUEL_SITES])
    centres, kinds = vtb_pin_lattice()

    flat = np.ones(len(centres))
    p_flat, _ = reconstruct_pin_powers(
        core.raster, r.flux_numpy, core.materials, core.material_map, sites,
        centres, flat, pin_kind=kinds, mix_material=core.mix_material,
        mix_weight=core.mix_weight, active=core.active)
    form = np.linspace(0.9, 1.1, len(centres))
    p_form, _ = reconstruct_pin_powers(
        core.raster, r.flux_numpy, core.materials, core.material_map, sites,
        centres, form, pin_kind=kinds, mix_material=core.mix_material,
        mix_weight=core.mix_weight, active=core.active)

    # the form function enters as a pure per-pin multiplier
    assert np.allclose(p_form[:, fuel], p_flat[:, fuel] * form[fuel])
    assert np.all(p_flat[:, ~fuel] == 0.0)
    # and the sampled shape actually varies across a single assembly
    tilt = np.ptp(p_flat[:, fuel], axis=1) / p_flat[:, fuel].mean(axis=1)
    assert tilt.max() > 0.01
