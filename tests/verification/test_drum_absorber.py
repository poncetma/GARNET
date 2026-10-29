"""Control-drum absorber area fractions are exact.

The absorber arc is an annular sector volume-mixed into the drum-body
triangles. Its per-cell fraction used to be estimated on a barycentric lattice
whose vertex and edge points were weighted like interior ones, biasing the
absorber area by -14..-17% at refine 2 (and -7% on HP-MR at refine 4). The
closed-form triangle / annular-sector intersection must reproduce the analytic
sector area at every refinement and rotation, for both the HexLattice front
end and the HP-MR benchmark builder.
"""

import math

import numpy as np
import pytest

from garnet.benchmarks.hpmr import (DRUM_ABSORBER_INNER, DRUM_ARC_HALF_DEG,
                                    DRUM_RADIUS, absorber_fraction_map,
                                    hpmr_raster)
from garnet.hexraster import hex_site_xy, rasterize_hex_sites
from garnet.model import _drum_absorber_mix, annular_sector_fraction, hex_disk


def _sector_area(inner, outer, arc_deg):
    return 0.5 * math.radians(arc_deg) * (outer ** 2 - inner ** 2)


@pytest.mark.parametrize("refine", [2, 4])
@pytest.mark.parametrize("inner, outer, arc_deg, angle_deg",
                         [(7.0, 8.5, 120.0, 0.0), (8.0, 9.0, 60.0, 37.0),
                          (0.0, 6.0, 300.0, 101.0)])
def test_lattice_drum_absorber_area_matches_the_sector(refine, inner, outer,
                                                       arc_deg, angle_deg):
    pitch = 20.0
    raster = rasterize_hex_sites({rc: 1 for rc in hex_disk(2)}, pitch, refine)
    x, y = hex_site_xy(2, 0, pitch)
    spec = [dict(rc=(2, 0), inner=inner, outer=outer,
                 arc_half=math.radians(arc_deg) / 2.0,
                 arc_az=math.atan2(y, x) + math.radians(180.0 + angle_deg),
                 absorber_id=9)]
    mix, weight = _drum_absorber_mix(raster, pitch, spec)
    cell_area = math.sqrt(3.0) / 4.0 * raster.side ** 2
    assert weight.sum() * cell_area == pytest.approx(
        _sector_area(inner, outer, arc_deg), rel=1e-12)
    assert weight.max() <= 1.0 and np.array_equal(weight > 0, mix == 9)


@pytest.mark.parametrize("angle", [0.0, 45.0, 180.0])
def test_hpmr_absorber_area_matches_the_sectors(angle):
    raster = hpmr_raster(4, np.full(12, angle), paint_absorber=False)
    frac = absorber_fraction_map(raster, np.full(12, angle))
    cell_area = math.sqrt(3.0) / 4.0 * raster.side ** 2
    exact = 12 * _sector_area(DRUM_ABSORBER_INNER, DRUM_RADIUS, 2 * DRUM_ARC_HALF_DEG)
    assert frac.sum() * cell_area == pytest.approx(exact, rel=1e-12)


def test_sector_fraction_against_fine_midpoint_quadrature():
    # Random triangles, convex and reflex arcs, annuli touching the centre.
    rng = np.random.default_rng(7)
    n = 400
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    up = (i + j) < n
    down = (i + j) < n - 1
    a = np.concatenate([(i[up] + 1 / 3) / n, (i[down] + 2 / 3) / n])
    b = np.concatenate([(j[up] + 1 / 3) / n, (j[down] + 2 / 3) / n])
    for _ in range(40):
        V = rng.uniform(-3.0, 3.0, (3, 2))
        inner = rng.uniform(0.0, 1.5)
        outer = inner + rng.uniform(0.1, 2.0)
        az, half = rng.uniform(-4.0, 4.0), rng.uniform(0.05, math.pi)
        P = V[0] + a[:, None] * (V[1] - V[0]) + b[:, None] * (V[2] - V[0])
        r = np.hypot(P[:, 0], P[:, 1])
        dphi = (np.arctan2(P[:, 1], P[:, 0]) - az + math.pi) % (2 * math.pi) - math.pi
        quad = ((r > inner) & (r <= outer) & (np.abs(dphi) <= half)).mean()
        exact = annular_sector_fraction(V, (0.0, 0.0), inner, outer, az, half)
        assert exact == pytest.approx(quad, abs=5e-4)
