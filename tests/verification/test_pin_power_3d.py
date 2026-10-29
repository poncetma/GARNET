"""Pin-power sampling of a 3-D (prismatic) TriReactor flux.

The power density of an extruded core is (nr, nc, 2, nz); flattening it
against the 2-D centroid indices used to read the wrong cells (zeros/NaN).
A 3-D flux must be sampled per layer, at one layer, or axially integrated, and
any other shape must be rejected.
"""

import numpy as np
import pytest

from garnet import HexLattice, PWR_TWO_GROUP
from garnet.pin_power import (reconstruct_pin_powers, sample_power_shape,
                              tri_cell_centroids)


@pytest.fixture(scope="module")
def core3d():
    core = HexLattice(pitch=10.0, refine=2).set_disk(1, PWR_TWO_GROUP).extrude(100.0, 4).build()
    flux = np.zeros((2,) + core.grid.shape)
    for k in range(core.grid.shape[-1]):
        flux[:, ..., k] = 10.0 ** k                    # layer-wise power 1:10:100:1000
    centroids = tri_cell_centroids(core.raster)
    cells = np.argwhere(core.active[..., 0])[:6]
    points = np.array([centroids[a, b, t] for a, b, t in cells])
    return core, flux, points


def _sample(core, flux, points, **kw):
    return sample_power_shape(core.raster, flux, core.materials, core.material_map,
                              points, active=core.active, **kw)


def test_3d_flux_is_sampled_per_layer(core3d):
    core, flux, points = core3d
    per_layer = _sample(core, flux, points)
    assert per_layer.shape == (len(points), 4)
    assert np.all(per_layer > 0)
    np.testing.assert_allclose(per_layer / per_layer[:, :1], [[1, 10, 100, 1000]] * 6)
    layer2 = _sample(core, flux, points, layer=2)
    np.testing.assert_allclose(layer2, per_layer[:, 2])
    np.testing.assert_allclose(_sample(core, flux, points, layer="integrate"),
                               per_layer.sum(axis=1))
    # Layer 0 matches the same points sampled from the equivalent 2-D problem.
    two_d = sample_power_shape(core.raster, flux[..., 0], core.materials,
                               core.material_map[..., 0], points,
                               active=core.active[..., 0])
    np.testing.assert_allclose(per_layer[:, 0], two_d)


def test_3d_pin_reconstruction_keeps_the_layer_axis(core3d):
    core, flux, points = core3d
    pins = np.array([[0.0, 0.0], [0.5, 0.0]])
    form = np.array([1.2, 0.8])
    power, xy = reconstruct_pin_powers(core.raster, flux, core.materials,
                                       core.material_map, points[:2], pins, form,
                                       pin_kind=["fuel", "moderator"],
                                       active=core.active)
    assert power.shape == (2, 2, 4) and xy.shape == (2, 2, 2)
    assert np.all(power[:, 1] == 0.0) and np.all(power[:, 0] > 0.0)
    total, _ = reconstruct_pin_powers(core.raster, flux, core.materials,
                                      core.material_map, points[:2], pins, form,
                                      active=core.active, layer="integrate")
    np.testing.assert_allclose(total[:, 0], power[:, 0].sum(axis=1))


def test_bad_shapes_and_layers_are_rejected(core3d):
    core, flux, points = core3d
    with pytest.raises(ValueError, match="layer"):
        _sample(core, flux, points, layer=4)
    with pytest.raises(ValueError, match="layer applies only"):
        sample_power_shape(core.raster, flux[..., 0], core.materials,
                           core.material_map[..., 0], points,
                           active=core.active[..., 0], layer=0)
    with pytest.raises(ValueError, match="matches neither"):
        sample_power_shape(core.raster, flux[:, :-1, ..., 0], core.materials,
                           core.material_map[:-1, ..., 0], points)
