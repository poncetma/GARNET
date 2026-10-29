"""Model / HexLattice front-end regressions from the API audit.

* ``Model.from_grid`` on a 2-D grid must treat the single-cell z axis as
  collapsed (reflective), like a model built directly in 2-D.
* ``add_box`` keeps its cell-centre selection but warns when that paints a
  visibly different interval than requested.
* An axial region painted over a control-drum site replaces the whole cell, so
  the drum's absorber blend must not survive under it.
"""

import warnings

import numpy as np
import pytest

from garnet import HexLattice, Material, Model, PWR_TWO_GROUP

_REFLECTOR = Material(name="refl", diffusion=[1.3, 0.3], sigma_a=[0.001, 0.02],
                      nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.03], [0.0, 0.0]])
_ABSORBER = Material(name="b4c", diffusion=[1.0, 0.3], sigma_a=[0.05, 0.5],
                     nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
TOL = dict(tol_k=1e-9, tol_source=1e-8)


def _two_region_2d():
    return (Model(size=(100.0, 80.0), cells=(20, 16)).fill(PWR_TWO_GROUP)
            .add_box(_REFLECTOR, x=(0.0, 20.0)).set_boundary("vacuum"))


def test_from_grid_keeps_a_2d_grid_two_dimensional():
    direct = _two_region_2d()
    k_direct = direct.run(device="cpu", **TOL).k_eff
    wrapped = Model.from_grid(direct.grid, direct.materials,
                              direct.material_map.reshape(direct.grid.shape))
    assert wrapped.ndim == 2
    assert wrapped._full_bc()[2] == "reflective"
    # Before the fix the collapsed 1 cm z axis got vacuum faces: k = 0.0175.
    assert wrapped.run(device="cpu", **TOL).k_eff == pytest.approx(k_direct, abs=1e-9)


def test_from_grid_honours_an_explicit_face_on_a_single_cell_axis():
    direct = _two_region_2d()
    mmap = direct.material_map.reshape(direct.grid.shape)
    slab = Model.from_grid(direct.grid, direct.materials, mmap,
                           bc=("vacuum", "vacuum", "vacuum"))
    assert slab.ndim == 3                       # deliberately a one-cell slab
    assert slab.run(device="cpu", **TOL).k_eff < 0.1
    one_d = Model.from_grid(Model(size=(50.0,), cells=(10,)).grid, PWR_TWO_GROUP)
    assert one_d.ndim == 1 and one_d.size == (50.0,)


def test_add_box_warns_when_cell_centres_paint_a_different_width():
    with pytest.warns(UserWarning, match=r"x range \(30, 90\).*51\.4286 cm instead of 60 cm"):
        Model(size=(120.0,), cells=(7,)).fill(_REFLECTOR).add_box(
            PWR_TWO_GROUP, x=(30.0, 90.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # face-aligned boxes stay silent
        Model(size=(120.0,), cells=(12,)).fill(_REFLECTOR).add_box(
            PWR_TWO_GROUP, x=(30.0, 90.0))
        Model(size=(120.0,), cells=(120,)).fill(_REFLECTOR).add_box(
            PWR_TWO_GROUP, x=(30.2, 89.9))      # sub-half-cell mismatch


def test_axial_region_over_a_drum_clears_the_absorber_blend():
    lattice = (HexLattice(pitch=10.0, refine=2).set_disk(1, PWR_TWO_GROUP)
               .set_drum((2, -1), _REFLECTOR, _ABSORBER, 3.0, 4.5, 120.0, 0.0)
               .extrude(100.0, 10).add_axial_region(_REFLECTOR, z=(0.0, 20.0)))
    core = lattice.build()
    weight, mix = core.mix_weight, core.mix_material
    assert np.all(weight[..., :2] == 0.0) and np.all(mix[..., :2] == -1)
    upper = weight[..., 2:]
    assert np.count_nonzero(upper) > 0 and np.all(upper == upper[..., :1])
