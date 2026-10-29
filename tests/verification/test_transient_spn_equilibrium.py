"""Unperturbed SPN-family transients must hold the steady state exactly.

Starting from the converged eigenstate with equilibrium precursors and no
perturbation, backward Euler reproduces the initial state for any step width:
the eliminated new-time precursor emission plus the old-inventory delayed
source rebuild exactly the steady fission spectrum, and the time source
cancels the operator's time term. Two defects broke that only for the SPN
family (diffusion was right):

* the implicit new-time delayed emission was given the total spectrum chi
  instead of chi_delayed, so a distinct delayed spectrum manufactured
  reactivity (P(0.5 s) ~ 6 instead of 1) -- shared and per-family delayed
  spectra are both checked;
* on r-z grids the operator (time term included) is metric-weighted but the
  backward-Euler time source was not, so the power decayed every step --
  catastrophically at small dt, where theta = 1/(v dt) dominates.

Tiny heterogeneous two-group fuel/reflector problem; the diffusion
TransientSolver is the control.
"""

import numpy as np
import pytest

from garnet import Grid, Kinetics, Material
from garnet.transient import (TransientSDP1Solver, TransientSDPNSolver,
                              TransientSolver, TransientSPNSolver)

FUEL = Material(name="fuel", diffusion=[1.4, 0.4], sigma_a=[0.010, 0.08],
                nu_sigma_f=[0.007, 0.135], chi=[0.97, 0.03],
                sigma_s=[[0.0, 0.018], [0.0005, 0.0]])
REFL = Material(name="refl", diffusion=[1.2, 0.2], sigma_a=[0.001, 0.02],
                nu_sigma_f=[0.0, 0.0],
                sigma_s=[[0.0, 0.03], [0.0, 0.0]])
MATS = [FUEL, REFL]

N = 6
_FUEL_CELLS = (slice(0, 4), slice(None), slice(0, 4))   # fuel at the origin corner
# Symmetry faces at the origin and on the thin y axis, vacuum outside.
BC = [("reflective", "vacuum"), "reflective", ("reflective", "vacuum")]


def _grid(geometry):
    if geometry == "cartesian":
        grid = Grid(shape=(N, 1, N), size=(60.0, 1.0, 60.0))
    else:
        grid = Grid(shape=(N, 1, N), size=(60.0, 1.0, 60.0),
                    geometry="cylindrical")
    mmap = np.ones(grid.shape, dtype=np.int32)
    mmap[_FUEL_CELLS] = 0
    return grid, mmap


_SPECTRA = {
    "chi_total": None,
    "shared": np.array([0.6, 0.4]),
    # One spectrum per precursor family (the DelayedEmission projection).
    "per_family": np.array([[0.7, 0.3], [0.5, 0.5]]),
}

_SOLVERS = {
    "diffusion": (TransientSolver, {}),
    "sdp1": (TransientSDP1Solver, {}),
    "sdpn1": (TransientSDPNSolver, dict(order=1)),
    "sdp2": (TransientSDPNSolver, dict(order=2)),
    "sdp3": (TransientSDPNSolver, dict(order=3)),
    "sp1": (TransientSPNSolver, dict(order=0)),
    "sp3": (TransientSPNSolver, dict(order=1)),
    "sp5": (TransientSPNSolver, dict(order=2)),
}

_TIGHT = dict(tol_k=1e-12, tol_source=1e-11)


@pytest.mark.parametrize("dt", [5e-4, 5e-2])
@pytest.mark.parametrize("spectrum", list(_SPECTRA))
@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical"])
@pytest.mark.parametrize("solver", list(_SOLVERS))
def test_unperturbed_transient_holds_power(solver, geometry, spectrum, dt):
    grid, mmap = _grid(geometry)
    kin = Kinetics(velocities=[1e7, 3e5], beta=[0.004, 0.0025],
                   decay=[0.08, 1.2], chi_delayed=_SPECTRA[spectrum])
    cls, kw = _SOLVERS[solver]
    res = cls(grid, lambda t: (MATS, mmap), kin, bc=BC, device="cpu",
              **kw).solve(t_end=4 * dt, dt=dt, tol_step=1e-11,
                          steady_kwargs=_TIGHT)
    assert len(res.power) == 5
    np.testing.assert_allclose(res.power, 1.0, rtol=0, atol=1e-8)
