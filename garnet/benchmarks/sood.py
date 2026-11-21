"""Analytic criticality benchmarks for the triangular S_N solver.

A. Sood, R. A. Forster and D. K. Parsons, "Analytical Benchmark Test Set for
Criticality Code Verification", LA-13511 (1999); Prog. Nucl. Energy 42 (2003)
55-106. Every finite problem is exactly critical (k_eff = 1 to at least five
decimals) at the tabulated critical dimension r_c: the half-thickness of a
bare slab, the radius of a bare cylinder. The cylinder and slab problems with
linearly anisotropic (P1) and P2 scattering verify the anisotropic path of
:class:`garnet.tri_sn.TriSNTransportSolver`; the isotropic problems on the same
meshes calibrate the geometric and discretization error.

Cross sections are those of the report's tables (macroscopic, 1/cm), with the
moments Sigma_s,l without the (2l+1) factor, as the report gives them
(Sigma_s(mu) = (Sigma_s0 + 3 mu Sigma_s1) / 2 for one group), which is
GARNET's ``Material.sigma_s_moments`` convention. GARNET's group 0 is the
report's fast group 2. In the U-235 one-group sets Sigma_s1 is exactly 0.1 and
0.5 of c Sigma_t = nu Sigma_f + Sigma_s0: the underlying solutions (Sanchez and
Ganapol) make all secondary neutrons linearly anisotropic, which at k = 1 is
the same source as anisotropic scattering alone.

Geometry on the triangular lattice (cell (i, j) spans the parallelogram
i b + j a, with b = (0, h) and a = (h sqrt(3)/2, h/2)):

* cylinder: vacuum boundary on the disk; cells the circle cuts are blended
  with a void at their exact covered area (the solver's volume mixing), so the
  disk keeps its exact area;
* slab: the columns j form vertical strips of width h sqrt(3)/2. Fuel columns
  are followed by a purely absorbing, non-scattering layer of 40 optical
  thicknesses, and the grid is periodic in both directions: the layer returns
  no neutrons, so each slab face sees an exact vacuum, and the slab is infinite
  in y. The periodic sweep needs ``engine="lu"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ..materials import Material
from ..tri import TriGrid

SQ3 = np.sqrt(3.0)

# name: (nu, Sigma_f, Sigma_c, P0 [from][to], P1, P2, Sigma_t, chi), group 0 fast
_ONE = lambda *v: np.array(v, dtype=float).reshape(1, 1)  # noqa: E731
_DATA = {
    # Table 5, Pu-239 (a), c = 1.50, isotropic
    "PUa": dict(nu=[3.24], sf=[0.081600], sc=[0.019584], s0=_ONE(0.225216),
                st=[0.32640]),
    # Table 24, Pu-239 (b), c = 1.40, P1 and P2
    "PUb-aniso": dict(nu=[2.5], sf=[0.266667], sc=[0.0], s0=_ONE(0.733333),
                      s1=_ONE(0.333333), s2=_ONE(0.125), st=[1.0]),
    # Table 12 / 26, U-235 (a), c = 1.30; P1 set (a) and (b)
    "Ua": dict(nu=[2.70], sf=[0.065280], sc=[0.013056], s0=_ONE(0.248064),
               st=[0.32640]),
    "Ua-aniso": dict(nu=[2.70], sf=[0.065280], sc=[0.013056], s0=_ONE(0.248064),
                     s1=_ONE(0.042432), st=[0.32640]),
    "Ub-aniso": dict(nu=[2.70], sf=[0.065280], sc=[0.013056], s0=_ONE(0.248064),
                     s1=_ONE(0.212160), st=[0.32640]),
    # Table 16, U-D2O reactor, c = 1.02, isotropic (nu from k_inf = 1.133333)
    "UD2O": dict(nu=[1.70], sf=[0.054628], sc=[0.027314], s0=_ONE(0.464338),
                 st=[0.54628]),
    # Table 28, U-D2O (b) and (c), P1 (set (a) has an unreadable Sigma_s1 in the
    # scanned report); nu reproduces the tabulated k_inf 1.227391 / 1.130933
    "UD2Ob-aniso": dict(nu=[1.841086], sf=[0.054628], sc=[0.027314], s0=_ONE(0.464338),
                        s1=_ONE(0.112982569), st=[0.54628]),
    "UD2Oc-aniso": dict(nu=[1.6964], sf=[0.054628], sc=[0.027314], s0=_ONE(0.464338),
                        s1=_ONE(-0.27850447), st=[0.54628]),
    # Tables 39/40 (isotropic) and 52/53 (P1), research reactor (a)
    "URRa": dict(nu=[2.50, 2.50], sf=[0.0010484, 0.050632], sc=[0.0010046, 0.025788],
                 s0=np.array([[0.62568, 0.029227], [0.0, 2.44383]]),
                 st=[0.65696, 2.52025], chi=[1.0, 0.0]),
    "URRa-aniso": dict(nu=[2.50, 2.50], sf=[0.0010484, 0.050632],
                       sc=[0.0010046, 0.025788],
                       s0=np.array([[0.62568, 0.029227], [0.0, 2.44383]]),
                       s1=np.array([[0.27459, 0.0075737], [0.0, 0.83318]]),
                       st=[0.65696, 2.52025], chi=[1.0, 0.0]),
}


@dataclass(frozen=True)
class SoodProblem:
    """One LA-13511 problem: k_eff = 1 at the critical dimension ``rc`` (cm)."""

    number: int
    identifier: str
    material: str
    order: int              # Legendre order of the scattering data
    geometry: str           # "slab" (rc = half-thickness), "cylinder" or "sphere" (radius)
    rc: float
    flux_positions: tuple = ()      # r / rc of the tabulated scalar fluxes
    flux: tuple = ()                # per group (fast first), normalized to the
                                    # fast flux at the centre


PROBLEMS = {p.number: p for p in (
    SoodProblem(2, "PUa-1-0-SL", "PUa", 0, "slab", 1.853722),
    SoodProblem(12, "Ua-1-0-SL", "Ua", 0, "slab", 2.872934),
    SoodProblem(13, "Ua-1-0-CY", "Ua", 0, "cylinder", 5.284935),
    SoodProblem(34, "PUb-1-1-SL", "PUb-aniso", 1, "slab", 0.79606),
    SoodProblem(35, "PUb-1-2-SL", "PUb-aniso", 2, "slab", 0.78396),
    SoodProblem(36, "Ua-1-1-CY", "Ua-aniso", 1, "cylinder", 5.514296811),
    SoodProblem(37, "Ub-1-1-CY", "Ub-aniso", 1, "cylinder", 6.940205668),
    SoodProblem(14, "Ua-1-0-SP", "Ua", 0, "sphere", 7.428998,
                (0.25, 0.5, 0.75, 1.0),
                ((0.93244907, 0.74553332, 0.48095413, 0.17177706),)),
    SoodProblem(24, "UD2O-1-0-SP", "UD2O", 0, "sphere", 22.017156),
    SoodProblem(41, "UD2Ob-1-1-SP", "UD2Ob-aniso", 1, "sphere", 18.30563081),
    SoodProblem(43, "UD2Oc-1-1-SP", "UD2Oc-aniso", 1, "sphere", 18.30563081),
    SoodProblem(55, "URRa-2-0-SP", "URRa", 0, "sphere", 16.049836),
    SoodProblem(54, "URRa-2-0-SL", "URRa", 0, "slab", 7.566853,
                (0.241394, 0.502905, 0.744300, 1.0),
                ((0.943363, 0.761973, 0.504012, 0.147598),
                 (0.340124, 0.273056, 0.173845, 0.0212324))),
    SoodProblem(71, "URRa-2-1-SL", "URRa-aniso", 1, "slab", 9.491600,
                (0.2, 0.5, 0.8, 1.0),
                ((0.963873, 0.781389, 0.472787, 0.189578),
                 (0.349006, 0.280870, 0.157376, 0.0277639))),
)}


def sood_material(name: str, order: int | None = None) -> Material:
    """GARNET material of a benchmark cross-section set, with its moments up
    to ``order`` (default: all it has)."""
    d = _DATA[name]
    s0 = np.asarray(d["s0"], dtype=float)
    G = s0.shape[0]
    moments = [s0] + [np.asarray(d[k], dtype=float) for k in ("s1", "s2") if k in d]
    if order is not None:
        moments = moments[:order + 1]
    st = np.asarray(d["st"], dtype=float)
    sf = np.asarray(d["sf"], dtype=float)
    return Material(name=name, diffusion=1.0 / (3.0 * st),
                    sigma_a=sf + np.asarray(d["sc"], dtype=float),
                    nu_sigma_f=np.asarray(d["nu"], dtype=float) * sf,
                    sigma_s=s0, chi=d.get("chi", [1.0] + [0.0] * (G - 1)),
                    total=st,
                    sigma_s_moments=np.array(moments) if len(moments) > 1 else None)


def _void(G):
    return Material(name="void", diffusion=[1e6] * G, sigma_a=[0.0] * G,
                    nu_sigma_f=[0.0] * G)


def _absorber(G, sigma):
    return Material(name="black", diffusion=[1.0 / (3.0 * sigma)] * G,
                    sigma_a=[sigma] * G, nu_sigma_f=[0.0] * G, total=[sigma] * G)


def cell_vertices(shape, side):
    """(nr, nc, 2, 3, 2) Cartesian vertices of a TriGrid's cells."""
    from ..pin_power import tri_cell_vertices
    return tri_cell_vertices(SimpleNamespace(material_map=np.zeros(shape[:2], int),
                                             side=side, origin=(1, 1)))


def build_sood_problem(number: int, cells: int):
    """Solver inputs for a problem at mesh resolution ``cells``: the number of
    cell layers across the slab's full thickness, or across the cylinder's
    diameter (triangle side h = 2 rc / (cells sqrt(3)/2) for the slab, 2 rc /
    cells for the cylinder).

    Returns SimpleNamespace(problem, grid, materials, material_map, active,
    mix_material, mix_weight, bc, engine, scattering_order, fuel_columns,
    column_x, centre_x) -- the last three for slab flux edits.
    """
    p = PROBLEMS[number]
    fuel = sood_material(p.material, p.order)
    G = fuel.n_groups
    if p.geometry == "slab":
        width = SQ3 / 2.0
        h = 2.0 * p.rc / (cells * width)
        n_abs = 40                                   # 40 optical thicknesses
        nr, nc = 2, cells + n_abs
        grid = TriGrid(shape=(nr, nc, 2), side=h)
        mmap = np.zeros(grid.shape, int)
        mmap[:, cells:, :] = 1
        materials = [fuel, _absorber(G, 1.0 / (h * width))]
        return SimpleNamespace(problem=p, grid=grid, materials=materials,
                               material_map=mmap, active=None, mix_material=None,
                               mix_weight=None, bc="periodic", engine="lu",
                               scattering_order=p.order, fuel_columns=cells,
                               column_x=(np.arange(cells) + 0.5) * h * width,
                               centre_x=p.rc)
    if p.geometry != "cylinder":
        raise ValueError(p.geometry)
    from ..pin_power import cell_footprints
    h = 2.0 * p.rc / cells
    n = int(np.ceil(4.0 * (p.rc + h) / (h * SQ3))) + 1
    grid = TriGrid(shape=(n, n, 2), side=h)
    V = cell_vertices(grid.shape, h)
    centre = 0.5 * n * (np.array([0.0, h]) + np.array([h * SQ3 / 2.0, h / 2.0]))
    W = cell_footprints(V.reshape(-1, 3, 2), centre[None], p.rc)
    covered = np.asarray(W.sum(axis=0)).reshape(grid.shape) / grid.cell_volume
    covered = np.clip(covered, 0.0, 1.0)
    active = covered > 1e-12
    mix_material = np.where(covered < 1.0 - 1e-12, 1, -1)
    return SimpleNamespace(problem=p, grid=grid, materials=[fuel, _void(G)],
                           material_map=np.zeros(grid.shape, int), active=active,
                           mix_material=mix_material, mix_weight=np.where(
                               mix_material >= 0, 1.0 - covered, 0.0),
                           bc="vacuum", engine="levels", scattering_order=p.order,
                           fuel_columns=None, column_x=None, centre_x=None)


def slab_flux_edits(case, flux):
    """Scalar flux at the problem's tabulated r/rc, per group, normalized to
    the fast flux at the centre: (G, n_positions) and (G,) at the centre.
    Column averages, linear interpolation, linear extrapolation to the face."""
    f = np.asarray(flux)
    G = f.shape[0]
    cols = f[:, :, :case.fuel_columns, :].mean(axis=(1, 3))       # (G, cells)
    x = case.column_x
    pos = case.centre_x + case.centre_x * np.asarray(case.problem.flux_positions)

    def at(g, xq):
        if xq <= x[-1]:
            return np.interp(xq, x, cols[g])
        return cols[g, -1] + (cols[g, -1] - cols[g, -2]) / (x[-1] - x[-2]) * (xq - x[-1])

    centre = np.array([at(g, case.centre_x) for g in range(G)])
    vals = np.array([[at(g, xq) for xq in pos] for g in range(G)]) / centre[0]
    return vals, centre / centre[0]
