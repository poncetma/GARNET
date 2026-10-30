"""SPN/SDPN verification and validation against Carreno et al. (2024), audit O10.

    PYTHONPATH=. python tools/assess_carreno2024.py bl --methods SP3 SDP3 \
        --vacuum marshak --meshes 40 80 160 --output bl.json
    PYTHONPATH=. python tools/assess_carreno2024.py c5g7 --methods SP3 \
        --vacuum marshak --cells-per-pin 4 8 --output c5g7.json

Reference: A. Carreno, A. Vidal-Ferrandiz, D. Ginestar, G. Verdu, "Development
of a diffusive simplified double PN approximation to the neutron transport
equation", Ann. Nucl. Energy 207 (2024) 110675 (CC BY 4.0; in literature/).

``bl`` is the one-group Brantley-Larsen problem (Fig. 3, Table 5, Table 6) on an
n x n mesh. ``c5g7`` is the 2-D C5G7 quarter core (Fig. 5, Table 7), either
pin-resolved (r = 0.54 cm cylinders, exact-area blended cells, the paper's
geometry) or pin-cell homogenized (``--geometry homogenized``, what
tests/validation/test_c5g7.py runs).

``--vacuum`` selects the vacuum-face treatment: ``marshak`` is the coupled
Marshak condition the paper uses, ``robin`` the per-moment alpha = 1/2 Robin
term, and, for C5G7 only, ``zero-flux`` replaces the outer vacuum faces by
phi = 0 (the builder's default). SP1 is diffusion, for which Marshak and Robin
coincide. ``SP3-SPN`` and ``SDP1-SDPN`` run SP3 and SDP1 through the general
SPN/SDPN block instead of their dedicated solvers; they must reproduce them.
``SDP3-A6`` is SDP3 with the coefficient tables as printed in the paper's Eq.
(A.6); garnet carries re-derived rows 2-3 (see the note in garnet/spn.py).
``SDP3-FFN`` uses the tables of the authors' FEMFFUSION code
(src/matrix_operators/matrix_operators_sdpn.cc), which differ from the printed
(A.6) in row 2 of c^(3): (9/25, -54/175) instead of (36/55, 54/175).

Each run records k, convergence, outer and inner iterations and setup and solve
seconds, plus the scalar flux along the lines the paper plots: y = 4.5 cm for
Brantley-Larsen (Fig. 4); the main diagonal of the inner UO2 assembly and the
anti-diagonal of a MOX assembly for C5G7 (Figs. 6-7, groups 1 and 7). C5G7 runs
also record the 51 x 51 pin fission-rate map, normalized to a mean of 1 over
the 1056 fuel pins. ``--repeats N`` keeps the fastest of N identical solves.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from garnet import (DiffusionEigenSolver, Grid, Material, SDP1EigenSolver,
                    SDP2EigenSolver, SDP3EigenSolver, SP3EigenSolver,
                    SP5EigenSolver, SP7EigenSolver)
from garnet.benchmarks.c5g7 import (K_REFERENCE_2D, N_PIN, PIN_PITCH, build_c5g7_2d,
                                   pin_powers)
from garnet.operator import _SDPN_C
from garnet.solver import SDPNEigenSolver, SPNEigenSolver

# Carreno et al. (2024) Table 6: k, dk (pcm), flux RMSE (%), CPU time (s).
PAPER_BL = {"SDP1": (0.80161, 375, 0.92, 1.5), "SDP2": (0.80373, 163, 0.76, 2.0),
            "SDP3": (0.80402, 134, 0.76, 2.6), "SP1": (0.77680, 2856, 4.16, 1.1),
            "SP3": (0.79904, 632, 1.01, 1.5), "SP5": (0.80280, 256, 0.74, 2.0),
            "SP7": (0.80354, 182, 0.74, 2.6)}
K_OPENMOC_BL = 0.80536
K_S16_BL = 0.80613              # TWODANT S16 (Brantley and Larsen, 2000)
# Table 7: k, |dk| (pcm), AVG (%), RMSE (%).
PAPER_C5G7 = {"SDP1": (1.18242, 413, 0.68, 0.03), "SDP2": (1.18219, 436, 0.57, 0.02),
              "SDP3": (1.18286, 369, 0.64, 0.03), "SP1": (1.18336, 319, 1.43, 0.05),
              "SP3": (1.18271, 384, 0.75, 0.03), "SP5": (1.18221, 434, 0.59, 0.02),
              "SP7": (1.18231, 424, 0.57, 0.02)}

# Table 5. D = 1/(3 Sigma_t); removal = Sigma_t - Sigma_s0.
BL_FUEL = Material(name="fuel", diffusion=[1.0 / 4.5], sigma_a=[0.15],
                   nu_sigma_f=[0.24], total=[1.5], chi=[1.0])
BL_MOD = Material(name="mod", diffusion=[1.0 / 3.0], sigma_a=[0.07],
                  nu_sigma_f=[0.0], total=[1.0], chi=[1.0])
BL_BARS = ((1.0, 2.0), (4.0, 5.0), (7.0, 8.0))
BL_LINE_Y = 4.5


class _SP3SPN(SPNEigenSolver):
    _order = 1


class _SDP1SDPN(SDPNEigenSolver):
    _order = 1


# The paper's Eq. (A.6), as printed: c^(1)..c^(4) of SDP3.
_A6 = [
    [[1, -2 / 3, 8 / 15, -16 / 35], [-2 / 3, 4 / 9, -16 / 45, 32 / 105],
     [8 / 15, -16 / 45, 64 / 225, -128 / 525],
     [-176 / 125, 352 / 375, -1408 / 1875, 2816 / 4375]],
    [[0, 0, 0, 0], [0, 5 / 9, -4 / 9, 8 / 21], [0, -4 / 9, 16 / 45, -32 / 105],
     [0, 88 / 75, -352 / 375, 704 / 875]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 36 / 55, 54 / 175], [0, 0, -594 / 625, 3564 / 4375]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 143 / 175]],
]


class _SDP3A6(SDP3EigenSolver):
    _coeffs = {**_SDPN_C, 3: _A6}


_FFN = [_A6[0], _A6[1], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 9 / 25, -54 / 175],
                         [0, 0, -594 / 625, 3564 / 4375]], _A6[3]]


class _SDP3FFN(SDP3EigenSolver):
    _coeffs = {**_SDPN_C, 3: _FFN}


METHODS = ("SP1", "SP3", "SP5", "SP7", "SDP1", "SDP2", "SDP3", "SP3-SPN", "SDP1-SDPN",
           "SDP3-A6", "SDP3-FFN")


def solver_for(method, vacuum):
    """(solver class, keyword arguments) for a method and vacuum treatment."""
    if method == "SP1":
        return DiffusionEigenSolver, {}
    if vacuum == "zero-flux":
        marshak = None                       # no vacuum faces: solver default
    else:
        marshak = vacuum == "marshak"
    table = {"SP3": SP3EigenSolver, "SP5": SP5EigenSolver, "SP7": SP7EigenSolver,
             "SDP2": SDP2EigenSolver, "SDP3": SDP3EigenSolver,
             "SP3-SPN": _SP3SPN, "SDP1-SDPN": _SDP1SDPN, "SDP3-A6": _SDP3A6, "SDP3-FFN": _SDP3FFN}
    if method == "SDP1":
        # The dedicated SDP1 block has no coupled Marshak term (see sp3.py).
        if marshak:
            return _SDP1SDPN, {"marshak_vacuum": True}
        return SDP1EigenSolver, {}
    kw = {} if marshak is None else {"marshak_vacuum": marshak}
    return table[method], kw


def brantley_larsen(n):
    """n x n mesh over the 10 x 10 cm square; reflective at x = 0 and y = 0."""
    h = 10.0 / n
    xc = (np.arange(n) + 0.5) * h
    in_bar = np.zeros(n, bool)
    for lo, hi in BL_BARS:
        in_bar |= (xc > lo) & (xc < hi)
    mmap = np.where(in_bar[:, None] & (xc < 9.0)[None, :], 0, 1).astype(int)
    grid = Grid(shape=(n, n, 1), size=(10.0, 10.0, h))
    bc = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    return grid, [BL_FUEL, BL_MOD], dict(material_map=mmap[:, :, None], bc=bc)


def bl_line(phi, n):
    """Scalar flux along y = 4.5 cm at the cell-centre x, linear in y."""
    h = 10.0 / n
    t = BL_LINE_Y / h - 0.5
    j = int(np.floor(t))
    w = t - j
    line = (1.0 - w) * phi[0, :, j, 0] + w * phi[0, :, j + 1, 0]
    return ((np.arange(n) + 0.5) * h).tolist(), line.tolist()


def c5g7_problem(s, geometry, vacuum, area="exact"):
    outer = "zero-flux" if vacuum == "zero-flux" else "vacuum"
    resolved = geometry == "pin-resolved"
    prob = build_c5g7_2d(cells_per_pin=s, pin_resolved=resolved,
                         exact_area=area == "exact", outer_bc=outer)
    kw = dict(material_map=prob.material_map, bc=prob.bc)
    if resolved:
        kw.update(mix_material=prob.mix_material, mix_weight=prob.mix_weight)
    return prob, kw


def c5g7_lines(prob, phi):
    """Groups 1 and 7 along the paper's Fig. 6 and Fig. 7 paths.

    Fig. 6: main diagonal of the inner UO2 assembly, from the core centre.
    Fig. 7: anti-diagonal of the MOX assembly at x in [21.42, 42.84] cm, from
    its reflector-side corner (42.84, 0) to (21.42, 21.42). Cell centres lie
    exactly on both diagonals; d is the distance along the path in cm.
    """
    s = prob.cells_per_pin
    m = N_PIN * s
    h = PIN_PITCH / s
    i = np.arange(m)
    d = (np.sqrt(2.0) * (i + 0.5) * h).tolist()
    out = {"d": d}
    for g in (0, 6):
        out[f"fig6_g{g + 1}"] = phi[g, i, i, 0].tolist()
        out[f"fig7_g{g + 1}"] = phi[g, 2 * m - 1 - i, i, 0].tolist()
    return out


def run_one(case, method, vacuum, mesh, args):
    cls, skw = solver_for(method, vacuum)
    if case == "bl":
        grid, mats, kw = brantley_larsen(mesh)
        prob = None
    else:
        prob, kw = c5g7_problem(mesh, args.geometry, vacuum, args.area)
        grid, mats = prob.grid, prob.materials
    best = None
    walls = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        solver = cls(grid, mats, device=args.device, **kw, **skw)
        setup = time.perf_counter() - t0
        res = solver.solve(tol_k=args.tol_k, tol_source=args.tol_source,
                           verbose=args.verbose)
        walls.append(res.solve_seconds)
        if best is None or res.solve_seconds < best[0].solve_seconds:
            best = (res, setup, solver)
    res, setup, solver = best
    phi = res.flux_numpy
    rec = dict(case=case, method=method, vacuum=vacuum, mesh=mesh,
               solver=type(solver).__name__, solver_kwargs=skw,
               linear_solver=getattr(solver._linsolve, "__name__", None),
               k=float(res.k_eff), converged=bool(res.converged),
               outers=int(res.outer_iterations), inners=int(res.inner_iterations),
               setup_seconds=setup, solve_seconds=res.solve_seconds,
               solve_seconds_all=walls, cells=int(np.prod(grid.shape)))
    if case == "bl":
        x, line = bl_line(phi, mesh)
        rec.update(line_x=x, line_phi=line)
        rec["paper"] = PAPER_BL.get(method.split("-")[0])
    else:
        rec["geometry"] = args.geometry
        rec["area"] = args.area
        if args.lines:
            rec["lines"] = c5g7_lines(prob, phi)
        rec["pin_powers"] = np.round(pin_powers(prob, phi), 7).tolist()
        rec["paper"] = PAPER_C5G7.get(method.split("-")[0])
    return rec


def environment():
    return dict(python=platform.python_version(), machine=platform.machine(),
                node=platform.node(), numpy=np.__version__,
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=["bl", "c5g7"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:7]))
    parser.add_argument("--vacuum", nargs="+", choices=["marshak", "robin", "zero-flux"],
                        default=["marshak", "robin"])
    parser.add_argument("--meshes", nargs="+", type=int, default=[40, 80, 160, 320],
                        help="bl: cells per side")
    parser.add_argument("--cells-per-pin", nargs="+", type=int, default=[4, 8],
                        help="c5g7: cells per pin-cell side")
    parser.add_argument("--geometry", choices=["pin-resolved", "homogenized"],
                        default="pin-resolved")
    parser.add_argument("--area", choices=["exact", "sampled"], default="exact",
                        help="c5g7 pin-resolved: exact or 16x16-sampled cell fractions")
    parser.add_argument("--no-lines", dest="lines", action="store_false",
                        help="c5g7: skip the Fig. 6/7 flux lines")
    parser.add_argument("--tol-k", type=float, default=1e-10)
    parser.add_argument("--tol-source", type=float, default=1e-9)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.case == "bl" and "zero-flux" in args.vacuum:
        parser.error("zero-flux applies to c5g7 only")
    meshes = args.meshes if args.case == "bl" else args.cells_per_pin
    out = dict(environment=environment(), args=vars(args), runs=[])
    for mesh in meshes:
        for method in args.methods:
            for vacuum in args.vacuum:
                if method == "SP1" and vacuum == "robin" and "marshak" in args.vacuum:
                    continue                  # identical for diffusion
                rec = run_one(args.case, method, vacuum, mesh, args)
                out["runs"].append(rec)
                ref = K_OPENMOC_BL if args.case == "bl" else K_REFERENCE_2D
                print(f"RESULT {args.case} {method:9s} {vacuum:9s} mesh {mesh:4d}  "
                      f"k={rec['k']:.9f}  dk={1e5 * (rec['k'] - ref):+8.1f} pcm  "
                      f"conv={rec['converged']}  outers={rec['outers']:4d}  "
                      f"inners={rec['inners']:7d}  t={rec['solve_seconds']:8.2f}s",
                      flush=True)
                if args.output:
                    with open(args.output, "w") as fh:
                        json.dump(out, fh)


if __name__ == "__main__":
    main()
