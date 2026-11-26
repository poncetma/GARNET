"""Compute every number in the GARNET theory report (docs/theory/report.tex).

Usage:
    python docs/theory/run_cases.py [STUDY ...]      # default: all studies

Each study writes data/<study>.json. Eigenvalue rows carry k, the error
against a named reference, outer and inner iterations and the solve time
(fastest of NREP identical runs, single thread). Run on an idle node:
    sbatch docs/theory/run_cases.sbatch

References:
  analytic        closed-form eigenvalue of the same discrete-free problem
  BIBLIS          FEMFFUSION FE3, refinements 0/1/2 extrapolated (data/biblis_femffusion.json)
  IAEA-3D         FEMFFUSION FE3 three-level limit (benchmark-results/garnet-vs-femffusion)
  VVER-440        FEMFFUSION 2-D diffusion (garnet.benchmarks.vver440.K_REFERENCE)
  TWIGL           Hageman & Yasinsky (1969) static k and P(t)
  kinetics ramp   Radau integration of the infinite-medium kinetics ODEs
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from garnet import (PWR_TWO_GROUP, CylindricalModel, DiffusionEigenSolver,  # noqa: E402
                    Grid, HexDiffusionEigenSolver, HexGrid, Kinetics, Material,
                    SP3EigenSolver, TransientSolver, TriDiffusionEigenSolver,
                    TriGrid, k_bare_box, k_bare_box_sp3, k_from_buckling,
                    k_infinite)

NREP = 3
TOL = dict(tol_k=1e-10, tol_source=1e-9)        # well below every error quoted
DATA = HERE / "data"


def best_of(run, n=NREP):
    """Fastest of n identical solves (each must converge)."""
    best = None
    for _ in range(n):
        r = run()
        assert getattr(r, "converged", True), r
        if best is None or r.solve_seconds < best.solve_seconds:
            best = r
    return best


def row(r, ref, **extra):
    return dict(extra, k=float(r.k_eff), err_pcm=1e5 * (float(r.k_eff) - ref),
                outers=int(r.outer_iterations), inners=int(r.inner_iterations),
                seconds=float(r.solve_seconds))


def save(name, obj):
    DATA.mkdir(exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(name, json.dumps(obj)[:400], flush=True)


# ---- verification --------------------------------------------------------------

def study_box():
    """Bare 90 cm cube, two groups, zero flux: diffusion and SP3."""
    size = (90.0, 90.0, 90.0)
    out = {}
    for label, cls, kref, ns in (("diffusion", DiffusionEigenSolver,
                                  k_bare_box(PWR_TWO_GROUP, size), (8, 16, 32, 64)),
                                 ("sp3", SP3EigenSolver,
                                  k_bare_box_sp3(PWR_TWO_GROUP, size), (8, 16, 32, 64))):
        rows = []
        for n in ns:
            g = Grid(shape=(n, n, n), size=size)
            r = best_of(lambda: cls(g, PWR_TWO_GROUP, bc="zero-flux", device="cpu").solve(**TOL))
            rows.append(row(r, kref, n=n, h=90.0 / n))
        out[label] = dict(k_ref=kref, rows=rows)
    save("box", out)


def study_cylinder():
    """Bare finite cylinder (r-z), one group: Bessel-mode eigenvalue."""
    mat = Material(name="1g", diffusion=[1.3], sigma_a=[0.030], nu_sigma_f=[0.035])
    R, H, j0 = 80.0, 160.0, 2.404825557695773
    kref = 0.035 / (0.030 + 1.3 * ((j0 / R) ** 2 + (math.pi / H) ** 2))
    rows = []
    for n in (10, 20, 40, 80, 160):
        def run():
            t0 = time.perf_counter()
            r = (CylindricalModel(radius=R, height=H, cells=(n, 2 * n)).fill(mat)
                 .set_boundary(r=("reflective", "zero-flux"), z="zero-flux")
                 .steady(device="cpu", **TOL))
            r.solve_seconds = getattr(r, "solve_seconds", time.perf_counter() - t0)
            return r
        r = best_of(run)
        rows.append(row(r, kref, n=n, h=R / n))
    save("cylinder", dict(k_ref=kref, rows=rows))


def study_prism():
    """Triangular prisms, reflective in-plane, zero flux in z: the 1-D slab."""
    H = 120.0
    kref = k_from_buckling(PWR_TWO_GROUP, (math.pi / H) ** 2)
    rows = []
    for nz in (8, 16, 32, 64, 128):
        g = TriGrid(shape=(3, 3, 2, nz), side=5.0, height=H)
        r = best_of(lambda: TriDiffusionEigenSolver(
            g, PWR_TWO_GROUP, bc=("reflective", "reflective", "zero-flux"),
            device="cpu").solve(**TOL))
        rows.append(row(r, kref, nz=nz, dz=H / nz))
    save("prism", dict(k_ref=kref, rows=rows))


def _diagonal_cut_square(n, L):
    from garnet.mesh import assemble_mesh
    xs = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    coords = np.c_[X.ravel(), Y.ravel()]
    idx = lambda i, j: i * (n + 1) + j          # noqa: E731
    cells = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            cells += [(a, b, c), (a, c, d)]
    return assemble_mesh(coords, cells, [0] * len(cells))


def study_skewed():
    """Bare square on triangles cut along one diagonal (non-orthogonal faces)."""
    from garnet.mesh import UnstructuredDiffusionSolver
    L = 10.0
    mat = Material(name="m", diffusion=[1.0], sigma_a=[0.1], nu_sigma_f=[0.3])
    kref = 0.3 / (0.1 + 2.0 * (math.pi / L) ** 2)
    out = {}
    for label, corr in (("corrected", True), ("two-point only", False)):
        rows = []
        for n in (8, 16, 32, 64):
            mesh = _diagonal_cut_square(n, L)
            r = best_of(lambda: UnstructuredDiffusionSolver(
                mesh, [mat], np.zeros(mesh.n_cells, int), alpha_boundary=1e10,
                device="cpu", nonorthogonal_correction=corr).solve(**TOL))
            rows.append(row(r, kref, n=n, cells=int(mesh.n_cells)))
        out[label] = rows
    save("skewed", dict(k_ref=kref, **out))


def study_reflective():
    """k = k_inf with reflective boundaries on every lattice (no leakage)."""
    mat = PWR_TWO_GROUP
    kinf = k_infinite(mat)
    rows = []
    cases = {
        "Cartesian": lambda: DiffusionEigenSolver(Grid(shape=(6, 5, 4), size=(30., 25., 20.)),
                                                  mat, bc="reflective", device="cpu"),
        "triangular prisms": lambda: TriDiffusionEigenSolver(
            TriGrid(shape=(4, 4, 2, 3), side=5.0, height=30.0), mat, bc="reflective",
            mask_bc="reflective", device="cpu"),
        "hexagonal prisms": lambda: HexDiffusionEigenSolver(
            HexGrid(shape=(5, 5, 3), pitch=10.0, height=30.0), [mat],
            np.zeros((5, 5, 3), int), bc="reflective", mask_bc="reflective", device="cpu"),
    }
    for name, make in cases.items():
        r = make().solve(**TOL)
        rows.append(row(r, kinf, case=name))
    save("reflective", dict(k_ref=kinf, rows=rows))


def study_kinetics():
    """Infinite-medium ramp (six families): time schemes against Radau."""
    from scipy.integrate import solve_ivp
    D, SA, NSF, V = 1.3, 0.030, 0.035, 2.2e5
    beta = np.array([0.00025, 0.0016, 0.0014, 0.003, 0.0009, 0.0003])
    lam = np.array([0.0127, 0.0317, 0.115, 0.311, 1.4, 3.87])
    kin = Kinetics(velocities=[V], beta=beta, decay=lam)
    T, RATE = 0.2, 0.0006
    grid = Grid(shape=(2, 2, 1), size=(10.0, 10.0, 1.0))

    def mat(dsa):
        return Material(name="1g", diffusion=[D], sigma_a=[SA - dsa], nu_sigma_f=[NSF])

    k0 = NSF / SA                                   # infinite medium

    def rhs(t, x):
        sa = SA - RATE * t
        bt = beta.sum()
        dphi = V * (-sa * x[0] + (1 - bt) * NSF / k0 * x[0] + lam @ x[1:])
        dc = beta * NSF / k0 * x[0] - lam * x[1:]
        return np.r_[dphi, dc]
    x0 = np.r_[1.0, beta * NSF / k0 / lam]
    sol = solve_ivp(rhs, (0.0, T), x0, method="Radau", rtol=1e-12, atol=1e-14)
    p_ref = float(sol.y[0, -1])

    cache = {}

    def problem_at(t):
        key = round(float(t), 15)
        if key not in cache:
            cache[key] = [mat(RATE * max(float(t), 0.0))]
        return cache[key], None

    out = {"P_ref": p_ref, "schemes": {}}
    schemes = {"backward Euler": dict(time_scheme="backward-euler"),
               "Crank-Nicolson": dict(time_scheme="crank-nicolson"),
               "BDF2": dict(time_scheme="bdf2"),
               "BDF3 (graded start)": dict(time_scheme="bdf3", bdf_startup="graded"),
               "BDF4 (graded start)": dict(time_scheme="bdf4", bdf_startup="graded")}
    for name, kw in schemes.items():
        rows = []
        for n in (10, 20, 40, 80, 160):
            t0 = time.perf_counter()
            r = TransientSolver(grid, problem_at, kin, bc="reflective", device="cpu").solve(
                t_end=T, dt=T / n, tol_step=1e-13, **kw)
            rows.append(dict(steps=n, dt=T / n, P=float(r.power[-1]),
                             rel_err=float(r.power[-1] / p_ref - 1.0),
                             seconds=time.perf_counter() - t0))
        out["schemes"][name] = rows
    save("kinetics", out)


# ---- validation --------------------------------------------------------------

def _cart(p, ref, **extra):
    r = best_of(lambda: DiffusionEigenSolver(p.grid, p.materials, p.material_map, bc=p.bc,
                                             active=p.active, mask_bc=p.mask_bc,
                                             device="cpu").solve(**TOL))
    return row(r, ref, cells=int(p.active.sum()), **extra)


def study_biblis():
    from garnet.benchmarks import build_biblis
    ref = json.loads((DATA / "biblis_femffusion.json").read_text())["k_limit"]
    save("biblis", dict(k_ref=ref, rows=[_cart(build_biblis(c), ref, cells_per_assembly=c)
                                         for c in (1, 2, 4, 8, 16)]))


def study_iaea():
    from garnet.benchmarks import build_iaea
    ref = 1.0290729                                 # FEMFFUSION FE3 limit
    save("iaea", dict(k_ref=ref, rows=[_cart(build_iaea(c), ref, cells_per_node=c)
                                       for c in (1, 2, 4, 8)]))


def study_vver():
    from garnet.benchmarks import build_vver440
    from garnet.benchmarks.vver440 import K_REFERENCE
    rows = []
    for refine in (1, 2, 4, 8):
        p = build_vver440(refine=refine)
        r = best_of(lambda: TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                                    active=p.active, mask_bc=p.mask_bc,
                                                    device="cpu").solve(**TOL))
        rows.append(row(r, K_REFERENCE, refine=refine, cells=int(p.active.sum())))
    save("vver", dict(k_ref=K_REFERENCE, rows=rows))


def study_twigl():
    from garnet.benchmarks import build_twigl
    from garnet.benchmarks.twigl import K_REFERENCE, P_REFERENCE
    static = []
    for c in (1, 2, 4, 8, 16):
        p = build_twigl("step", cells_per_8cm=c)
        mats, mmap = p.problem_at(0.0)
        r = best_of(lambda: DiffusionEigenSolver(p.grid, mats, mmap, bc=p.bc,
                                                 device="cpu").solve(**TOL))
        static.append(row(r, K_REFERENCE, cells_per_8cm=c, h=8.0 / c))
    transient = []
    for pert in ("step", "ramp"):
        for c, dt in ((2, 2e-3), (4, 1e-3), (8, 5e-4)):
            p = build_twigl(pert, cells_per_8cm=c)
            r = TransientSolver(p.grid, p.problem_at, p.kinetics, bc=p.bc, device="cpu").solve(
                t_end=0.5, dt=dt, time_scheme="bdf2")
            at = lambda tt: float(r.power[np.argmin(np.abs(r.times - tt))])   # noqa: E731
            transient.append(dict(perturbation=pert, cells_per_8cm=c, dt=dt,
                                  P01=at(0.1), P05=at(0.5),
                                  ref01=P_REFERENCE[pert][0.1], ref05=P_REFERENCE[pert][0.5],
                                  inners=int(r.total_inner_iterations),
                                  seconds=float(r.solve_seconds)))
    save("twigl", dict(k_ref=K_REFERENCE, static=static, transient=transient))


def study_hpmr():
    """2-D HP-MR, 11-group ENDF/B-VIII.0 (flat-flux assembly homogenization)."""
    from garnet.benchmarks.hpmr import build_hpmr2d
    from garnet.benchmarks.hpmr_thermal import hpmr_endfb8_builtin
    mats = hpmr_endfb8_builtin()
    angles = (0, 30, 60, 90, 120, 150, 180)
    out = {}
    for refine in (4, 8, 16):
        rows = []
        for a in angles:
            p = build_hpmr2d(refine=refine, drum_angle_deg=float(a), absorber="polar",
                             materials=mats)
            r = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, bc=p.bc,
                                        active=p.active, mask_bc=p.mask_bc,
                                        mix_material=p.mix_material, mix_weight=p.mix_weight,
                                        device="cpu").solve(**TOL)
            assert r.converged
            rows.append(dict(angle=a, k=float(r.k_eff), outers=int(r.outer_iterations),
                             inners=int(r.inner_iterations), seconds=float(r.solve_seconds),
                             cells=int(p.active.sum())))
        out[str(refine)] = rows
    save("hpmr", dict(groups=11, rows=out))


def _hpmr_k(refine, angle, mats):
    from garnet.benchmarks.hpmr import build_hpmr2d
    p = build_hpmr2d(refine=refine, drum_angle_deg=float(angle), absorber="polar",
                     materials=mats)
    r = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, bc=p.bc,
                                active=p.active, mask_bc=p.mask_bc,
                                mix_material=p.mix_material, mix_weight=p.mix_weight,
                                device=os.environ.get("GARNET_THEORY_DEVICE", "cpu")).solve(**TOL)
    assert r.converged
    return dict(refine=refine, angle=angle, device=r.device, k=float(r.k_eff), outers=int(r.outer_iterations),
                inners=int(r.inner_iterations), seconds=float(r.solve_seconds),
                cells=int(p.active.sum()))


def study_hpmr_noabs():
    """Control: the same core with the B4C arcs replaced by drum beryllium."""
    from garnet.benchmarks.hpmr import DRUM_ABSORBER, DRUM_BE
    from garnet.benchmarks.hpmr_thermal import hpmr_endfb8_builtin
    mats = list(hpmr_endfb8_builtin())
    mats[DRUM_ABSORBER] = mats[DRUM_BE]
    save("hpmr_noabs", dict(rows=[_hpmr_k(r, 0, mats) for r in (4, 8, 16, 32)]))


def study_hpmr32():
    """Refinement 32 (0.48 cm triangles) at the two end angles."""
    from garnet.benchmarks.hpmr_thermal import hpmr_endfb8_builtin
    mats = hpmr_endfb8_builtin()
    save("hpmr32", dict(rows=[_hpmr_k(32, a, mats) for a in (0, 180)]))


STUDIES = {n[6:]: f for n, f in globals().items() if n.startswith("study_")}

if __name__ == "__main__":
    import platform
    sel = sys.argv[1:] or list(STUDIES)
    DATA.mkdir(exist_ok=True)
    env_name = "environment" + ("" if sel == list(STUDIES) else "-" + "-".join(sel))
    (DATA / f"{env_name}.json").write_text(json.dumps(dict(
        host=platform.node(), machine=platform.machine(), python=platform.python_version(),
        numpy=np.__version__, processor=platform.processor(),
        date=time.strftime("%Y-%m-%d")), indent=1))
    for s in sel:
        t0 = time.perf_counter()
        STUDIES[s]()
        print(f"== {s} done in {time.perf_counter() - t0:.1f} s", flush=True)
