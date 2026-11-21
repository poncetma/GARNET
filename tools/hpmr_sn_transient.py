"""Full space-time S_N transient of the 2-D HP-MR drum manoeuvres: the transport
reference for the tabulated SPH-DF model where the quasi-static S_N is not
valid (deep insertions, near-prompt-critical moves).

    python tools/hpmr_sn_transient.py CASE [REFINE] [T_END] [DT] [DEVICE] [MODE]

CASE as in tools/hpmr_drum_transient.py (A, B, C, D, E, E2). The S_N problem at
each drum angle is the reference recipe of tools/hpmr_pin_reference.py
(``sn-p2:4:16``: P2 scattering, 4 x 16 product quadrature, SCB, net
absorption), cached per angle; ``TransientSNSolver`` with the level-scheduled
tri engine and in-place material updates. MODE ``rebalance`` (default) uses
the whole-core rebalance with plain Picard (anderson_depth 1); ``anderson``
the default Anderson fixed point. Library kinetics ``hpmr_kinetics_11g``.

Results: <CACHE>/drum_transients_sn/<case>_r<refine>_dt<dt>.npz (times, power,
step iterations, wall time).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hpmr_drum_transient import CASES  # noqa: E402
from hpmr_pin_reference import DEFAULT_CACHE, TOL_SN, parse, problem_for  # noqa: E402

from garnet import TransientSNSolver  # noqa: E402
from garnet.backend import asnumpy  # noqa: E402
from garnet.benchmarks.hpmr_assembly import FUEL as FUEL_COMPACT  # noqa: E402
from garnet.benchmarks.hpmr_core_pins import PIN_OFFSET, R_FUEL, build_hpmr2d_pins  # noqa: E402
from garnet.pin_power import footprint_flux, pin_footprints  # noqa: E402
from garnet.power import fission_energy_xs  # noqa: E402
from garnet.benchmarks.hpmr_thermal import hpmr_kinetics_11g  # noqa: E402
from garnet.tri_sn import TriSNTransportSolver  # noqa: E402

METHOD = "sn-p2:4:16"


def main(case, refine=32, t_end=None, dt=None, device="gpu", mode="rebalance"):
    theta, t_end0, dt0 = CASES[case]
    t_end = t_end0 if t_end is None else float(t_end)
    dt = dt0 if dt is None else float(dt)
    _, _, order, quad = parse(METHOD)
    problems = {}

    def problem(angle):
        a = round(float(angle), 9)
        if a not in problems:
            problems.clear()                       # keep one geometry resident
            problems[a] = problem_for(METHOD, refine, a)
        return problems[a]

    p0 = problem(theta(0.0))

    def problem_at(t):
        p = problem(theta(t))
        return p.materials, p.material_map, p.mix_material, p.mix_weight

    print(f"S_N transient {METHOD} refine {refine}, case {case}: theta {theta(0):g} -> "
          f"{theta(t_end):g} deg, t_end {t_end:g} s, dt {dt:g} s, {mode}, step acceleration "
          f"{os.environ.get('SN_STEP_ACCEL', 'auto')}, CMFD aggregate "
          f"{os.environ.get('SN_CMFD_AGG', '0')}, device {device}",
          flush=True)
    t0 = time.perf_counter()
    # pin powers (kappaSigma_f over the fuel-compact footprints) at every step,
    # their level carried by the solver power (pin sum / production ratio)
    pp, pins = build_hpmr2d_pins(refine=refine, drum_angle_deg=theta(0.0))
    kf = fission_energy_xs(pp.materials)[0][PIN_OFFSET + FUEL_COMPACT]
    W = pin_footprints(pp.raster, pins.xy[pins.fuel], R_FUEL)
    rec = dict(t=[], pins=[], ratio=[], power=[])
    cache = os.environ.get("GARNET_REF_CACHE", DEFAULT_CACHE)
    out_dir = os.path.join(cache, "drum_transients_sn")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{case}_r{refine}_dt{dt:g}_{mode}.npz")

    def save(partial):
        ratio = np.asarray(rec["ratio"])
        np.savez(out_file if not partial else out_file.replace(".npz", "_partial.npz"),
                 times=np.asarray(rec["t"]), power=np.asarray(rec["power"]),
                 theta=np.array([theta(x) for x in rec["t"]]), rec_t=np.asarray(rec["t"]),
                 pins=np.asarray(rec["pins"]),
                 P_pin=np.asarray(rec["power"]) * ratio / ratio[0], partial=partial)

    def on_step(t, phi, power):
        f = np.stack([asnumpy(x).reshape(-1) for x in phi])
        rate = (kf * footprint_flux(W, f.reshape((f.shape[0],) + p0.grid.shape))).sum(axis=1)
        p = problem(theta(t))
        nsf = np.array([np.asarray(m.nu_sigma_f, float) for m in p.materials])[
            np.asarray(p.material_map).reshape(-1)].T
        rec["t"].append(t)
        rec["power"].append(float(power))
        rec["pins"].append(rate.astype(np.float32))
        rec["ratio"].append(rate.sum() / float((nsf * f).sum()))
        if len(rec["t"]) % 20 == 1:                  # survive a time limit
            save(partial=True)

    print(f"problem built; steady solve ({METHOD}, refine {refine}) ...", flush=True)
    solver = TransientSNSolver(p0.grid, problem_at, hpmr_kinetics_11g(), bc="vacuum",
                               engine_cls=TriSNTransportSolver, device=device, engine="levels",
                               active=p0.active, n_polar=quad[0], n_azi=quad[1],
                               scheme="scb", scattering_order=order,
                               step_acceleration=os.environ.get("SN_STEP_ACCEL", "auto"),
                               cmfd_aggregate=int(os.environ.get("SN_CMFD_AGG", "0")),
                               cmfd_two_level=int(os.environ.get("SN_CMFD_2L", "0")),
                               on_step=on_step)
    kw = (dict(rebalance=True, anderson_depth=1) if mode == "rebalance" else {})
    in_place = os.environ.get("SN_UPDATE_IN_PLACE", "1") != "0"
    r = solver.solve(t_end, dt, tol_step=1e-6, max_sweeps=400, steady_kwargs=TOL_SN,
                     verbose=int(os.environ.get("SN_VERBOSE", "2")), update_in_place=in_place,
                     **kw)
    wall = time.perf_counter() - t0
    its = np.asarray(r.step_iterations)
    print(f"done: k0 {r.k0:.7f}, P(end) {r.power[-1]:.6f}, {len(its)} steps, sweeps per step "
          f"mean {its.mean():.1f} max {its.max()}, total {wall:.0f} s, inner sweeps "
          f"{r.total_inner_iterations}", flush=True)
    ratio = np.asarray(rec["ratio"])
    np.savez(os.path.join(out_dir, f"{case}_r{refine}_dt{dt:g}_{mode}.npz"),
             times=r.times, power=r.power, k0=r.k0, step_iterations=its, wall=wall,
             theta=np.array([theta(t) for t in r.times]), rec_t=np.asarray(rec["t"]),
             pins=np.asarray(rec["pins"]),
             P_pin=np.asarray(r.power) * ratio / ratio[0])


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 32, a[2] if len(a) > 2 else None,
         a[3] if len(a) > 3 else None, a[4] if len(a) > 4 else "gpu",
         a[5] if len(a) > 5 else "rebalance")
