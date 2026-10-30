"""O12 multigroup check: HP-MR 2-D refine 2, 11 groups, rebalanced Picard.

    python benchmark-results/transient-step-convergence/hpmr2d_ladder.py --label after --output out.json

A: the hpmr_transient_bench absorption step (dt 0.02, 6 steps).
B: a slow insertion (absorption x 0.99985, dt 0.05, 40 steps).
Each tol_step is compared with a tol_step 1e-10 run of the same code.
Stored runs: hpmr2d-ladder-{before,after}.json. The reference quoted in audit
O12 is the monolithic step at tol_step 1e-10, hpmr2d-monolithic-reference.json.
"""
import argparse
import json
import time

import numpy as np

from garnet import TransientSolver
from garnet.benchmarks.hpmr_transient_bench import (SIGMA_A_SCALE, SUBSWEEPS, build_case,
                                                   scale_absorption)
from garnet.tri import TriDiffusionEigenSolver, TriGroupOperator

CASES = {"A": dict(scale=SIGMA_A_SCALE, dt=0.02, steps=6),
         "B": dict(scale=0.99985, dt=0.05, steps=40)}


def run(case, tol_step, refine=2, **kw):
    c = CASES[case]
    p, kin = build_case(refine, 0, "11")
    mats1 = scale_absorption(p.materials, c["scale"])
    dt = c["dt"]
    problem_at = (lambda t: ((mats1 if t >= 0.5 * dt else p.materials),
                             p.material_map, p.mix_material, p.mix_weight))
    solver = TransientSolver(
        p.grid, problem_at, kin, bc=p.bc, active=p.active, mask_bc=p.mask_bc,
        mix_material=p.mix_material, mix_weight=p.mix_weight,
        group_operator=TriGroupOperator, eig_solver=TriDiffusionEigenSolver,
        device="cpu")
    opts = dict(dt=dt, tol_step=tol_step, anderson_depth=1, rebalance=True,
                max_sweeps=4000, scatter_subsweeps=SUBSWEEPS)
    opts.update(kw)
    t0 = time.perf_counter()
    res = solver.solve(t_end=c["steps"] * dt, **opts)
    return res, time.perf_counter() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--cases", default="AB")
    ap.add_argument("--tols", nargs="+", type=float, default=[1e-5, 1e-6, 1e-7, 1e-8])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--output")
    args = ap.parse_args()
    out = {}
    for case in args.cases:
        ref, ref_wall = run(case, 1e-10)
        step_ref = np.abs(np.diff(ref.power)) / ref.power[1:]
        print(f"[{args.label}] case {case}: ref P(end) {ref.power[-1]:.9f}  sweeps/step "
              f"{np.mean(ref.step_iterations):.1f}  cg/step {ref.total_inner_iterations / (len(ref.times) - 1):.0f}  "
              f"{ref_wall:.1f}s", flush=True)
        out[case] = dict(reference=dict(power=ref.power.tolist(), times=ref.times.tolist(),
                                        sweeps=list(map(int, ref.step_iterations)),
                                        inner=int(ref.total_inner_iterations), wall=ref_wall))
        for tol in args.tols:
            walls = []
            for _ in range(args.repeat):
                res, wall = run(case, tol)
                walls.append(wall)
            err = res.power / ref.power - 1.0
            step = np.abs(np.diff(res.power)) / res.power[1:]
            stalled = int(np.sum(step < 1e-2 * step_ref))
            print(f"[{args.label}] case {case} tol {tol:.0e}: P(end) {res.power[-1]:.9f}  "
                  f"err(end) {err[-1]:+.2e}  max|err| {np.abs(err).max():.2e}  stalled {stalled}/{len(step)}  "
                  f"sweeps/step {np.mean(res.step_iterations):6.1f}  cg/step "
                  f"{res.total_inner_iterations / len(step):7.0f}  wall {min(walls):6.2f}s", flush=True)
            out[case][f"{tol:.0e}"] = dict(power=res.power.tolist(), err_end=float(err[-1]),
                                            max_err=float(np.abs(err).max()), stalled=stalled,
                                            sweeps=list(map(int, res.step_iterations)),
                                            inner=int(res.total_inner_iterations), wall=min(walls))
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(out, fh)
    print(f"[{args.label}] DONE", flush=True)
