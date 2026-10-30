"""O12: slow steps on the exact-reference homogeneous medium of test_kinetics_reference.

    python benchmark-results/transient-step-convergence/homogeneous_slow_step.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "verification"))
from test_kinetics_reference import (GRID, KIN, SIGMA_A, _one_group,  # noqa: E402
                                     _step_problem, _step_reference)
from garnet import TransientSolver  # noqa: E402

BETA_TOT = KIN.beta.sum()


def run(dollars, t_end, dt, tol_step, **kw):
    perturbed = _one_group(dollars * BETA_TOT * SIGMA_A)
    t0 = time.perf_counter()
    res = TransientSolver(GRID, _step_problem(_one_group(), perturbed), KIN,
                          bc="reflective", device="cpu").solve(
        t_end=t_end, dt=dt, tol_step=tol_step, time_scheme="backward-euler", **kw)
    wall = time.perf_counter() - t0
    return res, wall, perturbed


def report(label, dollars=0.02, t_end=10.0, dt=0.05, tol_step=1e-6, **kw):
    res, wall, perturbed = run(dollars, t_end, dt, tol_step, **kw)
    tight, _, _ = run(dollars, t_end, dt, 1e-12, **kw)
    exact = _step_reference(_one_group(), perturbed, KIN, res.times)
    step = np.abs(np.diff(res.power)) / res.power[1:]
    stalled = int(np.sum(step[1:] < 1e-3 * np.median(np.abs(np.diff(tight.power)) / tight.power[1:])))
    err_tight = np.abs(res.power / tight.power - 1.0).max()
    print(f"{label:<34} stalled {stalled:4d}/{len(step)}  max|P/P_tight-1| {err_tight:.2e}  "
          f"P(end)/exact-1 {res.power[-1] / exact[-1] - 1:+.2e} (tight {tight.power[-1] / exact[-1] - 1:+.2e})  "
          f"sweeps {sum(res.step_iterations)}  inner {res.total_inner_iterations}  {wall:.2f}s")
    return res, tight


if __name__ == "__main__":
    for dollars in (0.002, 0.02, 0.1):
        print(f"--- {dollars} $ step, dt 0.05, BE, tol_step 1e-6")
        report("anderson 5 (default)", dollars)
        report("rebalance + picard", dollars, rebalance=True, anderson_depth=1)
