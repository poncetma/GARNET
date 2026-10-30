"""Compare HP-MR IQS guard runs against the full-diffusion reference (audit O3).

    python tools/assess_hpmr_iqs_guards.py RESULTS_DIR --output summary.json

``RESULTS_DIR`` holds one ``<job>-<case>/`` directory per case, as written by
``slurm/run_garnet_hpmr_iqs_gh.sh``: ``record.json`` from
``examples/hpmr_coupled_transient.py --json`` and, for the full-diffusion
``reference`` case, ``progress.txt`` (``t P/P0`` per accepted step). The
reference may be incomplete (a time-limit cut); comparisons stop at its last
sampled time, as in the 2026-08 calibration.

Per case it reports the maximum relative power error against the reference at
``--sample`` second intervals (10 s, as calibrated), the error at each sample,
transient wall time, shape/adjoint/fallback work, fallback and shape-update
times with reasons, and the largest IQS predictor disagreement.

It also flags where the reference itself is unreliable. Before the
2026-10-30 fix (audit O12) the full-diffusion step could pass its stopping
test on the first sweep when the true per-step change was small, and was then
accepted almost unchanged. The power history then stalls between feedback
updates and catches up in jumps. ``reference_stalls`` counts, per sample
window, the accepted steps whose relative power change is below
``--stall-tol`` (after the first step that moves). Errors inside those windows
are only resolved to about the size of the catch-up jumps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_case(directory):
    record = directory / "record.json"
    if record.exists():
        data = json.loads(record.read_text())
        return data, np.asarray(data["times"]), np.asarray(data["power"])
    progress = directory / "progress.txt"
    if progress.exists() and progress.stat().st_size:
        t, p = np.loadtxt(progress, ndmin=2).T
        return None, np.r_[0.0, t], np.r_[1.0, p]
    return None, None, None


def at(times, power, t):
    """Power at t: the accepted step ending there, else linear interpolation."""
    i = int(np.argmin(np.abs(times - t)))
    if abs(times[i] - t) <= 1e-9 * max(1.0, t):
        return float(power[i])
    return float(np.interp(t, times, power))


def stalled_windows(times, power, sample, tol):
    """Per sample window: accepted steps with |dP|/P < tol, and the largest jump."""
    change = np.abs(np.diff(power)) / np.abs(power[1:])
    t = times[1:]
    moving = np.flatnonzero(change >= tol)
    if moving.size == 0:
        return []
    out = []
    for lo in np.arange(0.0, t[-1], sample):
        m = (t > lo) & (t <= lo + sample) & (t >= t[moving[0]])
        stalled = int(np.sum(change[m] < tol))
        if stalled:
            out.append(dict(t_from=float(lo), t_to=float(lo + sample), stalled=stalled,
                            steps=int(m.sum()), largest_jump=float(change[m].max())))
    return out


def summarize(results, sample, stall_tol=2e-5):
    cases = {}
    for directory in sorted(Path(results).iterdir()):
        if directory.is_dir() and "-" in directory.name:
            job, case = directory.name.split("-", 1)
            cases[case] = (job, *load_case(directory))
    if "reference" not in cases or cases["reference"][2] is None:
        raise SystemExit("no reference history found")
    _, ref_record, ref_t, ref_p = cases["reference"]
    stalls = stalled_windows(ref_t, ref_p, sample, stall_tol)
    rows = {}
    for case, (job, record, times, power) in cases.items():
        row = dict(job=job, complete=record is not None)
        if times is None:
            rows[case] = dict(row, missing=True)
            continue
        horizon = min(times[-1], ref_t[-1])
        samples = np.arange(sample, horizon + 1e-9, sample)
        errors = [at(times, power, s) / at(ref_t, ref_p, s) - 1.0 for s in samples]
        i_peak = int(np.argmax(power))
        row.update(end_time=float(times[-1]), compared_through=float(horizon),
                   peak_power=float(power[i_peak]), peak_time=float(times[i_peak]),
                   final_power=float(power[-1]),
                   samples=[dict(t=float(s), power=at(times, power, s),
                                 reference=at(ref_t, ref_p, s), relative_error=float(e))
                            for s, e in zip(samples, errors)])
        if errors and case != "reference":
            worst = int(np.argmax(np.abs(errors)))
            row.update(max_abs_relative_error=float(abs(errors[worst])),
                       max_error_time=float(samples[worst]))
            # Every accepted step, outside the windows where the reference stalls.
            keep = (times > 0.0) & (times <= horizon)
            for w in stalls:
                keep &= ~((times > w["t_from"]) & (times <= w["t_to"]))
            if keep.any():
                every = power[keep] / np.interp(times[keep], ref_t, ref_p) - 1.0
                i = int(np.argmax(np.abs(every)))
                row.update(max_abs_relative_error_unstalled=float(abs(every[i])),
                           max_error_time_unstalled=float(times[keep][i]))
        if record is not None:
            counters = record.get("counters", {})
            disagreement = record.get("predictor_disagreement") or []
            row.update(
                mode=record.get("mode"), insertion=record.get("insertion"),
                transient_seconds=record.get("transient_seconds"),
                steps=record.get("steps"),
                shape_updates=counters.get("shape_updates"),
                iqs_shape_solves=counters.get("iqs_shape_solves"),
                adjoint_solves=counters.get("adjoint_eigen_solves"),
                fallbacks=counters.get("full_diffusion_fallbacks"),
                interval_rejections=counters.get("iqs_interval_rejections"),
                fallback_times=record.get("fallback_times"),
                shape_update_reasons=_count(record.get("shape_update_reasons") or []),
                max_predictor_disagreement=(max(disagreement) if disagreement else None),
                predictor_tol=record.get("args", {}).get("iqs_predictor_tol"))
        rows[case] = row
    return dict(sample_seconds=sample, reference_through=float(ref_t[-1]),
                reference_complete=ref_record is not None, stall_tol=stall_tol,
                reference_stalls=stalled_windows(ref_t, ref_p, sample, stall_tol),
                cases=rows)


def _count(reasons):
    out = {}
    for reason in reasons:
        out[str(reason)] = out.get(str(reason), 0) + 1
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="directory of <job>-<case>/ result directories")
    parser.add_argument("--sample", type=float, default=10.0, help="comparison interval, s")
    parser.add_argument("--stall-tol", type=float, default=2e-5,
                        help="relative per-step power change below which a "
                             "reference step counts as stalled")
    parser.add_argument("--output", help="write the summary JSON here")
    args = parser.parse_args()
    summary = summarize(args.results, args.sample, args.stall_tol)
    print(f"reference through {summary['reference_through']:.2f} s "
          f"({'complete' if summary['reference_complete'] else 'partial'})")
    for w in summary["reference_stalls"]:
        print(f"  reference stalls {w['t_from']:5.0f}-{w['t_to']:<5.0f} s: "
              f"{w['stalled']:3d}/{w['steps']} steps, largest jump {w['largest_jump']:.1e}")
    print(f"{'case':<15} {'max |err|':>9} {'at [s]':>7} {'unstalled':>9} {'at [s]':>7} "
          f"{'wall [s]':>9} {'shape':>6} "
          f"{'adjoint':>7} {'fallback':>8} {'peak P/P0':>10} {'final':>8}")
    for case, row in summary["cases"].items():
        if row.get("missing"):
            print(f"{case:<15} (no history yet)")
            continue
        err = row.get("max_abs_relative_error")
        every = row.get("max_abs_relative_error_unstalled")
        print(f"{case:<15} {'' if err is None else f'{100 * err:8.3f}%':>9} "
              f"{row.get('max_error_time', float('nan')):7.0f} "
              f"{'' if every is None else f'{100 * every:8.3f}%':>9} "
              f"{row.get('max_error_time_unstalled', float('nan')):7.2f} "
              f"{row.get('transient_seconds') or float('nan'):9.1f} "
              f"{row.get('iqs_shape_solves') or 0:6d} {row.get('adjoint_solves') or 0:7d} "
              f"{row.get('fallbacks') or 0:8d} {row['peak_power']:10.4f} {row['final_power']:8.4f}")
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
