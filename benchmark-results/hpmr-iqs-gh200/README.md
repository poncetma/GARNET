# Guarded adaptive IQS on GH200

Decision date: 2026-08-20. Re-checked 2026-10-29/30 under the
predictor–corrector IQS (audit item O3), with the decision unchanged.

Decision: **guarded adaptive IQS is the preferred method for production HP-MR
coupled transients**, with the `0.002/0.01/0.02` guard profile.
Fixed-cadence IQS remains a benchmark/reference mode, and full diffusion
remains the accuracy reference.

## Re-check under predictor–corrector IQS (2026-10-29, audit O3)

The 2026-10-29 audit made IQS a predictor–corrector: each shape interval is
replayed, and `iqs_predictor_tol` now rejects the current interval instead of
shortening the next one. The guard settings below were calibrated before that
change, so the calibration was repeated on the GH200 with the same command
(see [Workload](#workload)). It used three predictor tolerances, the
fixed-cadence control and a new full-diffusion reference, this time run to
200 s.

All five jobs ran one staged snapshot of commit `20967d3` plus that day's
uncommitted audit work ([PROVENANCE.txt](recheck-20261029/PROVENANCE.txt)).
They were submitted with `slurm/stage_and_submit_hpmr_iqs_gh.sh` to gh-daily.
Each `recheck-20261029/<job>-<case>/record.json` is the `--json` record of
`examples/hpmr_coupled_transient.py`, and the job logs are in
[recheck-20261029/logs](recheck-20261029/logs/). The comparison is
[summary-1s.json](recheck-20261029/summary-1s.json) and
[summary-10s.json](recheck-20261029/summary-10s.json), from
`tools/assess_hpmr_iqs_guards.py`.

The same `--dollars 0.3077` request now ends the drums at 90.345° and measures
**+0.309 $** inserted. In August it ended at 90.50° and +0.248 $. The likely
cause is the audit's exact drum absorber areas, which removed a −17% to +4%
bias in the cells the arc edge crosses and so changed the worth curve the
end-angle search uses. The transient is therefore stronger than in August
(peak P/P0 7.37 against 5.22), and the two calibrations are not comparable
number for number.

| quantity | guarded 0.005 | guarded 0.01 | guarded 0.02 | fixed cadence | full diffusion |
|---|---:|---:|---:|---:|---:|
| Slurm job, node | 241675, gpu001 | 241676, gpu001 | 241677, gpu002 | 241678, gpu002 | 241674, gpu002 |
| transient wall time | 4,564 s | 4,664 s | 3,602 s | 1,822 s | 45,638 s |
| IQS shape solves | 186 | 163 | 163 | 100 | — |
| adjoint solves | 50 | 46 | 46 | 21 | — |
| full-diffusion fallbacks | 12 | 12 | 12 | 0 | — |
| predictor interval rejections | 1 | 0 | 0 | not guarded | — |
| largest predictor disagreement | 0.47% | 0.64% | 0.64% | 6.7% (at 6 s) | — |
| peak P/P0 | 7.3762 at 37 s | 7.3762 at 37 s | 7.3762 at 37 s | 7.3742 at 37 s | 7.3746 at 37 s |
| final P/P0 (200 s) | 1.3203 | 1.3203 | 1.3203 | 1.3210 | 1.3241 |
| max error, 10 s samples | 0.288% (200 s) | 0.288% (200 s) | 0.288% (200 s) | 1.057% (10 s) | — |
| max error, every step outside reference stalls | 0.181% (71 s) | 0.181% (72 s) | 0.181% (72 s) | **8.155%** (4.95 s) | — |

The last row compares every accepted step, leaving out the 1 s windows where
the reference stalls (see [Reference reliability](#reference-reliability)).
Those windows are 1–2 s, 31–50 s and 165–200 s.

Representative values (guarded 0.02; the 0.005 and 0.01 runs agree with it to
5.3e-6):

| time | full diffusion | guarded IQS | guarded error | fixed IQS | fixed error |
|---:|---:|---:|---:|---:|---:|
| 1 s | 1.00000 | 1.00000 | 0.000% | 1.01142 | +1.142% |
| 3 s | 1.12574 | 1.12605 | +0.028% | 1.17612 | +4.475% |
| 5 s | 1.74538 | 1.74613 | +0.043% | 1.60473 | −8.059% |
| 10 s | 2.81328 | 2.81544 | +0.077% | 2.78354 | −1.057% |
| 20 s | 5.22037 | 5.22513 | +0.091% | 5.19134 | −0.556% |
| 40 s | 7.31198 | 7.31413 | +0.029% | 7.31720 | +0.071% |
| 60 s | 5.58607 | 5.57632 | −0.174% | 5.58900 | +0.052% |
| 110 s | 2.58163 | 2.57731 | −0.168% | 2.58160 | −0.001% |
| 160 s | 1.61852 | 1.61605 | −0.152% | 1.61767 | −0.053% |
| 200 s* | 1.32411 | 1.32029 | −0.288% | 1.32099 | −0.235% |

\* Inside a window where the reference stalls.

### Findings

- **The predictor tolerance does not bind.** At 0.01 and 0.02 the largest
  disagreement was 0.64%, nothing was rejected, and the two runs are bitwise
  identical. At 0.005 one interval was rejected. That run took 20 more
  residual-triggered updates, 23 more shape solves (+14%) and 4 more adjoint
  solves, and changed the power by at most 5.3e-6 relative. The guard is a
  working detector: on the unguarded schedule the disagreement reached 5.3%
  at 4 s and 6.7% at 6 s, where the fixed-cadence error is largest.
- **The residual guard drives the adaptivity, as in August.** All 12 fallbacks
  fall inside the drum ramp (t = 1.2, 1.5, 1.85, 2.2, 2.55, 2.85, 3.2, 3.5,
  3.85, 4.2, 4.55, 4.85 s), against 2 for the weaker August insertion. There
  were 63 residual-triggered shape updates. From about 32 s the residual crosses
  0.002 about 1.05 s after each scheduled update, so the shape cadence is in
  effect 1 s.
- **Guarded IQS is accurate through the ramp, where fixed cadence fails.**
  Guarded stays within 0.093% through 30 s. Fixed cadence is 1.1% high at 1 s,
  before the drum moves, because its first 2 s shape interval spans the ramp
  start. It is 8.2% low at the end of the ramp and still 1.1% low at 10 s.
- **After the peak, fixed cadence is closer.** Over 50–160 s, where the
  reference does not stall, fixed cadence stays within 0.054% at every step.
  Guarded moves away from the fixed-cadence history from 38 s, by −0.18% at
  50 s, and is at its furthest from the reference (−0.181%) at 72 s. It
  recovers slowly, to −0.15% at 160 s, despite twice the shape updates. The
  drift does not depend on `iqs_predictor_tol`. Its cause was not
  investigated; it stays within the August acceptance level.
- **Cost.** The fair wall-time comparison is gpu002, where guarded 0.02, fixed
  cadence and the reference ran at the same time. There, guarded took
  **1.98×** the fixed-cadence transient time; August measured 1.26× for the
  weaker insertion. The extra time is 63 more shape solves (18.3 s each against
  15.5 s), 25 more adjoint solves (9.1 s against 8.6 s) and 98 s of fallbacks.
  The 0.005 and 0.01 runs shared gpu001 and ran about 25% slower per solve.
  Their computations are identical to, or within 5.3e-6 of, the 0.02 run, so
  the wall-time spread between nodes is contention, not method. Full diffusion
  took 12.7× the guarded time.

### Reference reliability

This reference predates the fix for audit item O12: in three windows its power
stays flat between the 0.5 s feedback updates and catches up in jumps.
`tools/assess_hpmr_iqs_guards.py` reports them as `reference_stalls`:

| window | stalled steps | largest catch-up jump |
|---|---:|---:|
| 1–2 s (ramp start) | 6 | 1.1e-2 |
| 31–50 s (around the peak) | 165 | 4.1e-3 |
| 165–200 s (tail) | 327 | 9.2e-4 |

Inside them the reference is good only to about its jumps (0.4% near the peak,
0.1% in the tail). They are excluded from the comparison, and the 0.288%
maximum at the 200 s sample is not used for the decision.

### Decision

- Keep the `0.002/0.01/0.02` profile as the guarded HP-MR default. Where the
  reference is reliable, its maximum error is 0.181% under the
  predictor–corrector IQS, against 0.173% in August. Fixed cadence reaches
  8.2%. The cost is 2.0× fixed cadence on this transient, and full diffusion
  costs 12.7× more again.
- Keep `iqs_predictor_tol = 0.02`. A tighter value made no measurable
  difference here and cost 14% more shape solves at 0.005.

### Reproduce

```bash
bash slurm/stage_and_submit_hpmr_iqs_gh.sh          # 5 jobs on gh-daily
python tools/assess_hpmr_iqs_guards.py \
    /data/scratch/shared/poncet_m/garnet-hpmr-iqs/results --sample 1
# or, from the stored records:
python tools/assess_hpmr_iqs_guards.py \
    benchmark-results/hpmr-iqs-gh200/recheck-20261029 --sample 1
```

## Original calibration (2026-08-20)

Everything below is the August record, made with the explicit-shape IQS that
preceded the audit. The machine-readable record is
[`adaptive-fallback-r4-nz20.json`](adaptive-fallback-r4-nz20.json).

### Workload

The calibration used the serious 3-D HP-MR drum transient:

- 105,600 active cells at radial refinement 4 and 20 axial layers;
- the given 11-group ENDF/B-VIII cross sections and kinetics;
- all 12 drums moving from 90 to 90.50 degrees;
- measured insertion of +0.248 dollar;
- 4 s ramp starting at 1 s, followed through 200 s;
- `dt=0.05 s`, `dt_thermal=0.5 s`, and maximum `shape_dt=2 s`;
- one NVIDIA GH200 120GB GPU and one CPU task.

Guard settings:

```text
residual_tol       = 0.002
fallback_residual  = 0.01
iqs_predictor_tol  = 0.02
adjoint_every      = 5
```

The submitted command was equivalent to:

```bash
python -u examples/hpmr_coupled_transient.py \
  --refine 4 --nz 20 --groups 11 \
  --t-end 200 --dt 0.05 --dt-thermal 0.5 \
  --drum-from 90 --dollars 0.3076923076923077 \
  --t-start 1 --t-ramp 4 --n-angles 13 \
  --device gpu --profile \
  --quasistatic-shape-dt 2 --shape-method iqs --adjoint-every 5 \
  --residual-tol 0.002 --fallback-residual 0.01 \
  --iqs-predictor-tol 0.02
```

### Result

| quantity | guarded IQS | fixed-cadence IQS |
|---|---:|---:|
| Slurm job | 196799 | 196460 |
| transient | 3,343.2 s | 2,663.3 s |
| shape updates | 131 | 100 |
| IQS shape solves | 129 | 100 |
| adjoint solves | 29 | 21 |
| full-diffusion fallbacks | 2 | 0 |
| peak P/P0 | 5.2228 at 45 s | 5.2235 at 46 s |
| final P/P0 | 1.3580 | 1.3635 |

The guard added 29 IQS shape solves and two hard full-diffusion fallback steps.
Fallback work itself was only 16.4 s; most of the 25.5% transient overhead came
from the extra shape and adjoint solves. The independent predictor disagreement
never exceeded 0.59%, so the 2% predictor guard did not reduce the scheduled
interval. The adaptive behavior in this run was driven by the projected shape
residual.

### Accuracy decision

The full-diffusion reference job 196622 was preserved through physical time
110 s before cancellation. At 10 s sampling intervals through that point:

- guarded IQS maximum relative power error was **0.173%**;
- fixed-cadence IQS maximum relative power error was **5.708%**;
- guarded IQS stayed within 0.11% through the drum ramp and within 0.18%
  afterward;
- fixed-cadence IQS started 5.7% low and settled near 1% high late in the
  available reference.

Representative values:

| time | full diffusion | guarded IQS | guarded error | fixed IQS | fixed error |
|---:|---:|---:|---:|---:|---:|
| 10 s | 2.16912 | 2.17041 | +0.059% | 2.04530 | -5.708% |
| 20 s | 3.44761 | 3.45098 | +0.098% | 3.32260 | -3.626% |
| 60 s | 4.81154 | 4.80323 | -0.173% | 4.84990 | +0.797% |
| 110 s | 2.64905 | 2.64666 | -0.090% | 2.67650 | +1.036% |

This accuracy improvement is large enough to justify the 25.5% transient cost
increase. For this calibrated HP-MR problem, unguarded IQS should no longer be
used as the production answer merely because it is faster.

## Going-forward policy

- Use guarded adaptive IQS for production HP-MR coupled transients.
- Use the calibrated `0.002/0.01/0.02` guard profile as the starting point for
  comparable r4, 11-group drum maneuvers.
- Keep full diffusion as a shorter-window reference whenever the mesh,
  cross-section set, feedback model, or maneuver severity changes materially.
- Report guard settings, shape updates, fallbacks, adjoints, wall time, and the
  maximum error against the available reference.
- Use `--unguarded-iqs` only for fixed-cadence method studies and regression
  comparisons.
- Do not make these HP-MR-calibrated thresholds global low-level API defaults
  for unrelated reactor models.

## Limitations

Each configuration was run once, on shared nodes, so wall times carry about
25% node-to-node spread (see the cost finding above); solve counts are exact.
The August record did not serialize fallback or shape-update times. The
re-check records do (`fallback_times`, `shape_update_times`,
`shape_update_reasons`, `predictor_disagreement`, `shape_residual`).
