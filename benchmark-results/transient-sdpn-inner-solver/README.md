# Transient SDPN inner solver: BiCGStab vs GMRES (audit item O7)

Measured 2026-10-29 on CPU (merlin7 `cn000`, one thread per process).
This decides the inner Krylov solver that `TransientSDPNSolver` uses on a
non-symmetric block. That is SDP3 whenever a vacuum face rules out the
congruence basis; symmetric(-ized) blocks run CG and are not affected.

**Decision: keep BiCGStab as the default.** GMRES is now available through
`linear_solver="gmres"`. The steady `SDPNEigenSolver` moved to GMRES because
BiCGStab's irregular inner error stalled its Anderson outer loop on BIBLIS SDP3
when the inner tolerance was `0.1·src_err`. The transient sets its inner
tolerance to `1e-3` of the source change, and BiCGStab converged every step in
every run below, with the same trajectory error as GMRES and 27–32% less CPU
time. The eigen solver's 0.1 tightening for an explicit BiCGStab bought no
accuracy here, so the transient does not apply it.

## Setup

- **Brantley–Larsen-style core.** `_bl_problem(n)` from
  [test_sdpn_symmetric.py](../../tests/verification/test_sdpn_symmetric.py):
  one group, fuel bars in moderator, vacuum on the high x and y faces, n × n
  cells. The fuel `sigma_a` is lowered by 0.5% for t > 0. One precursor family
  (β = 0.0065, λ = 0.08 s⁻¹, v = 2.2e5 cm/s). It runs to t = 0.5 s with
  dt = 0.01 s.
- **BIBLIS.** `build_biblis(r)`: 17r × 17r cells, two groups, vacuum in-plane.
  Material 1's thermal `sigma_a` is lowered by 0.3% for t > 0. One family
  (β = 0.0065, λ = 0.08 s⁻¹, v = 1.25e7 / 2.5e5 cm/s). It runs to t = 0.2 s
  with dt = 0.01 s.
- **Common settings.** SDP3 (`order=3`, non-symmetric path asserted), steady
  start at `tol_k=1e-10`, `tol_source=1e-9`, default Anderson depth. The
  **reference** is GMRES at `tol_step=1e-11` on the same mesh; on the
  Brantley–Larsen core at n = 20, 1e-10 and 1e-11 differ by 1.4e-8 in P(t_end).
  The error column is the maximum relative power error over all steps.
- **Timing.** Best of 3 identical runs, one process per configuration. The
  node was shared (load average about 16 on 256 cores); the three runs of each
  configuration were within 4% of each other.
- **Variant switch.** Only the solver's `_linsolve` was swapped; nothing else
  differed from commit `20967d3`. `bicgstab-tight` is BiCGStab at `0.1·rtol`.

Inners are operator applies, comparable across solvers.

## Brantley–Larsen-style core (1 group)

| n | Solver | `tol_step` | max \|ΔP\|/P | Inners | t [s] | ms/apply |
|---|---|---|---|---|---|---|
| 20 | BiCGStab | 1e-6 | 1.33e-4 | 13370 | 2.17 | 0.162 |
| 20 | BiCGStab ×0.1 | 1e-6 | 1.44e-4 | 14986 | 2.40 | 0.160 |
| 20 | GMRES | 1e-6 | 9.28e-5 | 10681 | 3.07 | 0.287 |
| 20 | BiCGStab | 1e-8 | 7.03e-7 | 20383 | 3.26 | 0.160 |
| 20 | BiCGStab ×0.1 | 1e-8 | 1.32e-6 | 23049 | 3.61 | 0.157 |
| 20 | GMRES | 1e-8 | 1.41e-6 | 16227 | 4.69 | 0.289 |
| 40 | BiCGStab | 1e-6 | 4.34e-5 | 29969 | 6.36 | 0.212 |
| 40 | BiCGStab ×0.1 | 1e-6 | 1.39e-4 | 30109 | 6.40 | 0.213 |
| 40 | GMRES | 1e-6 | 1.33e-4 | 21140 | 8.75 | 0.414 |
| 40 | BiCGStab | 1e-8 | 1.72e-6 | 43422 | 9.21 | 0.212 |
| 40 | BiCGStab ×0.1 | 1e-8 | 1.27e-6 | 46013 | 9.58 | 0.208 |
| 40 | GMRES | 1e-8 | 1.02e-6 | 33897 | 13.63 | 0.402 |

k0 is identical across solvers: 0.8199421 at n = 20 and 0.8081683 at n = 40.
P(0.5 s) is 1.69267 and 1.67600.

## BIBLIS (2 groups)

| Mesh | Solver | `tol_step` | max \|ΔP\|/P | Inners | t [s] | ms/apply |
|---|---|---|---|---|---|---|
| 34×34 | BiCGStab | 1e-6 | 2.27e-4 | 4192 | 0.99 | 0.236 |
| 34×34 | GMRES | 1e-6 | 3.06e-4 | 4196 | 1.35 | 0.322 |
| 34×34 | BiCGStab | 1e-8 | 1.00e-6 | 10582 | 2.26 | 0.213 |
| 34×34 | GMRES | 1e-8 | 1.36e-6 | 10993 | 3.26 | 0.297 |
| 68×68 | BiCGStab | 1e-6 | 2.80e-4 | 5705 | 2.24 | 0.393 |
| 68×68 | GMRES | 1e-6 | 3.80e-4 | 5261 | 3.16 | 0.601 |

k0 is 1.0293732 on 34×34 and 1.0265597 on 68×68. P(0.2 s) is 1.11311 and
1.11370. A probe with a 3% absorption step (super-prompt-critical, P(0.2 s)
≈ 461) also converged with both solvers.

## Reading

- **Accuracy is set by `tol_step`, not by the solver.** At the default `1e-6`
  every variant sits at 4e-5 to 4e-4. Which solver is closer changes from case
  to case, and none is consistently more accurate. At `1e-8` all are near 1e-6.
- **GMRES saves applies but not time.** It needs between 30% fewer and 4% more
  applies, but each costs 1.4–2.0× more on CPU because of the restart-30
  Arnoldi basis, so it is 36–48% slower overall. This is a CPU measurement: on a GPU the
  per-apply balance differs and is not measured here.
- **Tightening BiCGStab ×0.1** adds up to 13% applies with no accuracy gain.
- A regression test
  (`test_transient_explicit_gmres_matches_default_bicgstab`) checks that the
  `linear_solver` swap leaves the trajectory unchanged to `1e-6` at
  `tol_step=1e-9`; the measured difference is 9.6e-8.

Data: [brantley-larsen-style.json](brantley-larsen-style.json),
[biblis.json](biblis.json). Each file records the environment and setup and,
per configuration, the full power history, inner count and all three timings.
The timing drivers were throwaway scripts; the setup above reproduces them.
