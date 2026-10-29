# Numerical implementation notes

These notes retain historical measurements and failure investigations moved
out of the numerical implementations during the readability review. They refer
to the original workloads and code state; they are not new benchmark results.
The source code keeps the equations and invariants needed to follow each method.

## Energy-group sweeps before amplitude rebalance

The diffusion transient defaults to six group subsweeps for upscattering
libraries. An amplitude rebalance assumes the swept flux satisfies the source
balance. Additional group sweeps settle the spectral distribution before that
amplitude correction.

The original 11-group HP-MR tolerance study compared three, six, and twelve
subsweeps. At refine 2, the converged powers were 1.3990072, 1.3990070, and
1.3990067 at `tol_step=1e-9`.

| Subsweeps | Outer sweeps, refine 2 / 3 | Inner CG work, refine 2 | Production-tolerance error |
|---|---|---|---|
| 3 | 248 / 247 | 43,371 | 2.5e-4 |
| 6 | 27 / 26 | 11,449 | 7.1e-5 |
| 12 | not recorded | 26,387 | 1.8e-4 |

For C5G7-TD1-1, raising the count from three to eight reduced outer sweeps
from 399 to 162 and inner CG work by a factor of 1.9. Errors changed from
1.24e-4 to 6.4e-5; tightly converged powers were 0.7814288 and 0.7814287.
Without rebalance, raising HP-MR subsweeps from three to six only reduced
outer sweeps from 1,347 to 1,080. The benefit depends on both mechanisms.

## SPH nonlinear solves

A non-positive trial region flux makes the logarithmic SPH residual undefined.
The recovery residual is proportional to `log(mu)`, pulling factors toward
`mu=1`. Its nonzero Jacobian gives Newton a recovery direction. The previous
constant penalty had a zero Jacobian: an HP-MR run stalled at a residual
magnitude of 106.826 while its line-search step changed without improving it.

The small HP-MR SPH problem had roughly twelve active region/group unknowns.
With production normalization, a Jacobian-free solve required 2,017 residual
evaluations without converging. A dense numerical Jacobian cost thirteen
coarse solves per construction and let the trust-region method resolve the
poor row scaling. This motivates offering dense and Krylov methods separately;
the appropriate choice depends on problem size and conditioning.

### Open issue: diffusion-coefficient scaling and flux normalisation

`garnet.sph.D_SCALING` selects how an SPH factor `mu` corrects the diffusion
coefficient; the default, `"inverse"`, is kept unchanged in the code for now:

| `D_SCALING` | Correction | Origin |
|---|---|---|
| `"inverse"` (default) | `D -> D / mu` | Transport consistency: `Sigma_t x mu` gives `D = 1/(3 Sigma_tr) x 1/mu` |
| `"direct"` | `D -> mu D` | Laboure et al. 2019, Eq. 7; Ortensi et al. |

The exact transform of the diffusion equation is `"direct"`. Substituting
`phi = mu phi~` into `-div(D grad phi) + Sigma phi = S`, with `mu` constant in
each region, gives `D~ = mu D` and `Sigma~ = mu Sigma` there. Under that
transform, multiplying every `mu` by one constant rescales the whole operator
and leaves `k` unchanged. The SPH flux normalisation fixes that constant, so
with `"direct"` the convention is nearly harmless.

With `"inverse"`, a common rescaling of `mu` multiplies the removal terms by
`c` and the leakage terms by `1/c`. That changes `k`, so the converged SPH
eigenvalue depends on the normalisation convention, not only on the reference.
The leaky three-region colorset below shows the effect. It is a 90 x 30 cm
PWR/poison slab, vacuum in x, with regions of unequal volume. The reference is
SP3, the coarse solve is diffusion, and the dense `hybr` method solves SPH.
The reference `k` is 1.0992249; uncorrected flux-volume homogenisation is
885 pcm high.

| `D_SCALING` | Sum-normalised `k` | Volume-weighted `k` | Normalisation spread | Sum-normalised error vs reference |
|---|---|---|---|---|
| `"inverse"` | 1.1228525 | 1.1193008 | 355 pcm | +2363 pcm |
| `"direct"` | 1.1159805 | 1.1157147 | 27 pcm | +1676 pcm |

Every run reports `converged=True` and satisfies `mu Phi = phi_ref` to
round-off. In this case SPH made the eigenvalue worse, from an 885 pcm error to
2363 pcm with the default settings. A `converged` flag shows only that the SPH
conditions were met. It does not show that the eigenvalue improved. The
observation appears to conflict with the measurement behind the `"inverse"`
default, recorded in the `D_SCALING` comment in `garnet/sph.py`, where
`"inverse"` closed four times the homogenisation error on another
leaky-colorset test. The default is therefore problem dependent and unresolved.

The recommended next step is to evaluate `"direct"` with per-group
flux-volume normalisation (`norm_weight` set to the region volumes) on the
verification colorsets and on HP-MR before changing the default. Until then,
compare SPH-corrected eigenvalues with the reference on the generation
geometry. The reproduction script is the audit script `sph_gauge.py`, which
calls `sph_correct(..., method="hybr")`. It compares `norm_weight=None` with
`norm_weight` set to the per-region volumes, once under each `D_SCALING`.

## Triangular equivalence currents

Partial-current reconstruction must use the same boundary law as the diffusion
operator. A previous reconstruction used `phi/4` independently of diffusion,
albedo, and the fitted boundary coefficient. The fit residual reached
1.2e-28 while the eigenvalue remained about 350 pcm away from the reference.

Per-cell discontinuity factors provide one unknown for a region's several
surface-current conditions. An HP-MR least-squares fit stalled at cost 0.678
with drum factors at their `exp(±1.2)` bounds. Per-face factors provide
independent unknowns for the ordered region interfaces.

Coarse and transport reference currents must also use the same face measure.
Omitting the edge length from coarse currents produced separate normalization
clusters for flux, interface current, and boundary current: approximately 31,
10, and 5. Current reconstruction now includes the face area explicitly.

## Transport sweep contractions

The level sweep uses persistent buffers for weighted angular reductions and
small corner-block products. The earlier broadcast-multiply implementation
was allocation-free and supported graph capture, but moved roughly three
times the data of a fused contraction. The fused kernels retain that storage
contract. Their broadcast fallbacks remain available, and the selected path
is fixed at setup so graph capture and replay use the same operations.

## Coupled-transient acceleration

The original 11-group HP-MR comparison used refine 3, one 50 ms step, and a
`tol=1e-10` rebalanced reference.

| Anderson depth | Rebalance | Tolerance | Power error | Time | Inner iterations |
|---|---|---|---|---|---|
| 1 | on | 1e-6 | -0.034% | 16.7 s | 112k |
| 1 | off | 1e-6 | -0.034% | 75.3 s | 535k |
| 5 | off | 1e-6 | -0.693% | 38.5 s | 273k |
| 5 | off | 1e-8 | -0.002% | 102.8 s | 652k |

Rebalanced Picard and plain Picard agreed within 1.2e-6 in that comparison.
A small change between Anderson-mixed iterates understated the residual on
the stiff, stateful group map; combining depth-five Anderson with rebalance
failed to converge. The transient therefore confirms apparent Anderson
convergence with an unaccelerated sweep.

The two-group demonstration behaved differently: Anderson took 2.0 s versus
15.0 s for rebalanced Picard, with answers agreeing within 1.5e-4. The coupled
driver retains a group-count heuristic and allows callers to override it
through `transient_kwargs`.

## Methods introduced by the October 2026 audit

The [code audit](qa/code_audit_2026-10-28.md) fixes added the methods below.
The docstrings named here hold the equations; these notes record the design
choice and the measurement behind it.

### Non-orthogonal finite-volume correction

`UnstructuredDiffusionSolver` (`garnet/mesh.py`) weights a face by
`A / (d_i/D_i + d_j/D_j)`, using normal distances from each centroid to the
face. The flux is the two-point flux between the points at those distances
along the face normal, so it is exact for a linear field. The tangential
offset term `w (t g_i + (1 − t) g_j) · d_tan` uses least-squares cell
gradients, with the Robin law closing boundary cells. This term is lagged into
the source of each inner solve, which keeps the CG operator symmetric and
matrix-free. On orthogonal meshes the setup detects zero offsets and returns
`None`, and the audit measured no change in results. Without the correction,
diagonal-cut squares were −316 pcm off at n = 64 and did not converge. The
unstructured HP-MR drum worth moved from 2462 to 2653 pcm and now matches the
structured-triangle result. Non-conforming faces are paired by overlap in
`_match_nonconforming`.

### Damped Neumann preconditioning

For SPD `A`, a degree-`d` Neumann polynomial maps an eigenvalue λ of `D⁻¹A` to
`1 − (1 − ωλ)^(d+1)`. Odd degrees are therefore indefinite once
`ω λ_max > 2`. The SP5/SP7 moment coupling pushes `λ_max(D⁻¹A)` to about 2.02,
and undamped SP7 degree 1 took about 350 CG iterations against 21 for plain
Jacobi. `linalg.jacobi_damping` sets `ω = 2/(λ_min + λ_max)` from 12 Lanczos
steps on `D^-1/2 A D^-1/2`. Both extremes are biased to the safe side, and
the estimate runs lazily once per preconditioner. `damping="auto"` is the
default in `neumann_preconditioner`; a float fixes ω, and `None` or `1.0`
restores the undamped splitting. Complex or non-positive diagonals, such as
the noise operator, are left undamped.

### Coupled SP3 Marshak vacuum

`SP3GroupOperator(marshak=None)` (`garnet/sp3.py`) adds the Brantley–Larsen
condition on vacuum faces. Its matrix `_SP3_MARSHAK_G` is the U-form SP3 matrix
`_SPN_G[1]` carried through the congruence `T = diag(1, 3)` that maps the
U-form block onto the `(Φ1, φ2)` block with its second row scaled by 5, so the
block stays symmetric and CG is unaffected. The per-face matrices come from
`spn._marshak_faces`, which now also covers active/inactive faces when
`mask_bc` is vacuum. The moment operators see those faces as reflective. The
coupled form needs a Cartesian `GroupOperator` grid without `hybrid_mask`;
SDP1, triangular, cylindrical and hybrid blocks fall back to the per-moment
`α = 1/2` term. The change moved Brantley–Larsen SP3 by +53 pcm (audit item
O8 tracks the remaining gap to the paper) and BIBLIS SP3 by +0.3 pcm.

### Adjoint transposes

The adjoint power iteration now uses the algebraic transpose wherever the
within-group operator is not symmetric:

| Operator | Transpose | Code |
|---|---|---|
| SDPN block | stencil with `A_ij → A_ji`, boundary `K → Kᵀ`; source and scalar-flux moment weights swapped | `SDPNGroupOperator.apply_transpose`, `SDPNEigenSolver._moment_weights` |
| r-z divergence form `W⁻¹A_w` | `W (W⁻¹A_w) W⁻¹`, the same operator conjugated by its row scale | `solver._conjugated_by_row_scale` |
| Triangular diffusion with discontinuity factors | stencil with the face weights `a` and `b` exchanged | `TriGroupOperator.apply_transpose` |
| Hybrid SP3/SDPN mask | mask moved from the source to the read-out | `_rhs` / `_phi` of `SP3EigenSolver` and `SDPNEigenSolver` |

Neumann preconditioners for the SDPN adjoint sweep the transpose; the
symmetric SPN companion preconditioner is reused unchanged. Before the fix,
SDP2 with Marshak gave an adjoint `k` 286 pcm off. The
`test_sdpn_adjoint.py` test compares with a dense left eigenvector.

### Eigenvalue stopping test

Plain power iteration contracts the error by the dominance ratio ρ per outer,
so a step of size `δ` leaves an error of about `δ ρ/(1 − ρ)`. In
`_PowerIterationSolver.solve` (`garnet/solver.py`), ρ̂ is the ratio of
successive three-outer sums of the source change. On the unaccelerated path,
used for `anderson_depth <= 1` and always for the adjoint, both tolerances are
multiplied by `1 − ρ̂`. They are floored at what the dtype and
`inner_rtol_floor` can resolve. Anderson iterates are not geometric, so that
test is unchanged. Two further guards apply. A step is accepted only after
inner solves at least as tight as `tol_source`, because a zero-iteration warm
start otherwise gives `dk = src_err = 0` on the adjoint. Three Anderson
restarts that fail to halve the source error drop the loop to plain power
iteration. Cost: adjoint shape errors are about 100 times smaller, at 25–80%
more outers.

### Time-step restarts and startup estimates

The event endpoint belongs to the old segment, so `TransientSolver` restarts
history on the step leaving the event (`restart_steps` in
`garnet/transient.py`). Crank–Nicolson has no multistep history, but its
carried field holds the `(1 − w) Fⁿ` half of the trapezoid. At a restart,
`ThetaMethod.start(u, increment=...)` re-seeds it from the post-event fields;
`Ψ = u` alone measured order 0.94. Fixed-step BDF3–6 stay second order
because of their full-width startup (audit item O1; graded steps measured
3.25). The adaptive controller's first step after t=0 or a restart uses
`startup_rates`, an explicit-Euler prediction evaluated with
`problem_at(t + 1e-6·dt)` and the accepted feedback state. This costs one
field and operator setup per startup.

### Predictor–corrector IQS

See [quasistatic_acceleration_plan.md](quasistatic_acceleration_plan.md#predictor-corrector-iqs-2026-10-29-audit-fix).

### Exact drum absorber area

`model.annular_sector_fraction` intersects each triangle with the annular
sector in closed form. A convex wedge is clipped by two half-planes, a reflex
wedge is the disk minus the opposite wedge, and each piece is intersected
with the outer and inner circles. The former barycentric sampling was biased
by −17% to +4% per cell. HP-MR drum worths moved by +6 to +7 pcm.
