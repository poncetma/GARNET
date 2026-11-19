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
The recovery residual is proportional to the distance from the starting
factors, so its Jacobian gives Newton a direction back. A constant penalty has
a zero Jacobian: an HP-MR run with one stalled at a residual of 106.826 while
its line-search step changed without improving it.

`sph_correct` keeps the best point it evaluated, with that point's own
eigenvalue, and sets `converged` from the measured residual at the returned
factors, never from a SciPy flag. A solver flag can be false at a residual of
1e-11 or true at a step-size stop. `SphResult.residual` holds the value.

The fixed-`k` formulation, `normalization="none"`, solves with MINPACK `hybr`
first. If that misses the tolerance it runs a Levenberg–Marquardt least-squares
solve from the best point, then `hybr` again from the least-squares endpoint.
Least squares alone stalled at a residual of 9e-6 on a case where the restart
reached 6e-15, and `hybr` alone can wander into infeasible factors. When no
solution exists, the least-squares floor is what the result reports. For many
unknowns, `method="jfnk"` avoids the dense Jacobian of one coarse solve per
unknown.

### Diffusion-coefficient convention and normalization (resolved 2026-11-13)

Until the SPH audit of 13 November 2026
([report](qa/sph_audit_2026-11-13.md)) the default correction was
`D -> D / mu`, with one normalization constant shared by all regions and
groups. Both were defects, and both are fixed on branch `SPH-dev`:

| Item | Before | Now |
|---|---|---|
| Diffusion coefficient | `D / mu` | `mu D`, the exact transform of `phi = mu Phi` (IGE-335 Eq. 3.43; Labouré et al. 2019 Eq. 7; Kumar et al. 2026 Eq. 2) |
| Normalization | one constant, unit or production weights | one flux–volume constant per group, `normalization="flux-volume"` |
| Vacuum-bounded generation geometry | normalized, `k` not reproduced | `normalization="none"`: `k` fixed at the reference, no normalization (Labouré et al. 2019 §3.2; Griffin) |
| Restriction to the fuel | none | `sph_regions` mask; other regions keep `mu = 1` |
| Transients | `1/v` | `sph_kinetics`: per-material `v/mu` |

`D / mu` breaks the per-group scaling symmetry of the corrected operator.
On a reflective colorset, multiplying one group's converged factors by 1.3
moves `k` by 830 to 980 pcm with `D / mu`, and by 0.00 pcm with `mu D`. A
normalization then picks one of many operators that all reproduce the
reference, with different eigenvalues. The table that this note used to carry
as an open issue is the symptom:

| `D_SCALING` | Sum-normalized `k` | Volume-weighted `k` | Spread | Sum-normalized error |
|---|---|---|---|---|
| `"inverse"` (old default) | 1.1228525 | 1.1193008 | 355 pcm | +2363 pcm |
| `"direct"` | 1.1159805 | 1.1157147 | 27 pcm | +1676 pcm |

The leaky three-region colorset is vacuum-bounded, so no normalized
formulation reproduces its `k`; the remaining 27 pcm is the vacuum boundary.
`tools/assess_sph_gauge.py` reproduces the table to every printed digit with
a copy of the old solver (part A) and runs the gauge, vacuum, transfer and
generation-time checks (parts B–E).

`d_scaling="inverse"` is still accepted by `_scale_material` to reproduce old
results. On vacuum-bounded problems it can fit the reference where `mu D`
cannot, because dividing `D` changes the diffusion length inside a region,
which acts as a hidden leakage factor. That is why it looked better on the
leaky-colorset test that kept it as the default in July 2026.

### When the fixed-`k` formulation has no solution

With `mu D` a region-uniform factor leaves the flux shape inside the region
unchanged. At a vacuum face it only moves the effective Marshak coefficient
between `alpha/mu -> 0` (reflective) and infinity (zero flux). Interfaces see
a flux discontinuity of ratio `mu_L/mu_R`. If the coarse model cannot produce
the reference region fluxes and `k` with these levers, no solution exists, and
`sph_correct` returns its least-squares floor with `converged=False`. Two
measured cases:

| Case | Fixed-`k` result |
|---|---|
| A corrected region at the vacuum face mixes poison stripe and fuel (three-region colorset, regions cut at 30 and 45 cm) | no solution; floor 7e-3 to 1e-2 and `k` off by 760 to 1060 pcm, depending on the method |
| Same colorset with regions aligned to the stripe | exact, residual 2e-15 |
| Water–fuel–water slab, fuel-only SPH, 2.5, 3.75 and 5 cm finite-volume nodes | exact |
| Same slab, 7.5 cm nodes | no solution, with or without the reflector corrected |

The 7.5 cm nodes are 2.6 thermal diffusion lengths of the water reflector
(2.85 cm). At that size even the uncorrected homogenized error changes sign:
−200 pcm on the fine mesh, +89 pcm on the 7.5 cm nodes. The coarse mesh must resolve the reflector, which is
also why Griffin meshes reflector and drums explicitly and corrects the fuel
only (Kumar et al. 2026 §3.5.1).

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

## Wielandt-shifted eigenvalue iteration

`solve(wielandt_shift=δk)` in `_PowerIterationSolver` (`garnet/solver.py`)
replaces each outer after the first with

    (M − λ_s F) Ψⁿ⁺¹ = (λⁿ − λ_s) F Ψⁿ,   λ = 1/k,   λ_s = 1/k_s,
    λⁿ⁺¹ = λ_s + (λⁿ − λ_s) Sⁿ/Sⁿ⁺¹,

with S the fission-source total. The fundamental-mode error contracts by
`(λ₀ − λ_s)/(λ₁ − λ_s)` per outer instead of `k₁/k₀`. The shifted system
couples every group through scattering and fission. `_ShiftedMultigroupSystem`
solves it with FGMRES on the stacked group states, so every angular family
(diffusion, SP3, SPN, SDPN, triangular prisms, r-z) and the adjoint share one
code path through `_rhs`, `_phi` and `_linear_components`. The right
preconditioner is a Gauss–Seidel sweep over the groups of `M` alone, with
inexact (`inner_rtol = 0.1`) within-group solves by the solver's own Krylov
method. The forward problem sweeps fast to thermal and the adjoint thermal to
fast; problems with upscatter get two sweeps. Fission stays with FGMRES: with
`P ≈ M` the preconditioned spectrum is `1 − kᵢ/k_s`, one isolated value near
`1 − k₀/k_s` and the rest towards 1. The first sweep starts from zero, and the
Krylov solvers then skip the initial `A·0` (`x0=None`). The FGMRES tolerance
is `min(0.1, 0.1·src_err)` relative to the right-hand side.

`wielandt_mode` chooses how `k_s` follows the iteration. `"fixed"` is the
classic method: `k_s = k¹ + δk` after the first outer, re-anchored only if
`k` climbs past half the offset. `"adaptive"` sets `k_s = k + δ` at every
outer and steers `δ` by `√(n/6)`, where `n` is the FGMRES count of the
previous shifted solve. `δ` stays within `[max(1e-4, 10·|Δk|), 10·δk]`.

Two alternatives were measured and rejected (HP-MR r2/nz20, 11 groups, and
IAEA r4, login node):

- A shift that shrinks with the outer change (`δ = 10·k·src_err`, Rayleigh
  quotient style) needed 6–7 times more within-group iterations than a
  constant offset. A small shift crowds every mode with `kᵢ` near `k₀`
  towards zero in `1 − kᵢ/k_s`, not only the fundamental, so the
  scatter-only preconditioner leaves FGMRES many slow modes.
- An FGMRES tolerance relative to the unshifted source, looser by
  `λ/(λ − λ_s)`, raised the within-group iterations from 3670 to 11777 at
  δk = 0.1. The noisier solves cost Anderson more outers than they saved.

Within-group iterations at `tol_k = 1e-7`, `tol_source = 1e-6`, Anderson depth
8 on the forward problem. The tight HP-MR reference is `k = 1.11731834`.

| Case | Plain | Fixed, δk = 0.3 | Adaptive, from 0.3 |
|---|---|---|---|
| HP-MR forward | 7466 (0.45 pcm early stop); 17996 converged | 3044 | 2998 |
| HP-MR adjoint | 21474 | 4962 | 4917 |
| IAEA forward | 700 | 1102 | 1543 |
| IAEA adjoint | 6869 | 4108 | 2458 |

On the HP-MR most of the gain comes from solving the upscatter in every outer.
A shift of 100 (`k_s ≫ k`) still needs only 3939 iterations, and δk = 0.3
removes another 30%. Plain power iteration sweeps the groups once per outer,
so its per-outer change understates the upscatter error. At the default
tolerances it stops 5×10⁻⁶ (0.45 pcm) short on that mesh. On two-group cores
without upscatter, Anderson-accelerated power iteration already resolves the
source in about 45 sweeps, while each shifted solve costs about ten. The shift
then pays only on the adjoint, which runs without Anderson. FGMRES keeps
`2·restart + 4` copies of the multigroup state, about 29 GB at restart 20 on
the 84-million-unknown HP-MR. The GH200 runs of that mesh therefore use
restart 8.

On exclusive EPYC 7742 cores and a GH200, the HP-MR from 0.44 to 84 million
unknowns went from about 250 outers to 10. The fixed shift (δk = 0.3) solved
2.2–3.5 times faster than the power iteration and 9–25% faster than the
adaptive one on every mesh, and moved `k` by at most 0.005 pcm. `"fixed"` is
therefore the default. At default tolerances on r2/nz30 its iteration error is
0.003 pcm and 1.8×10⁻⁷ in node power. See
`benchmark-results/garnet-vs-femffusion/` (Table "HP-MR steady").

The two-group cores were checked on the GH200 at the meshes that decide that
report: Langenbuch r12 (4.0 million unknowns) and IAEA r24 (127 million).
Anderson power iteration took 0.33 s and 34.1 s. The fixed (0.3, 0.1) and
adaptive shifts were 1.36–1.58 times slower. They halved the outers, but the
shifted solves cost more within-group iterations (IAEA 4263 against 3197).
Their FGMRES basis took GPU memory from 40 to 76 GB at IAEA r24 (restart 12).
Leave the shift off for two-group cores without upscatter.

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

The general SPN/SDPN block applies the same coupled condition through
`SDPNGroupOperator(boundary_g=...)`. Since 30 October 2026 it is the default
of every SPN/SDPN eigen, transient and noise solver wherever
`spn._marshak_applies` holds: a Cartesian, non-cylindrical `GroupOperator`
grid with a vacuum face (or a vacuum active-mask boundary) and no hybrid mask.
`marshak_vacuum=False` selects the per-moment Robin term, which is 52–73 pcm
low on the leaky Brantley–Larsen core and depends on the moment variables it
is applied to. The coupled term makes every SDPN block non-symmetric, so SDP2
loses its diagonal-similarity CG path on such problems; the SPN family's `g`
is symmetric and keeps CG. The dedicated SDP1 block has no coupled term and
keeps Robin. The transient solver passes its choice to its steady-state
solve, so an unperturbed core starts in equilibrium.

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
because of their full-width startup. For BDF3 and BDF4, `bdf_startup="graded"`
replaces it with the steps of `graded_bdf_widths` (audit item O1; measured
orders 2.97 and 4.19, and 3.14 for BDF3 across an interior restart). BDF5 and
BDF6 diverge on those growing steps (errors up to 1e2 for BDF6 at growth
ratios from 1.5 down to 1.1) and keep the cap. The adaptive controller's first step after t=0 or a restart uses
`startup_rates`, an explicit-Euler prediction evaluated with
`problem_at(t + 1e-6·dt)` and the accepted feedback state. This costs one
field and operator setup per startup.

### Predictor–corrector IQS

See [quasistatic_acceleration_plan.md](quasistatic_acceleration_plan.md#predictor-corrector-iqs-2026-10-29-audit-fix).

### Anisotropic scattering in tri-S_N (2026-11-16)

`TriSNTransportSolver(scattering_order=L)` expands the scattering source in
real spherical harmonics of the 2-D problem: Schmidt semi-normalized
associated Legendre functions with the polar axis along z, keeping only the
(l, k) with l + k even (the others are odd in the polar cosine and vanish in
2-D). Per group the solver carries the flux moments `S_j = Σ_m w_m R_j(Ω_m) ψ_m`
and rebuilds the angular source as
`q_m = Σ_j (2 l_j + 1) R_j(Ω_m) Σs,l_j S_j`, so a cell's angular source is a
flat value per ordinate (SCB: per ordinate and corner). On construction the
quadrature's Gram matrix of the basis is checked against `diag(1/(2l+1))` and
a quadrature that cannot integrate the requested order raises `ValueError`.
Moments use the Gauss–Seidel ordering of the scalar flux: down-scatter from
the current outer, up-scatter from the previous one. DSA builds its diffusion
operator from `Σt − Σs1(g→g)`. After a CMFD rebalance the higher moments are
rescaled with the cell's scalar-flux ratio. The volume-mixed second material
is blended into each cell's moment matrices like every other cross section.

The within-group P0 scattering is Σt − removal in every scheme. For a material
with an explicit `total` it is kept even when negative: out-scatter
transport-corrected data of a hydrogenous moderator have it below zero, and
clipping it at zero created neutrons wherever it applied (+1470 pcm on the
refine-16 HP-MR core). With Σt = 1/(3D) a negative value is still clipped,
since that total is only an estimate. A negative within-group term leaves the
sweep, DSA (`Σt − Σs,gg` > Σt) and CMFD well posed.

Verification (`tests/verification/test_tri_sn_anisotropic.py`): zero higher
moments reproduce the isotropic solve to 1e-11 in `k` on both engines and both
schemes; the LU and level-scheduled engines agree; a periodic infinite medium
keeps `k = k∞` with P2 scattering; and in a bare one-group core with
Σs1/Σt = 0.4 the gap between P1 scattering and the diagonal transport
correction is 7.0 %, 3.3 % and 1.8 % of the anisotropy effect at 1, 0.5 and
0.25 cm cells (SCB). It halves with the cell size, so it is the first-order
error of the cell-flat anisotropic source and not a difference in the
physics. On the HP-MR pin-resolved core the GPU and CPU paths give the same
`k` to 1e-7.

**Analytic verification.** The anisotropic path reproduces the critical
dimensions of the LA-13511 set (Sood, Forster & Parsons 1999;
`garnet/benchmarks/sood.py`, `tools/verify_sn_sood.py`,
`tests/verification/test_tri_sn_sood.py`).

- **Cylinders.** Bare one-group cylinders with P1 scattering, 36 and the
  strongly forward 37 (negative kernel near μ = −1), extrapolate in mesh and
  quadrature to −5.5 and −3.9 pcm. The isotropic control 13 gives −5.8 pcm.
- **Slabs.** The triangular solver has no reflective boundary, so a slab is
  built from fuel columns followed by a black, non-scattering layer of 40
  optical thicknesses on a periodic grid. Each face then sees an exact vacuum,
  and the slab is infinite along the strips.
  - P1 and P2 one-group slabs track their isotropic controls; all are within
    ±35 pcm, which is the 2-D product quadrature on optically thin slabs.
  - The two-group P1 slab 71 converges to −35 pcm.
- **Independent check of problem 71.** `tools/slab_sn_1d.py` is a 1-D
  Gauss–Legendre diamond-difference eigenvalue solver that shares no code with
  `tri_sn`, after OpenSN's `transport_anisotropic_scattering_verification`
  test. It puts 71 at −31.5 pcm at S256 and every other slab within 3 pcm of
  k = 1, so 71's residual is in its tabulated data.
- **Discrimination.** Without the anisotropy the same problems are 2900 to
  21 000 pcm off.

### Exact drum absorber area

`model.annular_sector_fraction` intersects each triangle with the annular
sector in closed form. A convex wedge is clipped by two half-planes, a reflex
wedge is the disk minus the opposite wedge, and each piece is intersected
with the outer and inner circles. The former barycentric sampling was biased
by −17% to +4% per cell. HP-MR drum worths moved by +6 to +7 pcm.

### Adjoint triangular S_N (2026-11-18)

`TriSNTransportSolver(..., adjoint=True)` solves the adjoint k-eigenproblem.
With psi~(r, O) = psi+(r, -O), the adjoint equation is a forward-type equation
with three changes:
- every group-to-group transfer is transposed, in every Legendre order;
- the fission spectrum and production are swapped (spectrum nu*Sigma_f,
  production chi in fissile cells);
- the total and within-group terms are unchanged.

The kernel depends only on O.O', and the symmetric quadrature maps vacuum
boundaries to vacuum boundaries. The transform acts on the solver's arrays
before the operators are factored, so the sweeps, DSA, CMFD and both engines
are shared with the forward solve.

For the step scheme this is the exact discrete adjoint. The streaming
matrix of direction O, transposed, is that of -O, since a closed cell's
outgoing and incoming face terms are equal. The scattering moments are
symmetric in the quadrature-weighted inner product.

The verification (`tests/verification/test_tri_sn_adjoint.py`) has four parts:
- **Three groups with P2 moments.** On a three-group, P2, upscattering core
  with a volume-mixed absorber, k_adj equals k_fwd to 1e-7 pcm for step and
  SCB alike.
- **Negative controls.** Omitting the transposition or the swap misses by
  14 000 pcm; transposing only P0 misses by 312 pcm.
- **One group.** The problem is self-adjoint: the adjoint scalar flux equals
  the forward one to 1e-15.
- **Perturbation reciprocity.** First-order perturbation estimates with the
  adjoint reproduce the direct change of 1/k to second order (the error falls
  10x per 10x smaller perturbation) for transfer, upscatter, within-group and
  nu*Sigma_f perturbations.

`region_partial_currents` refuses an adjoint solver, since its angular flux is
reversed.

### SPH-DF zones and node models for drum cores (2026-11-18)

`garnet.equivalence` and `garnet.drum_mesh` hold the drum-core equivalence
method of the SPH audit (report §10.6). It was ported from the HP-MR study
tools to `HexLattice` and to any raster or triangular mesh.

- **Gauge.** SPH factors scale D and every cross section of a zone, and a
  boundary factor scales the Robin albedo of its zone's vacuum faces. The
  triangular face term `8 D a / (h (h a + 2√3 D))` and the unstructured term
  `a D A / (d a + D)` are both homogeneous of degree one in (D, a). So
  rescaling all SPH and boundary factors of a group together is a gauge, and
  the flux-volume normalization only fixes it. The mesh solver's non-orthogonal
  gradient rows are of degree zero in (D, a) and do not break this.
- **Feasibility.** A boundary factor going to infinity makes a face zero-flux.
  A uniform SPH factor cannot raise that limit relative to the zone's average
  flux. A boundary zone therefore admits a solution only if its zero-flux
  leakage per unit flux exceeds the reference's.
  - That fails for a whole boundary hex (which is why the outer layer is split
    off) and for any boundary zone holding part of the arc (which is why the
    drums' outer layers carry no absorber).
  - The upper bound on log γ (+8) flags it. The lower bound is −30: γ → 0 is a
    reflective face, and an arc-shadowed zone legitimately needs 1e-6.
- **Transfer by containment.** Zones are defined on node cells. Each reference
  cell inherits the zone of the node cell containing its centroid (`locate`:
  k-d tree candidates, then a bounding-box search for large coarse cells). Its
  absorber fraction inherits the node cell's absorber zone. This replaces the
  former two-step raster transfer, reproduces it cell for cell on HP-MR, and
  works for drum-band meshes.
- **Outer layer of a mesh.** The cells with a vertex on the boundary. The
  boundary vertices are the end points of the boundary faces, so hanging nodes
  never mark an interior cell.

### Diffusion transients on unstructured meshes; boundary coefficients in time (2026-11-19)

The drum-band refined mesh is the default SPH-DF node, so drum transients must run on it. `garnet.mesh.MeshGrid` makes an unstructured mesh a geometry of the structured-solver framework (`shape = (n_cells,)`), next to the Cartesian, hex and triangular grids:

- **`MeshGroupOperator`.** It has the group-operator signature, so `TransientSolver(..., group_operator=MeshGroupOperator, eig_solver=MeshDiffusionEigenSolver)` reuses the whole time integration: precursors, BDF and theta schemes, per-material kinetics, materials and blends that change in time, events.
- **Volume-integrated operator.** The operator is the TPFA leakage plus removal, integrated over the cells, with `rhs_weight` = cell area. The transient's power integral takes its volume weights from `grid.cell_weights()`, the hook the cylindrical metric already used.
- **Non-orthogonal correction applied implicitly.** The least-squares-gradient correction of `UnstructuredDiffusionSolver` (needed on hanging-node meshes: 49 pcm on HP-MR `local:4:1`) is linear in the flux. `MeshGroupOperator.apply` therefore includes it, and the operator is exact but not symmetric: GMRES is required, and CG or monolithic steps are rejected. `UnstructuredDiffusionSolver` lags the same correction into the source. The two converge to the same solution (1e-6 pcm). The face, gradient and Robin assembly is shared by both (`MeshGrid.group_terms`, `MeshGrid.correction_terms`); the refactored steady solver reproduces its former output bit for bit.
- **Boundary coefficients in time.** `problem_at(t)` may return a fifth element `bcf` (G, *grid.shape). It reaches the initial eigen solve and every step's group operators as `bcf[g]` (triangular raster and mesh). A change of identity rebuilds the operators, like the other elements.

Point-kinetics parameters of an S_N state (`garnet.tri_sn.sn_effective_kinetics`):
- **β_eff** uses the forward and adjoint scalar fluxes, which is exact because emission is isotropic.
- **Λ** uses ⟨ψ†, ψ/v⟩ from the flux moments the anisotropic solve carries: ⟨ψ†, ψ⟩ = φ†φ/W + Σ_j (2l_j+1) m†_j m_j, with the adjoint's odd moments sign-flipped (ψ†(Ω) = ψ̃(−Ω)).
- **Accuracy of Λ on a three-group P2 test core:** against the 1/v insertion (Σ_t + ε/v), which is the exact first-order Λ, the moment Λ is within 1.2e-4 (step) and 2.7e-3 (SCB). The scalar-flux Λ is 4 % off.
- **The quasi-static reference march.** `garnet.quasistatic.adiabatic_point_kinetics` marches ρ(t) = 1 − k0/k(θ(t)) with these parameters, using the backward-Euler convention of `TransientSolver`. On a bare homogeneous one-group core it reproduces the space-time transient to 6e-8.

The tabulated SPH-DF model is `garnet.equivalence.SphDfTable`:
- **Interpolation.** It interpolates the zone constants, μ, γ and the form functions linearly in drum angle and applies μ to the constants, as the steady evaluation does.
- **State at an angle.** `state(angle)` returns the SPH-corrected zone materials, the node's zone maps and absorber blend at that angle, and the boundary coefficients γ on the boundary zones' cells.
- **Transient driver.** `problem_at(angle_of_t)` feeds the transient. It returns the same tuple while the angle holds, so the operators are not rebuilt.
- **Speeds.** `kinetics(base, angle)` gives the SPH speeds v/μ frozen at one angle, because the transient's kinetics are fixed in time.
- **HP-MR check** (`local:4:1` tables): the state reproduces the table k at 90° (0.00 pcm against the reference), and the evaluation tool's interpolated k at 97.5° and 108.75° (+12.29 and −36.62 pcm).

### Exact pin-resolved assembly fractions

`benchmarks.hpmr_assembly.build_hpmr_assembly2d` gives every triangle within
reach of a pin image its covered area per pin layer through the same
closed-form intersection (`pin_layer_fractions`, full circles; annuli are
differences of nested disks). Pins never overlap, so the graphite monolith is
the remainder, and the per-cell fractions (`AssemblyProblem.cell_fractions`)
sum to the analytic pin areas on any mesh. The former barycentric sampler used
only the upright sub-triangle centroids: at refine 4 it gave mod_shell 0.0183
and hp_shell 0.0278 against 0.0227 and 0.0383. The solver still sees each cell
as its two dominant constituents (`material_map` plus one mix partner), so a
cell holding core, shell and graphite loses its smallest part to the base
material. That truncation is the remaining error in `volume_fractions`: at
most 0.0074 at refine 4, 0.0030 at refine 8 and 5e-5 at refine 16. A cell
belongs to the pin that covers most of it, which is the limit of the former
sample vote. The periodic S_N k∞ moved from 1.24523 to 1.20632 at refine 4
and from 1.18232 to 1.18414 at refine 8.
