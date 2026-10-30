"""Multigroup k-eigenvalue solvers: power iteration over the fission source.

Outer loop: standard power iteration on  M phi = (1/k) F phi, where M is the
block (groups) transport-approximation operator and F the fission production
operator. Within one outer iteration the groups are swept Gauss-Seidel style
(downscatter uses the freshest fluxes), and each within-group system is solved
by matrix-free Jacobi-preconditioned CG with a warm start from the previous
outer iteration.

Two angular approximations share this machinery:

- DiffusionEigenSolver: classic diffusion; group state = scalar flux.
- SP3EigenSolver: simplified P3; group state = (Phi1, phi2) moment pair,
  solved as one SPD block system per group (see SP3GroupOperator).

The inner CG tolerance is adapted to the outer residual, so early outers are
cheap and the final ones are tight; combined with warm starts, late outer
iterations cost only a handful of stencil applications.

The same outer loop solves the adjoint (importance) problem via
``solve(adjoint=True)``. The fission and scattering couplings transpose; the
Cartesian ADF leakage operator and the non-symmetric SDPN blocks also use their
algebraic transpose, and the block solvers swap their source and flux-read-out
moment weights. The adjoint flux weights first-order perturbation theory and
adjoint kinetics parameters.

Everything — fluxes, coupling coefficients, sources — lives on the device for
the entire solve; only per-iteration convergence scalars cross the PCIe bus.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .blend import MaterialBlend
from .grid import Grid
from . import kernels
from .linalg import (bicgstab, get_linear_solver, neumann_preconditioner, pcg,
                     mix_source_history as _anderson_source)
from .materials import Material
from .sp3 import SP3GroupOperator
from .spn import (CongruentSDPNOperator, SDPNGroupOperator, _SDPN_C, _SDPN_G,
                  _SPN_C, _SPN_G, _congruence_available, _diag_similarity,
                  _marshak_applies)
from .stencil import BC_VACUUM, BC_ZERO_FLUX, GroupOperator


@dataclass
class Result:
    k_eff: float
    flux: object  # scalar flux, (G, nx, ny, nz) array on the solve device
    converged: bool
    outer_iterations: int
    inner_iterations: int
    solve_seconds: float
    device: str
    k_history: list = field(default_factory=list)
    source_error_history: list = field(default_factory=list)

    @property
    def flux_numpy(self) -> np.ndarray:
        return asnumpy(self.flux)

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"Result(k_eff={self.k_eff:.6f}, {status}, "
            f"{self.outer_iterations} outers / {self.inner_iterations} inners, "
            f"{self.solve_seconds:.2f} s on {self.device})"
        )


def scatter_stack(xp, sigma_s, G, adjoint, shape, dtype):
    """Dense in-scatter fields with shape (target groups, source groups, *shape).

    The diagonal and absent couplings are zero. Forward assembly reads
    sigma_s[source][target]; adjoint assembly reads sigma_s[target][source].
    Explicit axis names make this transpose testable independently of GPU
    source contractions.
    """
    scatter_to_from = xp.zeros((G, G) + tuple(shape), dtype=dtype)
    for target_group in range(G):
        for source_group in range(G):
            scatter = (sigma_s[target_group][source_group] if adjoint
                       else sigma_s[source_group][target_group])
            if source_group != target_group and scatter is not None:
                scatter_to_from[target_group, source_group] = scatter
    return scatter_to_from


class Fields:
    """Per-cell, per-group cross-section fields on the solve device.

    Attributes (lists of (nx, ny, nz) device arrays, length G, except
    sigma_s which is a G x G nested list with None for empty couplings):
    nu_sigma_f, chi, removal, diffusion, sigma_t, sigma_a, sigma_s.

    ``xs_update`` is an optional callable ``fields -> None`` run as the LAST
    step of construction, free to modify the assembled per-cell arrays in
    place. It is the seam for state-dependent cross sections -- a temperature
    feedback is one (see :mod:`garnet.feedback`) -- expressed per CELL, which is
    something a per-material table cannot represent.

    Two consequences worth knowing before using it:

    * Operators COPY these arrays when they are built, so mutating a
      ``Fields`` after its solver exists changes nothing. A driver that varies
      the state must rebuild the solver (cheap: ~0.1% of a solve).
    * ``TransientSolver`` rebuilds its fields only when ``problem_at(t)``
      returns different *objects* (an identity check), so a hook that depends
      on state rather than on t would silently freeze at t = 0. Feeding a
      transient back needs that trigger widened first.
    """

    def __init__(self, xp, grid, materials, material_map, dtype,
                 mix_material=None, mix_weight=None, xs_update=None):
        mats = [materials] if isinstance(materials, Material) else list(materials)
        G = mats[0].n_groups
        if any(m.n_groups != G for m in mats):
            raise ValueError("all materials must have the same number of groups")
        self.n_groups = G
        self.fissile = any(m.is_fissile for m in mats)

        # Optional per-cell two-material blend on top of the base integer map:
        # cell XS = (1 - w) * base + w * mix, for a per-cell mix material and
        # weight w (sentinel mix_material < 0 = no blend). This is the volume-
        # mixing homogenization used for a partially-present material -- e.g. a
        # control-drum absorber arc that covers a fraction w of the cell, a
        # partially-inserted rod tip, or a rasterized fuel-pin rim. Cross
        # sections blend linearly (exact reaction-rate averaging under a flat
        # flux); the diffusion coefficient blends *harmonically* (i.e. its
        # transport cross section 1/(3D) volume-averages), so a trace of a
        # strong absorber correctly chokes the cell; the emission spectrum chi
        # blends by fission-production share (see below). Non-blended cells
        # stay bit-identical to the pure-index map.
        mix = mix_material is not None
        blend = MaterialBlend(xp, grid.shape, material_map, len(mats),
                              dtype=dtype, mix_material=mix_material,
                              mix_weight=mix_weight)
        self.blend = blend
        lin, harm = blend.linear, blend.harmonic

        # Per-cell lookup of an arbitrary per-material table (linear blend on
        # mixed cells) -- also used by the transient solver to map per-material
        # kinetics data (1/v, beta) onto the grid, and by the thermal solver to
        # map conductivities onto the same cells under the same rules.
        self.map_table = lin

        def per_group(attr, lookup=None):
            lookup = lookup or lin
            table = np.array([getattr(m, attr) for m in mats])  # (M, G)
            return [lookup(table[:, g]) for g in range(G)]

        self.nu_sigma_f = per_group("nu_sigma_f")
        # chi is an emission *spectrum*, not a cross section: blending it
        # linearly by volume would leave a fissile material mixed with a
        # non-fissile one emitting a spectrum that sums to w -- silently losing
        # (1 - w) of the cell's fission neutrons. The correct merge weights
        # each component's spectrum by its share of the cell's fission
        # production, w * sum_g nuSigma_f (flat-flux proxy; exact whenever at
        # most one component is fissile, e.g. a fuel pin blended with
        # moderator). Cells where neither component is fissile keep the base
        # spectrum (it multiplies a zero source).
        if mix:
            prod = np.array([np.sum(m.nu_sigma_f) for m in mats])
            fission_weighted = lambda table: blend.fission_weighted(table, prod)
        else:
            fission_weighted = lin
        # Per-cell lookup of a per-material quantity that rides on the fission
        # source (chi, delayed fraction beta): mixed cells weight each
        # component by its share of the cell's fission production, so e.g. a
        # fuel/moderator rim cell keeps the fuel's spectrum and beta exactly.
        self.map_table_fission_weighted = fission_weighted

        chi_t = np.array([m.chi for m in mats])  # (M, G)
        self.chi = [fission_weighted(chi_t[:, g]) for g in range(G)]
        # sigma_a is kept alongside removal (= sigma_a + out-scatter) because a
        # temperature feedback adds to ABSORPTION: scaling removal instead
        # would silently scale the out-scatter with it.
        self.sigma_a = per_group("sigma_a")
        self.removal = per_group("removal")
        self.diffusion = per_group("diffusion", lookup=harm)
        self.sigma_t = per_group("sigma_t")

        sig_s = np.array([m.sigma_s for m in mats])  # (M, G, G)
        self.sigma_s = [[lin(sig_s[:, gf, gt]) if np.any(sig_s[:, gf, gt]) else None
                         for gt in range(G)] for gf in range(G)]

        # The per-material tables, for xs_update hooks that evaluate each
        # material's law separately before blending (ThermalFeedback).
        self.materials = mats
        if xs_update is not None:
            xs_update(self)

    def exclude_inactive(self, active):
        """Remove the group-coupling sources of excised (inactive) cells.

        The operators decouple an excised cell and give it a unit diagonal,
        which keeps its flux at zero only if nothing sources it. Whatever
        material the map assigns there, its fission production and emission
        (nu_sigma_f, chi) and in-scatter would otherwise drive an isolated
        one-cell eigenproblem phi = (nu_sigma_f / k) phi -- a fuel material
        with nu_sigma_f = 1.5 under the mask reported k = 1.5. The arrays are
        replaced (not modified in place), so shared tables stay untouched.
        """
        xp = kernels.module_of(self.nu_sigma_f[0])
        keep = xp.asarray(active).astype(bool)
        mask = lambda field: xp.where(keep, field, 0.0).astype(field.dtype)
        self.nu_sigma_f = [mask(f) for f in self.nu_sigma_f]
        self.chi = [mask(f) for f in self.chi]
        self.sigma_s = [[None if f is None else mask(f) for f in row]
                        for row in self.sigma_s]
        self.fissile = bool(any(float(xp.max(f)) > 0.0
                                for f in self.nu_sigma_f))

    def fission_source(self, phi_by_group):
        """Sum_g nuSigma_f,g * phi_g over an iterable of per-group fluxes."""
        src = self.nu_sigma_f[0] * phi_by_group[0]
        for g in range(1, self.n_groups):
            src += self.nu_sigma_f[g] * phi_by_group[g]
        return src


class _PowerIterationSolver:
    """Shared field construction and power-iteration outer loop.

    Subclasses set self.ops (one operator per group, exposing .apply and
    .inv_diag) in _build_operators() and define the group state layout via
    _initial_state / _rhs / _phi.

    Parameters
    ----------
    grid         : Grid
    materials    : a single Material, or a list of Materials indexed by
                   material_map.
    material_map : optional int array of shape grid.shape assigning a material
                   index to every cell (omit for a homogeneous reactor).
    bc           : "zero-flux" (default) or "reflective"; a single string
                   applies to all six faces, or pass 3 per-axis entries /
                   6 per-face values, e.g. bc=(("reflective", "zero-flux"),
                   ("reflective", "zero-flux"), "reflective") for a quarter
                   core with reflective symmetry planes, solved in 2D.
    device       : "auto" | "gpu" | "cpu"
    dtype        : floating dtype for the solve (float64 default; float32
                   roughly doubles GPU throughput at ~1e-6 k accuracy).
    precond_degree : degree of the Neumann-polynomial preconditioner for the
                   inner CG (0 = plain Jacobi, the default). Each degree adds
                   one stencil apply per CG iteration but cuts the iteration
                   count -- and with it the global reductions that are the
                   GPU's only synchronization points. Degree 1 or 3 is a good
                   setting (cf. E et al., NED 320 (2017), where degree-3
                   Neumann-PCG was the fastest GPU solver for 2e4-3e6 cells);
                   the splitting is damped automatically so odd degrees stay
                   SPD on the SP5/SP7 blocks too (see neumann_preconditioner).
    linear_solver : "cg" (default), "gmres", or "bicgstab" -- the within-group
                   Krylov solver. CG applies to symmetric positive definite
                   operators; nonsymmetric ADF or cylindrical divergence
                   operators require GMRES or BiCGStab.
    symmetric_operator : cylindrical grids only (True default). False builds
                   the natural divergence-form (per-unit-volume) stencil,
                   which is non-symmetric -- pair it with
                   linear_solver="gmres" or "bicgstab". Same discrete
                   solution as the default volume-weighted SPD form; exists
                   as a cross-check and a template for genuinely
                   non-symmetrizable operators. Diffusion only (the SP3
                   block symmetrization assumes SPD moment operators).
    """

    def __init__(self, grid: Grid, materials, material_map=None,
                 bc: str = BC_ZERO_FLUX, device: str = "auto", dtype=np.float64,
                 active=None, mask_bc=BC_VACUUM, precond_degree: int = 0,
                 mix_material=None, mix_weight=None, linear_solver="cg",
                 symmetric_operator: bool = True, hybrid_mask=None,
                 hybrid_confine: bool = False, xs_update=None):
        self.grid = grid
        self.xp = xp = get_backend(device)
        self.device = device_name(xp)
        self.dtype = np.dtype(dtype)
        self.active = active
        self.mask_bc = mask_bc
        # Hybrid transport/diffusion: a per-cell bool mask marking the cells that
        # keep the full angular block (e.g. the control-drum absorber); every
        # other cell runs pure diffusion. The angular blocks restrict their
        # higher moments to this mask, so the driver must likewise zero the
        # higher-moment rows of the RHS outside it (see _rhs). None (default) =
        # the uniform angular approximation everywhere. Diffusion has no higher
        # moments, so it ignores the mask.
        self.hybrid_mask = (None if hybrid_mask is None
                            else xp.asarray(hybrid_mask).astype(self.dtype))
        self.hybrid_confine = bool(hybrid_confine)
        self.symmetric_operator = bool(symmetric_operator)
        self._linsolve = get_linear_solver(linear_solver)
        # BiCGStab's residual is non-monotone: it stops anywhere below rtol, so
        # its solve-to-solve error is irregular at the rtol level. That noise
        # is invisible to plain power iteration but poisons the Anderson
        # history of the outer loop (default SDP3 on BIBLIS stalled at
        # src_err ~3e-4 for 2000 outers); one decade of extra inner accuracy
        # restores the GMRES outer count (BIBLIS SDP3: 28 outers vs 30).
        self._inner_rtol_scale = 0.1 if self._linsolve is bicgstab else 1.0
        self._precond_degree = int(precond_degree)
        if (not self.symmetric_operator and self._linsolve is pcg
                and getattr(grid, "geometry", "cartesian") == "cylindrical"):
            raise ValueError(
                "symmetric_operator=False builds the non-symmetric divergence-"
                "form cylindrical stencil; use linear_solver='gmres' or 'bicgstab'")

        # xs_update runs inside Fields, i.e. BEFORE _build_operators and before
        # the fissile check -- so the operators and the "is this a k-eigenvalue
        # problem at all" test both see the post-feedback data. Every solver
        # that subclasses this one (Tri*, SP3, SPN, SDPN) only overrides
        # _build_operators, so they inherit the hook unchanged.
        f = Fields(xp, grid, materials, material_map, self.dtype,
                   mix_material=mix_material, mix_weight=mix_weight,
                   xs_update=xs_update)
        if active is not None:
            f.exclude_inactive(active)
        if not f.fissile:
            raise ValueError("no fissile material: k-eigenvalue problem is undefined")
        self.fields = f
        self.n_groups = f.n_groups
        self.nu_sigma_f = f.nu_sigma_f  # list of (nx,ny,nz), len G
        self.chi = f.chi
        self.sigma_s = f.sigma_s

        self._build_operators(grid, f.diffusion, f.sigma_t, f.removal, bc)
        # Cylindrical grids: the operators are volume-weighted, so the
        # isotropic source must carry the same per-cell metric factor
        # (None on Cartesian grids -- the source is used as-is).
        self._src_weight = getattr(self.ops[0], "rhs_weight", None)
        self.preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                                int(precond_degree))
                         for op in self.ops]

    # ---- hooks implemented by the angular approximation -------------------
    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        raise NotImplementedError

    def _linear_components(self, g, adjoint):
        return self.ops[g].apply, self.preconds[g]

    def _initial_state(self):
        """Per-group solution state, as a list of device arrays."""
        raise NotImplementedError

    def _rhs(self, g, q0, adjoint=False):
        """Within-group right-hand side from the isotropic source q0.

        The block solvers couple groups through a rank-one map: the scalar
        source enters the moment rows by one weight vector (``_rhs``) and the
        scalar flux is read back by another (``_phi``). The transposed
        (adjoint) problem exchanges the two, hence the ``adjoint`` flag.
        """
        raise NotImplementedError

    def _phi(self, state_g, adjoint=False):
        """Scalar flux phi0 of one group's state (view or expression); with
        ``adjoint`` the scalar importance read by the transposed coupling."""
        raise NotImplementedError

    # -----------------------------------------------------------------------
    def _make_group_batch(self, state, adjoint, production_weights):
        r"""Stacked group data for the batched source assembly, or None.

        The per-group Python loops in ``solve`` cost O(G^2) kernel launches per
        outer -- two per (g, g') in-scatter pair plus O(G) for the fission
        source -- which for the 11-group HP-MR library is a few hundred small
        launches per outer, independent of grid size. Stacking the scattering
        matrix into (G, G, \*grid) and the scalar fluxes into (G, \*grid) lets
        one kernel walk a whole row, taking that to O(G).

        Returned only on GPU (the launch cost is the whole point; on CPU the
        sparse loop is strictly better because it skips absent couplings rather
        than multiplying by materialized zeros) and only when the stack is worth
        its memory: G >= 3, and the (G, G, \*grid) array must fit comfortably in
        free device memory. ``S[g][gf]`` is the in-scatter coupling g' -> g with
        the diagonal zeroed, matching the ``gf != g`` skip it replaces.
        """
        xp, G = self.xp, self.n_groups
        if not kernels.use_fused(xp, "groups") or G < 3:
            return None
        phi0 = self._phi(state[0], adjoint)
        nbytes = G * G * phi0.size * phi0.dtype.itemsize
        try:
            free = xp.cuda.Device().mem_info[0]
        except Exception:
            free = 0
        if free and nbytes > free // 8:
            return None                       # not worth the footprint
        scatter_to_from = scatter_stack(xp, self.sigma_s, G, adjoint, phi0.shape, phi0.dtype)
        production_stack = xp.stack([production_weights[g] for g in range(G)])
        phi = xp.stack([self._phi(state[g], adjoint) for g in range(G)])
        return {"scatter_to_from": scatter_to_from,
                "production_weights": production_stack, "phi": phi}

    def _fission_source(self, state, weight, adjoint=False):
        """Sum_g weight[g] * phi0_g -- the field that drives the outer loop.

        Forward: weight = nu_sigma_f (fission neutron production per cell).
        Adjoint: weight = chi (the transpose of F distributes production by
        nu_sigma_f and collects it against chi, so the driving field is the
        chi-weighted importance).
        """
        src = weight[0] * self._phi(state[0], adjoint)
        for g in range(1, self.n_groups):
            src += weight[g] * self._phi(state[g], adjoint)
        return src

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-6,
              max_outer: int = 2000, inner_rtol_floor: float = 1e-10,
              k_guess: float = 1.0, verbose: bool = False,
              adjoint: bool = False, anderson_depth: int = 8,
              state0=None) -> Result:
        """Run power iteration until |dk| < tol_k and the relative L2 change of
        the normalized fission source < tol_source.

        The fission-source fixed point is Anderson-accelerated (a short history of
        source residuals, the same scheme used in the transient step and SPH
        solve): plain power iteration converges at the dominance ratio, which is
        near 1 for a loosely-coupled core (hundreds of outers), whereas Anderson
        collapses the slow modes into a handful. ``anderson_depth`` <= 1 recovers
        the plain power iteration; the converged eigenpair is identical either way.
        If Anderson keeps restarting without progress (a map made noisy by its
        inner solves) the loop falls back to plain power iteration.

        Stopping test: with Anderson it is on the per-outer change (|dk|,
        src_err). Plain power iteration (anderson_depth <= 1, and always for
        the adjoint) converges geometrically at the dominance ratio rho, so its
        remaining error is ~ change / (1 - rho); there both tolerances are
        multiplied by (1 - rho_hat), rho_hat estimated from the src_err history,
        so tol_source bounds the error rather than the step (floored at what
        the dtype / inner_rtol_floor can resolve).

        adjoint : solve the adjoint (importance) k-eigenproblem M* phi* =
        (1/k) F* phi* instead of the forward one. Fission swaps the production
        weight nu_sigma_f and the emission spectrum chi, scattering swaps its group indices
        (sigma_s[g'->g] becomes sigma_s[g->g']). The dominant eigenvalue is
        identical to the forward k; the eigenvector is the adjoint flux, used
        for adjoint-weighted kinetics and first-order perturbation theory.
        Cartesian ADF diffusion and the non-symmetric SDPN blocks (SDP3 with
        vacuum faces, any order with Marshak) also transpose the within-group
        operator, and the block solvers exchange their source and scalar-flux
        moment weights; for them the returned flux is the scalar importance
        src_weights . U*.
        """
        xp, G = self.xp, self.n_groups
        synchronize(xp)
        t0 = time.perf_counter()

        # Anderson reliably accelerates the forward source, but on the adjoint
        # (chi-weighted) source it can lock onto a subdominant eigenpair, so it is
        # disabled there -- plain power iteration is used for the adjoint.
        if adjoint:
            anderson_depth = 1

        # Fission couples groups as F[g,g'] = chi_g * nu_sigma_f_g'. Its
        # transpose F* moves chi to the production weight and nu_sigma_f to the
        # emission spectrum; scattering transposes independently (below).
        production_weights = self.chi if adjoint else self.nu_sigma_f
        emission_spectrum = self.nu_sigma_f if adjoint else self.chi

        # state0 warm-starts the power iteration from a previous solve's group
        # state -- the payoff in an outer coupling loop, where each iteration
        # perturbs the cross sections only slightly and a flat start throws
        # away a converged flux shape. It is a starting point only: the
        # converged eigenpair is unchanged, and passing None (the default) is
        # the cold start every existing caller gets.
        state = self._initial_state() if state0 is None else [
            self.xp.asarray(s, dtype=self.dtype).copy() for s in state0]
        batch = self._make_group_batch(state, adjoint, production_weights)

        def fission_source():
            """Sum production_weights[g] * phi0_g with optional group batching."""
            if batch is None:
                return self._fission_source(state, production_weights, adjoint)
            out = xp.zeros_like(batch["phi"][0])
            return kernels.group_accumulate(xp, out, batch["production_weights"], batch["phi"])

        fsrc = fission_source()
        total = xp.sum(fsrc)
        if float(total) <= 0:
            raise RuntimeError("initial fission source is zero")
        k = float(k_guess)

        k_hist, err_hist = [], []
        inner_total = 0
        src_err = prev_src_err = 1.0
        converged = False
        hist = []                                # (fsrc_in, raw_iterate) for Anderson
        # Anderson safeguard: restarts that make no progress (the source error
        # at a restart not at least halved since the previous restart) mean the
        # map is too noisy for extrapolation; after a few the outer loop falls
        # back to plain power iteration, which always converges.
        restart_err, stalled_restarts = None, 0
        # Plain power iteration contracts the error by the dominance ratio rho
        # per outer, so the error left after a step of size src_err is
        # ~ src_err * rho / (1 - rho): the step alone overstates convergence by
        # 1/(1 - rho) (~100x on a 400 cm PWR). On the un-accelerated path the
        # stopping test is therefore scaled by an estimate of rho from the
        # src_err sequence (Anderson's error is not a geometric sequence, so
        # its test is left as is). The tightened tolerances never go below
        # what the inner solves / arithmetic can resolve.
        rho_hat = 0.0
        resolvable = 10.0 * max(float(np.finfo(self.dtype).eps), inner_rtol_floor)

        for outer in range(1, max_outer + 1):
            # Inner tolerance tracks the *outer residual* (src_err): cheap early,
            # tight late. Not tied to tol_source (that pinned the inner solve loose
            # when converging on k alone, stalling k and letting Anderson drift).
            # A non-monotone inner solver (BiCGStab) runs a decade tighter.
            rtol = max(self._inner_rtol_scale * min(1e-3, 0.1 * src_err),
                       inner_rtol_floor)
            fsrc_in = fsrc                       # this iteration's input (for Anderson)

            for g in range(G):
                # Keep an FP32 solve genuinely FP32. Host kinetics/eigenvalue
                # scalars may otherwise promote the seeded source before the
                # fused group contraction sees it.
                q0 = xp.asarray((emission_spectrum[g] / k) * fsrc, dtype=self.dtype)
                if batch is not None:
                    # One kernel for the whole in-scatter row. batch["phi"] is
                    # kept current group by group below, so this reads the
                    # updated flux for g' < g and the previous one for g' > g --
                    # the same Gauss-Seidel sweep the Python loop performed.
                    kernels.group_accumulate(xp, q0, batch["scatter_to_from"][g], batch["phi"])
                else:
                    for source_group in range(G):
                        # forward: in-scatter g'->g uses sigma_s[g'][g];
                        # adjoint transposes to sigma_s[g][g'].
                        s = self.sigma_s[g][source_group] if adjoint else self.sigma_s[source_group][g]
                        if source_group != g and s is not None:
                            q0 += s * self._phi(state[source_group], adjoint)
                if self._src_weight is not None:
                    q0 = q0 * self._src_weight
                apply_A, precond = self._linear_components(g, adjoint)
                state[g], n_it = self._linsolve(
                    apply_A, self._rhs(g, q0, adjoint), state[g],
                    self.ops[g].inv_diag, xp, rtol=rtol,
                    precond=precond)
                if batch is not None:
                    batch["phi"][g] = self._phi(state[g], adjoint)
                inner_total += n_it

            fsrc_new = fission_source()
            total_new = xp.sum(fsrc_new)
            k_new = k * float(total_new / total)

            diff = fsrc_new / total_new - fsrc / total
            src_err = float(xp.sqrt(xp.sum(diff * diff) / xp.sum((fsrc_new / total_new) ** 2)))
            dk = abs(k_new - k)

            # Normalize so the mean fission source stays at 1 (avoids drift).
            scale = self.grid.n_cells / total_new
            for g in range(G):
                state[g] *= scale
            if batch is not None:
                batch["phi"] *= scale         # phi0 is linear in the state
            raw_fission_source = fsrc_new * scale                     # raw power iterate, mean 1
            k = k_new

            k_hist.append(k)
            err_hist.append(src_err)
            if verbose:
                print(f"  outer {outer:4d}  k = {k:.8f}  dk = {dk:.2e}  src_err = {src_err:.2e}")

            # Dominance-ratio estimate for the un-accelerated stopping test:
            # the ratio of successive three-outer sums of src_err (smoother than
            # one step ratio). A non-contracting window (noise at the inner
            # tolerance floor, or the transient of a restart) keeps the last
            # trustworthy estimate.
            tol_k_eff, tol_source_eff = tol_k, tol_source
            if anderson_depth <= 1:
                if len(err_hist) >= 4:
                    num, den = sum(err_hist[-3:]), sum(err_hist[-4:-1])
                    if 0.0 < num < den:
                        rho_hat = min(num / den, 0.999)
                tol_k_eff = max(tol_k * (1.0 - rho_hat), min(tol_k, resolvable))
                tol_source_eff = max(tol_source * (1.0 - rho_hat),
                                     min(tol_source, resolvable))

            # Accept only if this outer's inner solves were at least as tight as
            # the source tolerance (or at the floor). A loose warm-started inner
            # solve can return in zero iterations; when that hits every group
            # that drives the source (e.g. the adjoint, driven by chi = fast
            # group only) the source is bit-identical, dk = src_err = 0, and
            # the loop would stop short. src_err is then tiny, so the next
            # outer runs tight inners and settles it honestly.
            if (dk < tol_k_eff and src_err < tol_source_eff
                    and rtol <= max(tol_source, inner_rtol_floor)):
                fsrc = raw_fission_source
                converged = True
                break

            # Anderson update of the fission source (state stays the plain iterate).
            fsrc = raw_fission_source
            if anderson_depth > 1 and src_err > 1.1 * prev_src_err:
                hist = []                                 # last mix overshot -> restart
                if restart_err is not None and src_err > 0.5 * restart_err:
                    stalled_restarts += 1
                else:
                    stalled_restarts = 0
                restart_err = src_err
                if stalled_restarts >= 3:
                    anderson_depth = 1                    # noisy map: plain power
            if anderson_depth > 1:
                hist.append((fsrc_in, raw_fission_source))
                if len(hist) > anderson_depth:
                    hist.pop(0)
                fsrc = _anderson_source(hist, raw_fission_source, xp)
                fsrc = fsrc * (self.grid.n_cells / float(xp.sum(fsrc)))
            prev_src_err = src_err
            total = xp.sum(fsrc)

        synchronize(xp)
        # Retain the converged per-group state (moment vectors for the block
        # solvers) so a transient can start from the exact steady moments.
        self.state = state
        return Result(
            k_eff=k,
            flux=xp.stack([self._phi(state[g], adjoint) for g in range(G)]),
            converged=converged,
            outer_iterations=outer,
            inner_iterations=inner_total,
            solve_seconds=time.perf_counter() - t0,
            device=self.device,
            k_history=k_hist,
            source_error_history=err_hist,
        )


def _conjugated_by_row_scale(op):
    """Transpose of a row-scaled symmetric operator diag(s) A_w: A_w diag(s)
    = diag(s)^-1 (diag(s) A_w) diag(s), applied through ``op.apply``."""
    scale = op.row_scale

    def apply_transpose(phi):
        return op.apply(phi * scale) / scale

    return apply_transpose


class DiffusionEigenSolver(_PowerIterationSolver):
    """Multigroup neutron diffusion k-eigenvalue solver (see base for args).

    ``adf`` is an optional three-tuple of (low-side, high-side) interior
    face arrays, each shaped (G, *face_shape). Factors are heterogeneous over
    homogeneous surface flux, and must already be unity on faces internal to
    a homogenized region. Only Cartesian grids are supported. Use
    GMRES or BiCGStab; supplying factors conservatively rejects CG even when
    the supplied arrays happen to produce a symmetric operator.
    """

    def __init__(self, *args, adf=None, **kwargs):
        if adf is not None and type(self) is not DiffusionEigenSolver:
            raise ValueError("Cartesian ADFs are not supported by this solver")
        self.adf = adf
        super().__init__(*args, **kwargs)
        if adf is not None and self._linsolve is pcg:
            raise ValueError("ADF may produce a nonsymmetric operator; use linear_solver='gmres' or 'bicgstab'")
        # Non-symmetric within-group operators solve the adjoint on their
        # algebraic transpose: the Cartesian ADF stencil has its own; the r-z
        # divergence form (symmetric_operator=False) is W^-1 A_w with A_w
        # symmetric, whose transpose is W (W^-1 A_w) W^-1 -- the same operator
        # conjugated by the row scale 1/W. The Jacobi diagonal is unchanged.
        self._adjoint_ops = None
        if adf is not None:
            self._adjoint_ops = [op.apply_transpose for op in self.ops]
        elif getattr(self.ops[0], "row_scale", None) is not None:
            self._adjoint_ops = [_conjugated_by_row_scale(op) for op in self.ops]
        if self._adjoint_ops is not None:
            self._adjoint_preconds = [neumann_preconditioner(
                apply_t, op.inv_diag, self._precond_degree)
                for apply_t, op in zip(self._adjoint_ops, self.ops)]

    def _linear_components(self, g, adjoint):
        if adjoint and self._adjoint_ops is not None:
            return self._adjoint_ops[g], self._adjoint_preconds[g]
        return super()._linear_components(g, adjoint)

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        if self.hybrid_mask is not None:
            raise ValueError("hybrid_mask has no effect on the diffusion solver "
                             "(diffusion has no higher moments); use an SP3/SDPN "
                             "solver to run transport on the masked subdomain")
        adf = self.adf
        if adf is not None:
            if len(adf) != 3 or any(len(pair) != 2 for pair in adf):
                raise ValueError("adf must contain three (low, high) face-array pairs")
            adf = tuple(tuple(self.xp.asarray(a) for a in pair) for pair in adf)
            for axis, pair in enumerate(adf):
                shape = list(grid.shape)
                shape[axis] -= 1
                expected = (self.n_groups, *shape)
                if any(a.shape != expected for a in pair):
                    raise ValueError(f"adf axis {axis} factors must have shape {expected}")
        self.ops = [GroupOperator(self.xp, grid, diffusion[g], removal[g], bc=bc,
                                  active=self.active, mask_bc=self.mask_bc,
                                  symmetric=self.symmetric_operator,
                                  adf=None if adf is None else tuple((lo[g], hi[g]) for lo, hi in adf))
                    for g in range(self.n_groups)]

    def _initial_state(self):
        return [self.xp.ones(self.grid.shape, dtype=self.dtype)
                for _ in range(self.n_groups)]

    def _rhs(self, g, q0, adjoint=False):
        return q0

    def _phi(self, state_g, adjoint=False):
        return state_g


class SP3EigenSolver(_PowerIterationSolver):
    """Multigroup simplified-P3 k-eigenvalue solver (see base for args).

    Same interface and cost structure as DiffusionEigenSolver with ~2x the
    work per group; captures leading transport effects that diffusion misses
    (steep flux gradients, strong absorbers, small cores).
    """

    # Angular variant of the two-moment block: "sp3" (Brantley & Larsen) or
    # "sdp1" (simplified double-P1); see SP3GroupOperator. Same solve otherwise.
    _sp_variant = "sp3"

    def __init__(self, *args, marshak_vacuum: bool | None = None, **kwargs):
        # marshak_vacuum: vacuum-face treatment of the block (see
        # SP3GroupOperator's ``marshak``). None (default) = the coupled
        # Brantley-Larsen Marshak condition wherever it is available (SP3 on
        # Cartesian grids, also on active-mask vacuum faces) and the per-moment
        # alpha = 1/2 Robin term elsewhere (SDP1, hybrid); True = require the
        # coupled form; False = per-moment Robin everywhere (the pre-2026
        # behaviour). The coupled form is symmetric, so CG is unaffected.
        self._marshak = marshak_vacuum
        super().__init__(*args, **kwargs)

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        if not self.symmetric_operator:
            raise ValueError("symmetric_operator=False is diffusion-only: the "
                             "SP3 block symmetrization assumes SPD moment operators")
        self.ops = [SP3GroupOperator(self.xp, grid, diffusion[g], sigma_t[g],
                                     removal[g], bc=bc, variant=self._sp_variant,
                                     active=self.active, mask_bc=self.mask_bc,
                                     hybrid_mask=self.hybrid_mask,
                                     hybrid_confine=self.hybrid_confine,
                                     marshak=self._marshak)
                    for g in range(self.n_groups)]

    def _initial_state(self):
        state = []
        for _ in range(self.n_groups):
            u = self.xp.zeros((2,) + self.grid.shape, dtype=self.dtype)
            u[0] = 1.0
            state.append(u)
        return state

    def _rhs(self, g, q0, adjoint=False):
        # Symmetrized block RHS: (q0, 5 * (-2/5) q0). Hybrid: the phi2 source is
        # confined to the phi2 subdomain (zero elsewhere), so phi2 stays exactly
        # zero outside it and the first row is pure diffusion there. The block
        # is symmetric and the source/flux weights coincide, so the adjoint
        # differs only in where that mask sits: the transpose of (masked
        # source, unmasked read-out) is (unmasked source, masked read-out).
        rhs = self.xp.empty((2,) + self.grid.shape, dtype=self.dtype)
        rhs[0] = q0
        rhs[1] = -2.0 * q0
        if self.hybrid_mask is not None and not adjoint:
            rhs[1] = rhs[1] * self.hybrid_mask
        return rhs

    def _phi(self, state_g, adjoint=False):
        if adjoint and self.hybrid_mask is not None:
            return state_g[0] - 2.0 * (self.hybrid_mask * state_g[1])
        return state_g[0] - 2.0 * state_g[1]


class SDP1EigenSolver(SP3EigenSolver):
    """Multigroup simplified double-P1 (SDP1) k-eigenvalue solver.

    The N=1 simplified double-PN approximation of Carreno et al., Ann. Nucl.
    Energy 207 (2024) 110675. Same two-moment block, cost, and interface as
    :class:`SP3EigenSolver` (SDP1 and SP3 have the identical number of degrees
    of freedom), differing only in the second-moment diffusion coefficient
    (see :class:`~garnet.operator.SP3GroupOperator`). Derived from a half-range
    (double-PN) angular expansion, it resolves discontinuous angular flux more
    faithfully than SP3, giving more accurate results across strongly
    heterogeneous media at equal cost.

    Vacuum faces always take the per-moment Robin (alpha = 1/2) term here: the
    coupled Marshak matrix is not symmetric in this two-moment basis, so it
    would cost the CG path this solver exists for. On a leaky core that is
    73 pcm low on Brantley-Larsen. ``SDPNEigenSolver`` of order 1 is the same
    SDP1 with the Marshak boundary by default (GMRES, about 5x the cost there).
    """

    _sp_variant = "sdp1"


def _peek_base_args(args, kwargs):
    """(grid, bc, active, mask_bc) as _PowerIterationSolver.__init__ will see
    them -- for subclasses that must make per-problem decisions before calling
    super().__init__. Falls back to the defaults on a bad call signature (the
    real error is then raised by super().__init__ itself)."""
    sig = inspect.signature(_PowerIterationSolver.__init__)
    try:
        bound = sig.bind_partial(None, *args, **kwargs).arguments
    except TypeError:
        bound = {}
    return (bound.get("grid"), bound.get("bc", BC_ZERO_FLUX),
            bound.get("active"), bound.get("mask_bc", BC_VACUUM))


class SDPNEigenSolver(_PowerIterationSolver):
    """Multigroup simplified double-PN (SDPN) k-eigenvalue solver, order N in
    {1, 2, 3}, in the diffusive U-form of Carreno et al., Ann. Nucl. Energy 207
    (2024) 110675 (see :class:`~garnet.operator.SDPNGroupOperator`).

    The group state is the M = N+1 even-moment vector U; the block reaction
    matrix is non-symmetric (the double-PN closure), so the within-group solve
    runs on CG in a symmetrizing basis whenever the tables and boundary
    conditions admit one (see ``symmetrize`` below) and on GMRES otherwise. Successive orders match
    the degrees of freedom of SP3/SP5/SP7 respectively and, for strongly
    heterogeneous media, converge toward the transport solution from fewer
    moments. Order 1 is equivalent to :class:`SDP1EigenSolver`, which is kept as
    a faster symmetric (CG) special case; use N = 2, 3 for the higher orders.
    """

    _order = 2
    _coeffs = None                 # None -> SDPN (operator default); _SPN_C -> SPN
    _g_coeffs = _SDPN_G            # Marshak boundary matrices for this family

    def __init__(self, *args, linear_solver: str | None = None,
                 marshak_vacuum: bool | None = None, symmetrize: bool | None = None,
                 spn_precondition: int | bool = 0, **kwargs):
        # marshak_vacuum: vacuum-face treatment. None (default) applies the
        # exact moment-coupled Marshak condition -n.D grad U = (g (x) I) U, the
        # boundary of Carreno et al. (2024), wherever it is available and has a
        # vacuum face to act on (spn._marshak_applies: Cartesian GroupOperator
        # grids without a hybrid mask), and the per-moment Robin (alpha = 1/2)
        # term elsewhere, or when symmetrize=True is requested (the coupled
        # term breaks the symmetrizing transforms). False selects per-moment
        # Robin everywhere, the default before 2026-10-30, kept for comparison:
        # it is 52-73 pcm low on the leaky Brantley-Larsen core and depends on
        # the moment variables it is applied to
        # (benchmark-results/carreno2024-sdpn-vv). True requires Marshak.
        # Marshak makes every SDPN block non-symmetric (GMRES), which costs
        # SDP2 its CG path (about 6x slower on Brantley-Larsen); the SPN
        # family's Marshak g is symmetric and keeps CG.
        #
        # symmetrize (default auto): put the block in a symmetric basis so
        # the within-group solve runs on CG. Auto turns on the *diagonal
        # similarity* whenever the tables admit one (SDP1/SDP2; works with
        # every boundary type except Marshak). SDP3 admits no diagonal
        # similarity, so auto falls back to the full symmetrizing *congruence*
        # (CongruentSDPNOperator) whenever this problem's boundaries allow it
        # -- every face (and the active-mask boundary, if any) reflective or
        # zero-flux, e.g. C5G7; vacuum-Robin faces are moment-dependent and
        # break the symmetry, so those problems keep the plain non-symmetric
        # form. The SPN family is symmetric as-is -- including its Marshak g
        # -- and takes CG without any transform. linear_solver=None resolves
        # to "cg" for a symmetric block and "gmres" otherwise: GMRES's
        # monotone true-residual stop gives the Anderson-accelerated outer loop
        # a smooth map, whereas BiCGStab's irregular error stalled it (default
        # SDP3 on vacuum-bounded BIBLIS: NOT converged after 2000 outers with
        # BiCGStab, 30 outers with GMRES). An explicit "bicgstab" still works
        # (the outer loop then tightens its inner tolerance a decade).
        #
        # spn_precondition: for the blocks that stay non-symmetric (SDP3 with
        # vacuum faces, any SDPN order with Marshak), precondition BiCGStab/
        # GMRES with `spn_precondition` Neumann (damped-Jacobi) sweeps of the
        # matched-order standard-SPN block -- the symmetric operator that
        # differs from SDPN only in the closure rows of the c tables (same
        # D_i, same boundary structure; with Marshak it carries SPN's own
        # symmetric g). The sweeps are a fixed linear operator, so both
        # BiCGStab and right-preconditioned GMRES remain valid. 0/False = off
        # (plain Jacobi, the default); True = 2 sweeps. The block's moment
        # coupling makes undamped Jacobi non-contractive (lambda_max(D^-1 A)
        # > 2), which made odd sweep counts indefinite (degree 1 was 10x
        # WORSE than plain Jacobi); the sweeps are now damped automatically
        # (see linalg.neumann_preconditioner), so every count helps. 1-3
        # sweeps cut the Krylov applies by 40-60% (Brantley-Larsen SDP3/vacuum
        # and SDP2/Marshak, GMRES: 568 -> 347/328/264) but each sweep costs one
        # block apply, so on CPU it is roughly break-even with Jacobi; it is
        # aimed at GPU runs, where applies are
        # streaming kernels and the saved iterations remove global reductions
        # (cf. the same trade-off for precond_degree / Neumann-PCG).
        # Meaningless on a path that already runs symmetric/CG (the companion
        # lives in the untransformed basis), so that combination raises.
        hybrid = kwargs.get("hybrid_mask") is not None
        grid, bc, active, mask_bc = _peek_base_args(args, kwargs)
        if marshak_vacuum is None:
            marshak_vacuum = symmetrize is not True and _marshak_applies(
                grid, bc, active, mask_bc, self._moment_op_cls, hybrid)
        self.marshak_vacuum = bool(marshak_vacuum)
        coeffs = self._coeffs if self._coeffs is not None else _SDPN_C
        r = _diag_similarity(coeffs[self._order])
        identity = r is not None and all(abs(x - 1.0) < 1e-12 for x in r)
        # The symmetrizing congruence (SDP3 CG path) mixes the moments over the
        # whole domain, so it cannot carry a per-moment hybrid mask; the plain
        # non-symmetric block can. Diagonal-similarity symmetrization (SDP1/SDP2)
        # keeps the moments separable and stays compatible.
        if symmetrize is None:
            if r is not None:
                symmetrize = not identity and not self.marshak_vacuum
            elif hybrid:
                symmetrize = False
            else:
                symmetrize = (not self.marshak_vacuum and
                              _congruence_available(self._order, grid, bc,
                                                    active, mask_bc,
                                                    coeffs=self._coeffs))
        if hybrid and symmetrize and r is None:
            raise ValueError("hybrid_mask cannot use the symmetrizing-congruence "
                             "path (it mixes moments across the whole domain); "
                             "pass symmetrize=False for a hybrid SDP3 solve")
        self._symmetrize = bool(symmetrize)
        self._diag_sym = r is not None                # similarity vs congruence
        if self._symmetrize and self.marshak_vacuum:
            raise ValueError("symmetrize is incompatible with marshak_vacuum")
        symmetric = identity or self._symmetrize
        self._spn_precond = 2 if spn_precondition is True else int(spn_precondition)
        if self._spn_precond < 0:
            raise ValueError("spn_precondition must be a non-negative sweep count")
        if self._spn_precond and symmetric:
            raise ValueError("spn_precondition applies only to the "
                             "non-symmetric SDPN path; this block is "
                             "symmetric(-ized) and already runs CG")
        if linear_solver is None:
            linear_solver = "cg" if symmetric else "gmres"
        super().__init__(*args, linear_solver=linear_solver, **kwargs)
        if self._spn_precond:
            self.preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                                    self._spn_precond)
                             for op in self._precond_ops]
        # Adjoint solve of a non-symmetric block runs on its algebraic
        # transpose, so the Neumann preconditioner must sweep the transpose
        # too (the Jacobi diagonal is shared). The SPN companion is symmetric,
        # hence equally close to the SDPN block and to its transpose: reuse it.
        self._adjoint_preconds = None
        if not all(op.symmetric for op in self.ops):
            self._adjoint_preconds = (self.preconds if self._spn_precond else [
                neumann_preconditioner(op.apply_transpose, op.inv_diag,
                                       self._precond_degree)
                for op in self.ops])

    def _linear_components(self, g, adjoint):
        if adjoint and not self.ops[g].symmetric:
            return self.ops[g].apply_transpose, self._adjoint_preconds[g]
        return super()._linear_components(g, adjoint)

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        if not self.symmetric_operator:
            raise ValueError("symmetric_operator=False is diffusion-only")
        if self._symmetrize and not self._diag_sym:
            self.ops = [CongruentSDPNOperator(
                self.xp, grid, diffusion[g], sigma_t[g], removal[g],
                order=self._order, bc=bc, active=self.active,
                mask_bc=self.mask_bc, op_cls=self._moment_op_cls,
                coeffs=self._coeffs) for g in range(self.n_groups)]
            return
        bg = self._g_coeffs[self._order] if self.marshak_vacuum else None
        self.ops = [SDPNGroupOperator(self.xp, grid, diffusion[g], sigma_t[g],
                                      removal[g], order=self._order, bc=bc,
                                      active=self.active, mask_bc=self.mask_bc,
                                      op_cls=self._moment_op_cls,
                                      coeffs=self._coeffs, boundary_g=bg,
                                      symmetrize=self._symmetrize,
                                      hybrid_mask=self.hybrid_mask,
                                      hybrid_confine=self.hybrid_confine)
                    for g in range(self.n_groups)]
        if self._spn_precond:
            # Matched-order standard-SPN companion blocks (symmetric tables,
            # same D_i and boundary treatment) for the Neumann preconditioner.
            bg_spn = _SPN_G[self._order] if self.marshak_vacuum else None
            self._precond_ops = [SDPNGroupOperator(
                self.xp, grid, diffusion[g], sigma_t[g], removal[g],
                order=self._order, bc=bc, active=self.active,
                mask_bc=self.mask_bc, op_cls=self._moment_op_cls,
                coeffs=_SPN_C, boundary_g=bg_spn,
                hybrid_mask=self.hybrid_mask,
                hybrid_confine=self.hybrid_confine)
                for g in range(self.n_groups)]

    # Spatial stencil for each moment (overridden by the triangular solver).
    _moment_op_cls = None

    def _initial_state(self):
        M = self._order + 1
        state = []
        for _ in range(self.n_groups):
            u = self.xp.zeros((M,) + self.grid.shape, dtype=self.dtype)
            u[0] = 1.0
            state.append(u)
        return state

    def _moment_weights(self, g, adjoint):
        """(source weights, flux read-out weights) of group g's block.

        Group coupling enters the block as the rank-one map src (x) phi0; its
        transpose is phi0 (x) src, so the adjoint swaps the two. For SDPN they
        genuinely differ (SDP2: src = c1[:, 0], phi0 = c1[0, :]).
        """
        op = self.ops[g]
        if adjoint:
            return op.phi0_weights, op.src_weights
        return op.src_weights, op.phi0_weights

    def _rhs(self, g, q0, adjoint=False):
        # Isotropic source (fission/k + in-scatter into phi0) distributed over
        # the moment rows by the first column of c^(1).
        w, _ = self._moment_weights(g, adjoint)
        M = self._order + 1
        rhs = self.xp.empty((M,) + self.grid.shape, dtype=self.dtype)
        for i in range(M):
            rhs[i] = w[i] * q0
            # Hybrid: confine the higher-moment source to the subdomain so those
            # moments stay exactly zero (pure diffusion) outside it. The
            # adjoint carries this mask on the read-out instead (transpose).
            if i > 0 and self.hybrid_mask is not None and not adjoint:
                rhs[i] = rhs[i] * self.hybrid_mask
        return rhs

    def _phi(self, state_g, adjoint=False):
        _, w = self._moment_weights(0, adjoint)
        phi = w[0] * state_g[0]
        for j in range(1, len(w)):
            if w[j] != 0.0:
                u = state_g[j]
                if adjoint and self.hybrid_mask is not None:
                    u = self.hybrid_mask * u
                phi = phi + w[j] * u
        return phi


class SDP2EigenSolver(SDPNEigenSolver):
    """Simplified double-P2 (SDP2) k-eigenvalue solver -- 3-moment block, the
    degrees of freedom of SP5. See :class:`SDPNEigenSolver`."""

    _order = 2


class SDP3EigenSolver(SDPNEigenSolver):
    """Simplified double-P3 (SDP3) k-eigenvalue solver -- 4-moment block, the
    degrees of freedom of SP7. See :class:`SDPNEigenSolver`."""

    _order = 3


class SPNEigenSolver(SDPNEigenSolver):
    """Standard simplified-PN (SPN) k-eigenvalue solver in the unified U-form,
    order N in {1, 2, 3} = SP3/SP5/SP7 (SP3 here reproduces the dedicated
    symmetric :class:`SP3EigenSolver`). Same block machinery as
    :class:`SDPNEigenSolver` with the standard SPN coefficient matrices, so SP5
    and SP7 -- which garnet otherwise lacks -- come for free and enable the
    matched-DoF SPN-vs-SDPN comparison of Carreno et al. (2024): SP3/SDP1,
    SP5/SDP2, SP7/SDP3."""

    _coeffs = _SPN_C
    _g_coeffs = _SPN_G


class SP1EigenSolver(SPNEigenSolver):
    """Standard SP1 k-eigenvalue solver (1-moment block) -- the P1/diffusion
    equation run through the general U-form machinery, so it must reproduce
    :class:`DiffusionEigenSolver` exactly (same D = 1/(3 Sigma_1), removal and
    boundary law; with marshak_vacuum=True the coupled condition degenerates
    to the same alpha = 1/2 Robin term). Exists as the order-0 member of the
    SPN family for like-for-like hierarchy comparisons and as an end-to-end
    consistency check of the block machinery."""

    _order = 0


class SP5EigenSolver(SPNEigenSolver):
    """Standard SP5 k-eigenvalue solver (3-moment block)."""

    _order = 2


class SP7EigenSolver(SPNEigenSolver):
    """Standard SP7 k-eigenvalue solver (4-moment block)."""

    _order = 3
