"""Matrix-free preconditioned Krylov solvers (CG and restarted GMRES).

Written against the NumPy/CuPy-common API: on GPU all vectors stay device-
resident and every operation is a CUDA kernel; the implicit device->host
syncs are the scalar reductions (one per CG iteration, one per orthogonalized
basis vector for GMRES).

CG is the default everywhere -- the discretized diffusion/SP3 operators are
kept symmetric positive definite by construction (volume-weighted stencils on
cylindrical grids exist for exactly this reason). GMRES and BiCGStab are the
escape hatches for operators that cannot be symmetrized: every solver in
``LINEAR_SOLVERS`` accepts pcg's common keywords (``COMMON_SOLVER_KWARGS``)
and shares its stopping rule, so any call site can swap them in via
``linear_solver="gmres"`` / ``"bicgstab"``. GMRES has a monotone residual but
stores `restart` basis vectors; BiCGStab runs in constant memory with two
operator applies per step but converges irregularly and can restart on
breakdown -- the usual Krylov trade-off (BiCGStab is what CFD codes like
OpenFOAM default to for asymmetric matrices).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect

import numpy as np

from . import kernels
from .backend import asnumpy


@dataclass
class PCGWorkspace:
    """Persistent vector storage for a sequence of same-shaped PCG solves.

    Transient group solves revisit the same operator shape thousands of times.
    Allocating ``x``, ``r``, ``z``, ``p``, and ``A p`` on every visit is cheap
    on a CPU but creates allocator traffic and prevents CUDA-graph capture on a
    GPU.  A workspace owns those five arrays and may be reused after the
    previous solution is no longer needed independently.  In garnet's
    transient drivers there is one workspace per energy group, and the
    returned solution remains the workspace's ``x`` array between calls.

    ``operator_out`` declares that ``apply_A(v, out=array)`` is supported.
    garnet's Cartesian and triangular finite-volume operators provide that
    interface.  Generic user callables can still use the workspace with
    ``operator_out=False``; only their operator result remains allocationful.
    """

    x: object
    r: object
    z: object
    p: object
    ap: object
    operator_out: bool = False
    fallback_start: object | None = None
    fallback_count: int = 0
    graph_scalars: dict = field(default_factory=dict)
    graph_key: tuple | None = None
    graph: object | None = None
    graph_error: str | None = None
    graph_captures: int = 0
    graph_replays: int = 0

    @classmethod
    def like(cls, template, *, operator_out=False, fallback=False):
        xp = kernels.module_of(template)
        values = [xp.empty_like(template) for _ in range(5)]
        start = xp.zeros_like(template) if fallback else None
        return cls(*values, operator_out=bool(operator_out),
                   fallback_start=start)

    def scalars(self, xp, dtype):
        """Persistent 0-D recurrence coefficients used by CUDA graphs."""
        dtype = np.dtype(dtype)
        if not self.graph_scalars:
            self.graph_scalars = {
                name: xp.empty((), dtype=dtype)
                for name in ("rz", "rz_new", "pap", "alpha", "beta")}
        return self.graph_scalars

    def clear_graph(self):
        self.graph_key = None
        self.graph = None
        self.graph_error = None

    def validate(self, template):
        """Reject accidental reuse across incompatible shape/dtype/backends."""
        xp = kernels.module_of(template)
        for name in ("x", "r", "z", "p", "ap"):
            value = getattr(self, name)
            if kernels.module_of(value) is not xp:
                raise ValueError("PCG workspace backend does not match the solve")
            if value.shape != template.shape or value.dtype != template.dtype:
                raise ValueError(
                    f"PCG workspace {name} has shape/dtype "
                    f"{value.shape}/{value.dtype}, expected "
                    f"{template.shape}/{template.dtype}")
        if self.fallback_start is not None:
            value = self.fallback_start
            if (kernels.module_of(value) is not xp
                    or value.shape != template.shape
                    or value.dtype != template.dtype):
                raise ValueError("PCG fallback workspace does not match the solve")


@dataclass
class FGMRESWorkspace:
    """Persistent storage for repeated same-shaped flexible GMRES solves.

    The contiguous ``V`` and ``Z`` arrays make GPU Arnoldi projections a
    matrix-vector reduction instead of one host-synchronizing dot product per
    stored vector.  Allocating the maximum restart basis once also avoids
    repeatedly growing and releasing two lists of large device arrays at every
    adaptive step attempt.
    """

    x: object
    r: object
    w: object
    V: object
    Z: object
    restart: int
    operator_out: bool = False

    @classmethod
    def like(cls, template, restart=30, *, operator_out=False):
        restart = int(restart)
        if restart < 1:
            raise ValueError("restart must be positive")
        xp = kernels.module_of(template)
        shape = tuple(template.shape)
        return cls(
            xp.empty_like(template), xp.empty_like(template),
            xp.empty_like(template),
            xp.empty((restart + 1,) + shape, dtype=template.dtype),
            xp.empty((restart,) + shape, dtype=template.dtype),
            restart, bool(operator_out))

    def validate(self, template, restart):
        xp = kernels.module_of(template)
        if int(restart) > self.restart:
            raise ValueError("FGMRES workspace restart capacity is too small")
        for name in ("x", "r", "w"):
            value = getattr(self, name)
            if (kernels.module_of(value) is not xp
                    or value.shape != template.shape
                    or value.dtype != template.dtype):
                raise ValueError(f"FGMRES workspace {name} is incompatible")
        expected = tuple(template.shape)
        if (kernels.module_of(self.V) is not xp
                or self.V.shape[1:] != expected
                or self.V.dtype != template.dtype
                or kernels.module_of(self.Z) is not xp
                or self.Z.shape[1:] != expected
                or self.Z.dtype != template.dtype):
            raise ValueError("FGMRES basis workspace is incompatible")


def pcg_workspaces(operators, templates, *, fallback=False):
    """Build one compatible :class:`PCGWorkspace` per energy group."""
    if len(operators) != len(templates):
        raise ValueError("operators and templates must have the same length")
    result = []
    for operator, value in zip(operators, templates):
        try:
            supports_out = "out" in inspect.signature(operator.apply).parameters
        except (TypeError, ValueError):
            supports_out = False
        result.append(PCGWorkspace.like(
            value, operator_out=supports_out, fallback=fallback))
    return result


class MixedPrecisionPreconditioner:
    """FP32 polynomial preconditioner for an FP64 outer Krylov solve.

    ``apply_A_low`` and ``inv_diag_low`` belong to a shadow low-precision
    operator. Input residuals are cast into persistent low-precision storage;
    only the approximate correction is cast back to the outer dtype. Physical
    operator applications, recurrence coefficients, true residuals, and
    stopping decisions remain in the outer solver's dtype.
    """

    garnet_out = True
    garnet_scratch = False
    mixed_precision = True

    def __init__(self, apply_A_low, inv_diag_low, degree, outer_dtype):
        self.apply_A = apply_A_low
        self.inv_diag = inv_diag_low
        self.degree = int(degree)
        self.outer_dtype = np.dtype(outer_dtype)
        self.xp = kernels.module_of(inv_diag_low)
        self.r = self.xp.empty_like(inv_diag_low)
        self.z = self.xp.empty_like(inv_diag_low)
        self.az = self.xp.empty_like(inv_diag_low)
        try:
            self.operator_out = (
                "out" in inspect.signature(apply_A_low).parameters)
        except (TypeError, ValueError):
            self.operator_out = False

    def __call__(self, residual, out=None):
        if self.degree and not getattr(self, "_damped", False):
            # Same damped splitting as neumann_preconditioner (estimated on the
            # low-precision shadow operator, once, on first use).
            self.inv_diag = jacobi_damping(
                self.apply_A, self.inv_diag) * self.inv_diag
            self._damped = True
        kernels.mixed_jacobi_start(
            self.xp, residual, self.inv_diag, self.r, self.z)
        for _ in range(self.degree):
            if self.operator_out:
                az = self.apply_A(self.z, out=self.az)
            else:
                self.xp.copyto(self.az, self.apply_A(self.z))
                az = self.az
            kernels.neumann_step(
                self.xp, self.z, self.r, az, self.inv_diag)
        if out is None:
            return self.z.astype(self.outer_dtype)
        out[...] = self.z
        return out


def mixed_precision_preconditioner(apply_A_low, inv_diag_low, degree,
                                   outer_dtype=np.float64):
    """Construct an allocation-stable low-precision polynomial preconditioner."""
    low_dtype = np.dtype(inv_diag_low.dtype)
    outer_dtype = np.dtype(outer_dtype)
    if low_dtype.itemsize >= outer_dtype.itemsize:
        raise ValueError("mixed preconditioner dtype must be narrower than outer dtype")
    return MixedPrecisionPreconditioner(
        apply_A_low, inv_diag_low, degree, outer_dtype)


def jacobi_damping(apply_A, inv_diag, steps=12):
    """Damping factor omega = 2 / (lambda_min + lambda_max) of D^-1 A.

    The extremes are estimated by ``steps`` Lanczos iterations on the
    symmetric scaling D^-1/2 A D^-1/2 (same spectrum as D^-1 A): the largest
    Ritz value plus its residual bound for lambda_max (an over-estimate, the
    safe side), the smallest Ritz value for lambda_min (also an over-estimate,
    again the safe side). Costs ``steps`` operator applies and ~2 ``steps``
    reductions, once. Returns 1.0 (undamped) when the estimate does not apply
    -- complex or non-positive diagonals (e.g. the frequency-domain noise
    operator) -- or comes out non-finite.
    """
    xp = kernels.module_of(inv_diag)
    if (not np.issubdtype(np.dtype(inv_diag.dtype), np.floating)
            or not bool(xp.all(inv_diag > 0))):
        return 1.0
    scale = xp.sqrt(inv_diag)
    # Deterministic pseudo-random start: overlaps every eigenvector,
    # including the high-frequency ones that set lambda_max.
    start = np.random.default_rng(12345).standard_normal(inv_diag.shape)
    v = xp.asarray(start, dtype=inv_diag.dtype)
    v = v / xp.sqrt(xp.sum(v * v))
    v_prev = xp.zeros_like(v)
    alphas, betas, beta = [], [], 0.0
    for _ in range(int(steps)):
        w = scale * apply_A(scale * v)
        alpha = float(xp.sum(w * v))
        w = w - alpha * v - beta * v_prev
        beta = float(xp.sqrt(xp.sum(w * w)))
        alphas.append(alpha)
        betas.append(beta)
        if not beta > 1e-12 * abs(alpha):      # invariant subspace (or NaN)
            break
        v_prev, v = v, w / beta
    n = len(alphas)
    tridiagonal = (np.diag(alphas) + np.diag(betas[:n - 1], 1)
                   + np.diag(betas[:n - 1], -1))
    ritz, vectors = np.linalg.eigh(tridiagonal)
    lam_max = ritz[-1] + abs(betas[-1] * vectors[-1, -1])
    lam_min = max(ritz[0], 0.0)
    omega = 2.0 / (lam_min + lam_max)
    return float(omega) if np.isfinite(omega) and omega > 0.0 else 1.0


def neumann_preconditioner(apply_A, inv_diag, degree, damping="auto"):
    """Truncated Neumann-series (polynomial) preconditioner.

    Approximates A^-1 by the first `degree`+1 terms of the Neumann series in
    the damped Jacobi splitting,  sum_k (I - w D^-1 A)^k w D^-1,  evaluated as
    `degree` damped-Jacobi correction sweeps: z <- z + w D^-1 (r - A z). Each
    application costs `degree` extra operator applies -- pure stencil
    streaming, no dot products -- and cuts the CG iteration count (and with it
    the per-iteration global reductions, the only synchronization points on
    the GPU). Degree 0 is plain Jacobi.

    Definiteness: for SPD A the polynomial maps an eigenvalue lam of D^-1 A to
    1 - (1 - w lam)^(degree+1), so EVEN degrees are always SPD but ODD degrees
    are SPD only while w lam_max < 2. Undamped (w = 1) that holds for a
    diagonally dominant diffusion stencil, but not for the SP5/SP7 moment
    blocks, whose inter-moment coupling pushes lam_max(D^-1 A) slightly past
    2 (~2.02): there an odd undamped degree is indefinite and CG crawls (SP7
    degree 1: ~350 iterations against 21 for plain Jacobi). ``damping="auto"``
    (default) therefore sets w = 2/(lam_min + lam_max) from a short Lanczos
    estimate (:func:`jacobi_damping`), computed lazily on the first
    application -- a one-off ``~12`` operator applies, no change to the
    per-application cost. For diffusion (lam_max < 2) w stays within a few
    percent of 1 and the iteration counts are unchanged to +-1; pass a float
    to fix w, or ``damping=None``/``1.0`` for the undamped splitting.
    GPU fine-mesh diffusion reference: Y. E et al., Nuclear Engineering and
    Design 320 (2017), 88-102, doi:10.1016/j.nucengdes.2017.04.001.
    Their comparison motivates the Neumann-PCG option; its degree and benefit
    are workload dependent.
    """
    xp = kernels.module_of(inv_diag)
    try:
        operator_out = "out" in inspect.signature(apply_A).parameters
    except (TypeError, ValueError):
        operator_out = False
    # Degree 0 is plain Jacobi, whose scale no Krylov method sees: no damping.
    weights = {"inv_diag": inv_diag if degree == 0 or damping is None
               or damping == "auto" else float(damping) * inv_diag,
               "pending": degree > 0 and damping == "auto"}

    def apply(r, out=None, scratch=None):
        if weights["pending"]:
            weights["inv_diag"] = jacobi_damping(apply_A, inv_diag) * inv_diag
            weights["pending"] = False
        inv_diag_w = weights["inv_diag"]
        if out is None:
            z = inv_diag_w * r
        else:
            xp.multiply(inv_diag_w, r, out=out)
            z = out
        for _ in range(degree):
            if scratch is None:
                az = apply_A(z)
            elif operator_out:
                az = apply_A(z, out=scratch)
            else:
                xp.copyto(scratch, apply_A(z))
                az = scratch
            kernels.neumann_step(xp, z, r, az, inv_diag_w)
        return z

    # PCG uses this opt-in instead of guessing whether an arbitrary user
    # preconditioner accepts ``out=`` (and potentially hiding a real TypeError).
    apply.garnet_out = True
    apply.garnet_scratch = True
    return apply


def _callable_owner(value):
    """Stable identity for a bound method, whose wrapper object is ephemeral."""
    return getattr(value, "__self__", value)


def _pcg_graph_capable(xp, workspace, precond, vector_size):
    """Whether one PCG iteration is allocation-free and capturable."""
    return (kernels.is_cupy(xp) and workspace is not None
            and workspace.operator_out
            and kernels.use_fused(xp, "krylov")
            and vector_size <= kernels.DOT_FUSED_MAX
            and (precond is None or getattr(precond, "garnet_out", False)))


def pcg(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
        precond=None, check_every=1, raise_on_fail=True, workspace=None,
        residual_replace_every=0, fallback_precond=None, graph_block=0):
    """Solve A x = b with preconditioned CG.

    apply_A  : callable, x -> A x (A symmetric positive definite)
    b        : right-hand side (any shape; treated as a flat vector)
    x0       : initial guess (warm start), not modified
    inv_diag : elementwise inverse of diag(A) (the Jacobi preconditioner)
    precond  : optional callable r -> z applying a custom SPD preconditioner
               (e.g. neumann_preconditioner); default is Jacobi via inv_diag
    check_every : test convergence only every k iterations. Each test is a
               ``float(...)`` device->host reduction (a sync); on GPU, spacing
               them out cuts the per-iteration stall when many iterations are
               expected (e.g. Jacobi-CG on a diffusion operator).
    raise_on_fail : if False, return the current iterate after ``maxiter``
               instead of raising -- for use as an *inexact* preconditioner
               (e.g. a synthetic-acceleration solve) where a partial result is
               still useful.
    workspace : optional :class:`PCGWorkspace` reused across same-shaped solves.
               Its solution array is returned directly and remains valid until
               that workspace is used again.  With ``operator_out=True`` the
               operator must accept ``apply_A(v, out=array)``.
    residual_replace_every : recompute the FP64 true residual and restart the
               recurrence every N iterations. Zero disables replacement. This
               limits recursive-residual drift when a lower-precision
               preconditioner is used.
    fallback_precond : optional safe preconditioner used to restart from zero
               when the first solve develops a non-finite checked residual or
               reaches ``maxiter``. Intended for automatic FP64 fallback from
               a mixed-precision preconditioner.
    graph_block : on CuPy, capture this many allocation-free PCG iterations
               into one CUDA graph. It must equal ``check_every`` so graph and
               uncaptured paths make convergence decisions at identical
               iterations. Unsupported configurations fall back transparently
               and record ``workspace.graph_error``.

    Returns (x, n_iterations).
    """
    # The vector updates and reductions go through garnet.kernels, which fuses
    # each of them into one CUDA kernel on GPU (and is the plain expression on
    # CPU). That matters here more than anywhere else: a CG iteration is three
    # reductions and three vector updates over arrays that fit in cache, so
    # written as separate array expressions it is almost entirely kernel-launch
    # overhead. The coefficients deliberately stay 0-d *device* scalars -- only
    # the convergence test (every `check_every` iterations) touches the host.
    dot = lambda u, v: kernels.dot(xp, u, v)
    M = precond if precond is not None else (lambda r: inv_diag * r)
    residual_replace_every = int(residual_replace_every)
    graph_block = int(graph_block)
    if residual_replace_every < 0:
        raise ValueError("residual_replace_every must be non-negative")
    if graph_block < 0:
        raise ValueError("graph_block must be non-negative")
    if graph_block and graph_block != check_every:
        raise ValueError("graph_block must equal check_every")

    def apply_preconditioner(source, out=None):
        if out is None:
            return M(source)
        if precond is None:
            xp.multiply(inv_diag, source, out=out)
            return out
        if getattr(precond, "garnet_out", False):
            kwargs = ({"scratch": ap_buf}
                      if getattr(precond, "garnet_scratch", False) else {})
            return precond(source, out=out, **kwargs)
        xp.copyto(out, precond(source))
        return out

    if workspace is None:
        x = x0.copy()
        r = b - apply_A(x)
        z_buf = p_buf = ap_buf = None
    else:
        workspace.validate(x0)
        x, r = workspace.x, workspace.r
        z_buf, p_buf, ap_buf = workspace.z, workspace.p, workspace.ap
        if x0 is not x:
            xp.copyto(x, x0)
        if workspace.operator_out:
            apply_A(x, out=ap_buf)
        else:
            xp.copyto(ap_buf, apply_A(x))
        xp.subtract(b, ap_buf, out=r)

    def retry_with_fallback(iterations):
        if fallback_precond is None:
            return None
        if workspace is None:
            safe_start = xp.zeros_like(x0)
        else:
            if workspace.fallback_start is None:
                workspace.fallback_start = xp.zeros_like(x0)
            safe_start = workspace.fallback_start
            workspace.fallback_count += 1
        solved, fallback_iterations = pcg(
            apply_A, b, safe_start, inv_diag, xp, rtol=rtol, atol=atol,
            maxiter=maxiter, precond=fallback_precond,
            check_every=check_every, raise_on_fail=raise_on_fail,
            workspace=workspace,
            residual_replace_every=residual_replace_every)
        return solved, iterations + fallback_iterations

    b_norm = float(xp.sqrt(dot(b, b)))
    if b_norm == 0.0:
        # A x = 0 has the exact solution 0; iterating from a warm start would
        # only chase a relative target of 0 (raising after maxiter).
        x[...] = 0
        return x, 0
    residual_tolerance_squared = max(rtol * b_norm, atol) ** 2
    if float(dot(r, r)) <= residual_tolerance_squared:
        return x, 0

    z = apply_preconditioner(r, z_buf)
    if workspace is None:
        p = z.copy()
    else:
        xp.copyto(p_buf, z)
        p = p_buf
    rz = dot(r, z)
    tiny = np.finfo(b.dtype).tiny

    graph_active = bool(graph_block and _pcg_graph_capable(
        xp, workspace, precond, b.size))
    if graph_block and not graph_active and workspace is not None:
        workspace.graph_error = (
            "capture needs CuPy fused reductions, an out= operator and an "
            f"allocation-free preconditioner; vector size must be <= "
            f"{kernels.DOT_FUSED_MAX}")

    if graph_active:
        scalars = workspace.scalars(xp, b.dtype)
        kernels.dot(xp, r, z, out=scalars["rz"])
        graph_key = (id(_callable_owner(apply_A)), id(precond), graph_block,
                     tuple(b.shape), np.dtype(b.dtype).str)
        if workspace.graph_key != graph_key:
            workspace.clear_graph()
            workspace.graph_key = graph_key

        def graph_iteration():
            apply_A(p, out=ap_buf)
            kernels.dot(xp, p, ap_buf, out=scalars["pap"])
            kernels.scalar_divide(
                xp, scalars["alpha"], scalars["rz"], scalars["pap"])
            kernels.cg_update(
                xp, x, r, p, ap_buf, scalars["alpha"])
            apply_preconditioner(r, z_buf)
            kernels.dot(xp, r, z_buf, out=scalars["rz_new"])
            kernels.scalar_divide(
                xp, scalars["beta"], scalars["rz_new"], scalars["rz"])
            kernels.cg_direction(xp, p, z_buf, scalars["beta"])
            kernels.scalar_copy(xp, scalars["rz"], scalars["rz_new"])

        def direct_block(width):
            for _ in range(width):
                graph_iteration()

        iterations = 0
        # The first direct block compiles every kernel and populates CuPy's
        # memory pool without adding artificial work: it is the solve's actual
        # first block. Capture starts only after that warm block unless this
        # workspace already owns a compatible graph from an earlier solve.
        warmed = workspace.graph is not None or workspace.graph_error is not None
        while iterations < maxiter:
            width = min(graph_block, maxiter - iterations)
            if width != graph_block or not warmed:
                direct_block(width)
                warmed = True
            elif workspace.graph is not None:
                workspace.graph.launch()
                workspace.graph_replays += 1
            elif workspace.graph_error is None:
                stream = xp.cuda.Stream(non_blocking=True)
                try:
                    with stream:
                        stream.begin_capture()
                        direct_block(graph_block)
                        workspace.graph = stream.end_capture()
                    workspace.graph_captures += 1
                    workspace.graph.launch()
                    workspace.graph_replays += 1
                except Exception as exc:
                    try:
                        stream.end_capture()
                    except Exception:
                        pass
                    workspace.graph_error = f"{type(exc).__name__}: {exc}"
                    direct_block(graph_block)
            else:
                direct_block(graph_block)
            iterations += width

            replaced = (residual_replace_every > 0
                        and iterations % residual_replace_every == 0)
            if replaced:
                apply_A(x, out=ap_buf)
                xp.subtract(b, ap_buf, out=r)
                apply_preconditioner(r, z_buf)
                xp.copyto(p_buf, z_buf)
                kernels.dot(xp, r, z_buf, out=scalars["rz"])

            if replaced or iterations % check_every == 0:
                residual_norm_squared = float(dot(r, r))
                if np.isfinite(residual_norm_squared) and residual_norm_squared <= residual_tolerance_squared:
                    return x, iterations
                if not np.isfinite(residual_norm_squared):
                    retried = retry_with_fallback(iterations)
                    if retried is not None:
                        return retried
                    break
        retried = retry_with_fallback(iterations)
        if retried is not None:
            return retried
        if raise_on_fail:
            raise RuntimeError(
                f"PCG failed to converge in {maxiter} iterations "
                f"(residual {float(xp.sqrt(dot(r, r))):.3e}, "
                f"target {residual_tolerance_squared**0.5:.3e})")
        return x, iterations

    for it in range(1, maxiter + 1):
        if workspace is None:
            Ap = apply_A(p)
        elif workspace.operator_out:
            Ap = apply_A(p, out=ap_buf)
        else:
            xp.copyto(ap_buf, apply_A(p))
            Ap = ap_buf
        # A delayed convergence check can leave an exactly converged recurrence
        # running for a few no-op iterations. Regularizing both denominators
        # keeps that benign 0/0 case from poisoning the solution with NaNs.
        alpha = rz / (dot(p, Ap) + tiny)
        kernels.cg_update(xp, x, r, p, Ap, alpha)
        replaced = (residual_replace_every > 0
                    and it % residual_replace_every == 0)
        if replaced:
            if workspace is None:
                r = b - apply_A(x)
            elif workspace.operator_out:
                apply_A(x, out=ap_buf)
                xp.subtract(b, ap_buf, out=r)
            else:
                xp.copyto(ap_buf, apply_A(x))
                xp.subtract(b, ap_buf, out=r)
        if replaced or it % check_every == 0:
            residual_norm_squared = float(dot(r, r))
            if np.isfinite(residual_norm_squared) and residual_norm_squared <= residual_tolerance_squared:
                return x, it
            if not np.isfinite(residual_norm_squared):
                retried = retry_with_fallback(it)
                if retried is not None:
                    return retried
                break
        z = apply_preconditioner(r, z_buf)
        if replaced:
            # A recomputed residual starts a new CG recurrence; old conjugacy
            # is no longer reliable after recursive-residual drift.
            if workspace is None:
                p = z.copy()
            else:
                xp.copyto(p_buf, z)
                p = p_buf
            rz = dot(r, z)
            continue
        rz_new = dot(r, z)
        kernels.cg_direction(xp, p, z, rz_new / (rz + tiny))
        rz = rz_new
    retried = retry_with_fallback(it)
    if retried is not None:
        return retried
    if raise_on_fail:
        raise RuntimeError(
            f"PCG failed to converge in {maxiter} iterations "
            f"(residual {float(xp.sqrt(dot(r, r))):.3e}, target {residual_tolerance_squared**0.5:.3e})"
        )
    return x, maxiter


def fixed_pcg(apply_A, b, x0, inv_diag, xp, iterations=1, precond=None,
              workspace=None):
    """Apply a fixed number of PCG steps without convergence synchronization.

    This is an inexact-preconditioner primitive, not a standalone converged
    solve.  Flexible GMRES supplies the true outer residual test.  Keeping the
    PCG work fixed means every recurrence coefficient remains a zero-dimensional
    device scalar and no per-group value is transferred to the host.  Tiny
    denominator regularization makes an already-exact warm start a no-op.

    Returns ``(x, iterations)`` with the requested deterministic work count.
    """
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("fixed PCG iterations must be non-negative")
    M = precond if precond is not None else (lambda r: inv_diag * r)

    if workspace is None:
        x = x0.copy()
        r = b - apply_A(x)
        z_buf = p_buf = ap_buf = None
    else:
        workspace.validate(x0)
        x, r = workspace.x, workspace.r
        z_buf, p_buf, ap_buf = workspace.z, workspace.p, workspace.ap
        if x0 is not x:
            xp.copyto(x, x0)
        if workspace.operator_out:
            apply_A(x, out=ap_buf)
        else:
            xp.copyto(ap_buf, apply_A(x))
        xp.subtract(b, ap_buf, out=r)
    if iterations == 0:
        return x, 0

    def apply_preconditioner(source, out=None):
        if out is None:
            return M(source)
        if precond is None:
            xp.multiply(inv_diag, source, out=out)
            return out
        if getattr(precond, "garnet_out", False):
            kwargs = ({"scratch": ap_buf}
                      if getattr(precond, "garnet_scratch", False) else {})
            return precond(source, out=out, **kwargs)
        xp.copyto(out, precond(source))
        return out

    z = apply_preconditioner(r, z_buf)
    if workspace is None:
        p = z.copy()
    else:
        xp.copyto(p_buf, z)
        p = p_buf
    rz = kernels.dot(xp, r, z)
    tiny = np.finfo(b.dtype).tiny
    for step in range(iterations):
        if workspace is None:
            ap = apply_A(p)
        elif workspace.operator_out:
            ap = apply_A(p, out=ap_buf)
        else:
            xp.copyto(ap_buf, apply_A(p))
            ap = ap_buf
        alpha = rz / (kernels.dot(xp, p, ap) + tiny)
        kernels.cg_update(xp, x, r, p, ap, alpha)
        if step + 1 == iterations:
            break
        z = apply_preconditioner(r, z_buf)
        rz_new = kernels.dot(xp, r, z)
        kernels.cg_direction(xp, p, z, rz_new / (rz + tiny))
        rz = rz_new
    return x, iterations


def gmres(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
          precond=None, restart=30, raise_on_fail=True, check_every=1,
          workspace=None):
    """Solve A x = b with restarted, right-preconditioned GMRES(m).

    Same interface, warm start and stopping rule as :func:`pcg` (true
    residual norm <= max(rtol * ||b||, atol)), but A need *not* be symmetric
    positive definite -- this is the option for operators that cannot be
    symmetrized (upwinded drift terms, non-symmetric acceleration schemes,
    asymmetric couplings). Right preconditioning solves A M^-1 u = b with
    x = M^-1 u, so the Arnoldi residual *is* the true residual and the
    stopping test matches pcg's exactly; the preconditioner must be linear
    (Jacobi and the Neumann polynomial both are), which lets the update be
    reassembled as x += M(V y) without storing the preconditioned basis.

    Costs relative to CG on an SPD system: `restart` basis vectors of extra
    memory and O(j) dot products at Arnoldi step j -- use CG whenever SPD
    holds. `restart` trades memory/orthogonalization work against the
    convergence penalty of discarding the Krylov space each cycle.

    raise_on_fail : if False, return the current iterate after ``maxiter``
               instead of raising -- the same escape hatch :func:`pcg` has, for
               call sites where a partial result is still useful (an inexact
               within-group solve inside an outer iteration that revisits it).
    check_every, workspace : accepted for signature compatibility with
               :func:`pcg` (see :data:`COMMON_SOLVER_KWARGS`) and ignored: the
               Arnoldi residual estimate is a host scalar at every step anyway,
               and the basis is allocated per restart cycle (use
               :func:`fgmres` for a persistent workspace).

    Returns (x, n_iterations) where an iteration is one operator apply.
    """
    del check_every, workspace          # inapplicable; see docstring
    dot = lambda u, v: float(xp.sum(u * v))
    norm = lambda u: float(xp.sqrt(xp.sum(u * u)))
    M = precond if precond is not None else (lambda r: inv_diag * r)

    b_norm = norm(b)
    if b_norm == 0.0:
        return xp.zeros_like(x0), 0       # A x = 0: exact solution
    x = x0.copy()
    residual_tolerance = max(rtol * b_norm, atol)
    r = b - apply_A(x)
    residual_norm = norm(r)
    if residual_norm <= residual_tolerance:
        return x, 0

    iteration_count = 0
    while iteration_count < maxiter:
        cycle_length = min(restart, maxiter - iteration_count)
        V = [r / residual_norm]                       # Arnoldi basis (device arrays)
        H = np.zeros((cycle_length + 1, cycle_length))            # Hessenberg, host
        givens_cos, givens_sin = np.zeros(cycle_length), np.zeros(cycle_length)   # Givens rotations
        g = np.zeros(cycle_length + 1)                 # rotated residual vector
        g[0] = residual_norm
        basis_size = 0
        for j in range(cycle_length):
            iteration_count += 1
            w = apply_A(M(V[j]))
            h = np.zeros(j + 2)
            for i in range(j + 1):          # modified Gram-Schmidt
                h[i] = dot(w, V[i])
                w = w - h[i] * V[i]
            h[j + 1] = norm(w)
            breakdown = h[j + 1] <= 1e-300  # exact solution in the space
            if not breakdown:
                V.append(w / h[j + 1])
            # Rotate the new column into upper-triangular form.
            for i in range(j):
                h[i], h[i + 1] = (givens_cos[i] * h[i] + givens_sin[i] * h[i + 1],
                                  -givens_sin[i] * h[i] + givens_cos[i] * h[i + 1])
            denom = float(np.hypot(h[j], h[j + 1]))
            givens_cos[j], givens_sin[j] = (1.0, 0.0) if denom == 0.0 else (h[j] / denom, h[j + 1] / denom)
            h[j], h[j + 1] = denom, 0.0
            H[:j + 2, j] = h
            g[j + 1] = -givens_sin[j] * g[j]
            g[j] = givens_cos[j] * g[j]
            basis_size = j + 1
            if abs(g[basis_size]) <= residual_tolerance or breakdown:
                break
        y = np.linalg.solve(
            H[:basis_size, :basis_size],
            g[:basis_size],
        )   # small upper-triangular system
        basis_correction = float(y[0]) * V[0]
        for j in range(1, basis_size):
            basis_correction += float(y[j]) * V[j]
        x += M(basis_correction)                              # M linear: M(V y) = (M V) y
        r = b - apply_A(x)
        residual_norm = norm(r)
        if residual_norm <= residual_tolerance:
            return x, iteration_count
    if raise_on_fail:
        raise RuntimeError(
            f"GMRES failed to converge in {maxiter} iterations "
            f"(residual {residual_norm:.3e}, target {residual_tolerance:.3e})"
        )
    return x, iteration_count


def _fgmres_workspace_solve(apply_A, b, x0, xp, M, *, rtol, atol,
                            maxiter, restart, raise_on_fail, workspace):
    """Allocation-stable FGMRES implementation used when storage is supplied."""
    workspace.validate(x0, restart)
    x, r, w = workspace.x, workspace.r, workspace.w
    V, Z = workspace.V, workspace.Z

    def apply_operator(value, out):
        if workspace.operator_out:
            return apply_A(value, out=out)
        xp.copyto(out, apply_A(value))
        return out

    def apply_preconditioner(value, out):
        if getattr(M, "garnet_out", False):
            return M(value, out=out)
        xp.copyto(out, M(value))
        return out

    def norm(value):
        return float(xp.sqrt(kernels.dot(xp, value, value)))

    xp.copyto(x, x0)
    residual_tolerance = max(float(rtol) * norm(b), float(atol))
    apply_operator(x, w)
    xp.subtract(b, w, out=r)
    residual_norm = norm(r)
    if residual_norm <= residual_tolerance:
        return x, 0

    iteration_count = 0
    while iteration_count < maxiter:
        cycle_length = min(restart, maxiter - iteration_count)
        xp.multiply(r, 1.0 / residual_norm, out=V[0])
        H = np.zeros((cycle_length + 1, cycle_length))
        givens_cos, givens_sin = np.zeros(cycle_length), np.zeros(cycle_length)
        g = np.zeros(cycle_length + 1)
        g[0] = residual_norm
        basis_size = 0
        for j in range(cycle_length):
            iteration_count += 1
            apply_preconditioner(V[j], Z[j])
            apply_operator(Z[j], w)
            if kernels.is_cupy(xp):
                # Two-pass classical Gram--Schmidt (CGS2) retains MGS-level
                # orthogonality, but both projection passes remain on device.
                basis = V[:j + 1].reshape(j + 1, -1)
                flat_w = w.reshape(-1)
                first_projection = basis @ flat_w
                kernels.basis_accumulate(
                    xp, w, V, first_projection, j + 1, alpha=-1.0)
                second_projection = basis @ flat_w
                kernels.basis_accumulate(
                    xp, w, V, second_projection, j + 1, alpha=-1.0)
                projection_device = first_projection + second_projection
                wnorm = xp.sqrt(kernels.dot(xp, w, w))
                packed = xp.concatenate((projection_device, wnorm.reshape(1)))
                projection_host = asnumpy(packed)  # one host synchronization per Arnoldi step
                H[:j + 1, j] = projection_host[:-1]
                H[j + 1, j] = float(projection_host[-1])
            else:
                # Preserve modified Gram--Schmidt on CPU, where scalar access
                # does not synchronize a device and small BLAS calls cost more.
                for i in range(j + 1):
                    H[i, j] = float(kernels.dot(xp, V[i], w))
                    kernels.axpy_inplace(xp, w, V[i], -H[i, j])
                H[j + 1, j] = norm(w)
            breakdown = H[j + 1, j] <= 1e-300
            if not breakdown:
                xp.multiply(w, 1.0 / H[j + 1, j], out=V[j + 1])
            for i in range(j):
                h0, h1_host = H[i, j], H[i + 1, j]
                H[i, j] = givens_cos[i] * h0 + givens_sin[i] * h1_host
                H[i + 1, j] = -givens_sin[i] * h0 + givens_cos[i] * h1_host
            denom = float(np.hypot(H[j, j], H[j + 1, j]))
            if denom == 0.0:
                givens_cos[j], givens_sin[j] = 1.0, 0.0
            else:
                givens_cos[j] = H[j, j] / denom
                givens_sin[j] = H[j + 1, j] / denom
            H[j, j], H[j + 1, j] = denom, 0.0
            g[j + 1] = -givens_sin[j] * g[j]
            g[j] = givens_cos[j] * g[j]
            basis_size = j + 1
            if abs(g[basis_size]) <= residual_tolerance or breakdown:
                break

        y = np.linalg.solve(H[:basis_size, :basis_size], g[:basis_size])
        y_device = xp.asarray(y, dtype=x.dtype)
        kernels.basis_accumulate(xp, x, Z, y_device, basis_size)
        apply_operator(x, w)
        xp.subtract(b, w, out=r)
        residual_norm = norm(r)
        if residual_norm <= residual_tolerance:
            return x, iteration_count
    if raise_on_fail:
        raise RuntimeError(
            f"FGMRES failed to converge in {iteration_count} applies "
            f"(residual {residual_norm:.3e}, target {residual_tolerance:.3e})")
    return x, iteration_count


def fgmres(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0,
           maxiter=5000, precond=None, restart=30, raise_on_fail=True,
           workspace=None, check_every=1):
    """Solve a real non-symmetric system with flexible restarted GMRES.

    Reference: Saad, SIAM J. Sci. Comput. 14 (1993), 461-469,
    doi:10.1137/0914028, for variable right preconditioning.

    This has the same public signature and true-residual stopping rule as
    :func:`gmres`, but stores every preconditioned Arnoldi vector ``Z[j]``.
    Consequently ``precond`` may be variable, nonlinear, or an inexact
    iterative solve.  That extra basis is the essential distinction for the
    monolithic transient prototype: one application of its right
    preconditioner is an energy-group Gauss--Seidel sweep whose within-group
    PCG iteration count can change from one Arnoldi vector to the next.

    ``inv_diag`` supplies the default Jacobi preconditioner for interface
    compatibility. ``workspace`` may be an :class:`FGMRESWorkspace`; on GPU
    this also batches all Arnoldi projections into one host synchronization per
    iteration. ``check_every`` is accepted for signature compatibility with
    :func:`pcg` and ignored (every Arnoldi step already returns its residual
    estimate to the host). A ``workspace`` that is not an
    :class:`FGMRESWorkspace` (e.g. a :class:`PCGWorkspace` handed to every
    interchangeable solver) is ignored. Returns ``(x, n_iterations)``,
    counting operator applies.
    """
    del check_every                     # inapplicable; see docstring
    dot = lambda u, v: float(xp.sum(u * v))
    norm = lambda u: float(xp.sqrt(xp.sum(u * u)))
    M = precond if precond is not None else (lambda r: inv_diag * r)

    restart = int(restart)
    if restart < 1:
        raise ValueError("restart must be positive")
    if not isinstance(workspace, FGMRESWorkspace):
        workspace = None
    if norm(b) == 0.0:
        if workspace is None:
            return xp.zeros_like(x0), 0   # A x = 0: exact solution
        workspace.validate(x0, restart)
        workspace.x[...] = 0
        return workspace.x, 0
    if workspace is not None:
        return _fgmres_workspace_solve(
            apply_A, b, x0, xp, M, rtol=rtol, atol=atol,
            maxiter=maxiter, restart=restart, raise_on_fail=raise_on_fail,
            workspace=workspace)
    x = x0.copy()
    residual_tolerance = max(rtol * norm(b), atol)
    r = b - apply_A(x)
    residual_norm = norm(r)
    if residual_norm <= residual_tolerance:
        return x, 0

    iteration_count = 0
    while iteration_count < maxiter:
        cycle_length = min(restart, maxiter - iteration_count)
        V = [r / residual_norm]
        Z = []
        H = np.zeros((cycle_length + 1, cycle_length))
        givens_cos, givens_sin = np.zeros(cycle_length), np.zeros(cycle_length)
        g = np.zeros(cycle_length + 1)
        g[0] = residual_norm
        basis_size = 0
        for j in range(cycle_length):
            iteration_count += 1
            zj = M(V[j])
            Z.append(zj)
            w = apply_A(zj)
            for i in range(j + 1):
                H[i, j] = dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = norm(w)
            breakdown = H[j + 1, j] <= 1e-300
            if not breakdown:
                V.append(w / H[j + 1, j])
            for i in range(j):
                h0, h1 = H[i, j], H[i + 1, j]
                H[i, j] = givens_cos[i] * h0 + givens_sin[i] * h1
                H[i + 1, j] = -givens_sin[i] * h0 + givens_cos[i] * h1
            denom = float(np.hypot(H[j, j], H[j + 1, j]))
            if denom == 0.0:
                givens_cos[j], givens_sin[j] = 1.0, 0.0
            else:
                givens_cos[j] = H[j, j] / denom
                givens_sin[j] = H[j + 1, j] / denom
            H[j, j], H[j + 1, j] = denom, 0.0
            g[j + 1] = -givens_sin[j] * g[j]
            g[j] = givens_cos[j] * g[j]
            basis_size = j + 1
            if abs(g[basis_size]) <= residual_tolerance or breakdown:
                break

        y = np.linalg.solve(H[:basis_size, :basis_size], g[:basis_size])
        for j in range(basis_size):
            x = x + float(y[j]) * Z[j]
        # A true residual at every restart protects the stopping decision from
        # loss of orthogonality and from an inexact/variable preconditioner.
        r = b - apply_A(x)
        residual_norm = norm(r)
        if residual_norm <= residual_tolerance:
            return x, iteration_count
    if raise_on_fail:
        raise RuntimeError(
            f"FGMRES failed to converge in {iteration_count} applies "
            f"(residual {residual_norm:.3e}, target {residual_tolerance:.3e})")
    return x, iteration_count


def cocg(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
         precond=None, raise_on_fail=True, check_every=1, workspace=None):
    """Solve A x = b with preconditioned COCG for *complex-symmetric* A.

    Conjugate Orthogonal CG (van der Vorst & Melissen, IEEE Trans. Magn. 26
    (1990) 706): the CG recurrence with the *bilinear* (unconjugated) inner
    product ``(u, v) = sum_j u_j v_j`` in place of the Hermitian one. It solves
    A x = b whenever A = A^T (complex symmetric) at CG's cost -- one apply and
    two short vector updates per iteration, constant memory -- which is exactly
    the frequency-domain within-group operator  -div(D grad .) + Sigma_r + i w/v
    (a real SPD stencil plus a purely imaginary diagonal shift, so A = A^T but
    not A = A^H). The convergence test still uses the *true* Euclidean norm
    ||r|| = sqrt(sum |r_j|^2); only the recurrence coefficients use the
    bilinear form. Same interface, warm start and stopping rule as :func:`pcg`
    (to which it reduces for real A), so it slots into the same call sites.

    COCG can break down if a bilinear product vanishes on a non-null vector;
    that is rare for these near-SPD operators and is reported as a failure
    (or, with ``raise_on_fail=False``, returns the current iterate, as after
    ``maxiter``). ``check_every`` and ``workspace`` are accepted for
    signature compatibility with :func:`pcg` and ignored (the true-norm test
    runs every iteration; vectors are allocated per solve).

    Returns (x, n_iterations).
    """
    del check_every, workspace          # inapplicable; see docstring
    bdot = lambda u, v: kernels.dot(xp, u, v)         # bilinear (unconjugated)
    rnorm = lambda u: float(xp.sqrt(xp.sum((u.conj() * u).real)))
    M = precond if precond is not None else (lambda r: inv_diag * r)

    b_norm = rnorm(b)
    if b_norm == 0.0:
        return xp.zeros_like(x0), 0       # A x = 0: exact solution
    x = x0.copy()
    r = b - apply_A(x)
    stop = max(rtol * b_norm, atol)
    if rnorm(r) <= stop:
        return x, 0

    z = M(r)
    p = z.copy()
    rz = bdot(r, z)
    it = 0
    for it in range(1, maxiter + 1):
        Ap = apply_A(p)
        pAp = bdot(p, Ap)
        if pAp == 0:
            if raise_on_fail:
                raise RuntimeError("COCG breakdown (p^T A p = 0)")
            return x, it
        alpha = rz / pAp
        kernels.cg_update(xp, x, r, p, Ap, alpha)
        if rnorm(r) <= stop:
            return x, it
        z = M(r)
        rz_new = bdot(r, z)
        kernels.cg_direction(xp, p, z, rz_new / rz)
        rz = rz_new
    if raise_on_fail:
        raise RuntimeError(
            f"COCG failed to converge in {maxiter} iterations "
            f"(residual {rnorm(r):.3e}, target {stop:.3e})"
        )
    return x, it


def bicgstab(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
             precond=None, raise_on_fail=True, check_every=1, workspace=None):
    """Solve A x = b with preconditioned BiCGStab (van der Vorst).

    Same interface, warm start and stopping rule as :func:`pcg`. Like
    :func:`gmres` it does not require symmetry, but it uses short recurrences:
    constant memory (7 work vectors, vs. `restart` basis vectors for GMRES)
    and two operator+preconditioner applies per iteration -- the same
    trade-off that makes it the standard asymmetric solver in CFD codes
    (e.g. OpenFOAM's PBiCGStab). The price is a non-monotone residual and
    the possibility of breakdown: (r_hat, r), (r_hat, v) or (t, s) vanishing
    *relative to the norms of its factors* (machine epsilon of the dtype, so
    the test is scale invariant). Breakdowns are handled by restarting the
    recurrence from the current residual, and reported as a failure only if
    no progress is possible.

    maxiter : bound on BiCGStab *steps*; each step costs two operator applies,
              so a solve may use up to 2 * maxiter applies.
    raise_on_fail : if False, return the current iterate on failure instead
              of raising (as :func:`pcg`).
    check_every, workspace : accepted for signature compatibility with
              :func:`pcg` and ignored (every half step already reduces its
              residual on the host; vectors are allocated per solve).

    Returns (x, n_iterations) where an iteration is one operator apply
    (each BiCGStab step counts 2, keeping counts comparable with pcg/gmres).
    """
    del check_every, workspace          # inapplicable; see docstring
    dot = lambda u, v: float(xp.sum(u * v))
    norm2 = lambda u: float(xp.sum(u * u))
    M = precond if precond is not None else (lambda r: inv_diag * r)
    eps = float(np.finfo(np.result_type(b.dtype, np.float32)).eps)

    b_norm2 = norm2(b)
    if b_norm2 == 0.0:
        return xp.zeros_like(x0), 0       # A x = 0: exact solution
    x = x0.copy()
    r = b - apply_A(x)
    stop2 = max(rtol * b_norm2 ** 0.5, atol) ** 2
    if norm2(r) <= stop2:
        return x, 0

    r_hat = r.copy()                      # fixed shadow residual
    r_hat_norm = norm2(r_hat) ** 0.5
    rho = alpha = omega = 1.0
    v = p = None
    restarted = False
    it = 0
    while it < 2 * maxiter:
        rho_new = dot(r_hat, r)
        if (abs(rho_new) <= eps * r_hat_norm * norm2(r) ** 0.5
                or (v is not None and omega == 0.0)):
            # Lanczos breakdown: restart the recurrence from where we are.
            if restarted and norm2(r) > stop2:
                break                     # twice in a row -> give up
            r_hat = r.copy()
            r_hat_norm = norm2(r_hat) ** 0.5
            rho = alpha = omega = 1.0
            v = p = None
            restarted = True
            continue
        if v is None:
            p = r.copy()
        else:
            beta = (rho_new / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
        rho = rho_new
        p_hat = M(p)
        v = apply_A(p_hat)
        it += 1
        denom = dot(r_hat, v)
        if abs(denom) <= eps * r_hat_norm * norm2(v) ** 0.5:
            r_hat = r.copy()
            r_hat_norm = norm2(r_hat) ** 0.5
            rho = alpha = omega = 1.0
            v = p = None
            if restarted:
                break
            restarted = True
            continue
        alpha = rho / denom
        s = r - alpha * v
        ss = norm2(s)
        if ss <= stop2:                   # converged at the half step
            x += alpha * p_hat
            return x, it
        s_hat = M(s)
        t = apply_A(s_hat)
        it += 1
        tt = norm2(t)
        ts = dot(t, s)
        # omega = 0 (stagnation: t orthogonal to s) triggers the restart above.
        omega = ts / tt if tt > 0.0 and abs(ts) > eps * (tt * ss) ** 0.5 else 0.0
        x += alpha * p_hat + omega * s_hat
        r = s - omega * t
        restarted = False
        if norm2(r) <= stop2:
            return x, it
    if raise_on_fail:
        raise RuntimeError(
            f"BiCGStab failed to converge in {it} operator applies "
            f"(residual {norm2(r)**0.5:.3e}, target {stop2**0.5:.3e})"
        )
    return x, it


def fgmres_c(apply_A, b, x0, xp, precond, rtol=1e-8, atol=0.0, maxiter=1000,
             restart=50):
    """Flexible, complex restarted GMRES for a general (non-symmetric) complex
    operator A with a *variable / nonlinear* right preconditioner.

    Unlike :func:`gmres` (real, linear preconditioner) this uses Hermitian inner
    products and complex Givens rotations, and -- being *flexible* -- stores the
    preconditioned basis Z_j = precond(V_j) so ``precond`` may itself be an
    iterative solve (e.g. a block Gauss-Seidel sweep whose within-group systems
    are solved by COCG). That is exactly the preconditioner the frequency-domain
    neutron-noise operator wants: the block-diagonal within-group solves are
    strong and cheap, and FGMRES/Krylov-accelerates the group (scatter/fission)
    coupling, which the unaccelerated fixed point resolves only slowly near
    criticality. ``precond`` need not be linear; the price over stored-basis
    GMRES is a second set of ``restart`` vectors (the Z_j).

    Returns (x, n_outer) with n_outer the number of A applies (Arnoldi steps).
    """
    hdot = lambda u, v: complex(xp.sum(u.conj() * v))
    nrm = lambda u: float(xp.sqrt(xp.sum((u.conj() * u).real)))

    x = x0.copy()
    residual_tolerance = max(rtol * nrm(b), atol)
    r = b - apply_A(x)
    residual_norm = nrm(r)
    if residual_norm <= residual_tolerance:
        return x, 0
    iteration_count = 0
    while iteration_count < maxiter:
        cycle_length = min(restart, maxiter - iteration_count)
        V = [r / residual_norm]
        Z = []
        H = np.zeros((cycle_length + 1, cycle_length), dtype=np.complex128)
        givens_cos = np.zeros(cycle_length, dtype=np.complex128)
        givens_sin = np.zeros(cycle_length, dtype=np.complex128)
        g = np.zeros(cycle_length + 1, dtype=np.complex128)
        g[0] = residual_norm
        basis_size = 0
        for j in range(cycle_length):
            iteration_count += 1
            zj = precond(V[j])                  # flexible: may be iterative
            Z.append(zj)
            w = apply_A(zj)
            for i in range(j + 1):              # modified Gram-Schmidt (Hermitian)
                H[i, j] = hdot(V[i], w)
                w = w - H[i, j] * V[i]
            hjj = nrm(w)
            H[j + 1, j] = hjj
            breakdown = hjj <= 1e-300
            if not breakdown:
                V.append(w / hjj)
            for i in range(j):                  # apply stored Givens to column j
                t1, t2 = H[i, j], H[i + 1, j]
                H[i, j] = givens_cos[i] * t1 + givens_sin[i] * t2
                H[i + 1, j] = -np.conj(givens_sin[i]) * t1 + givens_cos[i] * t2
            a, bb = H[j, j], H[j + 1, j]         # complex Givens to zero H[j+1,j]
            if bb == 0:
                c, s, rr = 1.0, 0.0 + 0j, a
            elif a == 0:
                c, s, rr = 0.0, 1.0 + 0j, bb
            else:
                aa, ab = abs(a), abs(bb)
                nu = np.hypot(aa, ab)
                c, s, rr = aa / nu, (a / aa) * np.conj(bb) / nu, (a / aa) * nu
            givens_cos[j], givens_sin[j] = c, s
            H[j, j], H[j + 1, j] = rr, 0.0
            g[j + 1] = -np.conj(s) * g[j]
            g[j] = c * g[j]
            basis_size = j + 1
            if abs(g[basis_size]) <= residual_tolerance or breakdown:
                break
        y = np.linalg.solve(H[:basis_size, :basis_size], g[:basis_size])
        for j in range(basis_size):
            x = x + complex(y[j]) * Z[j]
        r = b - apply_A(x)
        residual_norm = nrm(r)
        if residual_norm <= residual_tolerance:
            return x, iteration_count
    raise RuntimeError(
        f"FGMRES failed to converge in {iteration_count} applies "
        f"(residual {residual_norm:.3e}, target {residual_tolerance:.3e})")


#: Drop the Anderson history when the residual grows by more than this factor.
#: A fixed-point map whose evaluation is itself an iterative solve is not a
#: stationary map -- warm starts and inner tolerances make G slightly different
#: each time it is called -- so a history built across such a change can point
#: the extrapolation the wrong way. Restarting costs a few plain iterations;
#: not restarting can diverge. (Same constant, same reason, as the noise
#: solver's Gauss-Seidel sweep.)
ANDERSON_RESTART_GROWTH = 1.5


def mix_source_history(history, raw, xp, *, copy_raw=False):
    """Regularized Anderson mixing of ``(input, map_output)`` source pairs.

    Residual differences are measured against the latest pair. The small
    normal equations stay on the host; source arrays stay on their backend.
    A singular system or excessive coefficients falls back to the raw iterate.
    This source mixer has different safeguards from :func:`anderson_step`.
    """
    if len(history) < 2:
        return raw
    residuals = [output - source for source, output in history]
    differences = [residual - residuals[-1] for residual in residuals[:-1]]
    count = len(differences)
    gram = np.array([
        [float(xp.sum(left * right)) for right in differences]
        for left in differences])
    rhs = np.array([-float(xp.sum(delta * residuals[-1]))
                    for delta in differences])
    # Scale the Tikhonov floor with the residual energy to regularize
    # nearly dependent history vectors without imposing an absolute scale.
    regularization = 1e-12 * (np.trace(gram) + 1e-300)
    gram[np.diag_indices(count)] += regularization
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return raw
    if not np.all(np.abs(coefficients) < 1e4):
        return raw
    mixed = raw.copy() if copy_raw else raw
    for index, coefficient in enumerate(coefficients):
        correction = float(coefficient) * (history[index][1] - history[-1][1])
        if copy_raw:
            mixed += correction
        else:
            mixed = mixed + correction
    return mixed


def anderson_step(X, F, beta=1.0):
    """One Anderson update from histories of iterates and residuals.

    Reference: Walker and Ni, SIAM J. Numer. Anal. 49 (2011), 1715-1735,
    doi:10.1137/10078356X, for the least-squares difference-history form.

    X : list of past iterates x_k (flat arrays). F : list of residuals
    f_k = g(x_k) - x_k. Returns the next iterate: the least-squares mixture of
    the history minimizing the combined residual, damped by ``beta``.

    With a single point this is plain relaxed fixed-point ``x + beta*f`` --
    which is what makes ``depth <= 1`` directly comparable to an external
    coupling tool's constant-relaxation scheme.

    This is the unconstrained Walker-Ni form. It is deliberately NOT the same
    algorithm as ``solver._anderson_source`` / ``noise._anderson_complex``,
    which use the Type-II normal equations with a Tikhonov floor and are tuned
    for the fission-source fixed point; keeping both is intentional.
    """
    m = len(F)
    fk = F[-1]
    if m == 1:
        return X[-1] + beta * fk
    dF = np.column_stack([F[i + 1] - F[i] for i in range(m - 1)])   # (n, m-1)
    dX = np.column_stack([X[i + 1] - X[i] for i in range(m - 1)])
    gamma, *_ = np.linalg.lstsq(dF, fk, rcond=None)
    return X[-1] + beta * fk - (dX + beta * dF) @ gamma


class AndersonAccelerator:
    """Stateful Anderson acceleration of a fixed-point map, with restart.

    Feed it the current iterate and the map's output; it returns the next
    iterate. ``depth <= 1`` degenerates to under-relaxed Picard
    ``beta*g(x) + (1-beta)*x``, written in exactly that form so it matches an
    external coupling tool's constant-relaxation arithmetic bit for bit.
    """

    def __init__(self, depth=5, beta=1.0):
        self.depth = int(depth)
        self.beta = float(beta)
        self.reset()

    def reset(self):
        self._X, self._F, self._last = [], [], None

    def step(self, x, gx):
        x = np.asarray(x, dtype=float)
        gx = np.asarray(gx, dtype=float)
        shape = x.shape
        xf, gf = x.reshape(-1), gx.reshape(-1)
        if self.depth <= 1:
            # The literal relaxation form, not the algebraically identical
            # x + beta*(g - x): the two differ in the last bits, and those bits
            # propagate through an inner Krylov solve's stopping decision.
            return (self.beta * gf + (1.0 - self.beta) * xf).reshape(shape)

        f = gf - xf
        norm = float(np.linalg.norm(f))
        if self._last is not None and norm > ANDERSON_RESTART_GROWTH * self._last:
            self.reset()
        self._last = norm

        self._X.append(xf)
        self._F.append(f)
        if len(self._F) > self.depth:
            self._X.pop(0)
            self._F.pop(0)
        try:
            return anderson_step(self._X, self._F, self.beta).reshape(shape)
        except np.linalg.LinAlgError:                       # pragma: no cover
            self.reset()
            return (xf + self.beta * f).reshape(shape)


LINEAR_SOLVERS = {"cg": pcg, "cocg": cocg, "gmres": gmres,
                  "fgmres": fgmres, "bicgstab": bicgstab}

#: Keyword arguments every solver in :data:`LINEAR_SOLVERS` accepts, so call
#: sites can pass them regardless of ``linear_solver=``. A solver for which
#: one is meaningless (e.g. ``check_every`` for GMRES, which returns its
#: residual estimate every step anyway) accepts and ignores it explicitly.
COMMON_SOLVER_KWARGS = ("rtol", "atol", "maxiter", "precond", "check_every",
                        "raise_on_fail", "workspace")


def get_linear_solver(name):
    """Resolve a named solver or a callable with :func:`pcg`'s signature."""
    if callable(name):
        return name
    try:
        return LINEAR_SOLVERS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown linear solver {name!r}; use "
            f"{sorted(LINEAR_SOLVERS)} or a callable with pcg's signature")
