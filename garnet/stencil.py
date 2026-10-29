"""Matrix-free finite-volume diffusion stencil and boundary conditions.

Discretizes  A phi = -div(D grad phi) + Sigma_r phi  with a cell-centered
7-point stencil. Face diffusion coefficients use the harmonic mean, which is
exact for piecewise-constant D (so heterogeneous cores are handled correctly).

The operator is deliberately matrix-free: instead of assembling a sparse
matrix, we precompute one face-coupling array per axis plus the diagonal, and
`apply()` is six shifted multiply-adds -- both faster and lighter than sparse
CSR SpMV, and it keeps the entire operator on-device. On the GPU backend those
six multiply-adds (13 kernel launches, 7 temporaries) collapse into a single
hand-written CUDA kernel; see :mod:`garnet.kernels` and `apply` below.

Without discontinuity factors, the resulting operator is symmetric positive definite as long as removal is
positive somewhere (zero-flux BC) which is what lets us use CG. On
cylindrical grids the SPD property comes from volume-weighting the equations;
``symmetric=False`` opts out of that and builds the natural divergence form
instead, which is non-symmetric and needs GMRES/BiCGStab (see garnet.linalg).

This module also owns the boundary-condition vocabulary (``normalize_bc``,
``face_alpha``, ``robin_face_term``) shared by every spatial operator -- the
hex and triangular stencils and the SP3/SDPN angular blocks all reuse it.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

from .adf import _real_positive_factors
from .backend import asnumpy

BC_ZERO_FLUX = "zero-flux"
BC_REFLECTIVE = "reflective"
BC_VACUUM = "vacuum"

# Robin coefficient alpha in the boundary law  J_net = alpha * phi_surface
# (the leakage current out of the domain, per unit surface flux). The diffusion
# Marshak vacuum condition gives alpha = 1/2 (FEMFFUSION's bc_factor); a custom
# albedo boundary passes alpha directly. Reflective is alpha = 0 (zero current)
# and zero-flux is the alpha -> infinity (Dirichlet) limit.
_VACUUM_ALPHA = 0.5


def _face_valid(f):
    """Validate and canonicalize one face spec: a known string or an albedo."""
    if isinstance(f, str) and f in (BC_ZERO_FLUX, BC_REFLECTIVE, BC_VACUUM):
        return f
    # numbers.Real admits numpy scalars (np.float32(0.5)); a numeric string
    # (e.g. from an input file) is still parsed; bool is not an albedo.
    try:
        if not isinstance(f, (str, numbers.Real)) or isinstance(f, bool):
            raise TypeError
        alpha = float(f)
    except (TypeError, ValueError):
        raise ValueError(
            f"unknown bc {f!r}; use {BC_ZERO_FLUX!r}, {BC_REFLECTIVE!r}, "
            f"{BC_VACUUM!r}, or a non-negative albedo coefficient") from None
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"albedo coefficient must be finite (not NaN or inf) and non-negative, got {alpha}")
    return alpha


def face_alpha(spec):
    """Robin coefficient alpha for one face spec (see _VACUUM_ALPHA note)."""
    if spec == BC_REFLECTIVE:
        return 0.0
    if spec == BC_ZERO_FLUX:
        return math.inf
    if spec == BC_VACUUM:
        return _VACUUM_ALPHA
    return float(spec)


def normalize_bc(bc):
    """Expand a boundary-condition spec to ((xlo, xhi), (ylo, yhi), (zlo, zhi)).

    Accepts a single face spec (applied to all six faces), a length-3 sequence
    of per-axis entries (each a face spec or a (lo, hi) pair), or a flat
    length-6 sequence ordered x_lo, x_hi, y_lo, y_hi, z_lo, z_hi. A face spec is
    "zero-flux", "reflective", "vacuum", or a non-negative float albedo
    coefficient alpha (the boundary law J_net = alpha * phi_surface).
    """
    def is_scalar(x):
        return isinstance(x, (str, numbers.Real))

    if is_scalar(bc):
        faces = [bc] * 6
    else:
        bc = list(bc)
        if len(bc) == 3:
            faces = []
            for axis in bc:
                pair = [axis, axis] if is_scalar(axis) else list(axis)
                if len(pair) != 2:
                    raise ValueError(f"per-axis bc must be a face spec or (lo, hi) pair, got {axis!r}")
                faces += pair
        elif len(bc) == 6:
            faces = bc
        else:
            raise ValueError(f"bc must be a face spec, 3 per-axis entries, or 6 faces, got {bc!r}")
    faces = [_face_valid(f) for f in faces]
    return ((faces[0], faces[1]), (faces[2], faces[3]), (faces[4], faces[5]))


def harmonic_mean(Da, Db):
    """Face diffusion coefficient between two cells with piecewise-constant D.

    Requiring the two-point flux to be continuous across the shared face gives
    the harmonic mean 2*Da*Db/(Da + Db): exact for piecewise-constant D, and
    it correctly chokes transport at an interface where either D vanishes
    (the arithmetic mean would leak through strong absorbers).
    """
    return 2.0 * Da * Db / (Da + Db)


def robin_face_term(xp, Dface, d, alpha):
    """Boundary-face contribution to the operator diagonal, per unit volume,
    for a face perpendicular to a cell axis with centre-to-face distance d/2.

    The Robin law J_net = alpha * phi_s relates the outward current to the
    surface flux. Coupling the cell centre to the surface with the two-point
    current J = (2D/d)(phi_c - phi_s) and eliminating phi_s gives

        leakage / volume = 2 D alpha / (d (d alpha + 2 D)).

    alpha -> inf recovers the zero-flux Dirichlet term 2D/d^2; alpha = 0
    (reflective) contributes nothing; alpha = 1/2 is the Marshak vacuum.
    """
    if alpha == 0.0:
        return xp.zeros_like(Dface)
    if math.isinf(alpha):
        return 2.0 * Dface / d**2
    return 2.0 * Dface * alpha / (d * (d * alpha + 2.0 * Dface))


class GroupOperator:
    """A_g = -div(D_g grad .) + Sigma_r,g  on a uniform structured grid.

    Cartesian by default; a grid with geometry="cylindrical" yields the (r-z)
    revolution-body stencil instead (volume-weighted form; pair the operator
    with a source scaled by .rhs_weight, which is None on Cartesian grids).

    Parameters
    ----------
    xp        : array module (numpy or cupy)
    grid      : Grid
    D         : per-cell diffusion coefficient, shape grid.shape (device array)
    removal   : per-cell removal cross section, shape grid.shape (device array)
    bc        : boundary conditions, anything accepted by normalize_bc():
                "zero-flux" (phi = 0 on the outer boundary surface) or
                "reflective" (zero net current), globally, per axis, or per face.
    symmetric : cylindrical grids only. True (default) builds the
                volume-weighted form, which is SPD and pairs with CG plus a
                source scaled by .rhs_weight. False builds the natural
                divergence (per-unit-volume) form instead: the same equations
                with each row divided by its cell volume, so the source is
                used as-is (.rhs_weight is None) -- but the radial coupling
                seen from cell i (r_face/r_i) no longer equals its transpose
                (r_face/r_{i+1}), so the matrix is non-symmetric and must be
                paired with linear_solver="gmres" or "bicgstab", not CG.
                Ignored on Cartesian grids (already symmetric either way).
    adf       : optional three pairs (low-side, high-side) of positive,
                finite interior-face factor arrays for this energy group.
                Shape is grid.shape with the corresponding axis reduced by one.
                Factors are heterogeneous / homogeneous face flux. Cartesian
                only; external and active-mask boundaries are unaffected.
    """

    #: apply() accepts out=, so block operators can write straight into a row
    #: of their state instead of allocating a temporary per moment.
    supports_out = True

    def __init__(self, xp, grid, D, removal, bc=BC_ZERO_FLUX, active=None,
                 mask_bc=BC_VACUUM, symmetric=True, adf=None):
        bc = normalize_bc(bc)
        self.xp = xp
        self.shape = grid.shape
        dx, dy, dz = grid.spacing
        self.spacing = grid.spacing
        self._diffusion = D         # face_traces reconstructs half-cell drops

        # Cylindrical (r-z) grids: every term of the per-unit-volume Cartesian
        # stencil is scaled by its radial metric factor -- cell terms (removal,
        # y/z couplings, y/z boundary faces) by the cell-center radius, and
        # radial face terms by the face radius. This is the equation multiplied
        # by the cell volume, so the radial two-point stencil stays symmetric
        # (both neighbours see the shared face radius) and CG still applies;
        # the matching source-side weight is exposed as self.rhs_weight. On a
        # Cartesian grid all factors are 1 and the stencil is unchanged.
        cylindrical_metrics = getattr(grid, "cylindrical_metrics", lambda: None)()
        self._cartesian = cylindrical_metrics is None
        if cylindrical_metrics is None:
            cell_radius = radial_face_radius = 1.0
            inner_boundary_radius = outer_boundary_radius = 1.0
            self.rhs_weight = None
        else:
            center_radii, face_radii, inner_radius, outer_radius = cylindrical_metrics
            cell_radius = xp.asarray(center_radii, dtype=D.dtype)
            radial_face_radius = xp.asarray(face_radii, dtype=D.dtype)
            inner_boundary_radius, outer_boundary_radius = float(inner_radius), float(outer_radius)
            self.rhs_weight = cell_radius
            # The y "faces" of an r-z grid are the azimuthal seam, not a
            # surface: always reflective, whatever a scalar/all-face bc says
            # (as CylindricalModel does). The r = 0 axis needs no override --
            # its boundary radius is 0, so any bc there is inert below.
            bc = (bc[0], (BC_REFLECTIVE, BC_REFLECTIVE), bc[2])

        def robin(Dface, d, alpha):
            return robin_face_term(xp, Dface, d, alpha)

        # Interior face couplings (harmonic mean of the two adjacent cells).
        # wx[i] couples cells i and i+1 along x; shape (nx-1, ny, nz).
        def face(Da, Db, d):
            return harmonic_mean(Da, Db) / (d * d)

        self.wx = radial_face_radius * face(D[:-1, :, :], D[1:, :, :], dx)
        self.wy = cell_radius * face(D[:, :-1, :], D[:, 1:, :], dy)
        self.wz = cell_radius * face(D[:, :, :-1], D[:, :, 1:], dz)

        # Each pair contains the factors on the low- and high-coordinate side
        # of every interior face. The two directed coefficients multiply the
        # corresponding cell flux in the single conservative face current.
        self._adf = adf is not None
        self._high_weights = None
        if adf is not None:
            if cylindrical_metrics is not None:
                raise ValueError("ADF is currently supported on Cartesian grids only")
            if len(adf) != 3 or any(len(pair) != 2 for pair in adf):
                raise ValueError("adf must contain three (low, high) face-array pairs")
            directed = []
            for axis, (pair, weight) in enumerate(zip(adf, (self.wx, self.wy, self.wz))):
                factors = [_real_positive_factors(xp, a) for a in pair]
                if any(a.shape != weight.shape for a in factors):
                    raise ValueError(f"adf axis {axis} factors must have shape {weight.shape}")
                lo, hi = factors
                # Only the ratio matters. Normalize before narrowing the dtype
                # or forming resistances, so an arbitrary common factor scale
                # cannot overflow the denominator and erase face coupling.
                scale = xp.maximum(lo, hi)
                factor_dtype = np.result_type(D.dtype, np.float32)
                lo = _real_positive_factors(xp, lo / scale, dtype=factor_dtype)
                hi = _real_positive_factors(xp, hi / scale, dtype=factor_dtype)
                slices_lo = [slice(None)] * 3
                slices_hi = [slice(None)] * 3
                slices_lo[axis], slices_hi[axis] = slice(None, -1), slice(1, None)
                diffusion_low, diffusion_high = D[tuple(slices_lo)], D[tuple(slices_hi)]
                spacing = (dx, dy, dz)[axis]
                denom = 0.5 * spacing**2 * (lo / diffusion_low + hi / diffusion_high)
                weights = (lo / denom, hi / denom)
                if any(not bool(xp.all(xp.isfinite(w) & (w > 0))) for w in weights):
                    raise ValueError("ADF face coefficients must be finite and positive in the operator dtype")
                directed.append(weights)
            (self.wx, wx_hi), (self.wy, wy_hi), (self.wz, wz_hi) = directed
            self._high_weights = (wx_hi, wy_hi, wz_hi)

        # Non-rectangular cores: an "active" mask marks in-domain cells. Faces
        # touching an inactive (excised) cell carry no diffusion; the active
        # cell instead sees the core-surface boundary law mask_bc on that face.
        active_cells = None
        if active is not None:
            active_cells = xp.asarray(active).astype(bool)
            if active_cells.shape != grid.shape:
                raise ValueError(f"active mask shape {active_cells.shape} != grid shape {grid.shape}")
            self.wx = xp.where(active_cells[:-1, :, :] & active_cells[1:, :, :], self.wx, 0.0)
            self.wy = xp.where(active_cells[:, :-1, :] & active_cells[:, 1:, :], self.wy, 0.0)
            self.wz = xp.where(active_cells[:, :, :-1] & active_cells[:, :, 1:], self.wz, 0.0)
            if self._adf:
                self._high_weights = tuple(xp.where(mask, w, 0.0) for mask, w in zip(
                    (active_cells[:-1] & active_cells[1:], active_cells[:, :-1] & active_cells[:, 1:],
                     active_cells[:, :, :-1] & active_cells[:, :, 1:]), self._high_weights))

        # Diagonal: removal + sum of face couplings incident on each cell.
        diag = removal * cell_radius
        wx_hi, wy_hi, wz_hi = (self._high_weights if self._adf else (self.wx, self.wy, self.wz))
        diag[1:, :, :] += wx_hi
        diag[:-1, :, :] += self.wx
        diag[:, 1:, :] += wy_hi
        diag[:, :-1, :] += self.wy
        diag[:, :, 1:] += wz_hi
        diag[:, :, :-1] += self.wz

        # Outer box-boundary Robin terms (only where the boundary cell is
        # active), each scaled by its face's metric weight: the boundary radius
        # for the two radial faces (0 at the symmetry axis, so any bc there is
        # inert), the cell radius for y/z faces.
        cell_radius_at = (lambda sl: 1.0) if cylindrical_metrics is None else (lambda sl: cell_radius[sl])
        boundary_faces = [
            (face_alpha(bc[0][0]), (0, slice(None), slice(None)), dx, lambda sl: inner_boundary_radius),
            (face_alpha(bc[0][1]), (-1, slice(None), slice(None)), dx, lambda sl: outer_boundary_radius),
            (face_alpha(bc[1][0]), (slice(None), 0, slice(None)), dy, cell_radius_at),
            (face_alpha(bc[1][1]), (slice(None), -1, slice(None)), dy, cell_radius_at),
            (face_alpha(bc[2][0]), (slice(None), slice(None), 0), dz, cell_radius_at),
            (face_alpha(bc[2][1]), (slice(None), slice(None), -1), dz, cell_radius_at),
        ]
        for alpha, sl, d, wgt in boundary_faces:
            if alpha == 0.0:
                continue
            term = wgt(sl) * robin(D[sl], d, alpha)
            if active_cells is not None:
                term = xp.where(active_cells[sl], term, 0.0)
            diag[sl] += term

        # Internal core-surface faces (active cell facing an excised cell).
        if active_cells is not None:
            mask_alpha = face_alpha(_face_valid(mask_bc))
            if mask_alpha != 0.0:
                low_active, high_active = active_cells[:-1, :, :], active_cells[1:, :, :]
                diag[:-1, :, :] += xp.where(
                    low_active & ~high_active,
                    radial_face_radius * robin(D[:-1, :, :], dx, mask_alpha),
                    0.0,
                )
                diag[1:, :, :] += xp.where(
                    high_active & ~low_active,
                    radial_face_radius * robin(D[1:, :, :], dx, mask_alpha),
                    0.0,
                )
                low_active, high_active = active_cells[:, :-1, :], active_cells[:, 1:, :]
                diag[:, :-1, :] += xp.where(
                    low_active & ~high_active,
                    cell_radius * robin(D[:, :-1, :], dy, mask_alpha),
                    0.0,
                )
                diag[:, 1:, :] += xp.where(
                    high_active & ~low_active,
                    cell_radius * robin(D[:, 1:, :], dy, mask_alpha),
                    0.0,
                )
                low_active, high_active = active_cells[:, :, :-1], active_cells[:, :, 1:]
                diag[:, :, :-1] += xp.where(
                    low_active & ~high_active,
                    cell_radius * robin(D[:, :, :-1], dz, mask_alpha),
                    0.0,
                )
                diag[:, :, 1:] += xp.where(
                    high_active & ~low_active,
                    cell_radius * robin(D[:, :, 1:], dz, mask_alpha),
                    0.0,
                )
            # Excised cells: decouple (all face weights into them are already 0)
            # and give them a unit diagonal so Jacobi is well defined; with a
            # void material (no fission/scatter) their flux stays exactly zero.
            diag = xp.where(active_cells, diag, 1.0)

        # Divergence form (symmetric=False, cylindrical only): apply() runs
        # the weighted stencil and then divides each row by its cell weight,
        # which is exactly the matrix W^{-1} A_w -- the classic non-symmetric
        # finite-volume form. The Jacobi diagonal scales along.
        self._stencil_diag = diag
        self.row_scale = None
        if not symmetric and cylindrical_metrics is not None:
            self.row_scale = 1.0 / cell_radius
            self.rhs_weight = None
            diag = diag * self.row_scale
        self.diag = diag
        self.inv_diag = 1.0 / diag  # Jacobi preconditioner
        self._fused = {}            # dtype -> fused-kernel argument buffers
        self._adf_fused = {}
        self._op_dtype = None       # promoted dtype of the stencil arrays

    # -- fused GPU path ----------------------------------------------------
    # apply() below is 13 kernel launches and 7 full-size temporaries. On GPU
    # that is the dominant cost of every diffusion/SP3/SPN/DSA/CMFD solve at
    # the grid sizes this code runs, so it is collapsed into one launch by
    # garnet.kernels.stencil7_apply. Correctness is unchanged: the fused kernel
    # accumulates the six neighbours in the same order as the slice
    # expressions, and the NumPy path is what runs whenever the backend is not
    # CuPy or fusion is switched off (GARNET_FUSED=0).

    def _fusable(self, dtype):
        """True if A phi has dtype `dtype`, i.e. the cached stencil arrays can
        be cast to it losslessly. Refuses e.g. a real phi against the complex
        removal the noise solver builds, where casting down would drop the
        imaginary part -- there the generic path (which promotes) is correct.
        """
        if self._op_dtype is None:
            self._op_dtype = np.result_type(
                self._stencil_diag.dtype, self.wx.dtype, self.wy.dtype,
                self.wz.dtype)
        return np.result_type(self._op_dtype, dtype) == dtype

    def _fused_arrays(self, dtype):
        """Contiguous, common-dtype copies of the stencil arrays for the kernel.

        Built once per dtype and cached. The cast matters for the noise solver,
        where a complex removal makes the diagonal complex while the face
        couplings stay real -- the kernel wants one type placeholder.
        Zero-size coupling arrays (a grid one cell thick on some axis) become
        1-element dummies so the kernel always has a valid pointer; the
        corresponding branches are dead in that case.
        """
        buf = self._fused.get(dtype)
        if buf is None:
            xp = self.xp

            def prep(a):
                a = xp.ascontiguousarray(a, dtype=dtype)
                return a if a.size else xp.zeros(1, dtype=dtype)

            rs = None if self.row_scale is None else prep(
                xp.broadcast_to(self.row_scale, self.shape))
            buf = (prep(self._stencil_diag), prep(self.wx), prep(self.wy),
                   prep(self.wz), rs)
            self._fused[dtype] = buf
        return buf

    def apply(self, phi, out=None):
        """Return A phi (allocating the output array unless ``out`` is given)."""
        return self._apply(phi, out, transpose=False)

    def apply_transpose(self, phi, out=None):
        """Return the algebraic transpose on Cartesian grids."""
        if not self._cartesian:
            raise ValueError("transpose is currently supported on Cartesian grids only")
        return self._apply(phi, out, transpose=True)

    def _apply(self, phi, out, transpose):
        from . import kernels

        if (kernels.use_fused(self.xp, "stencil") and phi.flags.c_contiguous
                and self._fusable(phi.dtype)):
            diag, wx, wy, wz, rs = self._fused_arrays(phi.dtype)
            if self._adf:
                hi = self._adf_fused.get(phi.dtype)
                if hi is None:
                    hi = tuple(self.xp.ascontiguousarray(a, dtype=phi.dtype) if a.size
                               else self.xp.zeros(1, dtype=phi.dtype)
                               for a in self._high_weights)
                    self._adf_fused[phi.dtype] = hi
                low = (wx, wy, wz)
                if transpose:
                    low, hi = hi, low
                return kernels.stencil7_adf_apply(self.xp, phi, diag, *low, *hi, out=out)
            return kernels.stencil7_apply(self.xp, phi, diag, wx, wy, wz, rs,
                                          out=out)
        if out is None:
            out = self._stencil_diag * phi
        else:
            self.xp.multiply(self._stencil_diag, phi, out=out)
        wx_hi, wy_hi, wz_hi = (self._high_weights if self._adf else (self.wx, self.wy, self.wz))
        low = (self.wx, self.wy, self.wz)
        high = (wx_hi, wy_hi, wz_hi)
        if transpose:
            low, high = high, low
        out[1:, :, :] -= low[0] * phi[:-1, :, :]
        out[:-1, :, :] -= high[0] * phi[1:, :, :]
        out[:, 1:, :] -= low[1] * phi[:, :-1, :]
        out[:, :-1, :] -= high[1] * phi[:, 1:, :]
        out[:, :, 1:] -= low[2] * phi[:, :, :-1]
        out[:, :, :-1] -= high[2] * phi[:, :, 1:]
        if self.row_scale is not None:
            out *= self.row_scale
        return out

    def assemble(self):
        """Return the exact scalar stencil as a host CSR matrix.

        Matrix-free application remains the production path.  The explicit
        form is used to construct small conservative Galerkin/CMFD coarse
        operators and by diagnostics that need an algebraic reference.
        """
        import scipy.sparse as sp

        shape = self.shape
        idx = np.arange(int(np.prod(shape))).reshape(shape)
        rows, cols, values = [], [], []

        def add(row, col, value):
            rows.append(np.asarray(row).ravel())
            cols.append(np.asarray(col).ravel())
            values.append(np.asarray(asnumpy(value)).ravel())

        add(idx, idx, self._stencil_diag)
        add(idx[1:, :, :], idx[:-1, :, :], -asnumpy(self.wx))
        wx_hi, wy_hi, wz_hi = (self._high_weights if self._adf else (self.wx, self.wy, self.wz))
        add(idx[:-1, :, :], idx[1:, :, :], -asnumpy(wx_hi))
        add(idx[:, 1:, :], idx[:, :-1, :], -asnumpy(self.wy))
        add(idx[:, :-1, :], idx[:, 1:, :], -asnumpy(wy_hi))
        add(idx[:, :, 1:], idx[:, :, :-1], -asnumpy(self.wz))
        add(idx[:, :, :-1], idx[:, :, 1:], -asnumpy(wz_hi))
        matrix = sp.csr_matrix(
            (np.concatenate(values),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(idx.size, idx.size))
        if self.row_scale is not None:
            scale = np.broadcast_to(
                asnumpy(self.row_scale), shape).reshape(-1)
            matrix = sp.diags(scale) @ matrix
        return matrix.tocsr()

    def face_currents(self, phi):
        """Return interior current densities oriented in positive x, y, z.

        Each array has the corresponding interior-face shape. Masked faces
        are zero: external and active-to-inactive boundary currents are not
        included. The result uses exactly the coefficients of ``apply``.
        Cartesian grids only; multiply by face area to obtain integrated current.
        """
        if not self._cartesian:
            raise ValueError("face_currents currently supports Cartesian grids only")
        if phi.shape != self.shape:
            raise ValueError(f"flux shape {phi.shape} != grid shape {self.shape}")
        high = self._high_weights if self._adf else (self.wx, self.wy, self.wz)
        result = []
        for axis, (low_weight, high_weight) in enumerate(zip((self.wx, self.wy, self.wz), high)):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            result.append(self.spacing[axis] * (
                low_weight * phi[tuple(lo)] - high_weight * phi[tuple(hi)]))
        return tuple(result)

    def face_traces(self, phi):
        """Return the homogeneous surface-flux traces on both sides of each face.

        For every interior face, ``(low, high)`` are the two-point
        reconstructions ``phi_L - J h / (2 D_L)`` and ``phi_R + J h / (2 D_R)``
        from the cell values and the current of :meth:`face_currents`. With
        discontinuity factors the corrected traces agree,
        ``f_L * low == f_R * high``; without them ``low == high``. On masked
        faces the current is zero and both traces are the cell values.
        Cartesian grids only.
        """
        D = self._diffusion
        result = []
        for axis, current in enumerate(self.face_currents(phi)):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            half = 0.5 * self.spacing[axis]
            result.append((phi[lo] - current * half / D[lo],
                           phi[hi] + current * half / D[hi]))
        return tuple(result)
