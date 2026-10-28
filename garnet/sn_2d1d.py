"""CPU planar S_N / axial diffusion prototype.

One axial column is retained for every active radial triangle. Existing S_N
sweeps provide radial transport; ``AxialDiffusionOperator`` supplies axial
currents with the existing diffusion boundary laws. ``solve`` handles a
prescribed source; ``solve_eigenvalue`` adds fission power iteration around it.

The default Krylov method solves ``phi - T(S phi - Lz phi) = T q``. Here T is
the plane streaming/collision inverse, S includes all scattering, and Lz is
the axial diffusion leakage operator. This eliminates the angular unknowns
without approximating the intended coupled equations. A relaxed plane/column
Picard iteration is also available for comparison. Both paths return actual
balance and flux-consistency residuals, independently of iteration histories.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from ._sn_plane import _PlanarFixedSource
from .axial_diffusion import AxialDiffusionOperator
from .materials import Material
from .solver import Fields
from .tri import TriGrid
from .tri_sn import TriSNTransportSolver


@dataclass
class SN2D1DResult:
    """Owned fields from a fixed-source solve; scalar fields use ``(G, *grid.shape)``.

    ``radial_currents`` has shape ``(G, 3, nr, nc, 2, nz)`` (outward from
    each cell); ``axial_currents`` uses ``(G, nr, nc, 2, nz+1)`` (positive z).
    ``residual_history`` holds preconditioned Krylov residual norms, or Picard
    relative flux updates; the final three physical residuals govern success.
    """

    flux: np.ndarray
    radial_flux: np.ndarray
    axial_flux: np.ndarray
    radial_currents: np.ndarray
    axial_currents: np.ndarray
    radial_leakage: np.ndarray
    axial_leakage: np.ndarray
    balance_defect: np.ndarray
    converged: bool
    reason: str
    iterations: int
    n_sweeps: int
    n_axial_solves: int
    solve_seconds: float
    transport_residual: float
    balance_residual: float
    flux_disagreement: float
    negative_effective_sources: int
    minimum_effective_source: float
    residual_history: list[float]
    leakage_change_history: list[float]


@dataclass
class SN2D1DEigenResult(SN2D1DResult):
    """Eigenvalue result, with volume-integrated fission production equal to one.

    Inherited flux/current fields refer to the returned flux's own fission
    source, including when the outer iteration fails. ``iterations`` totals
    fixed-source iterations; ``outer_iterations`` counts accepted power updates.
    Histories contain one entry per accepted outer update. The inherited
    ``residual_history`` concatenates the inner fixed-source histories.
    """

    k_eff: float
    fission_source: np.ndarray
    outer_iterations: int
    k_history: list[float]
    k_change_history: list[float]
    source_change_history: list[float]
    flux_change_history: list[float]
    inner_iteration_history: list[int]


class SN2D1DSolver:
    """Transport on uniform triangular-prism extrusions, CPU only.

    ``active`` may be a radial mask or a full prism mask constant in z.
    Radial ``vacuum`` and CPU-LU ``periodic`` boundaries follow ``tri_sn``.
    ``axial_bc`` uses diffusion face specifications, independently at each end.
    Each reflective column needs positive removal somewhere for its axial
    solve. Material fields are frozen at construction; rebuild after changes.

    ``engine='levels'`` exercises existing CPU levels kernels, without adding
    GPU support or batching. Identical material planes share one transport
    instance. This experimental API is imported from ``garnet.sn_2d1d``.
    """

    def __init__(self, grid: TriGrid, materials, material_map=None, *, active=None,
                 radial_bc="vacuum", axial_bc="vacuum", n_polar=2, n_azi=8,
                 scheme="step", engine="lu", acceleration="dsa-gmres",
                 max_inner=800, mix_material=None, mix_weight=None):
        if not isinstance(grid, TriGrid) or len(grid.shape) != 4:
            raise ValueError("2D/1D needs an extruded TriGrid with shape (nr, nc, 2, nz)")
        if any(not isinstance(n, (int, np.integer)) or n < 1 for n in grid.shape):
            raise ValueError("grid dimensions must be positive integers")
        if not np.isfinite(grid.side) or not np.isfinite(grid.height):
            raise ValueError("grid dimensions must be finite")
        self.grid = grid
        self.radial_shape = grid.shape[:-1]
        self.nz = grid.nz
        if active is None:
            footprint = np.ones(self.radial_shape, bool)
        else:
            active = np.asarray(active, bool)
            if active.shape == self.radial_shape:
                footprint = active.copy()
            elif active.shape == grid.shape and np.all(active == active[..., :1]):
                footprint = active[..., 0].copy()
            else:
                raise ValueError("active footprint must match the grid and remain constant in z")
        if not footprint.any():
            raise ValueError("2D/1D needs at least one active radial cell")
        self.footprint = footprint
        self.active = np.broadcast_to(footprint[..., None], grid.shape).copy()
        materials = [materials] if isinstance(materials, Material) else list(materials)
        if not materials:
            raise ValueError("at least one material is required")
        material_map = (np.zeros(grid.shape, int) if material_map is None
                        else np.asarray(material_map))
        # Fields validates material IDs, group counts, and the existing blend
        # convention. Passing its maps to planes keeps both methods identical.
        fields = Fields(np, grid, materials, material_map, np.float64,
                        mix_material=mix_material, mix_weight=mix_weight)
        self.G = fields.n_groups
        self.shape = (self.G,) + grid.shape
        self.nu_sigma_f = np.stack(fields.nu_sigma_f)
        self.chi = np.stack(fields.chi)
        self._fission_fields = fields
        self.sigma_t = np.stack(fields.sigma_t)
        removal = np.stack(fields.removal)
        if np.any(removal > self.sigma_t * (1.0 + 1e-12)):
            raise ValueError("transport total must not be smaller than removal")
        self.self_scatter = np.maximum(self.sigma_t - removal, 0.0)
        self.removal = self.sigma_t - self.self_scatter
        self._total_columns = self._pack(self.sigma_t)
        self._self_scatter_columns = self._pack(self.self_scatter)
        self._transfers = [
            (source, target, fields.sigma_s[source][target][footprint].copy())
            for source in range(self.G) for target in range(self.G)
            if source != target and fields.sigma_s[source][target] is not None
        ]
        self.axial = AxialDiffusionOperator(
            1.0 / (3.0 * self._total_columns), self._pack(self.removal), grid.dz, axial_bc)
        # Check singular reflective columns at setup, and cache their factors.
        self.axial.solve(np.zeros(self.axial.shape))
        self.n_axial_solves = 0
        plane_grid = TriGrid(self.radial_shape, grid.side)
        mixed_materials = None if mix_material is None else np.asarray(mix_material)
        mixed_weights = None if mix_weight is None else np.asarray(mix_weight)
        plane_cache = {}
        planes = []
        for layer in range(self.nz):
            plane_map = material_map[..., layer]
            plane_mix = None if mixed_materials is None else mixed_materials[..., layer]
            plane_weight = None if mixed_weights is None else mixed_weights[..., layer]
            key = tuple(None if a is None else a.tobytes()
                        for a in (plane_map, plane_mix, plane_weight))
            if key not in plane_cache:
                transport = TriSNTransportSolver(
                    plane_grid, materials, material_map=plane_map, active=footprint,
                    mix_material=plane_mix, mix_weight=plane_weight,
                    n_polar=n_polar, n_azi=n_azi, bc=radial_bc, scheme=scheme,
                    require_fissile=False, engine=engine, device="cpu",
                    acceleration=acceleration, max_inner=max_inner)
                plane_cache[key] = _PlanarFixedSource(transport)
            planes.append(plane_cache[key])
        self.planes = tuple(planes)
        self._unique_planes = tuple(plane_cache.values())

    def _pack(self, field):
        return np.asarray(field)[:, self.footprint, :].copy()

    def _unpack(self, columns):
        field = np.zeros(self.shape)
        field[:, self.footprint, :] = columns
        return field

    def _field(self, value, name):
        try:
            field = np.broadcast_to(np.asarray(value, float), self.shape)
        except ValueError as error:
            raise ValueError(f"{name} must broadcast to {self.shape}") from error
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must be finite")
        return self._pack(field)

    def _scattering(self, flux, *, include_self=True):
        source = self._self_scatter_columns * flux if include_self else np.zeros_like(flux)
        for source_group, target_group, transfer in self._transfers:
            source[target_group] += transfer * flux[source_group]
        return source

    def _response(self, source):
        field = self._unpack(source)
        flux = np.empty_like(source)
        for layer, plane in enumerate(self.planes):
            for group in range(self.G):
                flux[group, :, layer] = plane.response(
                    group, field[group, ..., layer])[self.footprint]
        return flux

    def _axial_solve(self, source):
        self.n_axial_solves += 1
        return self.axial.solve(source)

    @staticmethod
    def _relative_change(new, old, scale_floor=0.0):
        scale = max(float(np.max(np.abs(new))), float(np.max(np.abs(old))),
                    scale_floor, np.finfo(float).tiny)
        return float(np.max(np.abs(new - old))) / scale

    def _reconstruct(self, flux, source):
        """Fold final currents and check the unmodified hybrid equations."""
        axial_leakage = self.axial.leakage(flux)
        effective_source = source + self._scattering(flux) - axial_leakage
        effective_field = self._unpack(effective_source)
        radial_flux = np.zeros(self.shape)
        radial_leakage = np.zeros(self.shape)
        radial_currents = np.zeros((self.G, 3) + self.grid.shape)
        for layer, plane in enumerate(self.planes):
            for group in range(self.G):
                state = plane.sweep(group, effective_field[group, ..., layer])
                radial_flux[group, ..., layer] = state.flux
                radial_leakage[group, ..., layer] = state.leakage
                radial_currents[group, ..., layer] = state.currents
        radial_columns = self._pack(radial_flux)
        radial_divergence = self._pack(radial_leakage)
        off_group_source = self._scattering(flux, include_self=False)
        axial_flux = self._axial_solve(source + off_group_source - radial_divergence)
        reactions = self._pack(self.removal) * flux
        defect = radial_divergence + axial_leakage + reactions - off_group_source - source
        balance_scale = max(float(np.max(
            np.abs(radial_divergence) + np.abs(axial_leakage) + np.abs(reactions)
            + np.abs(off_group_source) + np.abs(source))), np.finfo(float).tiny)
        currents = np.zeros((self.G,) + self.radial_shape + (self.nz + 1,))
        currents[:, self.footprint, :] = self.axial.currents(flux)
        return dict(
            flux=self._unpack(flux), radial_flux=radial_flux,
            axial_flux=self._unpack(axial_flux), radial_currents=radial_currents,
            axial_currents=currents, radial_leakage=radial_leakage,
            axial_leakage=self._unpack(axial_leakage), balance_defect=self._unpack(defect),
            transport_residual=self._relative_change(flux, radial_columns),
            balance_residual=float(np.max(np.abs(defect))) / balance_scale,
            flux_disagreement=self._relative_change(radial_columns, axial_flux),
            negative_effective_sources=int(np.count_nonzero(effective_source < 0)),
            minimum_effective_source=float(np.min(effective_source)),
        )

    def solve(self, source, *, initial_flux=None, coupling="krylov", tol=1e-9,
              maxiter=500, restart=40, relaxation=0.5):
        """Solve a prescribed group source; self/off-group scattering is implicit.

        ``source`` and ``initial_flux`` broadcast to ``(G, nr, nc, 2, nz)``.
        Inactive values are ignored. Signed sources and fluxes are preserved.
        ``maxiter`` caps inner Krylov iterations or Picard updates; exhaustion
        returns ``converged=False``. No eigenvalue/fission update is performed.
        """
        if coupling not in ("krylov", "picard"):
            raise ValueError("coupling must be 'krylov' or 'picard'")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if not isinstance(restart, (int, np.integer)) or restart < 1:
            raise ValueError("restart must be a positive integer")
        if not np.isfinite(relaxation) or not 0 < relaxation <= 1:
            raise ValueError("relaxation must be in (0, 1]")
        started = time.perf_counter()
        sweeps_before = sum(p.n_sweeps for p in self._unique_planes)
        axial_before = self.n_axial_solves
        source = self._field(source, "source")
        flux = np.zeros_like(source) if initial_flux is None else self._field(initial_flux, "initial_flux")
        history, leakage_history = [], []
        reason = "iteration limit reached"
        if coupling == "krylov":
            shape, size = flux.shape, flux.size

            def apply(vector):
                trial = vector.reshape(shape)
                response = self._response(self._scattering(trial) - self.axial.leakage(trial))
                return (trial - response).ravel()

            def precondition(vector):
                # T is approximately 1/Sigma_t locally. Inverting the axial
                # removal operator suppresses the stiffness as dz decreases.
                return self._axial_solve(self._total_columns * vector.reshape(shape)).ravel()

            operator = LinearOperator((size, size), matvec=apply, dtype=float)
            preconditioner = LinearOperator((size, size), matvec=precondition, dtype=float)
            solution, info = gmres(
                operator, self._response(source).ravel(), x0=flux.ravel(),
                M=preconditioner, rtol=tol * 0.01, atol=0.0,
                restart=restart, maxiter=maxiter, callback=history.append,
                callback_type="legacy")
            flux = solution.reshape(shape)
            if info == 0:
                reason = "Krylov solve completed"
        else:
            for _ in range(maxiter):
                old_leakage = self.axial.leakage(flux)
                base_source = source + self._scattering(flux, include_self=False)
                plane_source = self._unpack(base_source - old_leakage)
                flux_field = self._unpack(flux)
                radial_divergence = np.zeros_like(flux)
                planes_converged = True
                for layer, plane in enumerate(self.planes):
                    for group in range(self.G):
                        state = plane.solve_group(
                            group, plane_source[group, ..., layer],
                            flux_field[group, ..., layer], tol=tol * 0.05)
                        planes_converged &= state.converged
                        radial_divergence[group, :, layer] = state.leakage[self.footprint]
                axial_flux = self._axial_solve(base_source - radial_divergence)
                updated = flux + relaxation * (axial_flux - flux)
                history.append(self._relative_change(updated, flux))
                # In the 2D limit true axial leakage is zero. Scale its update
                # by physical source/collision terms so round-off leakage does
                # not prevent an otherwise converged plane/column iteration.
                leakage_scale = max(float(np.max(np.abs(source))),
                                    float(np.max(np.abs(self._total_columns * updated))))
                leakage_history.append(self._relative_change(
                    self.axial.leakage(updated), old_leakage, leakage_scale))
                flux = updated
                if not planes_converged:
                    reason = "a planar scattering solve did not converge"
                    break
                if max(history[-1], leakage_history[-1]) <= tol * 0.05:
                    reason = "Picard updates converged"
                    break
        state = self._reconstruct(flux, source)
        physical_residual = max(state[key] for key in (
            "transport_residual", "balance_residual", "flux_disagreement"))
        iteration_finished = reason in ("Krylov solve completed", "Picard updates converged")
        converged = bool(iteration_finished and np.isfinite(physical_residual) and physical_residual <= tol)
        if converged:
            reason = "converged"
        elif iteration_finished:
            reason = "coupled balance or flux consistency exceeds tolerance"
        return SN2D1DResult(
            **state, converged=converged, reason=reason, iterations=len(history),
            n_sweeps=sum(p.n_sweeps for p in self._unique_planes) - sweeps_before,
            n_axial_solves=self.n_axial_solves - axial_before,
            solve_seconds=time.perf_counter() - started,
            residual_history=[float(value) for value in history],
            leakage_change_history=leakage_history)

    def solve_eigenvalue(self, *, initial_flux=None, initial_k=1.0,
                         tol_k=1e-7, tol_source=1e-7, tol_inner=None,
                         max_outer=500, max_inner=500, coupling="krylov",
                         restart=40, relaxation=0.5):
        """Fission power iteration using the existing coupled fixed-source solve.

        Scattering, including upscatter, is converged inside each power update.
        All returned fluxes/currents use unit volume-integrated fission
        production. The initial active flux must be non-negative and produce
        fission. Inner failure or exhausted outer iterations returns an explicit
        nonconverged result; sources and fluxes are never clipped.

        Success requires relative k, normalized fission-source and scalar-flux
        changes, plus the final hybrid-equation residuals. ``tol_inner`` defaults
        to the smaller of 1e-9 and 5% of the requested outer tolerances.
        """
        for name, value in (("tol_k", tol_k), ("tol_source", tol_source), ("initial_k", initial_k)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(max_outer, (int, np.integer)) or max_outer < 1:
            raise ValueError("max_outer must be a positive integer")
        if tol_inner is None:
            tol_inner = min(1e-9, 0.05 * min(tol_k, tol_source))
        if not np.isfinite(tol_inner) or tol_inner <= 0:
            raise ValueError("tol_inner must be finite and positive")
        if not np.any(self.nu_sigma_f[:, self.footprint, :] > 0):
            raise ValueError("no active fissile material: k-eigenvalue is undefined")
        started = time.perf_counter()
        sweeps_before = sum(plane.n_sweeps for plane in self._unique_planes)
        axial_before = self.n_axial_solves
        flux = self._unpack(self._field(1.0 if initial_flux is None else initial_flux, "initial_flux"))
        if np.any(flux < 0):
            raise ValueError("initial_flux must be non-negative on active cells")
        fission_source = self._fission_fields.fission_source(flux)
        production = float(fission_source.sum() * self.grid.cell_volume)
        if not np.isfinite(production) or production <= 0:
            raise ValueError("initial_flux must have finite positive fission production")
        flux /= production
        fission_source /= production
        k_eff = float(initial_k)
        k_history, k_changes, source_changes, flux_changes = [], [], [], []
        inner_iterations, inner_history, leakage_history = [], [], []
        iteration_count = 0
        reason = "outer iteration limit reached"
        state = None
        converged = False
        for _ in range(max_outer):
            old_source = self.chi * fission_source / k_eff
            inner = self.solve(old_source, initial_flux=flux, coupling=coupling,
                               tol=tol_inner, maxiter=max_inner,
                               restart=restart, relaxation=relaxation)
            iteration_count += inner.iterations
            inner_history.extend(inner.residual_history)
            leakage_history.extend(inner.leakage_change_history)
            if not inner.converged:
                reason = f"fixed-source solve failed: {inner.reason}"
                break
            trial_fission = self._fission_fields.fission_source(inner.flux)
            trial_production = float(trial_fission.sum() * self.grid.cell_volume)
            if not np.isfinite(trial_production) or trial_production <= 0:
                reason = "power update produced non-positive or non-finite fission production"
                break
            # The old state has unit integrated production. The usual ratio
            # k_new/k_old = production_new/production_old therefore reduces
            # to multiplication by trial_production.
            trial_k = k_eff * trial_production
            if not np.isfinite(trial_k) or trial_k <= 0:
                reason = "power update produced an invalid eigenvalue"
                break
            trial_flux = inner.flux / trial_production
            trial_fission /= trial_production
            k_changes.append(abs(trial_k-k_eff) / trial_k)
            source_changes.append(self._relative_change(trial_fission, fission_source))
            flux_changes.append(self._relative_change(trial_flux, flux))
            k_history.append(float(trial_k))
            inner_iterations.append(inner.iterations)
            flux, fission_source, k_eff = trial_flux, trial_fission, float(trial_k)
            if (k_changes[-1] <= tol_k and source_changes[-1] <= tol_source
                    and flux_changes[-1] <= tol_source):
                # Reconstruct from the NEW eigen-source, not the lagged source
                # used in the last fixed-source solve. This also puts every
                # current, leakage, and defect on the same normalized state.
                state = self._reconstruct(self._pack(flux), self._pack(self.chi*fission_source/k_eff))
                residual = max(state[key] for key in (
                    "transport_residual", "balance_residual", "flux_disagreement"))
                if np.isfinite(residual) and residual <= tol_source:
                    converged = True
                    reason = "converged"
                    break
                state = None
                reason = "eigenvalue balance or flux consistency exceeds tolerance"
        if state is None:
            state = self._reconstruct(self._pack(flux), self._pack(self.chi*fission_source/k_eff))
        return SN2D1DEigenResult(
            **state, converged=converged, reason=reason, k_eff=k_eff,
            fission_source=fission_source.copy(), outer_iterations=len(k_history),
            iterations=iteration_count,
            n_sweeps=sum(plane.n_sweeps for plane in self._unique_planes) - sweeps_before,
            n_axial_solves=self.n_axial_solves - axial_before,
            solve_seconds=time.perf_counter()-started,
            residual_history=inner_history, leakage_change_history=leakage_history,
            k_history=k_history, k_change_history=k_changes,
            source_change_history=source_changes, flux_change_history=flux_changes,
            inner_iteration_history=inner_iterations)
