"""Fixed-source plane contract over the existing triangular S_N solver.

This adapter owns no streaming discretization. Step/SCB sweeps, within-group
scattering solves, and conservative face fluxes all come from ``tri_sn``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tri_sn import TriSNTransportSolver


@dataclass
class PlaneSolution:
    flux: np.ndarray
    currents: np.ndarray
    leakage: np.ndarray
    converged: bool
    residual: float
    n_sweeps: int


class _PlanarFixedSource:
    """CPU plane adapter; returned arrays survive subsequent sweeps.

    ``sweep`` takes the complete isotropic source, including self-scattering.
    ``solve_group`` takes a source excluding self-scattering. Sources can be
    signed. Face currents have shape ``(3, nr, nc, 2)`` and are outward from
    each cell; the three faces retain ``tri_sn._EDGES`` order for each type.
    """

    def __init__(self, transport: TriSNTransportSolver):
        if transport.is3d or transport.xp is not np:
            raise ValueError("the prototype plane adapter requires a 2D CPU S_N solver")
        self.transport = transport
        self.shape = transport.grid.shape
        self.n_sweeps = 0

    def _source(self, group, source):
        if not 0 <= group < self.transport.G:
            raise ValueError("energy group is outside the plane solver")
        source = np.asarray(source, float)
        if source.shape != self.shape or not np.all(np.isfinite(source)):
            raise ValueError(f"plane source must be finite with shape {self.shape}")
        return np.where(self.transport.active, source, 0.0)

    def sweep(self, group, source):
        source = self._source(group, source)
        flux, directed_currents = self.transport._sweep_currents(group, source.ravel())
        self.n_sweeps += 1
        flux = np.asarray(flux).reshape(self.shape).copy()
        nr, nc, _ = self.shape
        currents = directed_currents.reshape(2, 3, nr, nc).transpose(1, 2, 3, 0).copy()
        leakage = currents.sum(axis=0) * (self.transport.h / self.transport.area)
        defect = leakage + self.transport.st[group] * flux - source
        scale = max(float(np.max(np.abs(source))),
                    float(np.max(np.abs(leakage) + np.abs(self.transport.st[group] * flux))),
                    np.finfo(float).tiny)
        residual = float(np.max(np.abs(defect))) / scale
        return PlaneSolution(flux, currents, leakage, True, residual, 1)

    def response(self, group, source):
        """Collision/streaming inverse without a face-current fold."""
        source = self._source(group, source)
        flux = self.transport._sweep(group, source.ravel())
        self.n_sweeps += 1
        return np.asarray(flux).reshape(self.shape).copy()

    def solve_group(self, group, source, initial_flux=None, *, tol=1e-10):
        """Reuse the configured within-group solver and verify its fixed point."""
        source = self._source(group, source)
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("plane tolerance must be finite and positive")
        initial = np.zeros(self.shape) if initial_flux is None else self._source(group, initial_flux)
        count_before = self.transport._sweep_count
        iterate = self.transport._solve_group(group, source.ravel(), initial.ravel(), tol)
        inner_sweeps = self.transport._sweep_count - count_before
        self.n_sweeps += inner_sweeps
        iterate = np.asarray(iterate).reshape(self.shape)
        self_scatter = self.transport.ss_self[group]
        result = self.sweep(group, source + self_scatter * iterate)
        defect = result.leakage + (self.transport.st[group] - self_scatter) * result.flux - source
        scale = max(float(np.max(np.abs(source) + np.abs(result.leakage)
                                 + np.abs((self.transport.st[group] - self_scatter) * result.flux))),
                    np.finfo(float).tiny)
        flux_scale = max(float(np.max(np.abs(result.flux))), np.finfo(float).tiny)
        result.residual = max(float(np.max(np.abs(defect))) / scale,
                              float(np.max(np.abs(result.flux - iterate))) / flux_scale)
        result.converged = bool(np.isfinite(result.residual) and result.residual <= tol)
        result.n_sweeps += inner_sweeps
        return result
