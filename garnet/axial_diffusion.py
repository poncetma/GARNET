"""Conservative CPU diffusion columns for the 2D/1D transport prototype.

The last array axis is z; preceding axes are independent columns. Face currents
use the same two-point diffusion and Robin boundary laws as ``TriGroupOperator``.
Coefficients are copied at construction; rebuild the operator after a material
change. Thomas factors are cached on the first solve.
"""

from __future__ import annotations

import numpy as np

from .stencil import face_alpha, harmonic_mean, normalize_bc, robin_face_term


class AxialDiffusionOperator:
    """``-d/dz(D dphi/dz) + removal*phi`` on independent finite-volume columns.

    ``diffusion`` has shape ``(..., nz)``; ``removal`` broadcasts to that shape.
    ``widths`` is a positive scalar or an ``(nz,)`` array in cm. ``bc`` is a
    diffusion face specification or a ``(lower, upper)`` pair. Currents are
    positive in increasing z, including at the two exterior faces.
    """

    def __init__(self, diffusion, removal, widths, bc="vacuum"):
        diffusion = np.asarray(diffusion, dtype=float)
        if diffusion.ndim < 1 or not diffusion.size:
            raise ValueError("diffusion needs a nonempty axial dimension")
        self.diffusion = diffusion.copy()
        self.shape = diffusion.shape
        self.removal = np.broadcast_to(np.asarray(removal, float), self.shape).copy()
        self.widths = np.broadcast_to(np.asarray(widths, float), (self.shape[-1],)).copy()
        if not np.all(np.isfinite(diffusion)) or np.any(diffusion <= 0):
            raise ValueError("diffusion coefficients must be finite and positive")
        if not np.all(np.isfinite(self.removal)) or np.any(self.removal < 0):
            raise ValueError("removal must be finite and non-negative")
        if not np.all(np.isfinite(self.widths)) or np.any(self.widths <= 0):
            raise ValueError("axial widths must be finite and positive")
        self.bc = normalize_bc(("reflective", "reflective", bc))[2]
        lower_widths, upper_widths = self.widths[:-1], self.widths[1:]
        if np.all(self.widths == self.widths[0]):
            self.face_conductance = harmonic_mean(
                diffusion[..., :-1], diffusion[..., 1:]) / self.widths[0]
        else:
            self.face_conductance = 1.0 / (
                lower_widths / (2.0 * diffusion[..., :-1])
                + upper_widths / (2.0 * diffusion[..., 1:]))
        self.boundary_conductance = tuple(
            np.asarray(robin_face_term(np, diffusion[..., end], self.widths[end], face_alpha(spec))
                       * self.widths[end]).copy()
            for end, spec in zip((0, -1), self.bc)
        )
        self.lower = -self.face_conductance / upper_widths
        self.upper = -self.face_conductance / lower_widths
        self.diagonal = self.removal.copy()
        self.diagonal[..., :-1] -= self.upper
        self.diagonal[..., 1:] -= self.lower
        self.diagonal[..., 0] += self.boundary_conductance[0] / self.widths[0]
        self.diagonal[..., -1] += self.boundary_conductance[1] / self.widths[-1]
        self._inverse_pivots = None
        self._multipliers = None
        # Factors must remain consistent with apply() and reconstructed currents.
        for coefficient in (self.diffusion, self.removal, self.widths,
                            self.face_conductance, self.lower, self.upper,
                            self.diagonal, *self.boundary_conductance):
            coefficient.flags.writeable = False

    def _field(self, value):
        field = np.asarray(value, dtype=float)
        if field.shape != self.shape or not np.all(np.isfinite(field)):
            raise ValueError(f"field must be finite with shape {self.shape}")
        return field

    def currents(self, flux):
        """One shared current density per face, shape ``(..., nz+1)``."""
        flux = self._field(flux)
        current = np.empty(self.shape[:-1] + (self.shape[-1] + 1,))
        current[..., 1:-1] = self.face_conductance * (flux[..., :-1] - flux[..., 1:])
        current[..., 0] = -self.boundary_conductance[0] * flux[..., 0]
        current[..., -1] = self.boundary_conductance[1] * flux[..., -1]
        return current

    def leakage(self, flux):
        """Axial outward-current divergence per unit cell volume."""
        return np.diff(self.currents(flux), axis=-1) / self.widths

    def apply(self, flux):
        flux = self._field(flux)
        return self.leakage(flux) + self.removal * flux

    def _factor(self):
        if self._inverse_pivots is not None:
            return
        if all(face_alpha(spec) == 0 for spec in self.bc) and np.any(
                np.all(self.removal == 0, axis=-1)):
            raise ValueError("a reflective column with zero removal has a singular diffusion solve")
        pivots = self.diagonal.copy()
        multipliers = np.empty_like(self.lower)
        for layer in range(1, self.shape[-1]):
            multipliers[..., layer - 1] = self.lower[..., layer - 1] / pivots[..., layer - 1]
            pivots[..., layer] -= multipliers[..., layer - 1] * self.upper[..., layer - 1]
        if not np.all(np.isfinite(pivots)) or np.any(pivots <= 0):
            raise ValueError("axial diffusion factorization has a non-positive pivot")
        self._inverse_pivots = 1.0 / pivots
        self._multipliers = multipliers

    def solve(self, source):
        """Solve all columns for a prescribed source, using cached factors."""
        result = self._field(source).copy()
        self._factor()
        for layer in range(1, self.shape[-1]):
            result[..., layer] -= self._multipliers[..., layer - 1] * result[..., layer - 1]
        result[..., -1] *= self._inverse_pivots[..., -1]
        for layer in range(self.shape[-1] - 2, -1, -1):
            result[..., layer] -= self.upper[..., layer] * result[..., layer + 1]
            result[..., layer] *= self._inverse_pivots[..., layer]
        return result
