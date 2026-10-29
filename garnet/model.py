"""Simplified, human-friendly front ends for building and running a reactor.

The full solver classes are flexible but ask you to build grids, hand-assemble
material-map arrays, and read a terse ``Result``. These builders wrap the common
cases and return a :class:`ReactorResult` that prints a transparent report
(k_eff, reactivity, where the neutrons go, flux peaking, per-material power):

* :class:`Model`       -- a rectangular Cartesian core, painted with boxes in cm
                          (diffusion or SP3; 1-D / 2-D / 3-D; forward or adjoint).
* :class:`MeshModel`   -- an arbitrary unstructured mesh (a Gmsh file or an
                          assembled :class:`~garnet.Mesh`), 2-D or 3-D, materials
                          assigned by physical region or mesh tag.
* :class:`HexLattice`  -- a hexagonal/prismatic geometry builder whose
                          :meth:`~HexLattice.build` returns a reusable
                          :class:`TriReactor` for steady, transient and coupled
                          triangular-grid calculations.

Materials are ordinary :class:`~garnet.Material` objects, so defining new cross
sections is unchanged; these classes only remove the grid/array bookkeeping.

Example -- a reflected Cartesian core::

    import garnet
    m = garnet.Model(size=(120, 120), cells=(60, 60))     # 2-D, cm
    m.fill(reflector).add_box(fuel, x=(30, 90), y=(30, 90)).set_boundary("vacuum")
    print(m.run())
"""

from __future__ import annotations

import math
import numbers
import os
import warnings
from copy import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._model_support import (box_range, checked_kinetics, checked_materials,
                             material_map as checked_map, replace_materials,
                             solver_options, transient_schedule)
from .backend import asnumpy
from .grid import Grid
from .hexraster import hex_site_xy, rasterize_hex_sites
from .materials import Kinetics, Material
from .mesh import UnstructuredDiffusionSolver, read_gmsh
from .stencil import face_alpha, normalize_bc
from .solver import (DiffusionEigenSolver, SDP1EigenSolver, SDP2EigenSolver,
                     SDP3EigenSolver, SP1EigenSolver, SP3EigenSolver,
                     SP5EigenSolver, SP7EigenSolver)
from .transient import TransientSolver
from .tri import (TriDiffusionEigenSolver, TriGrid, TriSDP1EigenSolver,
                  TriSDP2EigenSolver, TriSDP3EigenSolver, TriSP1EigenSolver,
                  TriSP3EigenSolver, TriSP5EigenSolver, TriSP7EigenSolver)

_STRUCTURED = {"diffusion": DiffusionEigenSolver, "sp1": SP1EigenSolver,
               "sp3": SP3EigenSolver, "sp5": SP5EigenSolver,
               "sp7": SP7EigenSolver, "sdp1": SDP1EigenSolver,
               "sdp2": SDP2EigenSolver, "sdp3": SDP3EigenSolver}
_TRI = {"diffusion": TriDiffusionEigenSolver, "sp1": TriSP1EigenSolver,
        "sp3": TriSP3EigenSolver, "sp5": TriSP5EigenSolver,
        "sp7": TriSP7EigenSolver, "sdp1": TriSDP1EigenSolver,
        "sdp2": TriSDP2EigenSolver, "sdp3": TriSDP3EigenSolver}
_AXES = ("x", "y", "z")


def _albedo(spec) -> float:
    """Robin coefficient for a boundary name/albedo (zero-flux -> a large finite alpha)."""
    if np.ndim(spec) != 0:
        raise ValueError("MeshModel boundary must be a single face name or albedo")
    a = face_alpha(normalize_bc(spec)[0][0])
    if math.isnan(a):
        raise ValueError("boundary albedo cannot be NaN")
    return 1e8 if a == float("inf") else float(a)


# ---------------------------------------------------------------------------
# Shared diagnostics + report
# ---------------------------------------------------------------------------
def _diagnostics(materials, cell_material, flux_flat, cell_volume, active,
                 mix_material=None, mix_weight=None):
    """Reaction-rate diagnostics from the scalar flux, restricted to active cells.

    Returns (production, absorption, material_rows). Rates are formed straight
    from the cross sections and the physical scalar flux (so this works for
    diffusion or SP3), volume-weighted for non-uniform meshes.
    """
    groups = flux_flat.shape[0]
    cell_count = flux_flat.shape[1]
    absorption_xs = np.array([m.sigma_a for m in materials])          # (M, G)
    production_xs = np.array([m.nu_sigma_f for m in materials])
    base_material = np.asarray(cell_material).reshape(-1)
    cell_volumes = np.broadcast_to(np.asarray(cell_volume, float), (cell_count,))
    active_cells = (np.ones(cell_count, bool) if active is None
           else np.asarray(active).reshape(-1).astype(bool))
    volume_weighted_flux = flux_flat * cell_volumes * active_cells
    if mix_material is None:
        mixed_material = np.zeros(cell_count, dtype=np.int64)
        mixed_weight = np.zeros(cell_count)
        mixed = np.zeros(cell_count, dtype=bool)
    else:
        mixed_material = np.asarray(mix_material).reshape(-1)
        mixed_weight = np.asarray(mix_weight, dtype=float).reshape(-1)
        if mixed_material.shape != base_material.shape or mixed_weight.shape != base_material.shape:
            raise ValueError("mix_material/mix_weight must match the material map")
        mixed = mixed_material >= 0
        mixed_material = np.where(mixed, mixed_material, 0)
    base_fraction = np.where(mixed, 1.0 - mixed_weight, 1.0)
    mixed_fraction = np.where(mixed, mixed_weight, 0.0)

    production = 0.0
    absorption = 0.0
    for g in range(groups):
        group_production_xs = (production_xs[base_material, g] * base_fraction
                               + production_xs[mixed_material, g] * mixed_fraction)
        group_absorption_xs = (absorption_xs[base_material, g] * base_fraction
                               + absorption_xs[mixed_material, g] * mixed_fraction)
        production += float((group_production_xs * volume_weighted_flux[g]).sum())
        absorption += float((group_absorption_xs * volume_weighted_flux[g]).sum())

    total_volume = float((cell_volumes * active_cells).sum())
    rows = []
    for idx, mat in enumerate(materials):
        material_fraction = active_cells * (
            base_fraction * (base_material == idx)
            + mixed_fraction * (mixed_material == idx))
        if not np.any(material_fraction > 0.0):
            continue
        volume_fraction = float((cell_volumes * material_fraction).sum()) / total_volume
        material_production = float(sum(
            production_xs[idx, g] * (flux_flat[g] * cell_volumes * material_fraction).sum()
            for g in range(groups)))
        rows.append([mat.name or f"material {idx}", volume_fraction, material_production, 0.0])
    for row in rows:
        row[3] = row[2] / production if production > 0 else 0.0
    return production, absorption, rows


@dataclass(frozen=True)
class MaterialSummary:
    """Volume and fission-production fractions for one solved material."""

    name: str
    volume_fraction: float
    fission_fraction: float


class ReactorResult:
    """A solved reactor with a human-readable report.

    Printing it (or :meth:`summary`) shows k_eff, reactivity, the converged
    neutron balance, flux peaking and per-material power. Numeric fields
    (:attr:`k_eff`, :attr:`flux`, :attr:`leakage_fraction`, ...) are plain values.
    """

    def __init__(self, *, k_eff, converged, outer_iterations, inner_iterations,
                 solve_seconds, device, method, flux, materials, cell_material,
                 cell_volume=1.0, active=None, geometry_line="", boundary_line="",
                 adjoint=False, absorption_correction=0.0,
                 mix_material=None, mix_weight=None, grid=None,
                 boundary_conditions=None):
        self.k_eff = float(k_eff)
        self.reactivity_pcm = (1.0 - 1.0 / self.k_eff) * 1e5
        self.converged = bool(converged)
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.solve_seconds = solve_seconds
        self.device = device
        self.method = method
        self.adjoint = adjoint
        self.grid = grid
        self.boundary_conditions = boundary_conditions
        self._external_xs = False
        self.flux = flux                          # (G, *shape) numpy
        self.n_groups = flux.shape[0]
        # Keep postprocessing tied to the solved state when the builder or its
        # material arrays are subsequently edited.
        self._materials = tuple(m.replace() for m in materials)
        self._material_map = np.asarray(cell_material).copy()
        self.cell_volumes = np.broadcast_to(
            np.asarray(cell_volume, float), flux.shape[1:]).copy()
        self._active = (np.ones(flux.shape[1:], bool) if active is None else
                        np.asarray(active, bool).copy())
        self._mix_material = None if mix_material is None else np.asarray(mix_material).copy()
        self._mix_weight = None if mix_weight is None else np.asarray(mix_weight).copy()
        self._geometry_line = geometry_line
        self._boundary_line = boundary_line

        flat = flux.reshape(self.n_groups, -1)    # (G, ncells)
        act = (np.ones(flat.shape[1], bool) if active is None
               else np.asarray(active).reshape(-1).astype(bool))
        prod, absorb, rows = _diagnostics(
            materials, cell_material, flat, self.cell_volumes.reshape(-1), act,
            mix_material=mix_material, mix_weight=mix_weight)
        absorb += absorption_correction           # volume-mixed sub-cell absorbers (drum arcs)
        loss = prod / self.k_eff                  # neutrons available per generation
        self.absorbed_fraction = absorb / loss if loss > 0 else float("nan")
        self.leakage_fraction = 1.0 - self.absorbed_fraction
        thermal = flat[-1][act]
        self.peaking = (float(thermal.max() / np.average(
            thermal, weights=self.cell_volumes.reshape(-1)[act]))
            if thermal.size else float("nan"))
        self._material_stats = rows
        self.material_summary = tuple(MaterialSummary(name, volf, fisf)
                                      for name, volf, _fis, fisf in rows)

    @property
    def flux_numpy(self):
        """Host scalar flux; also available as ``flux`` on public results."""
        return self.flux

    def power_density(self, total_power):
        """Return a NumPy power-density field in W/cm³ at the given domain power.

        Its integral ``(q * result.cell_volumes).sum()`` equals ``total_power``
        in W. Two-dimensional calculations use a 1 cm slice unless their grid
        specifies a different thickness. Energy-release cross sections are used
        when supplied; otherwise the shape uses neutron fission production.
        """
        from .power import power_density

        if self.adjoint:
            raise ValueError("power density requires a forward flux, not an adjoint importance")
        if not np.isfinite(total_power) or total_power < 0.0:
            raise ValueError("total_power must be finite and non-negative")
        return power_density(
            self.flux, self._materials, self._material_map,
            total_power=total_power, volume_weight=self.cell_volumes,
            active=self._active, mix_material=self._mix_material,
            mix_weight=self._mix_weight)

    def summary(self) -> str:
        kind = "adjoint (importance) solution" if self.adjoint else "solution"
        status = (f"converged in {self.outer_iterations} outer / "
                  f"{self.inner_iterations} inner iterations, {self.solve_seconds:.2f} s"
                  if self.converged else "DID NOT CONVERGE")
        lines = [
            f"GARNET reactor {kind}",
            "=" * (14 + len(kind)),
            f"  geometry    : {self._geometry_line}",
            f"  groups      : {self.n_groups}     method: {self.method}     device: {self.device}",
            f"  boundary    : {self._boundary_line}",
            "",
            f"  k_eff       : {self.k_eff:.6f}",
            f"  reactivity  : {self.reactivity_pcm:+.0f} pcm   (rho = (k-1)/k)",
            f"  status      : {status}",
            "",
        ]
        if self.adjoint:
            lines.append("  (adjoint flux is neutron importance, not a physical flux)")
            lines.append(f"  importance peaking (thermal): peak / average = {self.peaking:.2f}")
            return "\n".join(lines)
        lines += [
            "  where the fission neutrons go (per neutron produced / k):",
            f"    absorbed  : {self.absorbed_fraction * 100:5.1f} %",
            f"    leaked    : {self.leakage_fraction * 100:5.1f} %",
            "",
            f"  flux peaking (thermal group): peak / average = {self.peaking:.2f}",
        ]
        if len(self._material_stats) > 1:
            lines.append("")
            lines.append(f"  {'material':<16}{'volume':>9}{'fission':>10}")
            for name, volf, _fis, fisf in self._material_stats:
                lines.append(f"    {name:<14}{volf * 100:7.1f}%{fisf * 100:9.1f}%")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        s = "converged" if self.converged else "NOT CONVERGED"
        tag = "adjoint " if self.adjoint else ""
        return (f"ReactorResult({tag}k_eff={self.k_eff:.6f}, {self.reactivity_pcm:+.0f} pcm, "
                f"{s}, {self.method} on {self.device})")


class ModelResult(ReactorResult):
    """:class:`ReactorResult` for a structured :class:`Model` (keeps ``.model``)."""

    def __init__(self, model, result, method, adjoint=False, *, problem=None):
        self.model = model
        self.raw = result
        materials, material_map, mix, weight = (
            (model.materials, model._map, None, None) if problem is None else problem)
        if isinstance(materials, Material):
            materials = [materials]
        if material_map is None:
            material_map = np.zeros(model.grid.shape, dtype=np.int64)
        super().__init__(
            k_eff=result.k_eff, converged=result.converged,
            outer_iterations=result.outer_iterations,
            inner_iterations=result.inner_iterations,
            solve_seconds=result.solve_seconds, device=result.device, method=method,
            flux=result.flux_numpy, materials=materials,
            cell_material=material_map, cell_volume=model.grid.cell_volumes, adjoint=adjoint,
            mix_material=mix, mix_weight=weight,
            grid=model.grid, boundary_conditions=(normalize_bc(model._full_bc()), None),
            geometry_line=model._geometry_line(), boundary_line=model._boundary_line())


_SPARK = " ▁▂▃▄▅▆▇█"


class TransientModelResult:
    """Common neutron-transient report for Cartesian, cylindrical and tri grids.

    Every transient begins from a steady-state solve, so that solution is kept
    here as :attr:`steady` (a :class:`ReactorResult`) and its eigenvalue as
    :attr:`k0`; the perturbation drives the power away from ``P(0) = 1``. The
    time and power arrays (:attr:`times`, :attr:`power`) are plain NumPy.
    """

    def __init__(self, model, tres, kinetics, *, steady_problem=None):
        self.model = model
        self.reactor = model
        self.raw = tres
        self.kinetics = kinetics
        self.k0 = float(tres.k0)
        self.times = np.asarray(tres.times)
        self.power = np.asarray(tres.power)
        self.device = tres.device
        self.solve_seconds = tres.solve_seconds
        self.total_inner_iterations = tres.total_inner_iterations
        self.flux = tres.flux_numpy                    # final scalar flux
        self.steady = model._wrap_result(tres.steady, problem=steady_problem)
        ip = int(np.argmax(self.power))
        self.peak_power, self.peak_time = float(self.power[ip]), float(self.times[ip])
        self.final_power, self.final_time = float(self.power[-1]), float(self.times[-1])
        self.dt = float(self.times[1] - self.times[0]) if self.times.size > 1 else 0.0

    def __getattr__(self, name):
        # Preserve solver telemetry, including fields added by later engines.
        raw = self.__dict__.get("raw")
        if raw is None:
            raise AttributeError(name)
        return getattr(raw, name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self.raw)))

    @property
    def flux_numpy(self):
        return self.flux

    @property
    def precursors_numpy(self):
        """Copy final spatial precursor fields to NumPy when requested."""
        return asnumpy(self.raw.precursors)

    @property
    def step_widths(self):
        """Accepted widths in seconds, for fixed and adaptive schedules alike."""
        return np.diff(self.times)

    @property
    def n_steps(self):
        return self.times.size - 1

    def power_at(self, times):
        """Linearly sample recorded P/P0 inside the solved time span.

        This interpolates the saved history; it is not a higher-order dense
        output formula. Extrapolation and nonfinite sample times are rejected.
        """
        values = np.asarray(times, dtype=float)
        if (not np.all(np.isfinite(values)) or np.any(values < self.times[0]) or
                np.any(values > self.times[-1])):
            raise ValueError("sample times must lie inside the recorded time span")
        sampled = np.interp(values, self.times, self.power)
        return float(sampled) if values.ndim == 0 else sampled

    def _sparkline(self, width=48):
        p = self.power
        idx = np.linspace(0, len(p) - 1, min(width, len(p))).round().astype(int)
        s = p[idx]
        lo, hi = float(s.min()), float(s.max())
        lvl = (np.zeros(len(s), int) if hi - lo < 1e-12
               else np.clip(((s - lo) / (hi - lo) * 8).astype(int), 0, 8))
        return "".join(_SPARK[i] for i in lvl)

    def summary(self) -> str:
        kin = self.kinetics
        fam = "family" if kin.n_families == 1 else "families"
        lines = [
            "GARNET reactor transient",
            "=======================",
            f"  geometry    : {self.model._geometry_line()}",
            f"  groups      : {self.steady.n_groups}     boundary: {self.model._boundary_line()}",
            f"  kinetics    : {kin.n_families} delayed {fam}, " +
            ("material-dependent beta" if kin.beta.ndim == 2 else
             f"beta = {kin.beta_total * 1e5:.0f} pcm"),
            "",
            f"  initial steady state : k0 = {self.k0:.6f}  "
            f"(fission source normalised by k0; unperturbed -> P/P0 stays 1)",
            f"  time span            : 0 -> {self.final_time:g} s, "
            f"{self.n_steps} steps ({self.time_scheme}), {self.solve_seconds:.2f} s on {self.device}",
            "",
            "  power P/P0:",
            f"    peak      : {self.peak_power:.4f} at t = {self.peak_time:g} s",
            f"    final     : {self.final_power:.4f} at t = {self.final_time:g} s",
            f"    trace     : {self._sparkline()}",
            f"                [{self.power.min():.3g} .. {self.power.max():.3g}] over 0..{self.final_time:g} s",
            "",
            "  (.steady holds the full t=0 solution report)",
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return (f"TransientModelResult(k0={self.k0:.6f}, P(end)={self.final_power:.4f} P0, "
                f"peak {self.peak_power:.4f} at t={self.peak_time:g}s, on {self.device})")


def _warn_painted_extent(axis, lo, hi, selected, length):
    """Warn when cell-centre selection paints a visibly different interval.

    Boxes keep their cell-centre semantics; each painted edge can move by up
    to half a cell, so the painted length may differ from the request by up to
    one cell. Half a cell or more of total mismatch (e.g. a 60 cm request
    painted as 51.4 cm on 17.1 cm cells) is reported rather than left silent.
    """
    if not np.any(selected):
        return
    d = length / selected.size
    index = np.flatnonzero(selected)
    painted = (index[0] * d, (index[-1] + 1) * d)
    wanted = (max(float(lo), 0.0), min(float(hi), length))
    mismatch = abs(painted[0] - wanted[0]) + abs(painted[1] - wanted[1])
    if mismatch >= 0.5 * d * (1.0 - 1e-9):
        warnings.warn(
            f"{axis} range ({float(lo):g}, {float(hi):g}) cm selects cell centres "
            f"painting ({painted[0]:g}, {painted[1]:g}) cm "
            f"({painted[1] - painted[0]:g} cm instead of {wanted[1] - wanted[0]:g} cm "
            f"on {d:g} cm cells); align the box edges with cell faces or refine "
            f"the mesh", UserWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Structured Cartesian model
# ---------------------------------------------------------------------------
class _CoupledModel:
    """Common thermal-coupling API for Cartesian and triangular models."""

    def _transient_problem(self, materials_at=None, state_at=None, at=None):
        callbacks = (materials_at, state_at, at)
        if sum(callback is not None for callback in callbacks) > 1:
            raise ValueError("use only one of materials_at, state_at or at/problem_at")
        if any(callback is not None and not callable(callback) for callback in callbacks):
            raise TypeError("transient callbacks must be callable")
        if at is not None:
            return at
        if state_at is not None:
            return lambda t: self._state_problem(state_at(t))
        geometry = self._coupling_geometry()
        base_materials, cache = self.materials, {}

        def problem_at(t):
            materials = (base_materials if materials_at is None else
                    replace_materials(base_materials, materials_at(t), context="materials_at"))
            key = tuple(id(m) for m in materials)
            if key not in cache:
                cache.clear()
                cache[key] = materials
            return (cache[key], geometry["material_map"],
                    geometry.get("mix_material"), geometry.get("mix_weight"))

        return problem_at

    def _initial_steady(self, result, problem, *, external_xs=False):
        """Validate public eigenpairs against the actual time-zero problem."""
        if not isinstance(result, ReactorResult):
            return result  # Low-level results retain their existing caller-owned contract.
        if result._external_xs or external_xs:
            raise ValueError("initial_steady has untracked cross-section updates; "
                             "solve a model with the explicit time-zero materials")
        if result.adjoint or result.method != "diffusion" or not result.converged:
            raise ValueError("initial_steady must be a converged forward diffusion result")
        geometry = self._coupling_geometry()
        boundary = (normalize_bc(geometry["bc"]), geometry.get("mask_bc"))
        if result.grid != self.grid or result.boundary_conditions != boundary:
            raise ValueError("initial_steady has incompatible grid or boundary conditions")
        materials, material_indices, mixed_material, mixed_weight = problem
        materials = [materials] if isinstance(materials, Material) else list(materials)
        material_indices = np.zeros(
            self.grid.shape,
            dtype=np.int64,
        ) if material_indices is None else material_indices
        keys = ("diffusion", "sigma_a", "nu_sigma_f", "sigma_s", "chi", "total")
        same_materials = len(materials) == len(result._materials) and all(
            all(np.array_equal(getattr(a, key), getattr(b, key)) for key in keys)
            for a, b in zip(materials, result._materials))
        active = geometry.get("active")
        active = np.ones(self.grid.shape, bool) if active is None else active
        if (not same_materials or not np.array_equal(material_indices, result._material_map) or
                not np.array_equal(mixed_material, result._mix_material) or
                not np.array_equal(mixed_weight, result._mix_weight) or
                not np.array_equal(active, result._active)):
            raise ValueError("initial_steady does not match the time-zero materials or maps")
        return result.raw

    def transient(self, t_end, dt, kinetics=None, materials_at=None, at=None,
                  device="auto", *, state_at=None, event_times=None, on_step=None,
                  precond_degree=0, precond_dtype=None, linear_solver="cg",
                  solver_kwargs=None, **solve_kwargs) -> TransientModelResult:
        """Solve diffusion kinetics from a time-zero critical equilibrium.

        ``materials_at(t)`` returns a partial name/index mapping or an ordered
        material list. ``state_at(t)`` selects cached models on the same grid;
        ``at(t)`` is the raw two/four-array escape hatch. Supply one callback.
        ``event_times`` splits fixed steps or aligns adaptive steps at control
        events and restarts its history. ``on_step(t, flux, power)`` runs after
        each accepted step; flux remains on the selected device.

        Time controls (time_scheme, adaptive_bdf, adaptive_ftm, tolerances) go here.
        Adaptive controllers select monolithic stepping unless overridden.
        Additional constructor settings belong in ``solver_kwargs``. A public
        ``initial_steady`` is checked against the actual t=0 model before reuse.
        ``kinetics.chi_delayed`` may supply a shared or family-specific delayed
        spectrum, distinct from the total fission spectrum in ``Material.chi``.

        ``materials_at`` must return NEW Material objects whenever cross
        sections change. The solver caches its operator by the identity of the
        returned materials, so mutating a Material's arrays in place and
        returning the same object leaves the time-zero operator in use, silently.
        """
        from .tri import TriGroupOperator

        checked_materials(self.materials)
        kinetics_data = self.kinetics if kinetics is None else kinetics
        if kinetics_data is None:
            raise ValueError("no kinetics: call set_kinetics() or pass kinetics=")
        checked_kinetics(kinetics_data, self.n_groups, material_count=len(self.materials))
        problem_at = self._transient_problem(materials_at, state_at, at)
        geometry = self._coupling_geometry()
        eig_solver = geometry.pop("solver_cls", TriDiffusionEigenSolver)
        constructor_options = dict(geometry, device=device, eig_solver=eig_solver,
                    precond_degree=precond_degree, precond_dtype=precond_dtype,
                    linear_solver=linear_solver)
        constructor_options.pop("material_map")
        constructor_options.pop("grid")
        if isinstance(self.grid, TriGrid):
            constructor_options["group_operator"] = TriGroupOperator
        options = dict(solver_kwargs or {})
        if on_step is not None:
            if "on_step" in options:
                raise ValueError("on_step was supplied both directly and in solver_kwargs")
            options["on_step"] = on_step
        if options.get("on_step") is not None and not callable(options["on_step"]):
            raise TypeError("on_step must be callable")
        solver = TransientSolver(
            self.grid,
            problem_at,
            kinetics_data,
            **solver_options(constructor_options, options),
        )
        initial_problem = solver._unpack(problem_at(0.0))
        if solve_kwargs.get("initial_steady") is not None:
            solve_kwargs["initial_steady"] = self._initial_steady(
                solve_kwargs["initial_steady"], initial_problem,
                external_xs=options.get("xs_update_at") is not None)
        dt, solve_kwargs = transient_schedule(t_end, dt, event_times, solve_kwargs)
        result = solver.solve(t_end=t_end, dt=dt, **solve_kwargs)
        result_view = TransientModelResult(self, result, kinetics_data, steady_problem=initial_problem)
        result_view.steady._external_xs = options.get("xs_update_at") is not None
        return result_view

    def _ordered_thermal_materials(self, spec):
        from .thermal import ThermalMaterial

        if not isinstance(spec, Mapping):
            values = list(spec)
            if len(values) != len(self.materials):
                raise ValueError("thermal material list must align with materials")
            return values
        names = [m.name for m in self.materials]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("named thermal mapping needs unique non-empty material names")
        geometry = self._coupling_geometry()
        material_map = geometry["material_map"]
        active = geometry.get("active")
        active = np.ones(material_map.shape, bool) if active is None else active
        unknown = set(spec) - set(names)
        if unknown:
            raise ValueError(f"unknown thermal material names: {sorted(unknown)}")
        used = set(np.unique(material_map[active]).tolist())
        mix = geometry.get("mix_material")
        if mix is not None:
            used.update(np.unique(mix[
                active & (mix >= 0) & (geometry["mix_weight"] > 0)]).tolist())
        out = []
        for i, name in enumerate(names):
            if name in spec:
                out.append(spec[name])
            elif i not in used:
                out.append(ThermalMaterial(1e-12, name=f"inactive {name}"))
            else:
                raise ValueError(f"no thermal material supplied for active {name!r}")
        return out

    def _feedback_from_mapping(self, feedback_specs, ambient_temperature):
        from .feedback import ThermalFeedback

        names = [m.name for m in self.materials]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("named feedback mapping needs unique non-empty material names")
        unknown = set(feedback_specs) - set(names)
        if unknown:
            raise ValueError(f"unknown feedback material names: {sorted(unknown)}")
        rows, reference_temperatures, expansion = [], [], []
        any_groupwise = False
        any_expansion = False
        for name in names:
            feedback_spec = feedback_specs.get(name, FeedbackSpec(ambient_temperature))
            if isinstance(feedback_spec, Mapping):
                feedback_spec = FeedbackSpec(**feedback_spec)
            if not isinstance(feedback_spec, FeedbackSpec):
                raise TypeError("feedback mapping values must be FeedbackSpec or dict")
            row = np.atleast_1d(np.asarray(feedback_spec.doppler, dtype=float))
            if row.size not in (1, self.n_groups):
                raise ValueError(f"feedback for {name!r} needs scalar or "
                                 f"{self.n_groups}-group doppler")
            any_groupwise |= row.size > 1
            rows.append(row)
            reference_temperatures.append(float(feedback_spec.reference_temperature))
            expansion.append(0.0 if feedback_spec.expansion is None else float(feedback_spec.expansion))
            any_expansion |= feedback_spec.expansion is not None
        doppler = (np.vstack([np.broadcast_to(
            group_coefficients,
            (self.n_groups,),
        ) for group_coefficients in rows])
                   if any_groupwise else np.array([group_coefficients[0] for group_coefficients in rows]))
        return ThermalFeedback(reference_temperatures, doppler,
                               np.array(expansion) if any_expansion else None)

    def configure_thermal(self, thermal_materials, *, total_power, feedback=None,
                          thermal_bc="adiabatic", thermal_mask_bc="adiabatic",
                          ambient_temperature=300.0, warm_start=True,
                          eigen_kwargs=None):
        """Attach conduction and feedback data used by coupled calculations.

        ``thermal_materials`` may be a list in solver material order or a
        ``{material_name: ThermalMaterial}`` mapping. ``feedback`` may be an
        existing feedback object or a named mapping of :class:`FeedbackSpec`.
        Missing feedback entries mean zero feedback; inactive padding materials
        need not be present in the thermal mapping.
        """
        from .feedback import ThermalFeedback

        checked_materials(self.materials)
        thermal = self._ordered_thermal_materials(thermal_materials)
        if feedback is None:
            feedback = ThermalFeedback(
                np.full(len(self.materials), float(ambient_temperature)),
                np.zeros(len(self.materials)))
        elif isinstance(feedback, Mapping):
            feedback = self._feedback_from_mapping(feedback, ambient_temperature)
        if (hasattr(feedback, "n_materials") and
                feedback.n_materials != len(self.materials)):
            raise ValueError("feedback material count does not match reactor materials")
        if not np.isfinite(total_power) or total_power < 0.0:
            raise ValueError("total_power must be finite and non-negative")
        self._thermal_config = dict(
            thermal_materials=thermal, total_power=float(total_power),
            feedback=feedback, thermal_bc=thermal_bc,
            thermal_mask_bc=thermal_mask_bc,
            ambient_temperature=float(ambient_temperature),
            warm_start=bool(warm_start),
            eigen_kwargs=dict(eigen_kwargs or {"tol_k": 1e-9,
                                               "tol_source": 1e-8}))
        return self

    def coupling_context(self, *, device="auto", dtype=np.float64, **overrides):
        """Create the low-level :class:`CouplingContext` for advanced use."""
        from .coupling import CouplingContext

        if self._thermal_config is None:
            raise ValueError("thermal physics is not configured; call configure_thermal()")
        kw = dict(self._thermal_config)
        kw.update(overrides)
        geometry = self._coupling_geometry()
        return CouplingContext(
            materials=self.materials, kinetics=self.kinetics,
            device=device, dtype=dtype, **dict(geometry, **kw))

    def coupled_steady(self, *, device="auto", context_kwargs=None,
                       **solve_kwargs):
        """Solve the self-consistent steady neutronics/thermal state."""
        from .coupling import CoupledSolver

        ctx = self.coupling_context(device=device, **(context_kwargs or {}))
        return CoupledSolver(ctx).solve(**solve_kwargs)

    def coupled_transient(self, t_end, dt, *, device="auto", problem_at=None,
                          state_at=None, materials_at=None, on_step=None, **kwargs):
        """Run a coupled diffusion/conduction transient.

        Named ``materials_at`` and cached ``state_at`` follow neutron kinetics.
        ``on_step(t, flux, power, temperature)`` observes accepted neutron steps,
        after any due thermal exchange. Arrays stay on their current device and
        must be treated as read-only; copy them to retain a history.
        """
        from .coupling import coupled_transient

        if self.kinetics is None:
            raise ValueError("no kinetics: call set_kinetics() first")
        if any(callback is not None for callback in (materials_at, state_at, problem_at)):
            problem_at = self._transient_problem(materials_at, state_at, problem_at)
        if kwargs.get("transient_kwargs") is not None:
            _, kwargs["transient_kwargs"] = transient_schedule(
                t_end, dt, None, kwargs["transient_kwargs"])
        ctx = self.coupling_context(device=device)
        return coupled_transient(ctx, t_end=t_end, dt=dt,
                                 problem_at=problem_at, on_step=on_step, **kwargs)

    def quasistatic_transient(self, t_end, dt, *, shape_dt=2.0,
                              device="auto", problem_at=None, state_at=None,
                              materials_at=None, **kwargs):
        """Run an improved quasi-static coupled transient.

        The adjoint-weighted population amplitude and precursors advance every
        ``dt`` while the full spatial shape is corrected every ``shape_dt``;
        the returned power includes the shape's adjoint-weighted fission-power
        factor. Time-dependent IQS is the default; ``shape_method="adiabatic"``
        selects instantaneous eigen shapes. Residual thresholds can force early
        corrections or full-diffusion fallback intervals for rapid localized
        motion; ``iqs_predictor_tol`` rejects the current shape interval and
        retries it at half width when the corrector's end power disagrees with
        the amplitude march by more than that relative amount.
        ``adjoint_residual_tol`` can refresh a reused importance shape before
        its configured ``adjoint_every`` maximum age.
        """
        from .quasistatic import quasistatic_coupled_transient

        if self.kinetics is None:
            raise ValueError("no kinetics: call set_kinetics() first")
        if any(callback is not None for callback in (materials_at, state_at, problem_at)):
            problem_at = self._transient_problem(materials_at, state_at, problem_at)
        ctx = self.coupling_context(device=device)
        return quasistatic_coupled_transient(
            ctx, t_end=t_end, dt=dt, shape_dt=shape_dt,
            problem_at=problem_at, **kwargs)


class Model(_CoupledModel):
    """A rectangular reactor defined in centimetres and painted with materials.

    size  : physical extent, ``(Lx,)``, ``(Lx, Ly)`` or ``(Lx, Ly, Lz)`` in cm;
            its length sets the dimensionality. cells : cells per axis.

    Call :meth:`fill` (the background material), then any :meth:`add_box` /
    :meth:`set_boundary`, then :meth:`run`. Methods return ``self`` for chaining.
    """

    def __init__(self, size, cells):
        size = tuple(float(s) for s in np.atleast_1d(size))
        counts = np.atleast_1d(cells)
        if (counts.ndim != 1 or not np.all(np.isfinite(counts)) or
                np.any(counts < 1) or np.any(counts != np.floor(counts)) or
                counts.dtype.kind == "b"):
            raise ValueError("cells must contain positive integers")
        cells = tuple(int(c) for c in counts)
        if len(size) != len(cells):
            raise ValueError("size and cells must have the same length")
        if not 1 <= len(size) <= 3:
            raise ValueError("size must be 1, 2 or 3 numbers (1D/2D/3D)")
        if any(not np.isfinite(s) or s <= 0 for s in size):
            raise ValueError("sizes must be positive and cell counts >= 1")
        self.ndim = len(size)
        self.size = size
        self.cells = cells
        self._shape = tuple(list(cells) + [1] * (3 - self.ndim))
        self._size3 = tuple(list(size) + [1.0] * (3 - self.ndim))
        self._materials: list[Material] = []
        self._map = np.zeros(self._shape, dtype=np.int64)
        self._bc = ["vacuum"] * self.ndim
        self._kinetics = None
        self._thermal_config = None
        self.grid = Grid(shape=self._shape, size=self._size3)
        self._centers = [(np.arange(n) + 0.5) * (L / n)
                         for n, L in zip(self._shape, self._size3)]

    @classmethod
    def from_grid(cls, grid, materials, material_map=None, *, bc="vacuum", kinetics=None):
        """Wrap an existing structured problem without repainting its regions.

        Cylindrical grids return a :class:`CylindricalModel`. On a Cartesian
        grid, trailing single-cell axes are collapsed exactly as in a model
        built directly from a 1-D/2-D ``size``: the model's ``ndim`` excludes
        them and a scalar ``bc`` applies to the real axes only, so collapsed
        axes stay reflective. Naming a non-reflective face on a single-cell
        axis in an explicit per-axis or per-face ``bc`` keeps that axis (a
        physical one-cell slab). The grid's physical thickness is retained.
        Maps and material ordering are copied; the grid is immutable and shared.
        """
        if not isinstance(grid, Grid):
            raise TypeError("from_grid needs a Grid; use TriReactor for triangular grids")
        if grid.geometry == "cylindrical":
            out = CylindricalModel(radius=grid.size[0], height=grid.size[2],
                                   cells=(grid.shape[0], grid.shape[2]))
            faces = normalize_bc(bc)
            out._bc = [faces[0], faces[2]]
        else:
            if cls is CylindricalModel:
                raise ValueError("CylindricalModel.from_grid needs cylindrical geometry")
            faces = normalize_bc(bc)
            ndim = 3
            # A scalar spec means "every real face"; an explicit sequence may
            # deliberately close a single-cell axis, which then stays real.
            explicit = not isinstance(bc, (str, numbers.Real))
            while (ndim > 1 and grid.shape[ndim - 1] == 1 and
                   not (explicit and any(face_alpha(f) != 0.0 for f in faces[ndim - 1]))):
                ndim -= 1
            out = cls(size=grid.size[:ndim], cells=grid.shape[:ndim])
            out._size3 = tuple(float(s) for s in grid.size)
            out._centers = [(np.arange(n) + 0.5) * (L / n)
                            for n, L in zip(out._shape, out._size3)]
            out._bc = list(faces[:ndim])
        out.grid = grid
        out._materials = checked_materials([materials] if isinstance(materials, Material) else materials)
        out._map = checked_map(material_map, grid.shape, len(out._materials))
        if kinetics is not None:
            out.set_kinetics(kinetics)
        return out

    def _wrap_result(self, result, method="diffusion", adjoint=False, *, problem=None):
        return ModelResult(self, result, method, adjoint=adjoint, problem=problem)

    def _index(self, material):
        checked_materials([material], groups=self.n_groups)
        for i, m in enumerate(self._materials):
            if m is material:
                return i
        self._materials.append(material)
        self._thermal_config = None
        return len(self._materials) - 1

    def fill(self, material) -> "Model":
        """Fill the whole core, clearing earlier material assignments."""
        self._materials = checked_materials([material])
        self._thermal_config = None
        self._map[...] = 0
        return self

    def add_box(self, material, x=None, y=None, z=None) -> "Model":
        """Paint an axis-aligned box (ranges in cm; ``None`` spans the whole core)."""
        if not self._materials:
            raise ValueError("call fill() to set the background material first")
        mask = np.ones(self._shape, dtype=bool)
        for ax, rng in enumerate((x, y, z)):
            if rng is None:
                continue
            if ax >= self.ndim:
                raise ValueError(f"cannot set {_AXES[ax]} range on a {self.ndim}D model")
            lo, hi = box_range(rng, _AXES[ax])
            sel = (self._centers[ax] >= lo) & (self._centers[ax] <= hi)
            _warn_painted_extent(_AXES[ax], lo, hi, sel, self._size3[ax])
            shape = [1, 1, 1]
            shape[ax] = self._shape[ax]
            mask &= sel.reshape(shape)
        if not np.any(mask):
            raise ValueError("box selects no cell centres; check its bounds and mesh resolution")
        self._map[mask] = self._index(material)
        return self

    def set_boundary(self, spec=None, *, x=None, y=None, z=None) -> "Model":
        """Boundary conditions by name (``"vacuum"``/``"reflective"``/``"zero-flux"``/
        albedo). ``spec`` sets every real axis; ``x``/``y``/``z`` override one axis
        (a face spec or a ``(lo, hi)`` pair). Collapsed axes stay reflective."""
        boundary = list(self._bc) if spec is None else [spec] * self.ndim
        for ax, val in enumerate((x, y, z)):
            if val is None:
                continue
            if ax >= self.ndim:
                raise ValueError(f"cannot set {_AXES[ax]} boundary on a {self.ndim}D model")
            boundary[ax] = val
        normalize_bc(tuple(boundary + ["reflective"] * (3 - self.ndim)))
        self._bc = boundary
        return self

    @property
    def material_map(self) -> np.ndarray:
        """Per-cell material index, squeezed to the model's dimension."""
        return self._map.reshape(self.cells).copy()

    @property
    def materials(self) -> list:
        """Materials in index order (index 0 is the background fill)."""
        return list(self._materials)

    @property
    def n_groups(self):
        if not self._materials:
            raise ValueError("empty model: call fill() with a material first")
        return self._materials[0].n_groups

    @property
    def kinetics(self):
        return self._kinetics

    def with_materials(self, materials) -> "Model":
        """Copy this model with a full material list or partial name/index mapping.

        ``core.with_materials({"fuel": hot_fuel})`` preserves the original
        geometry, boundary conditions, kinetics and thermal configuration.
        """
        values = replace_materials(self._materials, materials)
        out = copy(self)
        out._materials = values
        out._map = self._map.copy()
        out._bc = list(self._bc)
        out._thermal_config = (None if self._thermal_config is None else
                               dict(self._thermal_config))
        return out

    def _coupling_geometry(self):
        return dict(grid=self.grid, material_map=self._map, bc=self._full_bc(),
                    solver_cls=DiffusionEigenSolver)

    def _state_problem(self, state):
        if not isinstance(state, Model):
            raise TypeError("state_at must return a Model")
        if state.grid != self.grid or state.ndim != self.ndim:
            raise ValueError("state_at must preserve grid dimensions and physical size")
        if normalize_bc(state._full_bc()) != normalize_bc(self._full_bc()):
            raise ValueError("state_at must preserve boundary conditions")
        checked_materials(state.materials, count=len(self.materials),
                          groups=self.n_groups, context="state_at materials")
        return state._materials, state._map

    def _full_bc(self):
        return tuple(self._bc[ax] if ax < self.ndim else "reflective" for ax in range(3))

    def _geometry_line(self):
        dims = " x ".join(f"{s:g}" for s in self.size) + " cm"
        cells = " x ".join(str(c) for c in self.cells)
        return f"{dims},  {cells} cells ({int(np.prod(self.cells)):,})  [{self.ndim}D]"

    def _boundary_line(self):
        parts = []
        for ax in range(self.ndim):
            spec = self._bc[ax]
            parts.append(f"{_AXES[ax]}: {spec[0]}/{spec[1]}" if isinstance(spec, (list, tuple))
                         else f"{_AXES[ax]}: {spec}")
        if self.ndim < 3:
            parts.append("collapsed axes reflective")
        return " | ".join(parts)

    def run(self, method: str = "diffusion", device: str = "auto", adjoint: bool = False,
            tol_k: float = 1e-6, tol_source: float = 1e-5, *, solver_kwargs=None,
            **solve_kw) -> ModelResult:
        """Solve the k-eigenvalue problem and return a :class:`ModelResult`.

        method : ``"diffusion"``, ``"sp3"``, or the simplified double-PN
        approximations ``"sdp1"``/``"sdp2"``/``"sdp3"`` (equal-cost to
        SP3/SP5/SP7, more accurate than SPN in strongly heterogeneous media).
        device : ``"auto"``/``"cpu"``/``"gpu"``.
        adjoint : solve the adjoint (importance) problem instead of the forward one.
        solver_kwargs : constructor settings such as precond_degree or dtype;
                        the remaining keywords are passed to solve().
        """
        if not self._materials:
            raise ValueError("empty model: call fill() with a material first")
        method = method.lower()
        if method not in _STRUCTURED:
            raise ValueError(f"method must be one of {sorted(_STRUCTURED)}, got {method!r}")
        groups = self._materials[0].n_groups
        if any(m.n_groups != groups for m in self._materials):
            raise ValueError("all materials must have the same number of groups")

        grid = self.grid
        material = self._materials[0] if len(self._materials) == 1 else self._materials
        kw = dict(bc=self._full_bc(), device=device)
        if len(self._materials) > 1:
            kw["material_map"] = self._map
        solver = _STRUCTURED[method](grid, material, **solver_options(kw, solver_kwargs))
        res = solver.solve(tol_k=tol_k, tol_source=tol_source, adjoint=adjoint, **solve_kw)
        out = self._wrap_result(res, method, adjoint=adjoint)
        out._external_xs = (solver_kwargs or {}).get("xs_update") is not None
        return out

    steady = run

    def set_kinetics(self, velocities=None, beta=None, decay=None, chi_delayed=None,
                     *, kinetics=None) -> "Model":
        """Attach a Kinetics object or velocity/beta/decay component arrays.

        Both ``set_kinetics(kinetics_object)`` and the original
        ``set_kinetics(velocities=..., beta=..., decay=...)`` are supported.
        ``chi_delayed`` accepts a shared (groups,) spectrum or a
        (families, groups) table, distinct from total ``Material.chi``.
        """
        if isinstance(velocities, Kinetics):
            if kinetics is not None:
                raise ValueError("pass one Kinetics object")
            kinetics, velocities = velocities, None
        if kinetics is not None:
            if any(value is not None for value in (velocities, beta, decay, chi_delayed)):
                raise ValueError("pass a Kinetics object or its component arrays, not both")
        else:
            if velocities is None or beta is None or decay is None:
                raise ValueError("velocities, beta and decay are required")
            kinetics = Kinetics(velocities, beta, decay, chi_delayed)
        if self._materials:
            checked_kinetics(kinetics, self.n_groups, material_count=len(self.materials))
        elif not isinstance(kinetics, Kinetics):
            raise TypeError("kinetics must be a Kinetics object")
        self._kinetics = kinetics
        return self



# ---------------------------------------------------------------------------
# Cylindrical r-z model
# ---------------------------------------------------------------------------
class CylindricalModel(Model):
    """A full body of revolution painted in radial/axial centimetres.

    ``cells=(nr, nz)``; the internal grid/flux shape remains ``(nr, 1, nz)``.
    Powers and inventories use full annular volumes, independent of a dummy
    Cartesian thickness. The symmetry axis at r=0 has zero boundary area.
    """

    def __init__(self, radius, height, cells):
        if len(cells) != 2:
            raise ValueError("cylindrical cells must be (nr, nz)")
        super().__init__(size=(radius, 1.0, height), cells=(cells[0], 1, cells[1]))
        self.radius, self.height = float(radius), float(height)
        self.ndim = 2
        self.size = (self.radius, self.height)
        self.cells = (self._shape[0], self._shape[2])
        self.grid = Grid(self._shape, self._size3, geometry="cylindrical")
        self._bc = [("reflective", "vacuum"), "vacuum"]

    @property
    def radial_centers(self):
        return self.grid.cell_centers(0)

    @property
    def axial_centers(self):
        return self.grid.cell_centers(2)

    def add_box(self, material, *, r=None, z=None) -> "CylindricalModel":
        """Paint an annular axial region using cell-centre r/z ranges in cm."""
        if not self._materials:
            raise ValueError("call fill() first")
        mask = np.ones(self._shape, dtype=bool)
        for axis, name, bounds in ((0, "r", r), (2, "z", z)):
            if bounds is not None:
                lo, hi = box_range(bounds, name)
                shape = [1, 1, 1]
                shape[axis] = self._shape[axis]
                sel = (self._centers[axis] >= lo) & (self._centers[axis] <= hi)
                _warn_painted_extent(name, lo, hi, sel, self._size3[axis])
                mask &= sel.reshape(shape)
        if not np.any(mask):
            raise ValueError("region selects no cell centres; check r/z bounds and resolution")
        self._map[mask] = self._index(material)
        return self

    add_region = add_box

    def set_boundary(self, spec=None, *, r=None, z=None) -> "CylindricalModel":
        """Set the outer radial and axial boundaries; the axis is natural.

        ``r="vacuum"`` sets the cylinder wall. ``z=("reflective", "vacuum")``
        sets the bottom/top faces separately. A radial pair is accepted for
        compatibility with existing grids; its r=0 entry has zero area.
        """
        boundary = list(self._bc)
        for index, value in ((0, r if r is not None else spec),
                             (1, z if z is not None else spec)):
            if value is not None:
                boundary[index] = (("reflective", value)
                                   if index == 0 and np.ndim(value) == 0 else value)
        normalize_bc((boundary[0], "reflective", boundary[1]))
        self._bc = boundary
        return self

    def _full_bc(self):
        return self._bc[0], "reflective", self._bc[1]

    def _geometry_line(self):
        return (f"r=0..{self.radius:g} cm, z=0..{self.height:g} cm, "
                f"{self.cells[0]} x {self.cells[1]} annular cells [r-z]")

    def _boundary_line(self):
        return f"r: {self._bc[0]} | z: {self._bc[1]} | r=0 natural axis"


# ---------------------------------------------------------------------------
# Unstructured mesh model
# ---------------------------------------------------------------------------
class MeshModel:
    """An unstructured-mesh reactor: assign materials to the cells of a Gmsh mesh.

    mesh : a path to a Gmsh ``.msh`` file, or an assembled :class:`~garnet.Mesh`
           (2-D triangles/quads or 3-D tets/hexes/prisms).

    Assign with :meth:`fill` (all cells), then :meth:`assign` (by mesh tag or a
    boolean/callable selector) and :meth:`add_box` (by cell centroid), then
    :meth:`set_boundary` and :meth:`run`. Solves with the matrix-free
    finite-volume diffusion solver on CPU or GPU.
    """

    def __init__(self, mesh):
        self.mesh = read_gmsh(os.fspath(mesh)) if isinstance(mesh, (str, os.PathLike)) else mesh
        self._materials: list[Material] = []
        self._cellmat = np.zeros(self.mesh.n_cells, dtype=np.int64)
        self._bc = "vacuum"
        self._dim = self.mesh.coords.shape[1]

    def _index(self, material):
        checked_materials([material], groups=self._materials[0].n_groups)
        for i, m in enumerate(self._materials):
            if m is material:
                return i
        self._materials.append(material)
        return len(self._materials) - 1

    def fill(self, material) -> "MeshModel":
        """Fill every cell, clearing earlier material assignments."""
        self._materials = checked_materials([material])
        self._cellmat[...] = 0
        return self

    def assign(self, material, tag=None, where=None) -> "MeshModel":
        """Assign ``material`` to a subset of cells.

        tag   : match the mesh's per-cell tag (Gmsh physical id).
        where : a boolean mask over cells, or a callable ``centroid -> bool``
                receiving each cell centroid (an (x, y[, z]) array).
        """
        if not self._materials:
            raise ValueError("call fill() first")
        if tag is not None and where is not None:
            raise ValueError("use tag or where, not both")
        if tag is not None:
            sel = np.asarray(self.mesh.cell_tag) == tag
        elif where is not None:
            sel = (np.array([bool(where(c)) for c in self.mesh.centroid])
                   if callable(where) else np.asarray(where, bool))
        else:
            raise ValueError("assign needs either tag= or where=")
        if sel.shape != (self.mesh.n_cells,):
            raise ValueError(f"where mask must have shape ({self.mesh.n_cells},)")
        if not np.any(sel):
            raise ValueError("assignment selects no cells; check the tag or selector")
        self._cellmat[sel] = self._index(material)
        return self

    def add_box(self, material, x=None, y=None, z=None) -> "MeshModel":
        """Assign ``material`` to cells whose centroid falls in the given box (cm)."""
        if not self._materials:
            raise ValueError("call fill() first")
        c = self.mesh.centroid
        sel = np.ones(self.mesh.n_cells, bool)
        for ax, rng in enumerate((x, y, z)):
            if rng is None:
                continue
            if ax >= self._dim:
                raise ValueError(f"mesh is {self._dim}D; no {_AXES[ax]} axis")
            lo, hi = box_range(rng, _AXES[ax])
            sel &= (c[:, ax] >= lo) & (c[:, ax] <= hi)
        return self.assign(material, where=sel)

    def set_boundary(self, spec) -> "MeshModel":
        """Set the boundary condition on every boundary face (name or albedo)."""
        _albedo(spec)
        self._bc = spec
        return self

    @property
    def materials(self):
        """Materials in index order."""
        return list(self._materials)

    @property
    def material_map(self):
        """Per-cell material indices, in mesh cell order (a copy)."""
        return self._cellmat.copy()

    def with_materials(self, materials) -> "MeshModel":
        """Copy this mesh model with a full list or partial name/index mapping."""
        out = copy(self)
        out._materials = replace_materials(self._materials, materials)
        out._cellmat = self._cellmat.copy()
        return out

    def _geometry_line(self):
        return (f"{self.mesh.n_cells:,} cells, {len(self.mesh.faces):,} interior faces, "
                f"volume {self.mesh.area.sum():.3g} cm^{self._dim}  [{self._dim}D unstructured]")

    def run(self, device: str = "auto", tol_k: float = 1e-6, tol_source: float = 1e-5,
            *, solver_kwargs=None, **solve_kw) -> ReactorResult:
        """Solve on the unstructured mesh and return a :class:`ReactorResult`."""
        if not self._materials:
            raise ValueError("empty model: call fill() first")
        groups = self._materials[0].n_groups
        if any(m.n_groups != groups for m in self._materials):
            raise ValueError("all materials must have the same number of groups")
        solver = UnstructuredDiffusionSolver(self.mesh, self._materials, self._cellmat,
            **solver_options(dict(alpha_boundary=_albedo(self._bc), device=device),
                             solver_kwargs))
        res = solver.solve(tol_k=tol_k, tol_source=tol_source, **solve_kw)
        out = ReactorResult(
            k_eff=res.k_eff, converged=res.converged, outer_iterations=res.outer_iterations,
            inner_iterations=res.inner_iterations, solve_seconds=res.solve_seconds,
            device=res.device, method="diffusion", flux=res.flux, materials=self._materials,
            cell_material=self._cellmat, cell_volume=self.mesh.area,
            geometry_line=self._geometry_line(), boundary_line=f"all faces: {self._bc}")
        out.model = self
        out.raw = res
        return out

    steady = run


# ---------------------------------------------------------------------------
# Reusable triangular-grid reactor model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeedbackSpec:
    """Temperature feedback data for one named reactor material.

    ``reference_temperature`` is the cross-section tabulation temperature in K;
    ``doppler`` is a scalar or one value per energy group in K**-1/2; and
    ``expansion`` is the optional frozen-geometry density coefficient in 1/K.
    The high-level API expands these named specifications into the material-order
    arrays consumed by :class:`~garnet.ThermalFeedback`.
    """

    reference_temperature: float
    doppler: object = 0.0
    expansion: float | None = None


def hex_ring(radius: int) -> list[tuple[int, int]]:
    """Axial ``(R, C)`` coordinates on one hexagonal ring.

    Ring zero contains only ``(0, 0)``; ring ``r`` contains ``6*r`` sites.
    """
    radius = int(radius)
    if radius < 0:
        raise ValueError("hex ring radius must be non-negative")
    if radius == 0:
        return [(0, 0)]
    return [(R, C) for R in range(-radius, radius + 1)
            for C in range(-radius, radius + 1)
            if max(abs(R), abs(C), abs(-R - C)) == radius]


def hex_disk(radius: int) -> list[tuple[int, int]]:
    """Axial coordinates in a complete hexagonal disk through ``radius``."""
    radius = int(radius)
    if radius < 0:
        raise ValueError("hex disk radius must be non-negative")
    return [(R, C) for R in range(-radius, radius + 1)
            for C in range(-radius, radius + 1)
            if max(abs(R), abs(C), abs(-R - C)) <= radius]


class TriReactor(_CoupledModel):
    """A reusable reactor definition on :class:`~garnet.TriGrid`.

    This is the user-facing bridge between geometry construction and solver
    classes. It accepts any already-rasterized triangular/prismatic problem,
    while :meth:`HexLattice.build` creates one without exposing array plumbing.
    The same object can run steady diffusion/transport approximations, neutron
    kinetics, steady thermal coupling, and coupled transients.

    ``total_power`` configured through :meth:`configure_thermal` is the power in
    this modeled domain. For a 2-D slice, supply the slice power (W per modeled
    thickness), not the full 3-D plant rating.
    """

    def __init__(self, grid, materials, material_map, *, active=None,
                 bc="reflective", mask_bc="vacuum", mix_material=None,
                 mix_weight=None, kinetics=None, raster=None,
                 name="tri-grid reactor", geometry_line=None):
        if not isinstance(grid, TriGrid):
            raise TypeError("TriReactor needs a TriGrid")
        self.grid = grid
        self.materials = checked_materials(materials)
        self.material_map = np.array(material_map, dtype=np.int64, copy=True)
        if self.material_map.shape != tuple(grid.shape):
            raise ValueError(f"material_map shape {self.material_map.shape} != "
                             f"grid shape {tuple(grid.shape)}")
        self.active = (np.ones(grid.shape, dtype=bool) if active is None else
                       np.array(active, dtype=bool, copy=True))
        if self.active.shape != tuple(grid.shape):
            raise ValueError("active mask must match the grid shape")
        used = self.material_map
        if used.size and (used.min() < 0 or used.max() >= len(self.materials)):
            raise ValueError("material_map contains an out-of-range material index")
        if (mix_material is None) != (mix_weight is None):
            raise ValueError("mix_material and mix_weight must be supplied together")
        self.mix_material = (None if mix_material is None else
                             np.array(mix_material, dtype=np.int64, copy=True))
        self.mix_weight = (None if mix_weight is None else
                           np.array(mix_weight, dtype=float, copy=True))
        if self.mix_material is not None:
            if (self.mix_material.shape != tuple(grid.shape) or
                    self.mix_weight.shape != tuple(grid.shape)):
                raise ValueError("volume-mix arrays must match the grid shape")
            has_mix = self.mix_material >= 0
            if (np.any(self.mix_weight < 0.0) or np.any(self.mix_weight > 1.0) or
                    np.any(has_mix & (self.mix_material >= len(self.materials)))):
                raise ValueError("invalid volume-mix material index or weight")
        groups = self.materials[0].n_groups
        if any(m.n_groups != groups for m in self.materials):
            raise ValueError("all materials must have the same number of groups")
        self.bc, self.mask_bc = bc, mask_bc
        self.kinetics = (None if kinetics is None else checked_kinetics(
            kinetics, groups, material_count=len(self.materials)))
        self.raster = raster
        self.name = str(name)
        self._geometry = geometry_line or (
            f"{int(self.active.sum()):,} active triangular cells")
        self._thermal_config = None

    @property
    def n_groups(self):
        return self.materials[0].n_groups

    @property
    def shape(self):
        return tuple(self.grid.shape)

    @property
    def is_3d(self):
        return len(self.shape) == 4

    def _boundary_line(self):
        z_spec = (self.bc[2] if self.is_3d and isinstance(self.bc, (tuple, list))
                  and len(self.bc) == 3 else self.bc)
        z = f" | z faces: {z_spec}" if self.is_3d else ""
        return f"core edge: {self.mask_bc}{z}"

    def _geometry_line(self):
        return self._geometry

    def _wrap_result(self, result, method="diffusion", adjoint=False, *, problem=None):
        materials, mmap, mix, weight = (
            (self.materials, self.material_map, self.mix_material, self.mix_weight)
            if problem is None else problem)
        if isinstance(materials, Material):
            materials = [materials]
        if mmap is None:
            mmap = np.zeros(self.grid.shape, dtype=np.int64)
        out = ReactorResult(
            k_eff=result.k_eff, converged=result.converged,
            outer_iterations=result.outer_iterations, inner_iterations=result.inner_iterations,
            solve_seconds=result.solve_seconds, device=result.device, method=method,
            flux=result.flux_numpy, materials=materials, cell_material=mmap,
            cell_volume=self.grid.cell_volume, active=self.active, adjoint=adjoint,
            geometry_line=self._geometry, boundary_line=self._boundary_line(),
            mix_material=mix, mix_weight=weight, grid=self.grid,
            boundary_conditions=(normalize_bc(self.bc), self.mask_bc))
        out.reactor = self
        out.raw = result
        return out

    def _coupling_geometry(self):
        return dict(grid=self.grid, material_map=self.material_map,
                    active=self.active, mask_bc=self.mask_bc, bc=self.bc,
                    mix_material=self.mix_material, mix_weight=self.mix_weight)

    def _state_problem(self, state):
        if not isinstance(state, TriReactor):
            raise TypeError("state_at must return a TriReactor")
        if state.grid != self.grid:
            raise ValueError("state_at must preserve grid dimensions and physical size")
        if (normalize_bc(state.bc) != normalize_bc(self.bc) or
                state.mask_bc != self.mask_bc or not np.array_equal(state.active, self.active)):
            raise ValueError("state_at must preserve boundary conditions and active cells")
        checked_materials(state.materials, count=len(self.materials),
                          groups=self.n_groups, context="state_at materials")
        return state.materials, state.material_map, state.mix_material, state.mix_weight

    def set_kinetics(self, kinetics=None, *, velocities=None, beta=None,
                     decay=None, chi_delayed=None) -> "TriReactor":
        """Attach delayed-neutron data for :meth:`transient` and coupling.

        ``chi_delayed`` accepts a shared (groups,) spectrum or a
        (families, groups) table, distinct from total ``Material.chi``.
        """
        if kinetics is not None and any(x is not None for x in
                                        (velocities, beta, decay, chi_delayed)):
            raise ValueError("pass a Kinetics object or its component arrays, not both")
        if kinetics is None:
            if velocities is None or beta is None or decay is None:
                raise ValueError("velocities, beta and decay are required")
            kinetics = Kinetics(velocities, beta, decay, chi_delayed)
        self.kinetics = checked_kinetics(kinetics, self.n_groups,
                                        material_count=len(self.materials))
        return self

    def with_materials(self, materials, *, name=None) -> "TriReactor":
        """Return this geometry with a replacement material list.

        Accepts a full list in the same order, or a partial mapping such as
        ``{"fuel": hot_fuel}``. This is the direct hand-off for
        ``SphResult.corrected_materials`` or a new cross-section library while
        preserving the validated mesh, controls, kinetics, and thermal setup.
        """
        materials = replace_materials(self.materials, materials)
        out = TriReactor(
            self.grid, materials, self.material_map, active=self.active,
            bc=self.bc, mask_bc=self.mask_bc,
            mix_material=self.mix_material, mix_weight=self.mix_weight,
            kinetics=self.kinetics, raster=self.raster,
            name=self.name if name is None else name,
            geometry_line=self._geometry)
        if self._thermal_config is not None:
            out._thermal_config = dict(self._thermal_config)
        return out

    def steady(self, method="diffusion", device="auto", adjoint=False,
               tol_k=1e-6, tol_source=1e-5, solver_kwargs=None,
               **solve_kwargs) -> ReactorResult:
        """Solve a steady k-eigenvalue problem on this reusable geometry."""
        method = method.lower()
        if method not in _TRI:
            raise ValueError(f"method must be one of {sorted(_TRI)}, got {method!r}")
        init = dict(active=self.active, mask_bc=self.mask_bc, bc=self.bc,
                    mix_material=self.mix_material, mix_weight=self.mix_weight,
                    device=device)
        init = solver_options(init, solver_kwargs)
        solver = _TRI[method](self.grid, self.materials, self.material_map, **init)
        res = solver.solve(tol_k=tol_k, tol_source=tol_source,
                           adjoint=adjoint, **solve_kwargs)
        out = self._wrap_result(res, method, adjoint=adjoint)
        out._external_xs = (solver_kwargs or {}).get("xs_update") is not None
        return out

    run = steady



def _disk_polygon_area(poly, radius):
    """Exact area of a polygon intersected with the origin-centred disk.

    Sums the signed areas of disk & triangle(O, p, q) over the polygon edges;
    each edge is split where it crosses the circle, inner pieces contribute a
    triangle and outer pieces a circular sector. Valid for any simple polygon.
    """
    r2 = radius * radius
    total = 0.0
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        e = q - p
        a, b, c = e @ e, p @ e, p @ p - r2
        ts = [0.0]
        disc = b * b - a * c
        if a > 0.0 and disc > 0.0:
            root = math.sqrt(disc)
            ts += [t for t in ((-b - root) / a, (-b + root) / a) if 0.0 < t < 1.0]
        ts.append(1.0)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            u, v = p + t0 * e, p + t1 * e
            cross = u[0] * v[1] - u[1] * v[0]
            mid = p + 0.5 * (t0 + t1) * e
            if mid @ mid <= r2:
                total += 0.5 * cross
            else:
                total += 0.5 * r2 * math.atan2(cross, u @ v)
    return abs(total)


def _clip_half_plane(poly, normal):
    """Clip a convex polygon to ``normal . x >= 0`` (Sutherland-Hodgman)."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        sp, sq = normal @ p, normal @ q
        if sp >= 0.0:
            out.append(p)
        if (sp >= 0.0) != (sq >= 0.0):
            out.append(p + (sp / (sp - sq)) * (q - p))
    return out


def _sector_polygon_area(poly, radius, azimuth, half_angle):
    """Exact area of a convex polygon inside the origin-centred circular sector
    of ``radius`` spanning ``azimuth +- half_angle`` (half_angle in [0, pi])."""
    if half_angle > 0.5 * math.pi:
        # A reflex wedge is the disk minus the opposite convex wedge.
        return (_disk_polygon_area(poly, radius) -
                _sector_polygon_area(poly, radius, azimuth + math.pi,
                                     math.pi - half_angle))
    if half_angle <= 0.0:
        return 0.0
    lo, hi = azimuth - half_angle, azimuth + half_angle
    # Left of the lower ray and right of the upper ray: an intersection of two
    # half-planes is exactly the wedge while its opening is at most pi.
    for normal in (np.array([-math.sin(lo), math.cos(lo)]),
                   np.array([math.sin(hi), -math.cos(hi)])):
        poly = _clip_half_plane(poly, normal)
        if len(poly) < 3:
            return 0.0
    return _disk_polygon_area(poly, radius)


def annular_sector_fraction(vertices, centre, inner, outer, azimuth, half_angle):
    """Exact fraction of a triangle's area inside an annular sector.

    The sector is ``inner < r <= outer`` about ``centre`` with azimuth within
    ``azimuth +- half_angle`` (radians). The intersection is computed in closed
    form (polygon clipped to the wedge, then intersected with each circle), so
    the drum absorber volume is preserved independently of mesh resolution.
    """
    V = np.asarray(vertices, dtype=float) - np.asarray(centre, dtype=float)
    area = 0.5 * abs((V[1, 0] - V[0, 0]) * (V[2, 1] - V[0, 1]) -
                     (V[2, 0] - V[0, 0]) * (V[1, 1] - V[0, 1]))
    if area <= 0.0:
        return 0.0
    poly = list(V)
    covered = _sector_polygon_area(poly, outer, azimuth, half_angle)
    if inner > 0.0:
        covered -= _sector_polygon_area(poly, inner, azimuth, half_angle)
    # Round-off on a cell that only touches the sector is not an absorber blend.
    return 0.0 if covered <= 1e-12 * area else min(covered / area, 1.0)


def _drum_absorber_mix(raster, pitch, drums, samples=None):
    """Per-cell volume fraction of each drum's absorber arc (mix_material/weight).

    ``drums`` is a list of dicts with ``rc``, ``inner``, ``outer``, ``arc_half``
    (radians), ``arc_az`` (radians) and ``absorber_id``. For every triangle near a
    drum the fraction of its area inside that drum's annular sector -- radius in
    (inner, outer], azimuth within +-arc_half of the (rotated) arc centre -- is
    computed exactly (:func:`annular_sector_fraction`), so the mixed absorber
    area equals the analytic sector area at any refinement. The fraction is
    non-zero even where the thin arc only partly crosses a cell, and varies
    smoothly as the drum rotates, so it drives the volume-mix (mix a fraction of
    the absorber into the drum-body cell) rather than a staircase. ``samples``
    is accepted for compatibility and ignored (it set the resolution of the
    former, biased, barycentric-lattice estimate). Returns (mix_material,
    mix_weight) shaped like the material map (sentinel mix_material = -1 where
    there is no absorber).
    """
    mmap = raster.material_map
    ni, nj, _ = mmap.shape
    weight = np.zeros((ni, nj, 2))
    mix = np.full((ni, nj, 2), -1, dtype=np.int64)
    centres = np.array([hex_site_xy(rc[0], rc[1], pitch) for d in drums for rc in (d["rc"],)])
    outers = np.array([d["outer"] for d in drums])
    for a in range(ni):
        for b in range(nj):
            for t in (0, 1):
                if mmap[a, b, t] == 0:
                    continue
                V = raster.cell_vertices(a, b, t)
                cx, cy = V.mean(0)
                d2 = (centres[:, 0] - cx) ** 2 + (centres[:, 1] - cy) ** 2
                d = int(d2.argmin())
                reach = outers[d] + raster.side
                if d2[d] > reach * reach:
                    continue
                dm = drums[d]
                f = annular_sector_fraction(V, centres[d], dm["inner"], dm["outer"],
                                            dm["arc_az"], dm["arc_half"])
                if f > 0.0:
                    weight[a, b, t] = f
                    mix[a, b, t] = dm["absorber_id"]
    return mix, weight


# ---------------------------------------------------------------------------
# Triangular hex-lattice model
# ---------------------------------------------------------------------------
class HexLattice:
    """A hexagonal lattice of assemblies on the body-fitted triangular solver.

    pitch  : centre-to-centre hex spacing, cm. refine : triangles per hex is
             ``6 * refine**2`` (>= 2 for a usable mesh).

    Place assemblies with :meth:`set_site` (axial coordinates ``(R, C)``) or, for a
    control drum, :meth:`set_drum` (a hex with a rotatable absorber arc), set the
    lattice-edge boundary, then :meth:`run` with diffusion or SP3 transport.
    """

    def __init__(self, pitch: float, refine: int = 4):
        if pitch <= 0 or refine < 1:
            raise ValueError("pitch must be positive and refine >= 1")
        self.pitch = float(pitch)
        self.refine = int(refine)
        self._sites: dict = {}
        self._drums: dict = {}
        self._bc = "vacuum"
        self._height = None
        self._nz = None
        self._z_bc = "vacuum"
        self._axial_regions = []
        self._kinetics = None

    def set_site(self, rc, material) -> "HexLattice":
        """Place ``material`` at hex site ``rc = (R, C)`` (axial coordinates)."""
        rc = tuple(rc)
        if len(rc) != 2 or any(int(v) != v for v in rc):
            raise ValueError("a hex site must be two integer axial coordinates")
        self._sites[tuple(map(int, rc))] = material
        return self

    def set_sites(self, sites, material) -> "HexLattice":
        """Place one material at every coordinate in ``sites``."""
        for rc in sites:
            self.set_site(rc, material)
        return self

    def set_ring(self, radius, material) -> "HexLattice":
        """Fill one complete hex ring; shorthand for ``set_sites(hex_ring(...))``."""
        return self.set_sites(hex_ring(radius), material)

    def set_disk(self, radius, material) -> "HexLattice":
        """Fill a complete disk through ``radius`` (later calls may overwrite)."""
        return self.set_sites(hex_disk(radius), material)

    def set_drum(self, rc, body, absorber, inner_radius, outer_radius, arc_deg,
                 angle_deg=0.0) -> "HexLattice":
        """Place a control drum at hex site ``rc``: a ``body`` assembly carrying a
        rotatable ``absorber`` arc, volume-mixed by area fraction.

        The absorber occupies the annular sector between ``inner_radius`` and
        ``outer_radius`` (cm from the hex centre) spanning ``arc_deg`` degrees.
        ``angle_deg`` rotates it about the hex centre, measured from the inward
        radial direction: 0 points at the core centre (inserted), 180 away from
        the core (withdrawn). The "core centre" is the lattice origin, the
        centre of hex site ``(0, 0)``, whether or not a site is placed there
        and wherever the occupied sites lie: centre the core on ``(0, 0)``
        (or account for the offset in ``angle_deg``) when building a lattice
        that is not. Any number of drums may be placed, each rotated
        independently -- the drum-worth curve is swept by rerunning with different
        ``angle_deg``.
        """
        rc = tuple(rc)
        inner_radius, outer_radius, arc_deg = (float(inner_radius),
                                                float(outer_radius),
                                                float(arc_deg))
        if len(rc) != 2 or any(int(v) != v for v in rc):
            raise ValueError("a drum site must be two integer axial coordinates")
        rc = tuple(map(int, rc))
        if inner_radius < 0.0 or outer_radius <= inner_radius:
            raise ValueError("drum radii need 0 <= inner_radius < outer_radius")
        if not 0.0 < arc_deg <= 360.0:
            raise ValueError("drum arc_deg must lie in (0, 360]")
        self._sites[rc] = body
        self._drums[rc] = dict(absorber=absorber, inner=inner_radius,
                               outer=outer_radius,
                               arc_half=math.radians(arc_deg) / 2.0,
                               angle_deg=float(angle_deg))
        return self

    def set_drum_angle(self, rc, angle_deg) -> "HexLattice":
        """Rotate an already-defined control drum without redefining its arc."""
        rc = tuple(rc)
        if rc not in self._drums:
            raise KeyError(f"no control drum at site {rc}")
        self._drums[rc]["angle_deg"] = float(angle_deg)
        return self

    def extrude(self, height, nz, *, boundary="vacuum") -> "HexLattice":
        """Extrude the radial lattice into ``nz`` uniform triangular prisms.

        ``boundary`` applies to the lower/upper axial faces. Radial leakage is
        still controlled by :meth:`set_boundary`.
        """
        height, nz = float(height), int(nz)
        if height <= 0.0 or nz < 1:
            raise ValueError("extrusion height must be positive and nz >= 1")
        self._height, self._nz, self._z_bc = height, nz, boundary
        return self

    def add_axial_region(self, material, z, *, replace=None) -> "HexLattice":
        """Paint a material over an axial interval after extrusion.

        ``replace`` optionally supplies one material object or an iterable to
        replace. Omit it to paint every active assembly in the interval. This is
        the convenient way to add top/bottom reflectors while leaving radial
        reflectors or full-height control drums unchanged.
        """
        if self._height is None:
            raise ValueError("call extrude() before adding axial regions")
        lo, hi = map(float, z)
        if not 0.0 <= lo < hi <= self._height:
            raise ValueError("axial region must lie inside 0 <= z <= height")
        if replace is None or isinstance(replace, Material):
            replace = None if replace is None else (replace,)
        else:
            replace = tuple(replace)
        self._axial_regions.append((material, (lo, hi), replace))
        return self

    def set_kinetics(self, kinetics=None, **components) -> "HexLattice":
        """Attach kinetics that :meth:`build` copies to the TriReactor."""
        if kinetics is None:
            kinetics = Kinetics(**components)
        elif components:
            raise ValueError("pass a Kinetics object or component arrays, not both")
        if not isinstance(kinetics, Kinetics):
            raise TypeError("kinetics must be a Kinetics object")
        self._kinetics = kinetics
        return self

    def set_boundary(self, spec) -> "HexLattice":
        """Set the boundary condition on the lattice edge (name or albedo)."""
        self._bc = spec
        return self

    def build(self, samples: int = 8, name="hex-lattice reactor") -> TriReactor:
        """Rasterize once and return a reusable :class:`TriReactor`.

        The returned model exposes its grid/maps for inspection and supports
        steady, transient, thermal, and coupled calculations. Control-drum
        absorber area fractions are computed exactly; ``samples`` is accepted
        for compatibility and ignored.
        """
        if not self._sites:
            raise ValueError("place at least one assembly with set_site()")
        uniq, id_of = [], {}

        def register(mat):
            if id(mat) not in id_of:
                id_of[id(mat)] = len(uniq) + 1              # 1-based; 0 is void
                uniq.append(mat)
            return id_of[id(mat)]

        for mat in self._sites.values():
            register(mat)
        for dm in self._drums.values():
            register(dm["absorber"])
        for mat, _z, _replace in self._axial_regions:
            register(mat)
        groups = uniq[0].n_groups
        if any(m.n_groups != groups for m in uniq):
            raise ValueError("all assemblies must have the same number of groups")
        void = Material(name="void", diffusion=[1.0] * groups, sigma_a=[0.0] * groups,
                        nu_sigma_f=[0.0] * groups)
        materials = [void] + uniq

        site_material = {rc: id_of[id(mat)] for rc, mat in self._sites.items()}
        raster = rasterize_hex_sites(site_material, self.pitch, self.refine)
        mix_material = mix_weight = None
        if self._drums:
            specs = []
            for rc, dm in self._drums.items():
                x, y = hex_site_xy(rc[0], rc[1], self.pitch)
                specs.append(dict(rc=rc, inner=dm["inner"], outer=dm["outer"],
                                  arc_half=dm["arc_half"],
                                  arc_az=math.atan2(y, x) + math.radians(180.0 + dm["angle_deg"]),
                                  absorber_id=id_of[id(dm["absorber"])]))
            mix_material, mix_weight = _drum_absorber_mix(raster, self.pitch, specs, samples)
        mmap = raster.material_map
        if self._height is None:
            grid = TriGrid(shape=mmap.shape, side=raster.side)
            active = mmap > 0
            bc = "reflective"
            dimension = "2-D triangular"
        else:
            nz = self._nz
            mmap = np.repeat(mmap[..., None], nz, axis=3)
            mix_material = (None if mix_material is None else
                            np.repeat(mix_material[..., None], nz, axis=3))
            mix_weight = (None if mix_weight is None else
                          np.repeat(mix_weight[..., None], nz, axis=3))
            zc = (np.arange(nz) + 0.5) * self._height / nz
            for mat, (lo, hi), replace in self._axial_regions:
                zsel = (zc >= lo) & (zc <= hi if hi == self._height else zc < hi)
                if replace is None:
                    radial = mmap > 0
                else:
                    missing = [m.name for m in replace if id(m) not in id_of]
                    if missing:
                        raise ValueError("axial replace materials are not present "
                                         f"in the radial lattice: {missing}")
                    replace_ids = [id_of[id(m)] for m in replace]
                    radial = np.isin(mmap, replace_ids)
                painted = radial & zsel[None, None, None, :]
                mmap[painted] = id_of[id(mat)]
                if mix_weight is not None:
                    # A repainted cell is wholly the new material: a drum's
                    # absorber blend must not survive under an axial reflector.
                    mix_material[painted] = -1
                    mix_weight[painted] = 0.0
            grid = TriGrid(shape=mmap.shape, side=raster.side,
                           height=self._height)
            active = mmap > 0
            bc = ("reflective", "reflective", self._z_bc)
            dimension = f"3-D triangular prisms, {nz} axial layers"

        ntri = int(active.sum())
        drum_note = f", {len(self._drums)} drums" if self._drums else ""
        geo = (f"{len(self._sites)} hex sites{drum_note}, pitch {self.pitch:g} cm, "
               f"{6 * self.refine ** 2} triangles/hex, {ntri:,} active cells "
               f"[{dimension}]")
        return TriReactor(
            grid, materials, mmap, active=active, bc=bc, mask_bc=self._bc,
            mix_material=mix_material, mix_weight=mix_weight,
            kinetics=self._kinetics, raster=raster, name=name,
            geometry_line=geo)

    def run(self, method: str = "diffusion", device: str = "auto", adjoint: bool = False,
            tol_k: float = 1e-6, tol_source: float = 1e-5, samples: int = 8,
            **solve_kw) -> ReactorResult:
        """Backward-compatible shorthand for ``build(samples).steady(...)``."""
        return self.build(samples=samples).steady(
            method=method, device=device, adjoint=adjoint,
            tol_k=tol_k, tol_source=tol_source, **solve_kw)
