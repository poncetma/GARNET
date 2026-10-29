"""Implicit multigroup endpoint solve, separate from time-step acceptance."""

from dataclasses import dataclass

from .linalg import FGMRESWorkspace, fgmres
from .multigroup import (AdjointCoarseCorrectionPreconditioner,
                         EnergyGroupGaussSeidelPreconditioner,
                         MultigroupStepOperator)


def map_kinetic_fields(fields, kinetics, step_width, dtype):
    """Map family fractions and inverse speeds through the material geometry."""
    scalar = dtype.type
    if kinetics.beta.ndim == 2:
        # Delayed fractions follow fission production in a mixed cell.
        beta = [fields.map_table_fission_weighted(kinetics.beta[:, family])
                for family in range(kinetics.n_families)]
    else:
        beta = [scalar(value) for value in kinetics.beta]
    if kinetics.velocities.ndim == 2:
        inverse_velocity_dt = [fields.map_table(
            1.0 / (kinetics.velocities[:, group] * step_width))
            for group in range(kinetics.velocities.shape[-1])]
    else:
        inverse_velocity_dt = [scalar(1.0 / (velocity * step_width))
                               for velocity in kinetics.velocities]
    return beta, inverse_velocity_dt


def time_weights(scheme, step, inverse_velocity_dt, *, transformed):
    """Diagonal time shift and the matching carried-history source weight."""
    diagonal = [scheme.a0(step) * inverse for inverse in inverse_velocity_dt]
    history = ([scheme.history_factor(step) * inverse
                for inverse in inverse_velocity_dt] if transformed else diagonal)
    return diagonal, history


@dataclass
class EndpointSolution:
    flux: object
    outer_iterations: int
    inner_iterations: int
    workspace: FGMRESWorkspace

    @property
    def workspace_bytes(self):
        return sum(array.nbytes for array in (
            self.workspace.x, self.workspace.r, self.workspace.w,
            self.workspace.V, self.workspace.Z))


class MonolithicDiffusionStep:
    """Loss, scattering and implicit fission for one set of step coefficients.

    Precursor elimination makes the endpoint equation linear. Accepted flux
    and precursor history enter only through the fixed RHS; the block and
    group preconditioner can be reused until coefficients change.
    """

    def __init__(self, operators, fields, emission_weights, k0, *, rhs_weight,
                 group_batch, options, precond_degree):
        self.options = options
        self.block = MultigroupStepOperator(
            operators, fields.sigma_s, fields.nu_sigma_f, emission_weights, k0,
            rhs_weight=rhs_weight, group_batch=group_batch)
        self.preconditioner = EnergyGroupGaussSeidelPreconditioner(
            self.block,
            scatter_sweeps=int(options.get("scatter_sweeps", 3)),
            include_fission=False,
            inner_rtol=float(options.get("inner_rtol", 1e-3)),
            inner_atol=float(options.get("inner_atol", 0.0)),
            inner_maxiter=int(options.get("inner_maxiter", 300)),
            anderson_depth=int(options.get("energy_anderson", 0)),
            inner_fixed_relaxations=int(options.get("inner_fixed_relaxations", 0)),
            precond_degree=int(options.get("precond_degree", precond_degree)))

    def solve(self, flux, carried_flux, history_weights, delayed_sources, *,
              tolerance, workspace=None, coarse_adjoint=None):
        options = self.options
        if workspace is None:
            workspace = FGMRESWorkspace.like(
                flux, restart=int(options.get("restart", 30)), operator_out=True)
        rhs = self.block.fixed_rhs(history_weights, carried_flux, delayed_sources)
        self.preconditioner.reset_stats()
        preconditioner = self.preconditioner
        if coarse_adjoint is not None:
            # Recalibrate the amplitude correction to the latest forward shape.
            preconditioner = AdjointCoarseCorrectionPreconditioner(
                self.block, preconditioner, flux, coarse_adjoint)
        solved, iterations = fgmres(
            self.block.apply, rhs, flux, self.block.inv_diag, self.block.xp,
            precond=preconditioner,
            rtol=float(options.get("rtol", max(0.1 * tolerance, 1e-12))),
            atol=float(options.get("atol", 0.0)),
            restart=int(options.get("restart", 30)),
            maxiter=int(options.get("maxiter", 300)), workspace=workspace)
        return EndpointSolution(solved, iterations,
                                self.preconditioner.stats.inner_iterations,
                                workspace)
