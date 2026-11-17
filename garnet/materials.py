"""Multigroup macroscopic cross-section sets."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np


@dataclass
class Material:
    """Homogenized multigroup diffusion constants for one material region.

    All cross sections are macroscopic, in 1/cm; diffusion coefficients in cm.

    diffusion   : D_g, shape (G,)
    sigma_a     : absorption Sigma_a,g, shape (G,)
    nu_sigma_f  : production nu*Sigma_f,g, shape (G,)
    sigma_s     : scattering matrix Sigma_s[g_from, g_to], shape (G, G).
                  The diagonal (within-group scattering) is ignored: it cancels
                  identically between the removal term and the in-scatter source.
    chi         : total equilibrium fission spectrum (prompt plus delayed),
                  shape (G,); defaults to all fission neutrons born in group 0.
                  Must sum to 1 if fissile. Kinetics.chi_delayed may specify
                  separate delayed emission spectra for transient calculations.
    total       : optional total cross section Sigma_t,g, shape (G,). Used by
                  the SP3 solver for the higher angular moments; if omitted,
                  1/(3 D_g) is used (exact for transport-corrected data with
                  isotropic scattering).
    sigma_s_moments : optional Legendre moments of the scattering matrix,
                  shape (L+1, G, G), [l, g_from, g_to], moments WITHOUT the
                  (2l+1) factor (Sigma_s1 = mean-cosine-weighted scattering).
                  Moment 0 is the P0 matrix. With moments present, ``total``
                  must be the true total cross section: the anisotropy lives in
                  the moments, read by an anisotropic S_N solve. Isotropic
                  solvers need transport-corrected data instead
                  (:func:`transport_corrected`).
    """

    diffusion: np.ndarray
    sigma_a: np.ndarray
    nu_sigma_f: np.ndarray
    sigma_s: np.ndarray | None = None
    chi: np.ndarray | None = None
    total: np.ndarray | None = None
    # kappa*Sigma_f: energy release per unit flux, shape (G,). POWER is
    # proportional to this, NOT to nu_sigma_f -- nu varies by group, so
    # nu-weighting tilts a power distribution by the local spectrum. Optional:
    # solvers only need nu_sigma_f; power edits use this when present.
    kappa_fission: np.ndarray | None = None
    name: str = ""
    sigma_s_moments: np.ndarray | None = None

    def __post_init__(self):
        self.diffusion = np.atleast_1d(np.asarray(self.diffusion, dtype=np.float64))
        if self.diffusion.ndim != 1 or not self.diffusion.size:
            raise ValueError("diffusion must be a non-empty one-dimensional group array")
        G = self.n_groups
        self.sigma_a = np.asarray(self.sigma_a, dtype=np.float64).reshape(G)
        self.nu_sigma_f = np.asarray(self.nu_sigma_f, dtype=np.float64).reshape(G)
        if self.sigma_s is None:
            self.sigma_s = np.zeros((G, G))
        self.sigma_s = np.asarray(self.sigma_s, dtype=np.float64).reshape(G, G)
        if self.chi is None:
            self.chi = np.zeros(G)
            self.chi[0] = 1.0
        self.chi = np.asarray(self.chi, dtype=np.float64).reshape(G)
        for key in ("diffusion", "sigma_a", "nu_sigma_f", "sigma_s", "chi"):
            if not np.all(np.isfinite(getattr(self, key))):
                raise ValueError(f"{key} must contain finite values")
        if np.any(self.diffusion <= 0):
            raise ValueError("diffusion coefficients must be positive")
        if np.any(self.nu_sigma_f < 0) or np.any(self.sigma_s < 0):
            raise ValueError("cross sections must be non-negative")
        if np.any(self.sigma_a < 0) and np.any(self.removal <= 0):
            # A net absorption (see net_absorption) is negative where the
            # (n,2n) gain exceeds capture; the removal must still be positive.
            raise ValueError("cross sections must be non-negative; a negative (net) "
                             "absorption is accepted only while the removal "
                             "Sigma_a + out-scatter stays positive in every group")
        if np.any(self.chi < 0):
            raise ValueError("chi must be non-negative")
        if self.is_fissile and not np.isclose(self.chi.sum(), 1.0):
            raise ValueError(f"chi must sum to 1 for fissile material, got {self.chi.sum()}")
        if self.total is not None:
            self.total = np.asarray(self.total, dtype=np.float64).reshape(G)
            if not np.all(np.isfinite(self.total)) or np.any(self.total <= 0):
                raise ValueError("total cross sections must be finite and positive")
        if self.kappa_fission is not None:
            self.kappa_fission = np.asarray(self.kappa_fission,
                                            dtype=np.float64).reshape(G)
            if not np.all(np.isfinite(self.kappa_fission)) or np.any(self.kappa_fission < 0):
                raise ValueError("kappa_fission must be finite and non-negative")
        if self.sigma_s_moments is not None:
            m = np.asarray(self.sigma_s_moments, dtype=np.float64)
            if m.ndim != 3 or m.shape[1:] != (G, G) or m.shape[0] < 1:
                raise ValueError(f"sigma_s_moments must have shape (L+1, {G}, {G}), "
                                 f"got {m.shape}")
            if not np.all(np.isfinite(m)):
                raise ValueError("sigma_s_moments must be finite")
            self.sigma_s_moments = m

    @property
    def scattering_order(self) -> int:
        """Highest Legendre scattering moment carried (0 if isotropic)."""
        return 0 if self.sigma_s_moments is None else self.sigma_s_moments.shape[0] - 1

    def replace(self, **changes) -> "Material":
        """Return an independent, validated copy with selected fields changed.

        For example, ``fuel.replace(sigma_a=fuel.sigma_a * 1.01)`` preserves
        scattering, spectra and fission-energy data as well as the name.
        Array data are copied, so editing the new material leaves this one alone.
        """
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values.update(changes)
        values = {key: value.copy() if isinstance(value, np.ndarray) else value
                  for key, value in values.items()}
        return replace(self, **values)

    @property
    def sigma_t(self) -> np.ndarray:
        """Sigma_t,g, falling back to 1/(3 D_g) when not provided."""
        return self.total if self.total is not None else 1.0 / (3.0 * self.diffusion)

    @property
    def n_groups(self) -> int:
        return len(self.diffusion)

    @property
    def is_fissile(self) -> bool:
        return bool(np.any(self.nu_sigma_f > 0))

    @property
    def removal(self) -> np.ndarray:
        """Sigma_r,g = Sigma_a,g + sum_{g'!=g} Sigma_s,g->g' (out-scatter), shape (G,)."""
        out_scatter = self.sigma_s.sum(axis=1) - np.diag(self.sigma_s)
        return self.sigma_a + out_scatter


@dataclass
class Kinetics:
    """Kinetics data for transient calculations.

    velocities  : neutron speeds v_g in cm/s, shape (G,) -- or (M, G) with one
                  row per entry of the problem's materials list, for
                  material-dependent speeds (e.g. C5G7-TD Table 13).
    beta        : delayed neutron fractions per precursor family, shape (I,)
                  -- or (M, I) per material (non-fissile rows are zeros).
    decay       : decay constants lambda_i in 1/s, shape (I,). Always global:
                  the precursor families are nuclear data shared by every
                  material (a per-material lambda would make the families
                  incompatible between regions).
    chi_delayed : delayed emission spectrum: shape (G,) for a single spectrum,
                  (I, G) for one spectrum per precursor family, or None to
                  default to the total fission spectrum of each material.

    ``Material.chi`` is the total equilibrium (prompt plus delayed) spectrum.
    A distinct delayed spectrum implies the prompt production weight
    ``chi - sum_i beta_i * chi_delayed[i]``; delayed emission is added from
    each family's evolving precursor inventory during the transient.

    Per-material tables (2D velocities/beta) index the materials list passed
    to the transient solver's problem_at, whose order must therefore be stable
    in time. Solvers without per-material support reject 2D tables.
    """

    velocities: np.ndarray
    beta: np.ndarray
    decay: np.ndarray
    chi_delayed: np.ndarray | None = None

    def __post_init__(self):
        self.velocities = np.atleast_1d(np.asarray(self.velocities, dtype=np.float64))
        self.beta = np.atleast_1d(np.asarray(self.beta, dtype=np.float64))
        self.decay = np.atleast_1d(np.asarray(self.decay, dtype=np.float64))
        for key in ("velocities", "beta", "decay"):
            if not np.all(np.isfinite(getattr(self, key))):
                raise ValueError(f"{key} must contain finite values")
        if self.velocities.ndim > 2 or self.beta.ndim > 2 or self.decay.ndim != 1:
            raise ValueError("velocities/beta must be 1D or 2D (per-material), decay 1D")
        if self.beta.shape[-1] != self.decay.shape[0]:
            raise ValueError("beta and decay must have one value per precursor family")
        if np.any(self.velocities <= 0) or np.any(self.decay <= 0) or np.any(self.beta < 0):
            raise ValueError("velocities and decay constants must be positive, beta non-negative")
        if self.chi_delayed is not None:
            self.chi_delayed = np.asarray(self.chi_delayed, dtype=np.float64)
            if (self.chi_delayed.ndim not in (1, 2) or
                    not np.all(np.isfinite(self.chi_delayed)) or
                    np.any(self.chi_delayed < 0) or
                    np.any(self.chi_delayed.sum(axis=-1) <= 0)):
                raise ValueError("chi_delayed must contain finite, non-negative spectra with positive sums")
            if self.chi_delayed.ndim == 2 and len(self.chi_delayed) != self.n_families:
                raise ValueError("2D chi_delayed must have one spectrum per precursor family")
            self.chi_delayed = (self.chi_delayed
                                / self.chi_delayed.sum(axis=-1, keepdims=True))

    @property
    def n_families(self) -> int:
        return self.beta.shape[-1]

    @property
    def per_material(self) -> bool:
        """True when velocities or beta carry a leading material axis."""
        return self.velocities.ndim == 2 or self.beta.ndim == 2

    @property
    def beta_total(self) -> float:
        if self.beta.ndim == 2:
            raise ValueError("beta_total is undefined for per-material beta; "
                             "the solver builds a spatial field instead")
        return float(self.beta.sum())


# Classic homogenized two-group PWR-like constants (thermal reactor, no upscatter).
PWR_TWO_GROUP = Material(
    name="PWR two-group",
    diffusion=[1.2627, 0.3543],
    sigma_a=[0.01207, 0.1210],
    nu_sigma_f=[0.008476, 0.18514],
    sigma_s=[[0.0, 0.02619], [0.0, 0.0]],
    chi=[1.0, 0.0],
)

# Simple one-group set with k_inf ~ 1.17 (critical bare cube side ~ 87 cm).
ONE_GROUP_DEMO = Material(
    name="one-group demo",
    diffusion=[1.3],
    sigma_a=[0.030],
    nu_sigma_f=[0.035],
)


def net_absorption(material: Material) -> Material:
    """Copy whose absorption closes the balance on the TRUE total cross section:
    Sigma_a,g = Sigma_t,g - sum_g' Sigma_s0[g -> g'].

    GARNET's solvers remove Sigma_a + sum_{g' != g} Sigma_s[g -> g'] from each
    group (``Material.removal``; the within-group term cancels), while a
    transport code given the total and the P0 matrix removes
    Sigma_t - Sigma_s[g -> g]. The two agree only if
    Sigma_a + sum_g' Sigma_s0 = Sigma_t. A library whose scattering matrix
    counts (n,2n) with its multiplicity while the total counts the reaction
    once (Griffin ISOXML, Serpent's scattering production matrix) exceeds that
    by Sigma_(n,2n), and with the library absorption every GARNET solver loses
    the (n,2n) gain. The net absorption restores it, and is negative where the
    gain exceeds capture: Be in the fastest group of the HP-MR library,
    -0.0086 /cm against +0.0060 /cm. ``Material`` accepts that while the
    removal stays positive.

    Only for data whose ``sigma_t`` is the true total, not ``1/(3D)`` of
    transport-corrected diffusion data. It commutes with
    :func:`transport_corrected`, which lowers the total and the within-group
    scattering by the same amount.
    """
    return material.replace(sigma_a=material.sigma_t - material.sigma_s.sum(axis=1))


def transport_corrected(material: Material, method: str = "outscatter") -> Material:
    """Transport-corrected copy of a material with P1 scattering moments.

    Isotropic-scattering solvers (diffusion, SPN, isotropic S_N) need the
    anisotropy of scattering folded into the total cross section. With the P1
    moment Sigma_s1[g -> g'] (no (2l+1) factor) the corrected total is

        Sigma_tr,g = Sigma_t,g - c_g,   D_g = 1 / (3 Sigma_tr,g),

    and the within-group P0 scattering is reduced by the same c_g, which keeps
    the balance Sigma_tr = Sigma_a + sum_g' Sigma_s0~(g -> g'):

    * ``"outscatter"`` (the out-scatter approximation): c_g = sum_g'
      Sigma_s1[g -> g'], the P1 moment of everything scattered out of g;
    * ``"diagonal"``: c_g = Sigma_s1[g -> g], the within-group P1 term only,
      the usual transport approximation for isotropic S_N.

    The out-scatter correction can exceed the within-group P0 scattering
    (hydrogenous moderators in a few groups); the stored diagonal is then
    clipped at zero with a warning. No solver reads that diagonal: diffusion
    ignores it, and the triangular S_N solver takes its within-group
    scattering as total minus removal, which it keeps negative for data with
    an explicit total such as these, so the balance holds.

    The result carries no ``sigma_s_moments``: it is data for isotropic
    solvers and must not be combined with an anisotropic scattering solve.
    """
    import warnings

    if method not in ("outscatter", "diagonal"):
        raise ValueError(f"method must be 'outscatter' or 'diagonal', got {method!r}")
    if material.scattering_order < 1:
        raise ValueError(f"material {material.name!r} carries no P1 scattering "
                         "moment to transport-correct with")
    s1 = material.sigma_s_moments[1]
    corr = s1.sum(axis=1) if method == "outscatter" else np.diag(s1).copy()
    total = material.sigma_t - corr
    if np.any(total <= 0.0):
        raise ValueError(f"transport correction of {material.name!r} leaves a "
                         "non-positive total cross section")
    sigma_s = np.array(material.sigma_s, dtype=float, copy=True)
    diag = np.diag(sigma_s) - corr
    if np.any(diag < 0.0):
        warnings.warn(f"{method} transport correction of {material.name!r} makes "
                      f"the within-group scattering negative in groups "
                      f"{(np.flatnonzero(diag < 0.0) + 1).tolist()}; clipped at 0",
                      RuntimeWarning, stacklevel=2)
    sigma_s[np.diag_indices_from(sigma_s)] = np.maximum(diag, 0.0)
    return material.replace(diffusion=1.0 / (3.0 * total), total=total,
                            sigma_s=sigma_s, sigma_s_moments=None,
                            name=f"{material.name}-tc-{method}")
