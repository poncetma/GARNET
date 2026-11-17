"""Per-cell material property lookup with two-material volume mixing.

Extracted from :class:`garnet.solver.Fields` so that every physics built on the
same mesh -- neutron cross sections, thermal conductivity, heat-pipe sink
strength -- expands its per-material tables through *identical* rules. That
matters at the mixed cells: a control-drum arc that covers 30% of a triangle
must be 30% B4C to the neutronics and 30% B4C to the conduction solve, or the
two physics disagree about where the absorber is and the coupling quietly
solves a different problem than either code alone.

The rules themselves (why linear for reaction cross sections, harmonic for the
diffusion coefficient, fission-weighted for the emission spectrum) are
documented at each method.
"""

from __future__ import annotations

import numpy as np


class MaterialBlend:
    """Expands per-material tables onto the grid, honouring a volume blend.

    material_map : integer map into the materials list, shape ``shape``. None
        is allowed only for a single material with no mixing.
    mix_material / mix_weight : the per-cell second material and its volume
        fraction w, so a mixed cell is ``(1 - w) * base + w * mix``. The
        sentinel ``mix_material < 0`` means "no blend here"; those cells stay
        bit-identical to the pure-index lookup.
    """

    def __init__(self, xp, shape, material_map, n_materials, dtype=np.float64,
                 mix_material=None, mix_weight=None):
        self.xp = xp
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.mix = mix_material is not None
        self.material_map = material_map

        if material_map is None:
            if n_materials > 1:
                raise ValueError("material_map is required with multiple materials")
            if self.mix:
                raise ValueError("mixing requires an explicit material_map")
            self.mmap = None
            return

        mmap = xp.asarray(np.asarray(material_map))
        if mmap.shape != self.shape:
            raise ValueError(f"material_map shape {mmap.shape} != grid shape "
                             f"{self.shape}")
        if int(mmap.min()) < 0 or int(mmap.max()) >= n_materials:
            raise ValueError("material_map indexes outside the materials list")
        self.mmap = mmap

        if self.mix:
            mm2 = xp.asarray(np.asarray(mix_material))
            w = xp.asarray(np.asarray(mix_weight), dtype=self.dtype)
            if mm2.shape != self.shape or w.shape != self.shape:
                raise ValueError("mix_material/mix_weight shape must match grid")
            if int(mm2.max()) >= n_materials:
                raise ValueError("mix_material indexes outside the materials list")
            self.active_mix = mm2 >= 0
            self.mm2c = xp.where(self.active_mix, mm2, 0)
            self.w = w

    def _blend(self, table, combine):
        xp = self.xp
        if self.mmap is None:
            return xp.full(self.shape, float(table[0]), dtype=self.dtype)
        dev = xp.asarray(table, dtype=self.dtype)
        base = dev[self.mmap]
        if self.mix:
            base = xp.where(self.active_mix,
                            combine(base, dev[self.mm2c], self.w), base)
        return base

    def linear(self, table):
        """Volume-linear blend -- exact reaction-rate averaging under a flat
        flux. The rule for cross sections, and for volumetric quantities like a
        heat-pipe sink coefficient or a power density."""
        return self._blend(table, lambda b, o, wt: (1.0 - wt) * b + wt * o)

    def harmonic(self, table):
        """Harmonic blend -- the rule for a diffusion coefficient (its
        transport cross section 1/(3D) volume-averages, so a trace of a strong
        absorber correctly chokes the cell).

        NOT the rule for thermal conductivity: :class:`~garnet.thermal.
        ConductionSolver` blends k *arithmetically* within a cell (two
        materials sharing a cell conduct in parallel) and leaves the series
        (harmonic) mean to the operator's face coefficients."""
        return self._blend(table, lambda b, o, wt: 1.0 / ((1.0 - wt) / b + wt / o))

    def components(self):
        """The (at most two) materials sharing each cell, with their fractions.

        A list of ``(index, fraction)`` per-cell fields: the base material at
        ``1 - w`` and, where a blend is active, the partner at ``w`` (the
        partner entry has fraction 0 elsewhere). :meth:`linear` is exactly
        ``sum(fraction * table[index])`` over them. For a per-material LAW
        rather than a per-material number -- a temperature feedback, whose
        change depends on each material's own coefficient and reference
        temperature -- evaluate the law per component and blend the results,
        instead of blending the coefficients and evaluating once.
        """
        xp = self.xp
        ones = xp.ones(self.shape, dtype=self.dtype)
        if self.mmap is None:
            return [(xp.zeros(self.shape, dtype=np.int64), ones)]
        if not self.mix:
            return [(self.mmap, ones)]
        w = xp.where(self.active_mix, self.w, 0.0)
        return [(self.mmap, 1.0 - w), (self.mm2c, w)]

    def fission_weighted(self, table, production):
        """Blend by share of the cell's fission production, not by volume.

        For a quantity that rides on the fission source (the emission spectrum
        chi, the delayed fraction beta). Blending chi linearly would leave a
        fissile material mixed with a non-fissile one emitting a spectrum that
        sums to w -- silently losing (1 - w) of the cell's fission neutrons.
        ``production`` is the per-material total nu*Sigma_f (a flat-flux proxy;
        exact whenever at most one component is fissile, e.g. a fuel pin
        blended with moderator). Cells where neither component is fissile keep
        the base value: it multiplies a zero source.
        """
        xp = self.xp
        if not self.mix:
            return self.linear(table)
        pdev = xp.asarray(production, dtype=self.dtype)
        wb = (1.0 - self.w) * pdev[self.mmap]
        wo = self.w * pdev[self.mm2c]
        den = wb + wo
        blendable = self.active_mix & (den > 0)
        safe_den = xp.where(den > 0, den, 1.0)
        dev = xp.asarray(table, dtype=self.dtype)
        base = dev[self.mmap]
        merged = (wb * base + wo * dev[self.mm2c]) / safe_den
        return xp.where(blendable, merged, base)

    def field(self, values, harmonic=False):
        """Accept either a ready per-cell array or a per-material table.

        The convenience every physics wants: material properties are naturally
        written per material, but a caller with a computed field (a measured
        conductivity map, say) should be able to hand it over directly.
        """
        arr = np.asarray(values)
        if arr.shape == self.shape:
            return self.xp.asarray(arr, dtype=self.dtype)
        if arr.ndim != 1:
            raise ValueError(
                f"expected a per-material table (1-D) or a per-cell field with "
                f"shape {self.shape}, got shape {arr.shape}")
        return self.harmonic(arr) if harmonic else self.linear(arr)


def blend_materials(materials, fractions, name=None):
    """One Material equal to a volume blend of several, by the rules of
    :class:`MaterialBlend`: reaction cross sections (absorption, production,
    kappa*Sigma_f, total, scattering and its Legendre moments) linear in the
    volume fractions, the diffusion coefficient harmonic (its transport cross
    section 1/(3D) is linear), the emission spectrum weighted by each constituent's total
    nu*Sigma_f. This is flat-flux homogenization of a cell's constituents; for
    any number of them it is exact in the reaction rates a flat cell flux
    implies, which the two-material ``mix_material``/``mix_weight`` blend is
    only for two."""
    from .materials import Material

    mats = list(materials)
    f = np.asarray(fractions, dtype=float)
    if f.shape != (len(mats),) or np.any(f < 0.0) or not np.isclose(f.sum(), 1.0):
        raise ValueError("fractions must be one non-negative value per material, "
                         f"summing to 1; got {f}")
    keep = [i for i in range(len(mats)) if f[i] > 0.0]
    if len(keep) == 1:
        return mats[keep[0]]
    mats, f = [mats[i] for i in keep], f[keep]

    def lin(attr):
        return sum(w * np.asarray(getattr(m, attr), dtype=float) for w, m in zip(f, mats))

    prod = np.array([w * float(np.sum(m.nu_sigma_f)) for w, m in zip(f, mats)])
    chi = (sum(p * np.asarray(m.chi, dtype=float) for p, m in zip(prod, mats)) / prod.sum()
           if prod.sum() > 0.0 else np.asarray(mats[0].chi))
    kappa = None
    if any(m.kappa_fission is not None for m in mats):
        kappa = sum(w * (np.zeros_like(np.asarray(m.sigma_a, dtype=float))
                         if m.kappa_fission is None else np.asarray(m.kappa_fission))
                    for w, m in zip(f, mats))
    diffusion = 1.0 / sum(w / np.asarray(m.diffusion, dtype=float)
                          for w, m in zip(f, mats))
    # Legendre scattering moments blend linearly like every reaction rate; a
    # constituent without them scatters isotropically (moments l >= 1 zero).
    order = max(m.scattering_order for m in mats)
    moments = None
    if order > 0:
        G = mats[0].n_groups
        moments = np.zeros((order + 1, G, G))
        for w, m in zip(f, mats):
            own = (np.asarray(m.sigma_s_moments) if m.sigma_s_moments is not None
                   else np.asarray(m.sigma_s)[None])
            moments[:own.shape[0]] += w * own
    label = name or "+".join(f"{w:.4f}*{m.name}" for w, m in zip(f, mats))
    return Material(name=label, diffusion=diffusion, sigma_a=lin("sigma_a"),
                    nu_sigma_f=lin("nu_sigma_f"), sigma_s=lin("sigma_s"), chi=chi,
                    total=lin("sigma_t"), kappa_fission=kappa,
                    sigma_s_moments=moments)


def blend_material(base, partner, weight, name=None):
    """Two-material :func:`blend_materials`: ``base`` at ``1 - weight`` and
    ``partner`` at ``weight``, reproducing a ``mix_material``/``mix_weight`` cell
    of the raster solvers exactly. For solvers that take one material per cell
    (:class:`garnet.mesh.UnstructuredDiffusionSolver`)."""
    w = float(weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"weight must lie in [0, 1], got {weight}")
    if w == 0.0 or w == 1.0:
        return blend_materials([base, partner], [1.0 - w, w], name=name)
    return blend_materials([base, partner], [1.0 - w, w],
                           name=name or f"{base.name}+{w:.4f}*{partner.name}")


def per_cell_blended_materials(materials, cell_material, mix_material, mix_weight):
    """Materials and per-cell indices with every volume-mixed cell given its own
    blended material (:func:`blend_material`), for one-material-per-cell
    solvers. Unmixed cells keep their index; the returned list is ``materials``
    followed by one blend per mixed cell."""
    cell_material = np.asarray(cell_material).reshape(-1)
    mm = np.asarray(mix_material).reshape(-1)
    w = np.asarray(mix_weight, dtype=float).reshape(-1)
    out = list(materials)
    cells = np.array(cell_material, copy=True)
    for c in np.flatnonzero((mm >= 0) & (w > 0.0)):
        cells[c] = len(out)
        out.append(blend_material(materials[cell_material[c]], materials[mm[c]], w[c]))
    return out, cells
