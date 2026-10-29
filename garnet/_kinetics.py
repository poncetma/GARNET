"""Delayed-emission terms shared by diffusion time-integration stages."""


class DelayedEmission:
    """Project precursor-family sources onto neutron energy groups.

    Material ``chi`` is the total equilibrium fission spectrum. A supplied
    delayed spectrum may be shared by all families or specified per family;
    otherwise each material's total spectrum is used for delayed emission too.
    """

    def __init__(self, kinetics, groups):
        self.spectra = kinetics.chi_delayed
        self.groups = groups
        if self.spectra is not None and self.spectra.shape[-1] != groups:
            raise ValueError("kinetics.chi_delayed must have one entry per group")

    def project(self, family_sources, material_spectra):
        """Return ``sum_i chi_delayed[i, g] * family_sources[i]`` per group."""
        if self.spectra is not None and self.spectra.ndim == 2:
            sources = []
            for group in range(self.groups):
                source = 0.0
                for family, family_source in enumerate(family_sources):
                    weight = self.spectra[family, group]
                    if weight != 0.0:
                        source = source + weight * family_source
                sources.append(source)
            return sources
        total_source = family_sources[0]
        for source in family_sources[1:]:
            total_source = total_source + source
        spectra = material_spectra if self.spectra is None else self.spectra
        if self.spectra is not None:
            # Host spectrum values are scalar coefficients, so they must not
            # promote FP32 device fields through NumPy scalar dtype rules.
            return [float(spectra[group]) * total_source
                    for group in range(self.groups)]
        return [spectra[group] * total_source for group in range(self.groups)]

    def spectrum(self, family, group, material_spectra):
        """One family's emission weight, including the material fallback."""
        if self.spectra is None:
            return material_spectra[group]
        if self.spectra.ndim == 1:
            return self.spectra[group]
        return self.spectra[family, group]

    def fission_weights(self, removed_fractions, material_spectra):
        """Emission weights after eliminating the new precursor inventory.

        ``chi_total - sum_i chi_delayed[i] * removed_fractions[i]`` includes
        prompt production and the implicit new-time delayed contribution.
        """
        if self.spectra is None:
            total_removed = sum(removed_fractions)
            return [spectrum * (1.0 - total_removed)
                    for spectrum in material_spectra]
        delayed = self.project(removed_fractions, material_spectra)
        return [spectrum - removed
                for spectrum, removed in zip(material_spectra, delayed)]
