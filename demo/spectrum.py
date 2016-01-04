"""
Note: this code and the accompanying tests are only meant to illustrate
how to use a testing framework to write unit and functional tests.
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

class Spectrum(object):
    """
    Represents a spectrum with wavelength as the dispersion dimension.

    Parameters
    ----------
    wavelength : quantity_like
    flux : quantity_like
    """
    def __init__(self, wavelength, flux):

        self.wavelength = u.Quantity(wavelength)
        self.flux = u.Quantity(flux)

        if np.any(self.wavelength < 0.):
            raise ValueError("Wavelength values < 0 are not valid.")

        if self.wavelength.shape != self.flux.shape:
            raise ValueError("Shape of wavelength array must match shape"
                             " of flux array.")

    @classmethod
    def read_ascii(Cls, filename):
        """
        Read a spectrum object from an ASCII file containing two columns:
        wavelength and flux.

        Parameters
        ----------
        filename : str
            The full path to the file containing the spectrum data.
        """
        data = np.loadtxt(filename, dtype=float)
        return Cls(data[:,0], data[:,1])

    def integrate(self, wavelength_grid):
        """
        Integrate the spectrum flux over the specified grid of wavelengths.

        Parameters
        ----------
        wavelength_grid : quantity_like

        Returns
        -------
        integrated_flux : :class:`~astropy.units.Quantity`
        """
        grid = u.Quantity(wavelength_grid)
        grid = grid.to(self.wavelength.unit)

        interpolator = interp1d(self.wavelength.value, self.flux.value,
                                kind='cubic')
        new_flux = interpolator(grid.value)

        return simps(new_flux, x=grid.value) * self.flux.unit * grid.unit







