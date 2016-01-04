__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import tempfile

# Third-party
import astropy.units as u
import numpy as np
import pytest

from .spectrum import Spectrum

def test_creation():
    wvln = np.linspace(1000., 4000., 1024)
    flux = np.random.uniform(0., 1., wvln.size)

    # try with list, array, Quantity input
    s = Spectrum(list(wvln), list(flux))
    s = Spectrum(wvln, flux)
    s = Spectrum(wvln*u.angstrom, flux*u.erg/u.cm**2/u.angstrom)

    # ------------------------------------------------------------------------
    # Check that creation fails as expected if:

    # 1) shapes don't match
    with pytest.raises(ValueError):
        s = Spectrum(wvln[:-1], flux)
    with pytest.raises(ValueError):
        s = Spectrum(wvln, flux[:-1])

    # 2) object can't be coerced to a Quantity
    with pytest.raises(TypeError):
        s = Spectrum(wvln, None)
    with pytest.raises(TypeError):
        s = Spectrum(None, flux)
    with pytest.raises(TypeError):
        s = Spectrum(None, None)

    # 3) wavelength goes negative
    wvln2 = wvln.copy()
    wvln2[:-10] *= -1.
    with pytest.raises(ValueError):
        s = Spectrum(wvln2, flux)

def test_creation_from_file():
    wvln = np.linspace(1000., 4000., 1024)
    flux = np.random.uniform(0., 1., wvln.size)

    # make a temporary file with valid data and try reading
    with tempfile.NamedTemporaryFile() as f:
        np.savetxt(f, np.vstack((wvln,flux)).T)
        f.seek(0)
        s = Spectrum.read_ascii(f)

    # make sure this fails if the datatype is invalid (e.g., string)
    wvln2 = np.array(["spam"] * wvln.size)

    with tempfile.NamedTemporaryFile() as f:
        np.savetxt(f, np.vstack((wvln2,flux)).T, fmt="%s %s") # both become strings
        f.seek(0)

        with pytest.raises(ValueError):
            s = Spectrum.read_ascii(f)

def test_integrate():
    subslice = slice(100,200)
    wvln = np.linspace(1000., 4000., 1024)

    flux = np.zeros_like(wvln)
    flux[subslice] = 1./np.ptp(wvln[subslice]) # so the integral is 1

    s = Spectrum(wvln*u.angstrom, flux*u.erg/u.cm**2/u.angstrom)

    # the integration grid is a sub-section of the full wavelength array
    wvln_grid = s.wavelength[subslice]
    i_flux = s.integrate(wvln_grid)
    assert np.allclose(i_flux.value, 1.) # "close" because this is float comparison

