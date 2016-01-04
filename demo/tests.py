from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

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

