'''
Created on 08/24/2016

@author: ariel

Provides functions to correct spectra for galactic extinction
'''

from .wcs import get_galactic_coordinates_rad

import numpy as np
from os import path
from astropy import log

__all__ = ['extinction_corr', 'calc_extincion', 'get_EBV']


def get_EBV_map(file_name):
    '''

    Reads E(B-V) HEALPix map from Planck's dust map using healpy, if the map
    file is not found, it will be downloaded from:
    http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits


    '''
    import urllib.request, urllib.parse, urllib.error
    import healpy as hp

    if not path.exists(file_name):
        log.info(
            'Downloading dust map (1.5GB), this is probably a good time to check XKCD.')
        url = 'http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits'
        log.debug('Map: %s' % url)
        urllib.request.urlretrieve(url, file_name)

    log.info('Reading E(B-V) map from ' + file_name)
    EBV_map = hp.read_map(file_name, field=2)

    return EBV_map


def get_EBV(wcs, file_name):
    import healpy as hp

    l, b = get_galactic_coordinates_rad(wcs)
    EBV_map = get_EBV_map(file_name)
    # Get the corresponting HEALPix index and the E(B-V) value:
    index = hp.ang2pix(nside=2048, theta=(np.pi / 2) - b, phi=l)
    return EBV_map[index]


def CCM(wave, Rv=3.1):
    '''

    Calculates the Cardelli, Clayton & Mathis (CCM) extinction curve in the
    optical, wavelengths should be in the 3030-9090 Angstrons range.

    Input:   Wavelengths, Rv (Optional, default is 3.1)
    Returns: A_lambda/Av

    Reference: http://adsabs.harvard.edu/abs/1989ApJ...345..245C

    '''
    # Turn lambda from angstrons to microns:
    wave = wave / 10000.

    x = 1. / wave
    y = (x - 1.82)

    a = 1. + (0.17699 * y) - (0.50447 * (y ** 2)) - (0.02427 * (y ** 3))
    a += (0.72085 * (y ** 4)) + (0.01979 * (y ** 5)) - (0.77530 * (y ** 6))
    a += (0.32999 * (y ** 7))

    b = (1.41338 * y) + (2.28305 * (y ** 2)) + (1.07233 * (y ** 3))
    b += -(5.38434 * (y ** 4)) - (0.62251 * (y ** 5)) + (5.30260 * (y ** 6))
    b += -(2.09002 * (y ** 7))

    return a + (b / Rv)


def calc_extinction(wave, EBV, Rv=3.1):
    '''

    Gets the galactic extinction in a given wavelenght through a given line of
    sight.

    Input:   wavelenght, header with WCS, E(B-V), Rv (Optional, default is 3.1)
    Returns: A_lambda, E(B-V)

    '''
    Av = Rv * EBV
    A_lambda = Av * CCM(wave, Rv)

    return A_lambda


def extinction_corr(wave, EBV):
    '''

    Corrects spectra for the effects of galactic extinction.

    Input: Wavelenghts, Fluxes, RA, Dec, E(B-V) map
    Returns: Fluxes corrected for the effects of Milky Way dust.

    '''
    A_lambda = calc_extinction(wave, EBV)
    tau_lambda = A_lambda / (2.5 * np.log10(np.exp(1.)))
    return np.exp(tau_lambda)
