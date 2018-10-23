'''
Created on 23 de out de 2018

@author: andre
'''

from .resampling.core import resample_cube
import numpy as np

__all__ = ['synphot']

def synphot(l_obs, f_obs, l_filter, r_filter, redshift=0.0, mode='mag'):
    '''
    Calculate the synthetic photometry of a bandpass over a spectrum.

    Parameters
    ----------
    l_obs : array
        Wavelength array.

    f_obs : array
        [Masked] array with observed fluxes, in erg / s / cm2 / AA.

    l_filter : array
        Wavelength array of filter response.

    r_filter : array
        Filter response.

    redshift : float, optional
        If the spectrum is in rest frame, apply this redshift
        before extracting the photometry. Default: 0.0

    mode : string, optional
        One of:
            * ``'mag'``: Compute the AB magnitude of the flux (default).
            * ``'flux'``: Compute the flux in the bandpass.
                            Not available for ``exact=True``.

    Returns
    -------
    out : array or scalar.
        If f_obs is 1-d, the result is a scalar. For 2-d and 3-d, the
        result is an array. The argument ``mode`` controls the values are
        magnitudes or fluxes.
    '''
    lr_filter = l_filter * r_filter
    if f_obs.ndim == 2:
        lr_filter = lr_filter[:, np.newaxis]
    elif f_obs.ndim == 3:
        lr_filter = lr_filter[:, np.newaxis, np.newaxis]
    elif f_obs.ndim > 3:
        raise Exception('Unsupported number of dimensions in f_obs: %d' % f_obs.ndim)
    one_plus_z = 1.0 + redshift
    l_obs = l_obs * one_plus_z
    flux_filter = resample_cube(l_obs, l_filter, f_obs, interpolate=True)
    flux_filter /= one_plus_z
    A = np.trapz(lr_filter * flux_filter, l_filter, axis=0)
    B = np.trapz(r_filter / l_filter, l_filter)
    if mode == 'flux':
        l_central = np.sqrt(np.trapz(lr_filter.squeeze(), l_filter) / B)
        return (A / B) / l_central**2
    elif mode == 'mag':
        return -2.5 * np.log10(A / B) - 2.41
    else:
        raise Exception('Unsupported mode: %s' % mode)
