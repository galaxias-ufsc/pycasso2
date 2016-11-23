'''
Created on 09/12/2015

@author: andre
'''

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np

__all__ = ['redshift2lum_distance', 'velocity2redshift']

# Planck Collab 2015, Paper XIII
H0 = 67.7  # km / s / Mpc
omega0 = 0.307

c = 299792.458  # km / s


def redshift2lum_distance(z, simple=False):
    if simple:
        return z * c / H0

    cosmo = FlatLambdaCDM(H0, omega0)
    dl = cosmo.luminosity_distance(z) / u.Mpc
    return float(dl)


def velocity2redshift(v):
    return v / c


def wavelength_apply_redshift(wl, z, dest='rest'):
    '''
    Apply redshift correction to wavelength from to rest or observed frames.

    Parameters
    ----------
    l : array
        Wavelength array.

    z : array or float.
        Redshift.

    dest : string, optional
        Destination frame. Either ``'rest'`` or ``'observed'``.
        Default: ``'rest'``.

    Returns
    -------
    l_red : array
        Redshifted wavelength array. If ``z`` is an array,
        ``l_red`` will have the shape ``l.shape + z.shape``.
    '''
    if dest == 'rest':
        op = lambda x, y: x / y
    elif dest == 'observed':
        op = lambda x, y: x * y

    if np.isscalar(z):
        return op(wl, 1. + z)
    else:
        return op(wl[:, np.newaxis], 1. + z[np.newaxis, :])


def spectra2restframe(l_obs, f_obs, z, kcor=1.0):
    l_rest = wavelength_apply_redshift(l_obs, z, dest='rest')
    f_rest = f_obs * (1.0 + z)**kcor
    return l_rest, f_rest
