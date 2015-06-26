'''
Created on 26/06/2015

@author: andre
'''

from astropy.io import fits
import numpy as np

__all__ = ['get_axis_coordinates', 'set_axis_WCS', 'copy_WCS', 'get_cube_limits', 'get_shape']


def get_axis_coordinates(header, ax, dtype='float64'):
    naxes = header['NAXIS']
    if ax < 0 or ax > naxes:
        raise Exception('Axis %d not in range (1, %d)' % (ax, naxes))
    crpix, crval, cdelt, naxis = get_axis_WCS(header, ax)
    c_ini = crval - crpix * cdelt
    c_fin = c_ini + cdelt * (naxis - 1)
    return np.linspace(c_ini, c_fin, naxis, dtype=dtype)


def set_axis_WCS(header, ax, crpix=None, crval=None, cdelt=None, naxis=None):
    naxes = header['NAXIS']
    if ax < 0 or ax > naxes:
        raise Exception('Axis %d not in range (1, %d)' % (ax, naxes))
    if crpix is not None:
        header['CRPIX%d' % ax] = crpix + 1
    if crval is not None:
        header['CRVAL%d' % ax] = crval
    if cdelt is not None:
        header['CDELT%d' % ax] = cdelt
    if naxis is not None:
        header['NAXIS%d' % ax] = naxis


def get_axis_WCS(header, ax):
    naxes = header['NAXIS']
    if ax < 0 or ax > naxes:
        raise Exception('Axis %d not in range (1, %d)' % (ax, naxes))
    crpix = float(header['CRPIX%d' % ax]) - 1
    crval = header['CRVAL%d' % ax]
    cdelt = header['CDELT%d' % ax]
    N = header['NAXIS%d' % ax]
    return crpix, crval, cdelt, N 


def get_reference_pixel(header):
    crpix_x = float(header['CRPIX1']) - 1
    crpix_y = float(header['CRPIX2']) - 1
    crpix_l = float(header['CRPIX3']) - 1
    return (crpix_l, crpix_y, crpix_x)


def get_cube_limits(cube, ext):
    header = fits.getheader(cube, ext)
    l_obs = get_axis_coordinates(header, ax=3)
    yy = get_axis_coordinates(header, ax=2)
    xx = get_axis_coordinates(header, ax=1)
    return l_obs.min(), l_obs.max(), len(yy), len(xx) 


def get_shape(header):
    Nx = header['NAXIS1']
    Ny = header['NAXIS2']
    Nl = header['NAXIS3']
    return (Nl, Ny, Nx)

    
def copy_WCS(orig_header, dest_header, axes):
    if np.isscalar(axes):
        axes = [axes]
    for ax in axes:
        crpix, crval, cdelt, naxis = get_axis_WCS(orig_header, ax)
        set_axis_WCS(dest_header, ax, crpix, crval, cdelt, naxis)

