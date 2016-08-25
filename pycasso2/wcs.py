'''
Created on 26/06/2015

@author: andre
'''

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord as sc
import numpy as np

__all__ = ['get_axis_coordinates', 'get_wavelength_coordinates', 'copy_WCS', 'update_WCS',
           'get_cube_limits', 'get_shape', 'get_reference_pixel',
           'get_pixel_area', 'get_pixel_area_srad',
           'get_pixel_scale', 'get_pixel_scale_rad']


rad_per_deg = np.pi / 180.0
one_angstrom = 1e-10


def get_wavelength_coordinates(header):
    w = wcs.WCS(header, naxis=[3])
    pix_coords = np.arange(header['NAXIS3'])
    wave_coords = w.wcs_pix2world(pix_coords[:, np.newaxis], 0)
    if w.wcs.cunit == 'm':
        wave_coords /= one_angstrom
    return np.squeeze(wave_coords)


def get_celestial_coordinates(header, relative=True):
    # FIXME: Not sure how to make the coordinates relative.
    w = wcs.WCS(header, naxis=2)
    if relative:
        w.wcs.crval = (180.0, 0.0)
    xx_pix = np.arange(header['NAXIS1']) + 1
    yy_pix = np.arange(header['NAXIS2']) + 1
    x0_pix, y0_pix = np.rint(w.wcs.crpix).astype('int')
    xx_world, _ = w.wcs_pix2world(xx_pix, np.zeros_like(xx_pix) + y0_pix, 1)
    _, yy_world = w.wcs_pix2world(np.zeros_like(yy_pix) + x0_pix, yy_pix, 1)
    if relative:
        xx_world -= 180.0
    xx_world *= 3600.0
    yy_world *= 3600.0
    return xx_world, yy_world


def get_galactic_coordinates(ra, dec):
    '''
    
    Input: RA, Dec in degrees (J2000)
    Returns: Galactic coordinates l and b in radians to compare to HEALPix 
    maps.
    
    Note: l is consistent with HEALPix's phi, while HEALPix's theta will be 
    given by theta = pi/2 - b.
    
    '''
    
    coords    = sc(ra,dec,unit='deg',frame='fk5',equinox='j200')
    galcoords = coords.transform_to('galactic')

    l, b =  galcoords.l.radian, galcoords.b.radian
    
    return l, b


def get_reference_pixel(header):
    w = wcs.WCS(header)
    crpix = np.rint(w.wcs.crpix).astype('int') - 1
    return (crpix[2], crpix[1], crpix[0])


def get_cube_limits(cube, ext):
    header = fits.getheader(cube, ext)
    l_obs = get_wavelength_coordinates(header)
    Ny = header['NAXIS2']
    Nx = header['NAXIS1']
    return l_obs.min(), l_obs.max(), Ny, Nx 


def get_shape(header):
    Nx = header['NAXIS1']
    Ny = header['NAXIS2']
    Nl = header['NAXIS3']
    return (Nl, Ny, Nx)


def get_wavelength_sampling(header):
    w = wcs.WCS(header, naxis=[3])
    s = wcs.utils.proj_plane_pixel_scales(w)
    if w.wcs.cunit == 'm':
        s /= one_angstrom
    return np.asscalar(s)

    
def get_pixel_area(header):
    w = wcs.WCS(header, naxis=2)
    a = wcs.utils.proj_plane_pixel_area(w)
    return a 

    
def get_pixel_area_srad(header):
    a = get_pixel_area(header)
    return a * (rad_per_deg * rad_per_deg)

    
def get_pixel_scale(header):
    w = wcs.WCS(header, naxis=2)
    s = wcs.utils.proj_plane_pixel_scales(w)
    return s.mean()

    
def get_pixel_scale_rad(header):
    s = get_pixel_scale(header)
    return s * rad_per_deg

    
def copy_WCS(header, dest_header, axes):
    if np.isscalar(axes):
        axes = [axes]
    
    w = wcs.WCS(header, naxis=axes)
    dest_header.extend(w.to_header(), update=True)


def update_WCS(header, crpix, crval_wave, cdelt_wave):
    w = wcs.WCS(header)
    if w.wcs.cunit == 'm':
        crval_wave *= one_angstrom
        cdelt_wave *= one_angstrom
    crpix = np.array([crpix[2], crpix[1], crpix[0]]) + 1
    w.wcs.crpix = crpix
    crval_orig = w.wcs.crval
    w.wcs.crval = crval_orig[0], crval_orig[1], crval_wave
    if w.wcs.has_cd():
        w.wcs.cd[2, 2] = cdelt_wave
    else:
        w.wcs.cdelt[2] = cdelt_wave
    header.extend(w.to_header(), update=True)


def find_nearest_index(array, value):
    '''
    Return the array index that is closest to the valued provided. Note that
    this is intended for use with coordinates array.
    '''
    from numpy import abs
    idx = (abs(array-value)).argmin()
    return idx


