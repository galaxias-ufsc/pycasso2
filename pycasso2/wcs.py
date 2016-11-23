'''
Created on 26/06/2015

@author: andre
'''

from astropy.io import fits
from astropy import log, wcs
import numpy as np

__all__ = ['get_axis_coordinates', 'get_wavelength_coordinates', 'copy_WCS', 'update_WCS',
           'get_cube_limits', 'get_shape', 'get_reference_pixel', 'get_galactic_coordinates_rad',
           'get_pixel_area', 'get_pixel_area_srad', 'get_pixel_scale', 'get_pixel_scale_rad',
           'get_Nx', 'get_Ny', 'get_Nwave']


rad_per_deg = np.pi / 180.0
one_angstrom = 1e-10


def get_celestial(w):
    wc = w.celestial
    if wc.wcs.naxis != 2:
        log.warn('No celestial axes found, using axes 1,2.')
        wc = w.sub(2)
    return wc


def proj_plane_pixel_area(w):
    '''
    More tolerant version of `astropy.wcs.utils.proj_plane_pixel_area`.
    '''
    w = get_celestial(w)
    psm = w.pixel_scale_matrix
    a = np.abs(np.linalg.det(psm))
    x_unit, y_unit = w.wcs.cunit
    if x_unit == '' or y_unit == '':
        a /= 3600**2
    return a


def get_wavelength_coordinates(w, Nwave):
    w = w.sub([3])
    pix_coords = np.arange(Nwave)
    wave_coords = w.wcs_pix2world(pix_coords[:, np.newaxis], 0)
    if w.wcs.cunit[0] == 'm':
        wave_coords /= one_angstrom
    return np.squeeze(wave_coords)


def get_Nx(header):
    return header['NAXIS1']


def get_Ny(header):
    return header['NAXIS2']


def get_Nwave(header):
    return header['NAXIS3']


def get_celestial_coordinates(w, Nx, Ny, relative=True):
    # FIXME: Not sure how to make the coordinates relative for all cases. Nor
    # how to return arcsec always.
    w = get_celestial(w)
    x0_world, y0_world = w.wcs.crval
    make_relative = (x0_world != 0.0 and y0_world != 0.0 and relative)

    if make_relative:
        w.wcs.crval = (180.0, 0.0)
    xx_pix = np.arange(Nx) + 1
    yy_pix = np.arange(Ny) + 1
    x0_pix, y0_pix = np.rint(w.wcs.crpix).astype('int')
    xx_world, _ = w.wcs_pix2world(xx_pix, np.zeros_like(xx_pix) + y0_pix, 1)
    _, yy_world = w.wcs_pix2world(np.zeros_like(yy_pix) + x0_pix, yy_pix, 1)
    if make_relative:
        xx_world -= 180.0
        x_unit, y_unit = w.wcs.cunit
        if x_unit == 'deg' or y_unit == 'deg':
            xx_world *= 3600.0
            yy_world *= 3600.0
    return xx_world, yy_world


def get_galactic_coordinates_rad(w):
    '''

    Input: header with WCS information.
    Returns: Galactic coordinates l and b in radians to compare to HEALPix 
    maps.

    Note: l is consistent with HEALPix's phi, while HEALPix's theta will be 
    given by theta = pi/2 - b.

    '''

    w = get_celestial(w)
    x0, y0 = w.wcs.crpix
    coords = wcs.utils.pixel_to_skycoord(x0, y0, w, origin=1, mode='wcs')
    galcoords = coords.transform_to('galactic')

    l, b = galcoords.l.radian, galcoords.b.radian

    return l, b


def get_reference_pixel(w):
    crpix = np.rint(w.wcs.crpix).astype('int') - 1
    return (crpix[2], crpix[1], crpix[0])


def get_cube_limits(cube, ext):
    header = fits.getheader(cube, ext)
    l_obs = get_wavelength_coordinates(header)
    Ny = get_Ny(header)
    Nx = get_Nx(header)
    return l_obs.min(), l_obs.max(), Ny, Nx


def get_shape(header):
    Nx = get_Nx(header)
    Ny = get_Ny(header)
    Nwave = get_Nwave(header)
    return (Nwave, Ny, Nx)


def get_wavelength_sampling(w):
    w = w.sub([3])
    s = wcs.utils.proj_plane_pixel_scales(w)
    if w.wcs.cunit[0] == 'm':
        s /= one_angstrom
    return np.asscalar(s)


def get_pixel_area(w):
    return proj_plane_pixel_area(w)


def get_pixel_area_srad(w):
    a = get_pixel_area(w)
    return a * (rad_per_deg * rad_per_deg)


def get_pixel_scale(w):
    w = get_celestial(w)
    s = wcs.utils.proj_plane_pixel_scales(w)
    s = s.mean()
    x_unit, y_unit = w.wcs.cunit
    if x_unit == '' or y_unit == '':
        s /= 3600
    return s


def get_pixel_scale_rad(w):
    s = get_pixel_scale(w)
    return s * rad_per_deg


def copy_WCS(header, dest_header, axes):
    if np.isscalar(axes):
        axes = [axes]

    w = wcs.WCS(header, naxis=axes)
    dest_header.extend(w.to_header(), update=True)


def update_WCS(header, crpix, crval_wave, cdelt_wave):
    w = wcs.WCS(header)
    if w.wcs.cunit[2] == 'm':
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
