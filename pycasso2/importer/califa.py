'''
Created on 08/12/2015

@author: andre
'''
from ..cube import safe_getheader, FitsCube
from ..wcs import get_axis_coordinates, get_reference_pixel, set_axis_WCS
from ..resampling import resample_spectra, reshape_cube
from ..cosmology import redshift2lum_distance, spectra2restframe, velocity2redshift
from astropy import log
from astropy.io import fits
import numpy as np

__all__ = ['read_califa']


DR2_flux_unit = 1e-16


def read_califa(cube, cfg, **kwargs):
    '''
    FIXME: doc me! 
    '''
    # FIXME: sanitize kwargs
    l_ini = kwargs['l_ini']
    l_fin = kwargs['l_fin']
    dl = kwargs['dl']
    Nx = kwargs['width']
    Ny = kwargs['height']
    flux_unit = kwargs['flux_unit']
    name = kwargs['name']

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube)
    
    log.debug('Loading data from %s.' % cube)
    f_obs_orig = fits.getdata(cube, extname='PRIMARY')
    f_err_orig = fits.getdata(cube, extname='ERROR')
    badpix = fits.getdata(cube, extname='BADPIX') != 0
    l_obs = get_axis_coordinates(header, 3, dtype='float64')

    med_vel = float(header['MED_VEL'])
    z = velocity2redshift(med_vel)
    log.debug('Putting spectra in rest frame (z=%.2f, v=%.1f km/s).' % (z, med_vel))
    _, f_obs_rest = spectra2restframe(l_obs, f_obs_orig, z, kcor=1.0)
    l_rest, f_err_rest = spectra2restframe(l_obs, f_err_orig, z, kcor=1.0)

    log.debug('Resampling spectra in dl=%.2f \AA.' % dl)
    l_resam = np.arange(l_ini, l_fin + dl, dl)
    f_obs, f_err, f_flag = resample_spectra(l_rest, l_resam, f_obs_rest, f_err_rest, badpix)
    
    log.debug('Spatially reshaping cube into (%d, %d).' % (Ny, Nx))
    new_shape = (len(l_resam), Ny, Nx)
    center = get_reference_pixel(header)
    f_obs, f_err, f_flag, new_center = reshape_cube(f_obs, f_err, f_flag, center, new_shape)

    log.debug('Updating WCS.')
    set_axis_WCS(header, ax=1, crpix=new_center[2], naxis=new_shape[2])
    set_axis_WCS(header, ax=2, crpix=new_center[1], naxis=new_shape[1])
    set_axis_WCS(header, ax=3, crpix=0, crval=l_resam[0], cdelt=dl, naxis=new_shape[0])
    
    
    log.debug('Creating pycasso cube.')
    fix_flux_units = DR2_flux_unit / flux_unit
    K = FitsCube()
    K._initFits(f_obs * fix_flux_units, f_err * fix_flux_units, f_flag, header)
    K.flux_unit = flux_unit
    K.lumDistMpc = redshift2lum_distance(z)
    K.redshift = z    
    K.objectName = name
    
    return K
