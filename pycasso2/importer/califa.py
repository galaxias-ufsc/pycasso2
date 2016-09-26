'''
Created on 08/12/2015

@author: andre
'''
from ..cube import safe_getheader, FitsCube
from ..wcs import get_wavelength_coordinates, get_reference_pixel, update_WCS
from ..resampling import resample_spectra, reshape_cube
from ..cosmology import redshift2lum_distance, spectra2restframe, velocity2redshift
from astropy import log
from astropy.io import fits
import numpy as np

__all__ = ['read_califa']

califa_cfg_sec = 'califa'


def read_califa(cube, name, cfg):
    '''
    FIXME: doc me! 
    '''
    # FIXME: sanitize kwargs
    l_ini = cfg.getfloat(califa_cfg_sec, 'import_l_ini')
    l_fin = cfg.getfloat(califa_cfg_sec, 'import_l_fin')
    dl = cfg.getfloat(califa_cfg_sec, 'import_dl')
    Nx = cfg.getint(califa_cfg_sec, 'import_Nx')
    Ny = cfg.getint(califa_cfg_sec, 'import_Ny')
    flux_unit = cfg.getfloat(califa_cfg_sec, 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube)
    
    log.debug('Loading data from %s.' % cube)
    f_obs_orig = fits.getdata(cube, extname='PRIMARY')
    f_err_orig = fits.getdata(cube, extname='ERROR')
    badpix = fits.getdata(cube, extname='BADPIX') != 0
    l_obs = get_wavelength_coordinates(header)

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
    update_WCS(header, crpix=new_center, crval_wave=l_resam[0], cdelt_wave=dl)    
    
    log.debug('Creating pycasso cube.')
    K = FitsCube()
    K._initFits(f_obs, f_err, f_flag, header)
    K.flux_unit = flux_unit
    K.lumDistMpc = redshift2lum_distance(z)
    K.redshift = z    
    K.name = name
    
    return K
