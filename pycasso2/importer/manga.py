'''
Created on 08/12/2015

@author: andre
'''
from ..cube import safe_getheader, FitsCube
from ..wcs import get_axis_coordinates, get_reference_pixel, set_axis_WCS
from ..resampling import resample_spectra, reshape_cube
from ..cosmology import redshift2lum_distance, spectra2restframe
from astropy import log
from astropy.io import fits
import numpy as np

__all__ = ['read_manga']

def read_drpall(mangaid, filename):
    with fits.open(filename) as f:
        t = f[1].data
    i = np.where(t['mangaid'] == mangaid)[0]
    return t[i]


def read_manga(cube, cfg, **kwargs):
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
    header = safe_getheader(cube, ext='FLUX')

    drp = read_drpall(header['MANGAID'], cfg.get('manga', 'drpall'))
    z = np.asscalar(drp['nsa_z'])
    
    log.debug('Loading data from %s.' % cube)
    with fits.open(cube) as f:
        f_obs_orig = f['FLUX'].data
        # FIXME: make sense of IVAR extension.
        #f_err_orig = f['IVAR'].data**-0.5
        f_err_orig = np.zeros_like(f_obs_orig)
        # FIXME: Check mask bits.
        badpix = f['MASK'].data > 0
        l_obs = f['WAVE'].data

    log.debug('Putting spectra in rest frame (z=%.2f).' % z)
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
    K = FitsCube()
    K._initFits(f_obs, f_err, f_flag, header)
    K.flux_unit = flux_unit
    K.lumDistMpc = redshift2lum_distance(z)
    K.redshift = z    
    K.objectName = name
    
    return K
