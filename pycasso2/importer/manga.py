'''
Created on 08/12/2015

@author: andre
'''
from ..cube import safe_getheader, FitsCube
from ..wcs import update_WCS, get_reference_pixel
from ..resampling import resample_spectra, reshape_cube, vac2air
from ..cosmology import redshift2lum_distance, spectra2restframe
from ..reddening import get_EBV, extinction_corr
from astropy import log, wcs
from astropy.io import fits
import numpy as np

__all__ = ['read_manga', 'read_drpall']

manga_cfg_sec = 'manga'


def read_drpall(filename, mangaid=None):
    with fits.open(filename) as f:
        t = f[1].data
    if mangaid is not None:
        i = np.where(t['mangaid'] == mangaid)[0]
        t = t[i]
    return t


def read_manga(cube, name, cfg):
    '''
    FIXME: doc me! 
    '''
    # FIXME: sanitize kwargs
    l_ini = cfg.getfloat(manga_cfg_sec, 'import_l_ini')
    l_fin = cfg.getfloat(manga_cfg_sec, 'import_l_fin')
    dl = cfg.getfloat(manga_cfg_sec, 'import_dl')
    Nx = cfg.getint(manga_cfg_sec, 'import_Nx')
    Ny = cfg.getint(manga_cfg_sec, 'import_Ny')
    flux_unit = cfg.getfloat(manga_cfg_sec, 'flux_unit')
    
    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube, ext='FLUX')
    drp = read_drpall(cfg.get(manga_cfg_sec, 'drpall'), header['MANGAID'])
    z = np.asscalar(drp['nsa_z'])
    
    log.debug('Loading data from %s.' % cube)
    with fits.open(cube) as f:
        f_obs_orig = f['FLUX'].data
        # FIXME: Check mask bits.
        badpix = f['MASK'].data > 0
        goodpix = ~badpix
        f_err_orig = np.zeros_like(f_obs_orig)
        f_err_orig[goodpix] = f['IVAR'].data[goodpix]**-0.5
        l_obs = f['WAVE'].data
        
    log.debug('Vacuum to air wavelengths.')
    l_obs = vac2air(l_obs)
    
    # FIXME: Dust maps in air or vacuum?
    dust_map = cfg.get('tables', 'dust_map')
    log.debug('Extinction correction (map = %s).' % dust_map)
    #EBV = get_EBV(header, dust_map)
    EBV = header['EBVGAL']
    log.debug('    E(B-V) = %f.' % EBV)
    f_obs_orig = extinction_corr(l_obs, f_obs_orig, EBV)
    
    log.debug('Putting spectra in rest frame (z=%.2f).' % z)
    _, f_obs_rest = spectra2restframe(l_obs, f_obs_orig, z, kcor=1.0)
    l_rest, f_err_rest = spectra2restframe(l_obs, f_err_orig, z, kcor=1.0)

    log.debug('Spatially reshaping cube into (%d, %d).' % (Ny, Nx))
    new_shape = (len(l_rest), Ny, Nx)
    center = get_reference_pixel(wcs.WCS(header))
    f_obs_rest, f_err_rest, badpix, new_center = reshape_cube(f_obs_rest, f_err_rest, badpix, center, new_shape)

    log.debug('Resampling spectra in dl=%.2f \AA.' % dl)
    l_resam = np.arange(l_ini, l_fin + dl, dl)
    f_obs, f_err, f_flag = resample_spectra(l_rest, l_resam, f_obs_rest, f_err_rest, badpix)
    
    log.debug('Updating WCS.')
    update_WCS(header, crpix=new_center, crval_wave=l_resam[0], cdelt_wave=dl)
    
    log.debug('Creating pycasso cube.')
    K = FitsCube()
    K._initFits(f_obs, f_err, f_flag, header)
    K.flux_unit = flux_unit
    K.lumDistMpc = redshift2lum_distance(z)
    K.redshift = z    
    K.objectName = name
    
    return K
