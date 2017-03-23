'''
Created on 08/12/2015

@author: andre
'''
from ..cube import safe_getheader, FitsCube
from ..wcs import get_wavelength_coordinates, get_Naxis, get_reference_pixel, update_WCS
from ..resampling import resample_spectra
from ..cosmology import velocity2redshift, spectra2restframe
from ..reddening import extinction_corr

from astropy import log, wcs
from astropy.io import fits
import numpy as np

__all__ = ['read_gmos', 'gmos_read_masterlist']

gmos_cfg_sec = 'gmos'


def read_gmos(redcube, name, cfg, sl=None):
    '''
    FIXME: doc me! 
    '''
    l_ini = cfg.getfloat(gmos_cfg_sec, 'import_l_ini')
    l_fin = cfg.getfloat(gmos_cfg_sec, 'import_l_fin')
    dl = cfg.getfloat(gmos_cfg_sec, 'import_dl')
    flux_unit = cfg.getfloat(gmos_cfg_sec, 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % redcube)
    header = safe_getheader(redcube, ext=1)
    w = wcs.WCS(header)
    l_obs_orig = get_wavelength_coordinates(w, get_Naxis(header, 3))
    crpix = get_reference_pixel(w)
    
    masterlist = cfg.get(gmos_cfg_sec, 'masterlist')
    log.debug('Loading masterlist for %s: %s.' % (name, masterlist))
    ml = gmos_read_masterlist(masterlist, name)
        
    log.debug('Loading data from cube %s.' % redcube)
    f_obs_orig = fits.getdata(redcube, extname='SCI') / flux_unit
    f_err_orig = fits.getdata(redcube, extname='ERR') / flux_unit
    badpix = fits.getdata(redcube, extname='NCUBE') < 1

    if sl is not None:
        log.debug('Taking a slice of the cube...')
        y_slice, x_slice = sl
        f_obs_orig = f_obs_orig[:, y_slice, x_slice]
        f_err_orig = f_err_orig[:, y_slice, x_slice]
        badpix = badpix[:, y_slice, x_slice]
        crpix = (crpix[0], crpix[1] - y_slice.start, crpix[2] - x_slice.start)
        log.debug('New shape: %s.' % str(f_obs_orig.shape))

    log.debug('Extinction correction')
    EBV = ml['EBVGAL']
    log.debug('    E(B-V) = %f.' % EBV)
    f_obs_orig = extinction_corr(l_obs_orig, f_obs_orig, EBV)
    f_err_orig = extinction_corr(l_obs_orig, f_err_orig, EBV)
    
    z = velocity2redshift(ml['V_hel'])
    log.debug('Putting spectra in rest frame (z=%.2f).' % z)
    _, f_obs_rest = spectra2restframe(l_obs_orig, f_obs_orig, z, kcor=1.0)
    l_rest, f_err_rest = spectra2restframe(l_obs_orig, f_err_orig, z, kcor=1.0)
    
    log.debug('Resampling spectra in dl=%.2f \AA.' % dl)
    l_resam = np.arange(l_ini, l_fin + dl, dl)
    f_obs, f_err, f_flag = resample_spectra(
        l_rest, l_resam, f_obs_rest, f_err_rest, badpix)
    crpix = (0, crpix[1], crpix[2])
    
    log.debug('Updating WCS.')
    update_WCS(header, crpix=crpix, crval_wave=l_resam[0], cdelt_wave=dl)

    log.debug('Creating pycasso cube.')
    cube = FitsCube()
    cube._initFits(f_obs, f_err, f_flag, header, w)
    cube.flux_unit = flux_unit
    cube.lumDistMpc = np.asscalar(ml['DL'])
    cube.redshift = z
    cube.name = name
    return cube


masterlist_dtype = [('id', '|S08'),
                    ('name', '|S12'),
                    ('cube', '|S0128'),
                    ('cube_obs', '|S0128'),
                    ('ra', np.float64),
                    ('dec', np.float64),
                    ('V_hel', 'float64'),
                    ('morph', '|S05'),
                    ('DL', 'float64'),
                    ('EBVGAL', 'float64'),
                    ]


def gmos_read_masterlist(filename, galaxy_id=None):
    '''
    Read the whole masterlist, or a single entry.

    Parameters
    ----------
    filename : string
        Path to the file containing the masterlist.

    galaxy_id : string, optional
        ID of the masterlist entry, the first column of the table.
        If set, return only the entry pointed by ``galaxy_id'``.
        Default: ``None``

    Returns
    -------
    masterlist : recarray
        A numpy record array containing either the whole masterlist
        or the entry pointed by ``galaxy_id``.
    '''
    ml = np.genfromtxt(filename, masterlist_dtype, skip_header=1, delimiter=',')
    if galaxy_id is not None:
        index = np.where(ml['id'] == galaxy_id)[0]
        if len(index) == 0:
            raise Exception(
                'Entry %s not found in masterlist %s.' % (galaxy_id, filename))
        return np.squeeze(ml[index])
    else:
        return ml
