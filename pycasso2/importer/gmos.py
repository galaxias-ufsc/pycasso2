'''
Created on 08/12/2015

@author: andre
'''
from ..wcs import get_wavelength_coordinates, get_Naxis
from ..cosmology import velocity2redshift
from .core import import_spectra, safe_getheader, fill_cube

from astropy import log, wcs
from astropy.io import fits
import numpy as np

__all__ = ['read_gmos', 'gmos_read_masterlist']

gmos_cfg_sec = 'gmos'


def read_gmos(cube, name, cfg, destcube=None):
    '''
    FIXME: doc me! 
    '''
    if len(cube) != 1:
        raise Exception('Please specify a single cube.')
    cube = cube[0]

    flux_unit = cfg.getfloat(gmos_cfg_sec, 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube, ext=1)
    w = wcs.WCS(header)
    l_obs = get_wavelength_coordinates(w, get_Naxis(header, 3))
    
    masterlist = cfg.get(gmos_cfg_sec, 'masterlist')
    log.debug('Loading masterlist for %s: %s.' % (name, masterlist))
    ml = gmos_read_masterlist(masterlist, name)
    EBV = ml['EBVGAL']
    z = velocity2redshift(ml['V_hel'])
        
    log.debug('Loading data from cube %s.' % cube)
    f_obs_orig = fits.getdata(cube, extname='SCI') / flux_unit
    f_err_orig = fits.getdata(cube, extname='ERR') / flux_unit
    badpix = fits.getdata(cube, extname='NCUBE') < 1

    l_obs, f_obs, f_err, f_flag, w, _ = import_spectra(l_obs, f_obs_orig,
                                                       f_err_orig, badpix,
                                                       cfg, gmos_cfg_sec,
                                                       w, z, vaccuum_wl=False,
                                                       EBV=EBV)

    destcube = fill_cube(f_obs, f_err, f_flag, header, w,
                         flux_unit, np.asscalar(ml['DL']), z, name, cube=destcube)
    return destcube


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
