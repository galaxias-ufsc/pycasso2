'''
Created on 08/12/2015

@author: andre
'''
from ..wcs import get_wavelength_coordinates, get_Naxis
from ..cosmology import redshift2lum_distance, velocity2redshift
from .core import import_spectra, safe_getheader, fill_cube

from astropy import log, wcs
from astropy.io import fits
from astropy.table import Table
import numpy as np

__all__ = ['read_califa', 'califa_read_masterlist']

califa_cfg_sec = 'califa'


def read_califa(cube, name, cfg, destcube=None):
    '''
    FIXME: doc me!
    '''
    if len(cube) != 1:
        raise Exception('Please specify a single cube.')
    cube = cube[0]

    flux_unit = cfg.getfloat(califa_cfg_sec, 'flux_unit')
    DL_from_masterlist = cfg.getboolean(califa_cfg_sec, 'DL_from_masterlist')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube)
    w = wcs.WCS(header)
    l_obs = get_wavelength_coordinates(w, get_Naxis(header, 3))

    med_vel = float(header['MED_VEL'])
    z = velocity2redshift(med_vel)

    log.debug('Loading data from %s.' % cube)
    f_obs_orig = fits.getdata(cube, extname='PRIMARY')
    f_err_orig = fits.getdata(cube, extname='ERROR')
    badpix = fits.getdata(cube, extname='BADPIX') != 0

    # Get luminosity distance from master list
    if DL_from_masterlist:
        masterlist = cfg.get(califa_cfg_sec, 'masterlist')
        galaxy_id = name
        log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
        ml = califa_read_masterlist(masterlist, galaxy_id)
        lum_dist_Mpc = ml['d_Mpc']
    else:
        lum_dist_Mpc = redshift2lum_distance(z)

    
    l_obs, f_obs, f_err, f_flag, w, _ = import_spectra(l_obs, f_obs_orig,
                                                       f_err_orig, badpix,
                                                       cfg, califa_cfg_sec,
                                                       w, z, vaccuum_wl=False,
                                                       EBV=0.0)

    destcube = fill_cube(f_obs, f_err, f_flag, header, w,
                         flux_unit, lum_dist_Mpc, z, name, cube=destcube)
    return destcube

def califa_read_masterlist(filename, galaxy_id=None):
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
    ml = Table.read(filename, format='csv')
    if galaxy_id is not None:
        index = np.where(ml['#CALIFA_ID'] == galaxy_id)[0]
        if len(index) == 0:
            raise Exception(
                'Entry %s not found in masterlist %s.' % (galaxy_id, filename))
        return ml[index][0]
    else:
        return ml
