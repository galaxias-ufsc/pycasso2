'''
Created on 08/12/2015

@author: andre
'''
from ..wcs import get_wavelength_coordinates, get_Naxis
from ..cosmology import velocity2redshift
from .. import flags
from .core import safe_getheader,  ObservedCube

from astropy import log, wcs
from astropy.io import fits
from astropy.table import Table
import numpy as np

__all__ = ['read_muse', 'muse_read_masterlist']

def read_muse(cube, name, cfg):
    '''
    FIXME: doc me! 
    '''
    if len(cube) != 1:
        raise Exception('Please specify a single cube.')
    cube = cube[0]

    flux_unit = cfg.getfloat('import', 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube, ext=1)
    w = wcs.WCS(header)
    l_obs = get_wavelength_coordinates(w, get_Naxis(header, 3))

    log.debug('Loading data from %s.' % cube)
    f_obs = fits.getdata(cube, extname='DATA').astype('float64')
    f_err = fits.getdata(cube, extname='STAT').astype('float64')
    np.sqrt(f_err, out=f_err)
    # Try to get bad pixel extension. If not, follow the MUSE pipeline manual 1.6.2,
    # > DQ The data quality flags encoded in an integer value according to the Euro3D standard (cf. [RD05]).
    # > However, by default, the data quality extension is not present, instead pixels which do not have a clean data quality status are directly encoded as Not-a-Number (NaN) values in the DATA extension itself.
    try:
        badpix = fits.getdata(cube, extname='DQ') != 0
    except:
        badpix = ~np.isfinite(f_obs) | ~np.isfinite(f_err)
        badpix |= (f_obs <= 0.0)
        badpix |= (f_err <= 0.0)
    f_obs[badpix] = 0.0
    f_err[badpix] = 0.0

    # Get data from master list
    masterlist = cfg.get('tables', 'master_table')
    galaxy_id = name
    log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
    ml = muse_read_masterlist(masterlist, galaxy_id)
    z = velocity2redshift(ml['V_r [km/s]'])
    muse_save_masterlist(header, ml)
    f_flag = np.where(badpix, flags.no_data, 0)
 
    obs = ObservedCube(name, l_obs, f_obs, f_err, f_flag, flux_unit, z, header)
    obs.EBV = float(ml['E(B-V)'])
    obs.lumDist_Mpc = ml['D [Mpc]']
    return obs

def muse_save_masterlist(header, ml):
    header_ignored = ['cube', 'cube_obs']
    for key in ml.dtype.names:
        if key in header_ignored:
            continue
        hkey = 'HIERARCH MASTERLIST %s' % key.upper()
        header[hkey] = ml[key]

def muse_read_masterlist(filename, galaxy_id=None):
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
        index = np.where(ml['Name'] == galaxy_id)[0]
        if len(index) == 0:
            raise Exception(
                'Entry %s not found in masterlist %s.' % (galaxy_id, filename))
        return ml[index][0]
    else:
        return ml
