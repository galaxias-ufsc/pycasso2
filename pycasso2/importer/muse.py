'''
Created on 08/12/2015

@author: andre
'''
from ..wcs import get_wavelength_coordinates, get_Naxis, write_WCS
from ..cosmology import velocity2redshift
from .. import flags
from ..config import parse_slice
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

    # We slice the cube early here. Some MUSE cubes are huge and will thrash the system memory.    
    sl_string = cfg.get('import', 'slice', fallback=None)
    sl = parse_slice(sl_string)
    if sl is not None:
        log.info('Slicing cube while reading.')
        y_slice, x_slice = sl
        log.debug('Slice will be: %d:%d, %d:%d' % (y_slice.start, y_slice.stop, x_slice.start, x_slice.stop))
        w = w[:, y_slice, x_slice]
        write_WCS(header, w)
    else:
        Nx = w.pixel_shape[0]
        x_slice = slice(Nx)
        Ny = w.pixel_shape[1]
        y_slice = slice(Ny)
    with fits.open(cube) as f:
        f_obs = f['DATA'].section[:, y_slice, x_slice].astype('float64')
        f_err = f['STAT'].section[:, y_slice, x_slice].astype('float64')
        if 'CONTRIB' in f:
            log.info('Loading combined MUSE cube.')
            badpix = (f['CONTRIB'].section[:, y_slice, x_slice] == 0)
        else:
            log.info('Loading default MUSE cube.')
            badpix = ~np.isfinite(f_obs)
            badpix |= ~np.isfinite(f_err)

    np.sqrt(f_err, out=f_err)

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
    if sl is not None:
        obs.addKeyword('SLICE', sl_string)
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
