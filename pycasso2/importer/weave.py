'''
Created on 22/09/2019

@author: RGB
'''
from ..wcs import get_wavelength_coordinates, get_Naxis
from ..cosmology import velocity2redshift
from .. import flags
from .core import safe_getheader,  ObservedCube

from astropy import log, wcs
from astropy.io import fits
from astropy.table import Table
import numpy as np

__all__ = ['read_weave', 'weave_read_masterlist']

def read_weave(cube, name, cfg):
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

    hdus = fits.open(cube)
    hdu_names = {}
    for hdu in hdus:
        if hdu.name.endswith('DATA'):
            hdu_names['DATA'] = hdu.name
        elif hdu.name.endswith('IVAR'):
            hdu_names['IVAR'] = hdu.name
        elif hdu.name.endswith('SENSFUNC'):
            hdu_names['SENSFUNC'] = hdu.name
    hdus.close()

    log.debug('Loading data from %s.' % cube)
    f_obs = fits.getdata(cube, extname=hdu_names['DATA'])
    ivar  = fits.getdata(cube, extname=hdu_names['IVAR'])
    goodpix = ivar > 0.0
    f_err = np.zeros_like(f_obs)
    f_err[goodpix] = ivar[goodpix]**-0.5
    f_flag = np.where(~goodpix, flags.no_data, 0)

    if 'SENSFUNC' in hdu_names and cfg.getboolean('import', 'sensfunc', fallback=True):
        sensfunc = fits.getdata(cube, extname=hdu_names['SENSFUNC'])
        f_obs *= sensfunc[:, np.newaxis, np.newaxis]
        f_err *= sensfunc[:, np.newaxis, np.newaxis]
        
    # Get data from master list
    masterlist = cfg.get('tables', 'master_table')
    galaxy_id = name
    log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
    ml = weave_read_masterlist(masterlist, galaxy_id)
    if 'velocity' in ml.colnames:
        z = velocity2redshift(ml['velocity'])
    elif 'z' in ml.colnames:
        z = ml['z']
    weave_save_masterlist(header, ml)

    obs = ObservedCube(name, l_obs, f_obs, f_err, f_flag, flux_unit, z, header)
    if 'EBV' in ml.colnames:
        obs.EBV = float(ml['EBV'])
    if 'DL_Mpc' in ml.colnames:
        obs.lumDist_Mpc = ml['DL_Mpc']
    return obs

def weave_save_masterlist(header, ml):
    header_ignored = ['cube', 'cube_obs']
    for key in ml.dtype.names:
        if key in header_ignored:
            continue
        hkey = 'HIERARCH MASTERLIST %s' % key.upper()
        header[hkey] = ml[key]

def weave_read_masterlist(filename, galaxy_id=None):
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
    ml = Table.read(filename, format='ascii.fixed_width_two_line')
    if galaxy_id is not None:
        index = np.where(ml['Name'] == galaxy_id)[0]
        if len(index) == 0:
            raise Exception(
                'Entry %s not found in masterlist %s.' % (galaxy_id, filename))
        return ml[index][0]
    else:
        return ml
