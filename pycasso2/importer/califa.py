'''
Created on 08/12/2015

@author: andre
'''
from ..wcs import get_wavelength_coordinates, get_Naxis
from ..cosmology import velocity2redshift
from .. import flags
from .core import safe_getheader, ObservedCube

from astropy import log, wcs
from astropy.io import fits
from astropy.table import Table, Column
import numpy as np

__all__ = ['read_califa', 'califa_read_masterlist']


def read_califa(cube, name, cfg):
    '''
    FIXME: doc me!
    '''
    if len(cube) != 1:
        raise Exception('Please specify a single cube.')
    cube = cube[0]

    flux_unit = cfg.getfloat('import', 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube)
    w = wcs.WCS(header)
    l_obs = get_wavelength_coordinates(w, get_Naxis(header, 3))

    masterlist = cfg.get('tables', 'master_table')
    galaxy_id = name
    log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
    ml = califa_read_masterlist(masterlist, galaxy_id)

    med_vel = ml['MED_VEL']
    if (med_vel <= -990):
        med_vel = float(header['MED_VEL'])
        log.warn('Could not find the systemic velocity in the master list to restframe the spectra. Using MED_VEL from the fits file, which might be wrong!')
    z = velocity2redshift(med_vel)

    log.debug('Loading data from %s.' % cube)
    f_obs = fits.getdata(cube, extname='PRIMARY')
    f_err = fits.getdata(cube, extname='ERROR')
    badpix = fits.getdata(cube, extname='BADPIX') != 0
    f_flag = np.where(badpix, flags.no_data, 0)

    
    obs = ObservedCube(name, l_obs, f_obs, f_err, f_flag, flux_unit, z, header)
    obs.EBV = 0.0
    obs.lumDist_Mpc = ml['d_Mpc']
    return obs


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
    ml = Table.read(filename, format='ascii.fixed_width_two_line')

    if galaxy_id is not None:
        index = np.where(ml['CALIFA_ID'] == galaxy_id)[0]
        if len(index) == 0:
            raise Exception('Entry %s not found in masterlist %s.' % (galaxy_id, filename))
        return ml[index][0]
    else:
        return ml
