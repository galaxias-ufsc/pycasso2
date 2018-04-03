'''
Created on 08/12/2015

@author: andre
'''
from ..wcs import get_wavelength_coordinates, get_Naxis
from ..cosmology import velocity2redshift
from ..starlight.io import read_wavelength_mask
from .. import flags
from .core import safe_getheader, ObservedCube

from astropy import log, wcs
from astropy.io import fits, ascii
import numpy as np

__all__ = ['read_diving3d', 'd3d_read_masterlist', 'd3d_get_galaxy_id']

d3d_cfg_sec = 'diving3d'

def read_diving3d(cubes, name, cfg):
    '''
    FIXME: doc me! 
    '''
    if len(cubes) != 2:
        raise Exception('Please specify the reduced and observed cubes.')
        redcube, obscube = cubes
    
    flux_unit = cfg.getfloat('import', 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from reduced cube %s.' % redcube)
    header = safe_getheader(redcube)
    d3d_fix_crpix(header, 1)
    d3d_fix_crpix(header, 2)
    log.debug('Loading header from observed cube %s.' % obscube)
    obs_header = safe_getheader(obscube)
    for k in list(obs_header.keys()):
        if k in header or k == 'COMMENT' or k == '':
            continue
        header[k] = obs_header[k]
    w = wcs.WCS(header)
    l_obs = get_wavelength_coordinates(w, get_Naxis(header, 3))

    log.debug('Loading data from reduced cube %s.' % redcube)
    f_obs = fits.getdata(redcube)
    f_err = np.zeros_like(f_obs)
    badpix = np.zeros(f_obs.shape, dtype='bool')
    f_flag = np.where(badpix, flags.no_data, 0)

    masterlist = cfg.get('tables', 'master_table')
    galaxy_id = d3d_get_galaxy_id(redcube)
    log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
    ml = d3d_read_masterlist(masterlist, galaxy_id)
    d3d_save_masterlist(header, ml)
    z = velocity2redshift(ml['V_hel_km/s'])

    log.debug('Applying CCD gap mask (z = %f)' % z)
    gap_mask_template = cfg.get(d3d_cfg_sec, 'gap_mask_template')
    gap_mask_file = gap_mask_template % ml['grating']
    gap_mask = read_wavelength_mask(gap_mask_file, l_obs, z, dest='rest')
    f_flag[gap_mask] |= flags.ccd_gap

    obs = ObservedCube(name, l_obs, f_obs, f_err, f_flag, flux_unit, z, header)
    obs.EBV = 0.0
    obs.lumDist_Mpc = np.asscalar(ml['DL_Mpc'])
    obs.inRestFrame = True
    return obs


def d3d_read_masterlist(filename, galaxy_id=None):
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
    ml = ascii.read(filename)
    if galaxy_id is not None:
        index = np.where(ml['Name'] == galaxy_id)[0]
        if len(index) == 0:
            raise Exception(
                'Entry %s not found in masterlist %s.' % (galaxy_id, filename))
        return np.squeeze(ml[index][0])
    else:
        return ml


def d3d_fix_crpix(header, ax):
    '''
    Check for crazy bugs in the Diving3D cubes WCS.
    '''
    naxes = header['NAXIS']
    if ax < 1 or ax > naxes:
        raise Exception('Axis %d not in range (1, %d)' % (ax, naxes))
    crpix = header['CRPIX%d' % ax]
    if type(crpix) == str:
        log.warn('Converting CRPIX to float for axis %d.' % ax)
        crpix = float(crpix)
    if crpix <= 0.0:
        log.warn('Fixing negative CRPIX for axis %d.' % ax)
        naxis = header['NAXIS%d' % ax]
        crpix = naxis / 2.0 + 0.5
    header['CRPIX%d' % ax] = crpix


def d3d_save_masterlist(header, ml):
    header_ignored = ['Cube', 'Observed_cube']
    for key in ml.dtype.names:
        if key in header_ignored:
            continue
        hkey = 'HIERARCH MASTERLIST %s' % key.upper()
        header[hkey] = np.asscalar(ml[key])


def d3d_get_galaxy_id(cube):
    '''
    Return the ID of the cube, which can be used to index the masterlist.
    '''
    from os.path import basename
    return basename(cube).split('_')[0]
