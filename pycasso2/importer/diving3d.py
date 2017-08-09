'''
Created on 08/12/2015

@author: andre
'''
from ..wcs import get_wavelength_coordinates, get_Naxis
from ..cosmology import velocity2redshift
from ..starlight.tables import read_wavelength_mask
from .. import flags
from .core import import_spectra, safe_getheader, fill_cube

from astropy import log, wcs
from astropy.io import fits
import numpy as np

__all__ = ['read_diving3d', 'd3d_read_masterlist', 'd3d_get_galaxy_id']

d3d_cfg_sec = 'diving3d'

def read_diving3d(cubes, name, cfg, destcube=None):
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
    f_obs_orig = fits.getdata(redcube)
    f_err_orig = np.zeros_like(f_obs_orig)
    badpix = np.zeros(f_obs_orig.shape, dtype='bool')

    l_obs, f_obs, f_err, f_flag, w, _ = import_spectra(l_obs, f_obs_orig,
                                                       f_err_orig, badpix,
                                                       cfg, w, z=0.0,
                                                       vaccuum_wl=False,
                                                       EBV=0.0)

    masterlist = cfg.get('tables', 'master_table')
    galaxy_id = d3d_get_galaxy_id(redcube)
    log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
    ml = d3d_read_masterlist(masterlist, galaxy_id)
    d3d_save_masterlist(header, ml)
    z = velocity2redshift(ml['V_hel'])

    log.debug('Applying CCD gap mask (z = %f)' % z)
    gap_mask_template = cfg.get(d3d_cfg_sec, 'gap_mask_template')
    gap_mask_file = gap_mask_template % ml['grating']
    gap_mask = read_wavelength_mask(gap_mask_file, l_obs, z, dest='rest')
    f_flag[gap_mask] |= flags.ccd_gap

    destcube = fill_cube(f_obs, f_err, f_flag, header, w,
                         flux_unit, ml['DL'], z, name, cube=destcube)
    return destcube


masterlist_dtype = [('id', '|S05'),
                    ('name', '|S12'),
                    ('V_hel', 'float64'),
                    ('morph', '|S05'),
                    ('T', 'float64'),
                    ('R_e', 'float64'),
                    ('M_K', 'float64'),
                    ('n_s', 'float64'),
                    ('epsilon', 'float64'),
                    ('DL', 'float64'),
                    ('eDL', 'float64'),
                    ('EL', '|S05'),
                    ('grating', '|S04'),
                    ('cube', '|S0128'),
                    ('cube_obs', '|S0128'),
                    ]


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
    ml = np.genfromtxt(filename, masterlist_dtype, skip_header=2)
    if galaxy_id is not None:
        index = np.where(ml['id'] == galaxy_id)[0]
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
    header_ignored = ['cube', 'cube_obs']
    for key in ml.dtype.names:
        if key in header_ignored:
            continue
        hkey = 'HIERARCH MASTERLIST %s' % key.upper()
        header[hkey] = ml[key]


def d3d_get_galaxy_id(cube):
    '''
    Return the ID of the cube, which can be used to index the masterlist.
    '''
    from os.path import basename
    return basename(cube).split('_')[0]
