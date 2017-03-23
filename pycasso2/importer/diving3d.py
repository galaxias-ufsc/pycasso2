'''
Created on 08/12/2015

@author: andre
'''
from ..cube import safe_getheader, FitsCube
from ..wcs import get_wavelength_coordinates, get_Naxis, get_reference_pixel, update_WCS
from ..resampling import resample_spectra
from ..cosmology import velocity2redshift
from ..starlight.tables import read_wavelength_mask
from .. import flags

from astropy import log, wcs
from astropy.io import fits
import numpy as np

__all__ = ['read_diving3d', 'd3d_read_masterlist', 'd3d_get_galaxy_id']

d3d_cfg_sec = 'diving3d'


def read_diving3d(redcube, obscube, name, cfg, sl=None):
    '''
    FIXME: doc me! 
    '''
    l_ini = cfg.getfloat(d3d_cfg_sec, 'import_l_ini')
    l_fin = cfg.getfloat(d3d_cfg_sec, 'import_l_fin')
    dl = cfg.getfloat(d3d_cfg_sec, 'import_dl')
    flux_unit = cfg.getfloat(d3d_cfg_sec, 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from reduced cube %s.' % redcube)
    header = safe_getheader(redcube)
    d3d_fix_crpix(header, 1)
    d3d_fix_crpix(header, 2)
    log.debug('Loading header from observed cube %s.' % obscube)
    obs_header = safe_getheader(obscube)
    for k in obs_header.keys():
        if k in header or k == 'COMMENT' or k == '':
            continue
        header[k] = obs_header[k]
    w = wcs.WCS(header)
    l_obs_orig = get_wavelength_coordinates(w, get_Naxis(header, 3))
    crpix = get_reference_pixel(w)

    log.debug('Loading data from reduced cube %s.' % redcube)
    f_obs_orig = fits.getdata(redcube)
    f_err_orig = np.zeros_like(f_obs_orig)
    badpix = np.zeros(f_obs_orig.shape, dtype='bool')

    if sl is not None:
        log.debug('Taking a slice of the cube...')
        y_slice, x_slice = sl
        f_obs_orig = f_obs_orig[:, y_slice, x_slice]
        f_err_orig = f_err_orig[:, y_slice, x_slice]
        badpix = badpix[:, y_slice, x_slice]
        crpix = (crpix[0], crpix[1] - y_slice.start, crpix[2] - x_slice.start)
        log.debug('New shape: %s.' % str(f_obs_orig.shape))

    log.debug('Resampling spectra in dl=%.2f \AA.' % dl)
    l_obs = np.arange(l_ini, l_fin + dl, dl)
    f_obs, f_err, f_flag = resample_spectra(
        l_obs_orig, l_obs, f_obs_orig, f_err_orig, badpix)
    crpix = (0, crpix[1], crpix[2])

    log.debug('Updating WCS.')
    update_WCS(header, crpix=crpix, crval_wave=l_obs[0], cdelt_wave=dl)

    masterlist = cfg.get(d3d_cfg_sec, 'masterlist')
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

    log.debug('Creating pycasso cube.')
    cube = FitsCube()
    cube._initFits(f_obs, f_err, f_flag, header, w)
    cube.flux_unit = flux_unit
    cube.lumDistMpc = ml['DL']
    cube.redshift = z
    cube.name = name
    return cube


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
