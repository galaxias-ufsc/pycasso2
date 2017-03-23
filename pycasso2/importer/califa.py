'''
Created on 08/12/2015

@author: andre
'''
from ..cube import safe_getheader, FitsCube
from ..wcs import get_wavelength_coordinates, get_reference_pixel, update_WCS, get_Naxis
from ..resampling import resample_spectra
from ..cosmology import redshift2lum_distance, spectra2restframe, velocity2redshift
from astropy import log, wcs
from astropy.io import fits
from astropy.table import Table
import numpy as np

__all__ = ['read_califa', 'califa_read_masterlist']

califa_cfg_sec = 'califa'


def read_califa(cube, name, cfg, sl=None):
    '''
    FIXME: doc me!
    '''
    l_ini = cfg.getfloat(califa_cfg_sec, 'import_l_ini')
    l_fin = cfg.getfloat(califa_cfg_sec, 'import_l_fin')
    dl = cfg.getfloat(califa_cfg_sec, 'import_dl')
    flux_unit = cfg.getfloat(califa_cfg_sec, 'flux_unit')
    DL_from_masterlist = cfg.getboolean(califa_cfg_sec, 'DL_from_masterlist')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube)
    w = wcs.WCS(header)
    l_obs = get_wavelength_coordinates(w, get_Naxis(header, 3))
    crpix = get_reference_pixel(w)

    log.debug('Loading data from %s.' % cube)
    f_obs_orig = fits.getdata(cube, extname='PRIMARY')
    f_err_orig = fits.getdata(cube, extname='ERROR')
    badpix = fits.getdata(cube, extname='BADPIX') != 0

    if sl is not None:
        log.debug('Taking a slice of the cube...')
        y_slice, x_slice = sl
        f_obs_orig = f_obs_orig[:, y_slice, x_slice]
        f_err_orig = f_err_orig[:, y_slice, x_slice]
        badpix = badpix[:, y_slice, x_slice]
        crpix = (crpix[0], crpix[1] - y_slice.start, crpix[2] - x_slice.start)
        log.debug('New shape: %s.' % str(f_obs_orig.shape))

    med_vel = float(header['MED_VEL'])
    z = velocity2redshift(med_vel)
    log.debug('Putting spectra in rest frame (z=%.2f, v=%.1f km/s).' %
              (z, med_vel))
    _, f_obs_rest = spectra2restframe(l_obs, f_obs_orig, z, kcor=1.0)
    l_rest, f_err_rest = spectra2restframe(l_obs, f_err_orig, z, kcor=1.0)

    log.debug('Resampling spectra in dl=%.2f \AA.' % dl)
    l_resam = np.arange(l_ini, l_fin + dl, dl)
    f_obs, f_err, f_flag = resample_spectra(
        l_rest, l_resam, f_obs_rest, f_err_rest, badpix)
    crpix = (0, crpix[1], crpix[2])

    log.debug('Updating WCS.')
    update_WCS(header, crpix=crpix, crval_wave=l_resam[0], cdelt_wave=dl)

    log.debug('Creating pycasso cube.')
    K = FitsCube()
    K._initFits(f_obs, f_err, f_flag, header, w)
    K.flux_unit = flux_unit
    K.lumDistMpc = redshift2lum_distance(z)
    K.redshift = z
    K.name = name

    # Get luminosity distance from master list
    if DL_from_masterlist:
        masterlist = cfg.get(califa_cfg_sec, 'masterlist')
        galaxy_id = name
        log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
        ml = califa_read_masterlist(masterlist, galaxy_id)
        K.lumDistMpc = ml['d_Mpc']

    return K

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
