'''
Created on 08/12/2015

@author: andre
'''
from ..cube import safe_getheader, FitsCube
from ..wcs import get_wavelength_coordinates, get_reference_pixel, update_WCS, get_Naxis
from ..resampling import resample_spectra
from ..cosmology import redshift2lum_distance, spectra2restframe, velocity2redshift
from ..reddening import extinction_corr
from astropy import log, wcs
from astropy.io import fits
from astropy.table import Table
import numpy as np

__all__ = ['read_muse', 'muse_read_masterlist']

muse_cfg_sec = 'muse'


def read_muse(cube, name, cfg, sl=None):
    '''
    FIXME: doc me! 
    '''
    l_ini = cfg.getfloat(muse_cfg_sec, 'import_l_ini')
    l_fin = cfg.getfloat(muse_cfg_sec, 'import_l_fin')
    dl = cfg.getfloat(muse_cfg_sec, 'import_dl')
    flux_unit = cfg.getfloat(muse_cfg_sec, 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube, ext=1)
    w = wcs.WCS(header)
    l_obs = get_wavelength_coordinates(w, get_Naxis(header, 3))
    crpix = get_reference_pixel(w)

    log.debug('Loading data from %s.' % cube)
    f_obs_orig = fits.getdata(cube, extname='DATA')
    f_err_orig = fits.getdata(cube, extname='STAT')

    # Try to get bad pixel extension. If not, follow the MUSE pipeline manual 1.6.2,
    # > DQ The data quality flags encoded in an integer value according to the Euro3D standard (cf. [RD05]).
    # > However, by default, the data quality extension is not present, instead pixels which do not have a clean data quality status are directly encoded as Not-a-Number (NaN) values in the DATA extension itself.
    try:
        badpix = fits.getdata(cube, extname='DQ') != 0
    except:
        badpix = ~np.isfinite(f_obs_orig) | (f_obs_orig <= 0.0) | (f_err_orig <= 0.0)
    f_obs_orig[badpix] = 0.0
    f_err_orig[badpix] = 0.0

    if sl is not None:
        print((f_obs_orig.shape))
        print(crpix)
        log.debug('Taking a slice of the cube...')
        y_slice, x_slice = sl
        f_obs_orig = f_obs_orig[:, y_slice, x_slice]
        f_err_orig = f_err_orig[:, y_slice, x_slice]
        badpix = badpix[:, y_slice, x_slice]
        crpix = (crpix[0], crpix[1] - y_slice.start, crpix[2] - x_slice.start)
        log.debug('New shape: %s.' % str(f_obs_orig.shape))

    # Get distance from master list
    masterlist = cfg.get(muse_cfg_sec, 'masterlist')
    galaxy_id = name
    log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
    ml = muse_read_masterlist(masterlist, galaxy_id)
    muse_save_masterlist(header, ml)
    
    # Correction for galactic extinction
    ebv = float(ml['E(B-V)'])
    log.debug('Applying galactic extinction correction: E(B-V) = %.4f.' % ebv)
    f_obs_orig = extinction_corr(l_obs, f_obs_orig, ebv)
        
    z = velocity2redshift(ml['V_r [km/s]'])
    log.debug('Putting spectra in rest frame (z=%.4f, v=%.1f km/s).' %
              (z, ml['V_r [km/s]']))
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

    return K

masterlist_dtype = [('Name', '|S05'),
                    ('Galaxy name', '|S12'),
                    ('V_r [km/s]', 'float64'),
                    ('D (Mpc)', 'float64'),
                    ]

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
