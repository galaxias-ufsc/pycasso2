'''
Created on 24 de jul de 2017

@author: andre
'''

from ..wcs import get_reference_pixel, get_updated_WCS
from ..resampling import vac2air, resample_spectra, find_nearest_index
from ..reddening import get_EBV, extinction_corr
from ..config import parse_slice
from ..modeling import cube_continuum
from ..segmentation import bin_spectra, get_cov_factor
from ..geometry import convex_hull_mask
from .. import flags

from astropy import log
from astropy.io import fits
import numpy as np


def safe_getheader(f, ext=0):
    with fits.open(f) as hl:
        hdu = hl[ext]
        hdu.verify('fix')
        return hdu.header


def fill_cube(f_obs, f_err, f_flag, header, wcs,
              flux_unit, lum_dist_Mpc, redshift, name, cube=None):
    if cube is None:
        from ..cube import FitsCube
        log.warn('Creating pycasso cube.')
        cube = FitsCube()
    header.extend(wcs.to_header(), update=True)
    cube._initFits(f_obs, f_err, f_flag, header, wcs, segmask=None)
    cube.flux_unit = flux_unit
    cube.lumDistMpc = lum_dist_Mpc
    cube.redshift = redshift
    cube.name = name
    return cube


def import_spectra(l_obs, f_obs, f_err, badpix, cfg, wcs,
                   z=None, vaccuum_wl=False, EBV=None, vector_resam=False):
    '''
    FIXME: doc me! 
    '''
    cfg_import_sec = 'import'
    cfg_tables_sec = 'tables'
    cfg_starlight_sec = 'starlight'

    l_ini = cfg.getfloat(cfg_import_sec, 'l_ini')
    l_fin = cfg.getfloat(cfg_import_sec, 'l_fin')
    dl = cfg.getfloat(cfg_import_sec, 'dl')
    
    if vaccuum_wl:
        log.debug('Converting vacuum to air wavelengths.')
        l_obs = vac2air(l_obs)
    
    crpix = get_reference_pixel(wcs)
    sl_string = cfg.get(cfg_import_sec, 'slice', fallback=None)
    sl = parse_slice(sl_string)
    if sl is not None:
        log.debug('Taking a slice of the cube...')
        y_slice, x_slice = sl
        f_obs = f_obs[:, y_slice, x_slice]
        f_err = f_err[:, y_slice, x_slice]
        badpix = badpix[:, y_slice, x_slice]
        crpix = (crpix[0], crpix[1] - y_slice.start, crpix[2] - x_slice.start)
        log.debug('New shape: %s.' % str(f_obs.shape))

    bin_size = cfg.getint(cfg_import_sec, 'binning', fallback=1)
    if bin_size > 1:
        log.debug('Binning cube (%d x %d).' % (bin_size, bin_size))
        A = cfg.getfloat('import', 'spat_cov_a', fallback=0.0)
        B = cfg.getfloat('import', 'spat_cov_b', fallback=1.0)
        cov_factor = get_cov_factor(bin_size**2, A, B)
        log.debug('    Covariance factor: %.2f.' % cov_factor)
        crpix = (crpix[0], crpix[1] / bin_size, crpix[2] / bin_size)
        f_obs, f_err, good_frac = bin_spectra(f_obs, f_err, badpix, bin_size, cov_factor)
        badpix = good_frac == 0

    if EBV is None:
        # FIXME: Dust maps in air or vacuum?
        dust_map = cfg.get(cfg_tables_sec, 'dust_map')
        log.debug('Calculating extinction correction (map = %s).' % dust_map)
        EBV = get_EBV(wcs, dust_map)

    log.debug('Dereddening spectra, E(B-V) = %f.' % EBV)
    ext_corr = extinction_corr(l_obs, EBV)[:, np.newaxis, np.newaxis]
    f_obs *= ext_corr
    f_err *= ext_corr

    if z is not None:
        log.debug('Putting spectra in rest frame (z=%.2f).' % z)
        z_plus_1 = 1.0 + z
        f_obs *= z_plus_1
        f_err *= z_plus_1
        l_obs /= z_plus_1

    log.debug('Resampling spectra in dl=%.2f \AA.' % dl)
    l_resam = np.arange(l_ini, l_fin + dl, dl)
    f_obs, f_err, f_flag = resample_spectra(
        l_obs, l_resam, f_obs, f_err, badpix, vectorized=vector_resam)
    crpix = (0, crpix[1], crpix[2])

    log.debug('Updating WCS.')
    wcs = get_updated_WCS(wcs, crpix=crpix, crval_wave=l_resam[0], cdelt_wave=dl)

    return l_resam, f_obs, f_err, f_flag, wcs, EBV
