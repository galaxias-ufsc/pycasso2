'''
Created on 24 de jul de 2017

@author: andre
'''

from ..wcs import scale_celestial_WCS, replace_wave_WCS, write_WCS
from ..resampling import vac2air, resample_spectra, find_nearest_index
from ..reddening import Galactic_reddening_corr
from ..config import parse_slice
from ..modeling import cube_continuum
from ..segmentation import bin_spectra, get_cov_factor, clean_mask_polygons
from ..geometry import convex_hull_mask
from ..cosmology import redshift2lum_distance
from .. import flags
from .. import __version__
from .git_helpers import get_git_devstr

from astropy import log, wcs
from astropy.io import fits
import numpy as np
import sys

__all__ = ['ObservedCube', 'preprocess_obs', 'safe_getheader']

class ObservedCube(object):
    _hdr_prefix = 'HIERARCH PYCASSO'

    def __init__(self, name, l_obs, f_obs, f_err, f_flag, 
                 flux_unit, redshift, header, f_disp=None,
                 good_frac=None, cov_matrix=False):
        self.name = name
        self.header = header
        self.l_obs = l_obs
        self.f_obs = f_obs
        self.f_err = f_err
        self.f_flag = f_flag
        self.f_disp = f_disp
        self.good_frac = good_frac
        self.flux_unit = flux_unit
        self.redshift = redshift
        self.EBV = 0.0
        self.cov_matrix = cov_matrix
        self.vaccuum_wl = False
        self.inRestFrame = False
        self._lumDist_Mpc = None
        self._wcs = None

        if self.redshift <= 0.:
            log.warn('Your redshift is %.4f; this will cause problems when running Starlight.' % self.redshift)

        
    def addKeyword(self, key, value):
        key = self._hdr_prefix + ' ' + key
        self.header[key] = value

    @property
    def wcs(self):
        if self._wcs is None:
            self._wcs = wcs.WCS(self.header)
        return self._wcs

    @wcs.setter
    def wcs(self, value):
        self._wcs = value
    
    def updateHeaderWCS(self):
        write_WCS(self.header, self.wcs)
        
    @property
    def lumDist_Mpc(self):
        if self._lumDist_Mpc is None:
            self._lumDist_Mpc = redshift2lum_distance(self.redshift)
        return self._lumDist_Mpc
    
    @lumDist_Mpc.setter
    def lumDist_Mpc(self, value):
        self._lumDist_Mpc = value
        
    def deredden(self):
        if self.EBV == 0.0:
            log.debug('E(B-V) = 0, not dereddening spectra.')
            return
        log.info('Dereddening spectra, E(B-V) = %f.' % self.EBV)
        ext_corr = Galactic_reddening_corr(self.l_obs, self.EBV)[:, np.newaxis, np.newaxis]
        self.f_obs *= ext_corr
        self.f_err *= ext_corr
        
    def slice(self, x_slice, y_slice):
        log.info('Taking a slice of the cube...')
        self.f_obs = self.f_obs[:, y_slice, x_slice].copy()
        self.f_err = self.f_err[:, y_slice, x_slice].copy()
        self.f_flag = self.f_flag[:, y_slice, x_slice].copy()
        self.wcs = self.wcs[:, y_slice, x_slice]
        log.debug('New shape: %s.' % str(self.f_obs.shape))

    def bin(self, bin_size, cov_factor_A=0, cov_factor_B=1.0, cov_matrix=False):
        if not cov_matrix:
            cov_factor = get_cov_factor(bin_size**2, cov_factor_A, cov_factor_B)
            log.info('Binning cube (%d x %d), cov. factor=%.2f.' % (bin_size, bin_size, cov_factor))
            f_obs, f_err, good_frac = bin_spectra(self.f_obs, self.f_err, self.f_flag,
                                                  bin_size, cov_factor_A, cov_factor_B)
        else:
            log.info('Binning cube (%d x %d), cov. matrices.' % (bin_size, bin_size))
            f_obs, f_err, good_frac = bin_spectra(self.f_obs, self.f_err, self.f_flag,
                                                  bin_size, cov_matrix=cov_matrix)
        if self.f_disp is not None:
            # Get mean intrumental dispersion; ignore covariance.
            f_disp, _, _ =  bin_spectra(self.f_disp, self.f_err, self.f_flag,
                                        bin_size, cov_factor_A=0., cov_factor_B=1.)
            self.f_disp = f_disp / bin_size**2
        # This needs to be assigned *after* the f_disp binning.
        self.f_obs, self.f_err = f_obs, f_err
        self.good_frac = good_frac
        self.f_flag = np.where(good_frac == 0, flags.no_data, 0)
        self.wcs = scale_celestial_WCS(self.wcs, scaling=bin_size)

    def toRestFrame(self):
        if self.inRestFrame:
            log.debug('Already in rest frame (z=%.2f).' % self.redshift)
            return
        log.info('Putting spectra in rest frame (z=%.2f).' % self.redshift)
        z_plus_1 = 1.0 + self.redshift
        self.f_obs *= z_plus_1
        self.f_err *= z_plus_1
        # Note: do nothing to f_disp.
        self.l_obs /= z_plus_1
        self.inRestFrame = True

    def toAirWavelength(self):
        if not self.vaccuum_wl:
            log.debug('Already in air wavelengths.')
            return
        log.info('Converting vacuum to air wavelengths.')
        self.l_obs = vac2air(self.l_obs)
        self.vaccuum_wl = False

    def resample(self, l_ini, l_fin, dl, vectorized=False):
        log.info('Resampling spectra in [%.1f, %.1f], dl=%.2f \AA.' % (l_ini, l_fin, dl))
        l_resam = np.arange(l_ini, l_fin + dl, dl)
        badpix = (self.f_flag & flags.no_obs) > 0
        f_obs, f_err, f_flag = resample_spectra(self.l_obs, l_resam,
                                                self.f_obs, self.f_err, badpix,
                                                vectorized=vectorized)
        if self.good_frac is not None:
            good_frac, _, _ = resample_spectra(self.l_obs, l_resam,
                                               self.good_frac, self.f_err, badpix,
                                               vectorized=vectorized)
            self.good_frac = good_frac
        if self.f_disp is not None:
            f_disp, _, _ = resample_spectra(self.l_obs, l_resam,
                                            self.f_disp, self.f_err, badpix,
                                            vectorized=vectorized)
            self.f_disp = f_disp
        # This needs to be assigned *after* the f_disp binning.
        self.f_obs, self.f_err, self.f_flag = f_obs, f_err, f_flag
        self.l_obs = l_resam
        self.wcs = replace_wave_WCS(self.wcs, crpix_wave=0, crval_wave=l_resam[0], cdelt_wave=dl)

    def flagLowSN(self, llow_sn, lupp_sn, sn_min, convex_hull=False, clean_mask=False, largest=True, 
                  formal_sn=False, nmin=0):
        log.info('Masking pixels with S/N < %.1f (%.1f-%.1f AA) [Formal SN: %s]' % (sn_min, llow_sn, lupp_sn, formal_sn))
        if formal_sn:
            sn = _calc_formal_sn(self.l_obs, self.f_obs, self.f_err, self.f_flag, llow_sn, lupp_sn)
        else:
            sn = _calc_sn(self.l_obs, self.f_obs, self.f_flag, llow_sn, lupp_sn)
        high_sn = (sn > sn_min).filled(False)
        if clean_mask:
            log.info('Cleaning S/N mask.')
            high_sn = clean_mask_polygons(high_sn, largest=largest, nmin=nmin, masked_value=False)
        if convex_hull:
            log.info('Calculating convex hull of S/N mask.')
            high_sn = convex_hull_mask(high_sn)
        self.f_flag[:, ~high_sn] |= flags.low_sn


def preprocess_obs(obs, cfg, mask_file=None):
    cfg_import_sec = 'import'
    cfg_starlight_sec = 'starlight'
    
    obs.addKeyword('VERSION', __version__)
    obs.addKeyword('GITHASH', get_git_hash())

    obs.toAirWavelength()
    obs._wcs = wcs.WCS(obs.header)

    sl_string = cfg.get(cfg_import_sec, 'slice', fallback=None)
    sl = parse_slice(sl_string)
    if 'PYCASSO SLICE' not in obs.header and sl is not None:
        obs.addKeyword('SLICE', sl_string)
        y_slice, x_slice = sl
        obs.slice(x_slice, y_slice)

    if mask_file is not None:
        # Get mask
        log.info('Reading mask file %s' % mask_file)
        _hdu = fits.open(mask_file)
        m = _hdu['mask'].data
        obs.f_flag[:, (m == 1)] |= flags.overlapping_spaxel
        # Fix wcs
        obs._wcs.wcs.crval[:2] = wcs.WCS(_hdu[0].header).wcs.crval
        
    bin_size = cfg.getint(cfg_import_sec, 'binning', fallback=1)
    if bin_size > 1:
        A = cfg.getfloat(cfg_import_sec, 'spat_cov_a', fallback=0.0)
        B = cfg.getfloat(cfg_import_sec, 'spat_cov_b', fallback=1.0)
        cov_matrix = cfg.getboolean(cfg_import_sec, 'spat_cov_matrix', fallback=False)
        if cov_matrix:
            obs.bin(bin_size, A, B, cov_matrix=[obs.C, obs.flagC])
        else:
            obs.bin(bin_size, A, B)
        obs.addKeyword('SPAT_COV A', A)
        obs.addKeyword('SPAT_COV B', B)
        obs.addKeyword('SPAT_COV MATRIX', cov_matrix)
        obs.addKeyword('BIN SIZE', bin_size)
    
    obs.deredden()
    obs.toRestFrame()

    l_ini = cfg.getfloat(cfg_import_sec, 'l_ini')
    l_fin = cfg.getfloat(cfg_import_sec, 'l_fin')
    dl = cfg.getfloat(cfg_import_sec, 'dl')
    obs.resample(l_ini, l_fin, dl)

    l1 = cfg.getfloat(cfg_starlight_sec, 'llow_SN', fallback=obs.l_obs.min())
    l2 = cfg.getfloat(cfg_starlight_sec, 'lupp_SN', fallback=obs.l_obs.max())
    sn_min = cfg.getfloat(cfg_import_sec, 'SN_min', fallback=0.0)
    convex_hull = cfg.getboolean(cfg_import_sec, 'convex_hull_mask', fallback=False)
    clean_mask_sn = cfg.getboolean(cfg_import_sec, 'clean_mask_sn', fallback=False)
    sn_largest = cfg.getboolean(cfg_import_sec, 'sn_largest', fallback=True)
    sn_minsize = cfg.getint(cfg_import_sec, 'sn_minsize', fallback=2)
    formal_sn  = cfg.getboolean(cfg_import_sec, 'formal_sn', fallback=False)
    if sn_min > 0.0:
        obs.addKeyword('MASK SN_MIN', sn_min)
        obs.addKeyword('MASK LLOW', l1)
        obs.addKeyword('MASK LUPP', l2)
        obs.flagLowSN(l1, l2, sn_min, convex_hull=convex_hull, clean_mask=clean_mask_sn, 
                      largest=sn_largest, formal_sn=formal_sn, nmin=sn_minsize)
    
    obs.updateHeaderWCS()
    
    
def safe_getheader(f, ext=0):
    with fits.open(f) as hl:
        hdu = hl[ext]
        hdu.verify('fix')
        return hdu.header


def _calc_sn(l_obs, f_obs, f_flag, l1, l2):
    i1, i2 = find_nearest_index(l_obs, [l1, l2])
    l_obs = l_obs[i1:i2]
    f_obs = f_obs[i1:i2]
    f_flag = f_flag[i1:i2]
    bad = (f_flag & flags.no_obs) > 0
    f_obs = np.ma.masked_where(bad, f_obs)
    y = cube_continuum(l_obs, f_obs, degr=1, niterate=0)
    signal = y.mean(axis=0)
    noise = (f_obs - y).std(axis=0)
    return signal / noise

def _calc_formal_sn(l_obs, f_obs, f_err, f_flag, l1=None, l2=None):
    if l1 is None:
        l1 = l_obs.min()
    if l2 is None:
        l2 = l_obs.max()
    i1, i2 = find_nearest_index(l_obs, [l1, l2])
    l_obs = l_obs[i1:i2]
    f_obs = f_obs[i1:i2]
    f_err = f_err[i1:i2]
    f_flag = f_flag[i1:i2]
    bad = (f_flag & flags.no_obs) > 0
    f_obs = np.ma.masked_where(bad, f_obs)
    f_err = np.ma.masked_where(bad, f_err)
    return (f_obs / f_err).mean(axis=0)

def get_git_hash():
    module = sys.argv[0]
    return get_git_devstr(sha=True, path=module)
