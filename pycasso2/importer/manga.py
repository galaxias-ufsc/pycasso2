'''
Created on 08/12/2015

@author: andre
'''
from ..cosmology import redshift2lum_distance
from .core import safe_getheader, fill_cube, import_spectra
from astropy import log, wcs
from astropy.io import fits
import numpy as np


__all__ = ['read_manga', 'read_drpall']

manga_cfg_sec = 'manga'

CRITICAL_BIT = 1 << 30


def read_drpall(filename, mangaid=None):
    with fits.open(filename) as f:
        t = f[1].data
    if mangaid is not None:
        i = np.where(t['mangaid'] == mangaid)[0]
        t = t[i]
    return t


def read_manga(cube, name, cfg, destcube=None):
    '''
    FIXME: doc me! 
    '''
    if len(cube) != 1:
        raise Exception('Please specify a single cube.')
    cube = cube[0]

    flux_unit = cfg.getfloat(manga_cfg_sec, 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube, ext='FLUX')
    w = wcs.WCS(header)
    
    drp = read_drpall(cfg.get(manga_cfg_sec, 'drpall'), header['MANGAID'])
    z = np.asscalar(drp['nsa_z'])

    if header['DRP3QUAL'] & CRITICAL_BIT:
        log.warn('Critical bit set. There are problems with this cube.')

    log.debug('Loading data from %s.' % cube)
    with fits.open(cube) as f:
        f_obs_orig = f['FLUX'].data
        # FIXME: Check mask bits.
        badpix = f['MASK'].data > 0
        goodpix = ~badpix
        f_err_orig = np.zeros_like(f_obs_orig)
        f_err_orig[goodpix] = f['IVAR'].data[goodpix]**-0.5
        l_obs = f['WAVE'].data

    EBV = header['EBVGAL']
    l_obs, f_obs, f_err, f_flag, w, _ = import_spectra(l_obs, f_obs_orig,
                                                       f_err_orig, badpix,
                                                       cfg, manga_cfg_sec,
                                                       w, z, vaccuum_wl=True,
                                                       EBV=EBV)

    destcube = fill_cube(f_obs, f_err, f_flag, header, w,
                         flux_unit, redshift2lum_distance(z), z, name, cube=destcube)
    return destcube


def get_bitmask_indices(bitmask):
    if bitmask == 0:
        return 0
    true_indices = []
    binary = bin(bitmask)[:1:-1]
    for x in range(len(binary)):
        if int(binary[x]):
            true_indices.append(x)
    return np.array(true_indices)


def bitmask2string(targ1, targ2, targ3):
    bits = {

        'targ1': np.array(['NONE', 'PRIMARY_PLUS_COM', 'SECONDARY_COM',
                           'COLOR_ENHANCED_COM', 'PRIMARY_v1_1_0', 'SECONDARY_v1_1_0',
                           'COLOR_ENHANCED_v1_1_0', 'PRIMARY_COM2', 'SECONDARY_COM2',
                           'COLOR_ENHANCED_COM2', 'PRIMARY_v1_2_0', 'SECONDARY_v1_2_0',
                           'COLOR_ENHANCED_v1_2_0', 'FILLER', 'RETIRED']),

        'targ2': np.array(['NONE', 'SKY', 'STELLIB_SDSS_COM', 'STELLIB_2MASS_COM', 'STELLIB_KNOWN_COM', 'STELLIB_COM_mar2015', 'STELLIB_COM_jun2015', 'STELLIB_PS1', 'STELLIB_APASS', 'STELLIB_PHOTO_COM', 'STELLIB_aug2015', 'STD_FSTAR_COM', 'STD_WD_COM', 'STD_STD_COM', 'STD_FSTAR', 'STD_WD', 'STD_APASS_COM', 'STD_PS1_COM']),

        'targ3': np.array(['NONE', 'AGN_BAT', 'AGN_OIII', 'AGN_WISE', 'AGN_PALOMAR', 'VOID', 'EDGE_ON_WINDS', 'PAIR_ENLARGE', 'PAIR_RECENTER', 'PAIR_SIM', 'PAIR_2IFU', 'LETTERS', 'MASSIVE', 'MWA', 'DWARF', 'RADIO_JETS', 'DISKMASS', 'BCG', 'ANGST', 'DEEP_COMA'])

    }

    targ1_bits = bits['targ1'][get_bitmask_indices(targ1)]
    targ2_bits = bits['targ2'][get_bitmask_indices(targ2)]
    targ3_bits = bits['targ3'][get_bitmask_indices(targ3)]

    return np.hstack((targ1_bits, targ2_bits, targ3_bits))


def isgalaxy(targ1, targ3):
    return (targ1 > 0) | (targ3 > 0)


def isprimary(targ1):
    return (targ1 & 1024) > 0


def issecondary(targ1):
    return (targ1 & 2048) > 0


def iscolorenhanced(targ1):
    return (targ1 & 4096) > 0


def isprimaryplus(targ1):
    return (targ1 & (1024 | 4096)) > 0


def isancillary(targ3):
    return (targ3 > 0)
