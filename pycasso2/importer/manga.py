'''
Created on 08/12/2015

@author: andre
'''
from .. import flags
from .core import safe_getheader, ObservedCube

from astropy import log
from astropy.io import fits
import numpy as np


__all__ = ['read_manga', 'read_drpall']

CRITICAL_BIT = 1 << 30


def read_drpall(filename, plateifu=None):
    with fits.open(filename) as f:
        t = f[1].data
    if plateifu is not None:
        i = np.where(t['plateifu'] == plateifu)[0]
        t = t[i]
    return t


def read_manga(cube, name, cfg):
    '''
    FIXME: doc me! 
    '''
    if len(cube) != 1:
        raise Exception('Please specify a single cube.')
    cube = cube[0]
    
    flux_unit = cfg.getfloat('import', 'flux_unit')

    # FIXME: sanitize file I/O
    log.debug('Loading header from cube %s.' % cube)
    header = safe_getheader(cube, ext='FLUX')

    drp = read_drpall(cfg.get('tables', 'master_table'), header['PLATEIFU'])
    z = np.asscalar(drp['nsa_z'])

    if header['DRP3QUAL'] & CRITICAL_BIT:
        log.warn('Critical bit set. There are problems with this cube.')

    log.debug('Loading data from %s.' % cube)
    with fits.open(cube) as f:
        f_obs = f['FLUX'].data
        # FIXME: Check mask bits.
        badpix = f['MASK'].data > 0
        goodpix = ~badpix
        f_err = np.zeros_like(f_obs)
        f_err[goodpix] = f['IVAR'].data[goodpix]**-0.5
        f_flag = np.where(badpix, flags.no_data, 0)
        l_obs = f['WAVE'].data

    obs = ObservedCube(name, l_obs, f_obs, f_err, f_flag, flux_unit, z, header)
    obs.EBV = header['EBVGAL']
    obs.vaccuum_wl = True
    return obs


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
