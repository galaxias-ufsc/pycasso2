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
from astropy.table import Table
import numpy as np
from os import path

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
    f_obs[badpix] = 0.0
    f_err[badpix] = 0.0
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


################################################################################
morph_bin_labels = np.array(['E0', 'S0', 'Sa', 'Sb', 'Sbc', 'Sc', 'Sd'])
morph_bins = np.arange(len(morph_bin_labels))
mtype2num = dict(zip(morph_bin_labels, morph_bins))
mtype2num_califa = {
                    'E0': 0,
                    'E1': 1,
                    'E2': 2,
                    'E3': 3,
                    'E4': 4,
                    'E5': 5,
                    'E6': 6,
                    'E7': 7,

                    'S0': 8,
                    'S0a': 9,
                    'Sa': 10,
                    'Sab': 11,
                    'Sb': 12,
                    'Sbc': 13,
                    'Sc': 14,
                    'Scd': 15,
                    'Sd': 16,
                    'Sdm': 17,
                    'Sm': 18,
                    'I': 19,
}
num2mtype_califa = dict(zip(mtype2num_califa.values(), mtype2num_califa.keys()))


def califa_read_main_sample(dr3_table):
    dr3_sample = fits.getdata(dr3_table)
    combo_main = (dr3_sample['flag_release_comb'] == 1) & (dr3_sample['califaid'] < 1000) \
        & (dr3_sample['califaid'] != 788)# & (dr3_sample['califaid'] != 799)
    sample = dr3_sample[combo_main]
    return sample


def califa_read_morph(morph_table, cubes=None):
    mc = fits.getdata(morph_table)
    # Change to our morphology classification scheme.
    bins = np.array([mtype2num_califa[t] for t in morph_bin_labels])
    mc.hubtyp = np.digitize(mc.hubtyp, bins=bins, right=False) - 1
    
    if cubes is None:
        return mc
    
    obs_califa_str = np.array([califa_id_from_cube(f) for f in cubes])
    obs_califa_id = np.array([califa_id_to_int(c_id) for c_id in obs_califa_str])
    # Make the CALIFA IDs into indices.
    obs_keys = obs_califa_id - 1
    
    return mc[obs_keys]


def califa_id_from_cube(f):
    '''
    Return the CALIFA ID (as string) from a PyCASSO datacube filename.
    '''
    base = path.basename(f)
    return base[0:5]


def califa_id_to_str(califa_id):
    '''
    Convert the CALIFA ID from integer to string (KNNNN).
    '''
    return 'K%04d' % califa_id


def califa_id_to_int(califa_id):
    '''
    Convert the CALIFA ID from string (KNNNN) to int.
    '''
    return int(califa_id[1:])

