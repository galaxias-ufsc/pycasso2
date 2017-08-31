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
from astropy.table import Table, Column
import numpy as np

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

    med_vel = float(header['MED_VEL'])
    z = velocity2redshift(med_vel)

    log.debug('Loading data from %s.' % cube)
    f_obs = fits.getdata(cube, extname='PRIMARY')
    f_err = fits.getdata(cube, extname='ERROR')
    badpix = fits.getdata(cube, extname='BADPIX') != 0
    f_flag = np.where(badpix, flags.no_data, 0)

    masterlist = cfg.get('tables', 'master_table')
    galaxy_id = name
    log.debug('Loading masterlist for %s: %s.' % (galaxy_id, masterlist))
    ml = califa_read_masterlist(masterlist, galaxy_id)
    
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
    ml = Table()
    rec = np.genfromtxt(filename, dtype=_dtype, skip_header=1, delimiter=',')
    for name in rec.dtype.names:
        c = Column(rec[name], name=name)
        ml.add_column(c)

    if galaxy_id is not None:
        index = np.where(ml['CALIFA_ID'] == galaxy_id)[0]
        if len(index) == 0:
            raise Exception('Entry %s not found in masterlist %s.' % (galaxy_id, filename))
        return ml[index][0]
    else:
        return ml


_dtype = np.dtype([
         ('CALIFA_ID', 'U20'),
         ('ned_name', 'U20'),
         ('ppak_file', 'U40'),
         ('ra', np.float64),
         ('dec', np.float64),
         ('b', np.float64),
         ('l', np.float64),
         ('d_Mpc', np.float64),
         ('modz', np.float64),
         ('INFOmodz', 'U1'),
         ('IRSA_E(B-V)', np.float64),
         ('e_IRSA_E(B-V)', np.float64),
         ('Mr', np.float64),
         ('u-r', np.float64),
         ('objID', np.int64),
         ('run', np.int),
         ('rerun', np.int),
         ('camcol', np.int),
         ('field', np.int),
         ('obj', np.int),
         ('type', np.int),
         ('flags', np.int64),
         ('fiberMag_r', np.float64),
         ('petroMag_u', np.float64),
         ('petroMag_g', np.float64),
         ('petroMag_r', np.float64),
         ('petroMag_i', np.float64),
         ('petroMag_z', np.float64),
         ('extinction_u', np.float64),
         ('extinction_g', np.float64),
         ('extinction_r', np.float64),
         ('extinction_i', np.float64),
         ('extinction_z', np.float64),
         ('petroRad_r', np.float64),
         ('petroR50_r', np.float64),
         ('petroR90_r', np.float64),
         ('isoA_g', np.float64),
         ('isoB_g', np.float64),
         ('isoA_r', np.float64),
         ('isoB_r', np.float64),
         ('isoAgrad_r', np.float64),
         ('isoBgrad_r', np.float64),
         ('specObjID', np.int64),
         ('z', np.float64),
         ('zErr', np.float64),
         ('zConf', np.float64),
         ('zStatus', np.int),
         ('specClass', np.int),
         ('velDisp', np.float64),
         ('velDispErr', np.float64),
         ('eClass', np.float64),
         ('J', np.float64),
         ('eJ', np.float64),
         ('H', np.float64),
         ('eH', np.float64),
         ('K', np.float64),
         ('eK', np.float64),
         ('FUV', np.float64),
         ('eFUV', np.float64),
         ('NUV', np.float64),
         ('eNUV', np.float64),
         ('Fnu_12', np.float64),
         ('Fnu_25', np.float64),
         ('Fnu_60', np.float64),
         ('Fnu_100', np.float64),
         ('NEDNAME', 'U20'),
         ('CHANDRA_flux_aper_b', np.float64),
         ('CHANDRA_hard_hs', np.float64),
         ('mjd', np.int),
         ('plate', np.int),
         ('fiberid', np.int),
         ('FIRST_flux_int', np.float64),
         ('FIRST_e_flux_int', np.float64),
         ('FIRST_MajAx', np.float64),
         ('FIRST_MinAx', np.float64),
         ('FIRST_PosAng', np.float64),
         ('abs_U', np.float64),
         ('abs_G', np.float64),
         ('abs_R', np.float64),
         ('abs_I', np.float64),
         ('abs_Z', np.float64),
         ('abs_J', np.float64),
         ('abs_H', np.float64),
         ('abs_K', np.float64),
         ('eabs_U', np.float64),
         ('eabs_G', np.float64),
         ('eabs_R', np.float64),
         ('eabs_I', np.float64),
         ('eabs_Z', np.float64),
         ('eabs_J', np.float64),
         ('eabs_H', np.float64),
         ('eabs_K', np.float64),
         ('MU50R', np.float64),
         ('PETROTHETA', np.float64),
         ('PETROTH50', np.float64),
         ('PETROTH90', np.float64),
         ('SERSIC_N', np.float64),
         ('SERSIC_TH50', np.float64),
         ('SERSIC_FLUX', np.float64),
         ('CLASS', 'U20'),
         ('SUBCLASS', 'U20'),
         ('int_r_mag', np.float64),
         ('_PA', np.float64),       # Original name: 'PA'
         ('_PA_align', np.float64), # Original name: 'PA_align'
         ('_PA_err', np.float64),   # Original name: 'PA_err'
         ('_BA', np.float64),       # Original name: 'BA'
         ('_BA_err', np.float64),   # Original name: 'BA_err'
         ('re', np.float64),
         ('r90', np.float64),
         ('r20', np.float64),
         ('u_mag', np.float64),
         ('g_mag', np.float64),
         ('r_mag', np.float64),
         ('i_mag', np.float64),
         ('z_mag', np.float64),
         ('el_r_hlr', np.int),
         ('circ_r_mag', np.float64),
         ('pa', np.float64),
         ('pa_align', np.float64),
         ('ba', np.float64),
         ('_flags', np.int),    # Original name: 'flags'
         ('CMr', np.float64),
         ('Cu-r', np.float64)
])

