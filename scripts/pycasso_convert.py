#!/usr/bin/env python
'''
Created on 10/11/2017

@author: andre
'''

from pycasso2 import FitsCube, flags, __version__
from pycasso2.importer.core import get_git_hash
from pycasso2.wcs import replace_wave_WCS, write_WCS

from astropy.io import fits
from astropy.wcs.wcs import WCS
from astropy import log
import ast
#import matplotlib.pyplot as plt
import numpy as np
import argparse

###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert old CALIFA cubes to new format.')

    parser.add_argument('cubeIn', type=str, nargs='+',
                        help='Input cube. Ex.: K0001.fits')
    parser.add_argument('--out', dest='cubeOut',
                        help='Output cube. Ex.: K0001_new.fits')
    parser.add_argument('--name', dest='name',
                        help='Object name. Ex.: NGC0123')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output.')

    return parser.parse_args()
###############################################################################


###############################################################################
def qplane_map(header):
    '''
    Create a dictionary of plane names to index in image.
    '''   
    nplanes = header['NAXIS3']
    plane_index = {}
    for pid in range(nplanes):
        key = 'PLANE_%d' % pid
        pname = header[key].split(':')[0]
        plane_index[pname] = pid
    return plane_index
###############################################################################


###############################################################################
def qzones2segmask(qZones):
    Nzone = int(qZones.max()) + 1
    segmask = np.zeros((Nzone,) + qZones.shape, dtype='int32')
    for z in range(Nzone):
        this_zone = (qZones == z)
        segmask[z, this_zone] = 1
    return segmask
###############################################################################


log.setLevel('DEBUG')
args = parse_args()

f = fits.open(args.cubeIn[0])
header = f[0].header

if args.name is None:
    name = header['CALIFA_ID']
else:
    name = args.name

flux_unit = header['SYN FLUX_UNIT']

qp_map = qplane_map(header)
w = WCS(header)
f_obs = f['f_obs'].data / flux_unit
f_err = f['f_err'].data / flux_unit
f_flag_orig = f['f_flag'].data
badpix = (f_flag_orig >= 2.0)
sky_lines = (f_flag_orig == 12.0)
f_flag = np.zeros(f_obs.shape, dtype='int32')
f_flag[badpix] = flags.no_obs
f_flag[sky_lines] = flags.telluric

qZones = f['primary'].data[qp_map['Zones']] - 1
segmask = qzones2segmask(qZones)
good_frac = np.ones_like(f_obs)

g = FitsCube()
header['HIERARCH PYCASSO ORIG_VERSION'] = header['VPYCASSO']
del(header['VPYCASSO'])
header['HIERARCH PYCASSO VERSION'] = __version__
header['HIERARCH PYCASSO GITHASH'] = get_git_hash()

l_ini = header['SYN L_INI']
dl = header['SYN DL']
w = replace_wave_WCS(w, crpix_wave=0, crval_wave=l_ini, cdelt_wave=dl)
write_WCS(header, w)

g._initFits(f_obs, f_err, f_flag, header, w, segmask, good_frac)
g.flux_unit = flux_unit
g.lumDistMpc = header['D_MPC']
g.redshift = header['REDSHIFT']
g.name = name

age_base = np.array(ast.literal_eval(header['SYN_AGEB']), dtype=float)
met_base = np.array(ast.literal_eval(header['SYN_METB']), dtype=float)

popage_base = np.zeros((len(age_base), len(met_base)))
popZ_base = np.zeros((len(age_base), len(met_base)))
popage_base[...] = age_base[:, np.newaxis]
popZ_base[...] = met_base[np.newaxis, :]

pop_len = len(age_base) * len(met_base)
Nz = segmask.shape[0]

g.createSynthesisCubes(pop_len)
g.f_syn[...] = f['f_syn'].data / flux_unit
g.f_wei[...] = f['f_wei'].data
g.f_flag |= np.where(g.f_wei == -1.0, flags.starlight_clipped, 0)
g.f_flag |= np.where(g.f_wei == 0.0, flags.starlight_masked, 0)

g.popage_base[...] = popage_base.T.ravel()
g.popage_base_t2[...] = popage_base.T.ravel()
g.popZ_base[...] = popZ_base.T.ravel()
g.Mstars[...] = f['mstars'].data.T.ravel()
g.fbase_norm[...] = f['fbase_norm'].data.T.ravel()

g.popx[...] = np.transpose(f['popx'].data, (1, 0, 2)).reshape(-1, Nz)
g.popmu_ini[...] = np.transpose(f['popmu_ini'].data, (1, 0, 2)).reshape(-1, Nz)
g.popmu_cor[...] = np.transpose(f['popmu_cor'].data, (1, 0, 2)).reshape(-1, Nz)

g.Lobs_norm[...] = f['lobs_norm'].data
g.Mini_tot[...] = f['mini_tot'].data
g.Mcor_tot[...] = f['mcor_tot'].data
g._HDUList['fobs_norm'].data[...] = f['fobs_norm'].data
g.A_V[...] = f['A_V'].data
g.v_0[...] = f['v_0'].data
g.v_d[...] = f['v_d'].data
g.adev[...] = f['adev'].data
g._HDUList['ntot_clipped'].data[...] = f['ntot_clipped'].data
g._HDUList['nglobal_steps'].data[...] = f['nglobal_steps'].data
g.chi2[...] = f['chi2'].data

sn_image = f['primary'].data[qp_map['ZonesSn']]
sn_zones = (sn_image * segmask).sum(axis=(1, 2)) / segmask.sum(axis=(1, 2))
g.SN_normwin[...] = sn_zones

syn_keyword_list = ['arq_config', 'N_chains', 'l_norm', 'q_norm',
                    'llow_norm', 'lupp_norm', 'i_SaveBestSingleCompFit', 'IsFIRcOn' ,
                    'IsPHOcOn', 'IsQHRcOn' , 'llow_SN', 'lupp_SN', 'q_norm',
                    'red_law_option', 'flux_unit' , 'l_ini', 'l_fin',
                    'dl', 'Nl_obs', 'arq_base', 'N_base', 'N_exAV', 'LumDistInMpc']
for k in syn_keyword_list:
    g._header['HIERARCH STARLIGHT ' + k] = header['SYN ' + k.upper()]

t = g._getTableExtensionData(g._ext_integ_spectra)
t['f_obs'] = f['integrated_f_obs'].data / flux_unit
t['f_err'] = f['integrated_f_err'].data / flux_unit
t['f_syn'] = f['integrated_f_syn'].data / flux_unit
t['f_wei'] = f['integrated_f_wei'].data
t['f_flag'] = np.where(f['integrated_f_flag'].data > 1.0, flags.no_data, 0)
t['f_flag'] |= np.where(g.integ_f_wei == -1.0, flags.starlight_clipped, 0)
t['f_flag'] |= np.where(g.integ_f_wei == 0.0, flags.starlight_masked, 0)

g.integ_popx[:] = f['integrated_popx'].data.T.ravel()
g.integ_popmu_ini[:] = f['integrated_popmu_ini'].data.T.ravel()
g.integ_popmu_cor[:] = f['integrated_popmu_cor'].data.T.ravel()

 # NOTE: The last one should be SN_NORMWIN, but does not exist in legacy cubes.
 # We hackishly load another header card, then replace with zero.
leg_keyword_list = ['Lobs_norm', 'Mini_tot', 'Mcor_tot', 'fobs_norm',
                    'A_v', 'v_0', 'v_d', 'adev', 'Ntot_clipped',
                    'Nglobal_steps', 'chi2', 'chi2']
integ_header = g._HDUList[g._ext_integ_pop].header
for kd, ko in zip(g._ext_keyword_list, leg_keyword_list):
    integ_header['HIERARCH STARLIGHT ' + kd] = header['SYN INTEG ' + ko.upper()]
integ_header['HIERARCH STARLIGHT SN_NORMWIN'] = 0.0
integ_header['HIERARCH STARLIGHT A_V'] = integ_header['HIERARCH STARLIGHT AV']
integ_header['HIERARCH STARLIGHT V_0'] = integ_header['HIERARCH STARLIGHT V0']
integ_header['HIERARCH STARLIGHT V_D'] = integ_header['HIERARCH STARLIGHT VD']
g._HDUList.append(fits.ImageHDU(f[0].data, f[0].header, 'qPlanes'))

# Testing at_flux

popx = f['popx'].data
at_flux = (popx * np.log10(age_base[:, np.newaxis, np.newaxis])).sum(axis=(0, 1)) / popx.sum(axis=(0, 1))
assert(np.allclose(at_flux, g.at_flux))

log.info('Saving cube %s.' % args.cubeOut)
g.write(args.cubeOut, overwrite=args.overwrite)

#plt.imshow(g.spatialize(g.flux_norm_window, extensive=True))