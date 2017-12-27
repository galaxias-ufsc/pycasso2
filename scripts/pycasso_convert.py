'''
Created on 10/11/2017

@author: andre
'''

from pycasso2 import FitsCube, flags, __version__
from pycasso2.importer.core import get_git_hash
from pycasso2.wcs import replace_wave_WCS, write_WCS
from pycasso2.legacy import qplane_map, qzones2segmask

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

log.setLevel('DEBUG')
args = parse_args()

f = fits.open(args.cubeIn[0])
header = f[0].header

if args.name is None:
    name = header['CALIFA_ID']
else:
    name = args.name

qp_map = qplane_map(header)
w = WCS(header)
f_obs = f['f_obs'].data
f_err = f['f_err'].data
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
g.flux_unit = header['SYN FLUX_UNIT']
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
g.f_syn[...] = f['f_syn'].data
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
    g._header['HIERARCH STARLIGHT ' + k.upper()] = header['SYN ' + k.upper()]

g._HDUList.append(fits.ImageHDU(f[0].data, f[0].header, 'qPlanes'))

# Testing at_flux

popx = f['popx'].data
at_flux = (popx * np.log10(age_base[:, np.newaxis, np.newaxis])).sum(axis=(0, 1)) / popx.sum(axis=(0, 1))
assert(np.allclose(at_flux, g.at_flux))

log.info('Saving cube %s.' % args.cubeOut)
g.write(args.cubeOut, overwrite=args.overwrite)

#plt.imshow(g.spatialize(g.flux_norm_window, extensive=True))