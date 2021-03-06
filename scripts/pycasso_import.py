#!/usr/bin/env python
'''
Created on 15/06/2015

@author: andre

'''

from pycasso2.importer import read_type
from pycasso2.starlight.io import read_wavelength_mask
from pycasso2.config import get_config, default_config_path
import pycasso2.segmentation as seg
from pycasso2 import flags
from pycasso2 import FitsCube

from astropy import log
import argparse
import sys
from os import path
import numpy as np

###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description='Import data into a pycasso cube.')

    parser.add_argument('cubeIn', type=str, nargs='+',
                        help='Input cube. Ex.: T001.fits [other_cubes.fits ...]')
    parser.add_argument('--out', dest='cubeOut',
                        help='Output cube. Ex.: T001_pycasso.fits')
    parser.add_argument('--name', dest='name',
                        help='Object name. Ex.: NGC0123')
    parser.add_argument('--cube-type', dest='cubeType', default='diving3d',
                        help='Cube type. Ex.: diving3d, califa')
    parser.add_argument('--config', dest='configFile', default=default_config_path,
                        help='Config file. Default: %s' % default_config_path)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output.')
    parser.add_argument('--mask', dest='mask_file', default=None,
                        help='Fits file with spatial mask for overlapping spaxels. Default: None')

    return parser.parse_args()
###############################################################################

log.setLevel('DEBUG')
args = parse_args()
cfg = get_config(args.configFile)

if args.name is None:
    name = path.basename(args.cubeIn[0])
else:
    name = args.name

if args.cubeType not in read_type.keys():
    log.error('Unknown cube type %s' % args.cubeType)
    sys.exit()

g = FitsCube(args.cubeIn, cube_type=args.cubeType, name=name, mask_file=args.mask_file,
             import_cfg=cfg)

seg_type = cfg.get('import', 'segmentation', fallback=None)
if seg_type is not None:
    pa = cfg.getfloat('import', 'seg_pa', fallback=g.pa)
    ba = cfg.getfloat('import', 'seg_ba', fallback=g.ba)
    x0 = cfg.getfloat('import', 'seg_x0', fallback=g.x0)
    y0 = cfg.getfloat('import', 'seg_y0', fallback=g.y0)
    if seg_type == 'ring':
        step = cfg.getfloat('import', 'seg_step')
        log.info('Creating rings with step = %.1f pix.' % step)
        segmask = seg.ring_segmentation((g.Ny, g.Nx), x0, y0, pa, ba, step)
    elif seg_type == 'aperture':
        step = cfg.getfloat('import', 'seg_step')
        log.info('Creating apertures with step = %.1f pix.' % step)
        segmask = seg.aperture_segmentation((g.Ny, g.Nx), x0, y0, pa, ba, step)
    elif seg_type == 'voronoi':
        sn = cfg.getfloat('import', 'seg_target_sn')
        log.info('Creating voronoi zones with S/N = %.1f.' % sn)
        llow_SN = cfg.getfloat('starlight', 'llow_SN')
        lupp_SN = cfg.getfloat('starlight', 'lupp_SN')
        g.l_norm = (lupp_SN + llow_SN) / 2.0
        g.dl_norm = lupp_SN - llow_SN
        segmask = seg.voronoi_segmentation(g.flux_norm_window, g.noise_norm_window, sn)
    else:
        log.info('Loading segmentation from file %s.' % seg_type)
        segmask = seg.read_segmentation_map(seg_type)
    
   
    spatial_flags = flags.no_obs
    if cfg.getboolean('import', 'seg_low_sn', fallback=False):
        spatial_flags |= flags.low_sn
    threshold_flag = cfg.getfloat('import', 'threshold_flag', fallback=0.8)
    spatial_mask = g.getSpatialMask(spatial_flags, threshold=threshold_flag)
    segmask = seg.prune_segmask(segmask, spatial_mask)

    A = cfg.getfloat('import', 'spat_cov_a', fallback=0.0)
    B = cfg.getfloat('import', 'spat_cov_b', fallback=1.0)
    cov_matrix = cfg.getboolean('import', 'spat_cov_matrix', fallback=False)

    f_obs, f_err, good_frac = seg.sum_spectra(segmask, g.f_obs, g.f_err,
                                              cov_factor_A=A, cov_factor_B=B)
    # Get mean intrumental dispersion; ignore covariance.
    # FIXME: this calculates the sum, not the mean!
    f_disp, _, _ =  seg.sum_spectra(segmask, g.f_disp, g.f_err, good=np.ones_like(g.f_disp),
                                    cov_factor_A=0., cov_factor_B=1.)
    
    gs = FitsCube()
    gs._initFits(f_obs, f_err, None, g._header, g._wcs, segmask, good_frac, f_disp=f_disp)
    gs.name = g.name
    gs._header['HIERARCH PYCASSO SPAT_COV A'] = A
    gs._header['HIERARCH PYCASSO SPAT_COV B'] = B
    gs._header['HIERARCH PYCASSO SPAT_COV MATRIX'] = cov_matrix
    g = gs

telluric_mask_file = cfg.get('tables', 'telluric_mask', fallback=None)
if telluric_mask_file is not None:
    log.info('Applying telluric lines masks (z = %f)' % g.redshift)
    telluric_mask = read_wavelength_mask(
    telluric_mask_file, g.l_obs, g.redshift, dest='rest')
    g.f_flag[telluric_mask] |= flags.telluric
else:
    log.warn('Telluric mask not informed or not found.')

log.info('Saving cube %s.' % args.cubeOut)
g.write(args.cubeOut, overwrite=args.overwrite)

