'''
Created on 15/06/2015

@author: andre

'''

from pycasso2.importer import read_type
from pycasso2.starlight.tables import read_wavelength_mask
from pycasso2.config import get_config, default_config_path, parse_slice
import pycasso2.segmentation as seg
from pycasso2 import flags
from pycasso2 import FitsCube

from astropy import log
import argparse
import sys
from os import path

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
    parser.add_argument('--slice', dest='slice', default=None,
                        help='Import only a slice of the cube. ' \
                        'Example: y1:y2,x1:x2. Default: full cube.')
    parser.add_argument('--seg', dest='seg', default=None,
                        help='If specified, segment the cube using a custom ' \
                        'segmentation file or built-in segmentation ' \
                        '(mosaic | ring | aperture | voronoi).')
    parser.add_argument('--seg-npix', dest='npix', type=float,
                        help='Scale of segmentation. Zone width for mosaic, radial ' \
                        'step for ring and aperture.')
    parser.add_argument('--seg-sn', dest='sn', type=float, default=20,
                        help='Target S/N when using Voronoi segmentation. Default: 20')
    parser.add_argument('--seg-pa', dest='pa', type=float, default=None,
                        help='Position angle in degrees. Only used for ring and aperture. '\
                        'Default: calculate from cube.')
    parser.add_argument('--seg-ba', dest='ba', type=float, default=None,
                        help='b/a fraction. Only used for ring and aperture. '\
                        'Default: Default: calculate from cube.')
    parser.add_argument('--seg-x0', dest='x0', type=float, default=None,
                        help='x coordinate of the center. '\
                        'Only used for ring and aperture. Default: calculate from cube.')
    parser.add_argument('--seg-y0', dest='y0', type=float, default=None,
                        help='y coordinate of the center. Only used for ring and aperture. '\
                        'Default: Default: calculate from cube.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output.')

    return parser.parse_args()
###############################################################################

log.setLevel('DEBUG')
args = parse_args()
cfg = get_config(args.configFile)

if args.name is None:
    name = path.basename(args.cubeIn[0])
else:
    name = args.name

try:
    sl = parse_slice(args.slice)
except:
    log.error('Error reading or bad slice definition: %s' % sl)
    sys.exit()

if args.cubeType not in read_type.keys():
    log.error('Unknown cube type %s' % args.cubeType)
    sys.exit()

g = FitsCube(args.cubeIn, cube_type=args.cubeType, name=name,
             import_cfg=cfg, import_slice=sl)

if args.seg:
    pa = g.pa if args.pa is None else args.pa
    ba = g.ba if args.ba is None else args.ba
    x0 = g.x0 if args.x0 is None else args.x0
    y0 = g.y0 if args.y0 is None else args.y0
    if args.seg == 'mosaic':
        npix = int(args.npix)
        log.info('Creating mosaic with width = %d pix.' % npix)
        segmask = seg.mosaic_segmentation((g.Ny, g.Nx), bin_size=npix)
    elif args.seg == 'ring':
        log.info('Creating rings with step = %.1f pix.' % args.npix)
        segmask = seg.ring_segmentation((g.Ny, g.Nx), x0, y0, pa, ba)
    elif args.seg == 'aperture':
        log.info('Creating apertures with step = %.1f pix.' % args.npix)
        segmask = seg.aperture_segmentation((g.Ny, g.Nx), x0, y0, pa, ba)
    elif args.seg == 'voronoi':
        log.info('Creating voronoi zones with S/N = %.1f.' % args.sn)
        segmask = seg.voronoi_segmentation(g.flux_norm_window, g.noise_norm_window, args.sn)
    else:
        log.info('Loading segmentation from file %s.' % args.seg)
        segmask = seg.read_segmentation_map(args.seg)
    
    spatial_mask = g.getSpatialMask(flags.no_obs)
    segmask = seg.prune_segmask(segmask, spatial_mask)
    
    f_obs, f_err, good_frac = seg.sum_spectra(segmask, g.f_obs, g.f_err, g.f_flag)
    gs = FitsCube()
    gs._initFits(f_obs, f_err, None, g._header, g._wcs, segmask, good_frac)
    gs.name = g.name
    g = gs

log.info('Applying telluric lines masks (z = %f)' % g.redshift)
telluric_mask_file = cfg.get('tables', 'telluric_mask')
telluric_mask = read_wavelength_mask(
    telluric_mask_file, g.l_obs, g.redshift, dest='rest')
g.f_flag[telluric_mask] |= flags.telluric

log.info('Saving cube %s.' % args.cubeOut)
g.write(args.cubeOut, overwrite=args.overwrite)

