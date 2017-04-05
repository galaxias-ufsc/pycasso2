'''
Created on 22 de mar de 2017

@author: andre
'''

from pycasso2.config import default_config_path
from pycasso2 import FitsCube, flags
import pycasso2.segmentation as seg

from astropy import log
import argparse

###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Create segmented cube from pycasso cube.')

    parser.add_argument('cubeIn', type=str, nargs=1,
                        help='Cube. Ex.: T001.fits')
    parser.add_argument('--out', dest='cubeOut', required=True,
                        help='Output cube.')
    parser.add_argument('--rename', dest='newName',
                        help='Rename the output cube.')
    parser.add_argument('--config', dest='configFile', default=default_config_path,
                        help='Config file. Default: %s' % default_config_path)
    parser.add_argument('--config-section', dest='configSection', default='starlight',
                        help='Config section with starlight settings. Default: starlight')
    parser.add_argument('--seg', dest='seg',
                        help='Custom segmentation file or built-in segmentation (mosaic | ring | aperture | voronoi).')
    parser.add_argument('--npix', dest='npix', type=float,
                        help='Scale of segmentation. Zone width for mosaic, radial ' \
                        'step for ring and aperture.')
    parser.add_argument('--sn', dest='sn', type=float, default=20,
                        help='Target S/N when using Voronoi segmentation. Default: 20')
    parser.add_argument('--pa', dest='pa', type=float, default=None,
                        help='Position angle in degrees. Only used for ring and aperture. '\
                        'Default: calculate from cube.')
    parser.add_argument('--ba', dest='ba', type=float, default=None,
                        help='b/a fraction. Only used for ring and aperture. '\
                        'Default: Default: calculate from cube.')
    parser.add_argument('--x0', dest='x0', type=float, default=None,
                        help='x coordinate of the center. '\
                        'Only used for ring and aperture. Default: calculate from cube.')
    parser.add_argument('--y0', dest='y0', type=float, default=None,
                        help='y coordinate of the center. Only used for ring and aperture. '\
                        'Default: Default: calculate from cube.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output.')

    return parser.parse_args()
###############################################################################

args = parse_args()

c = FitsCube(args.cubeIn[0])
pa = c.pa if args.pa is None else args.pa
ba = c.ba if args.ba is None else args.ba
x0 = c.x0 if args.x0 is None else args.x0
y0 = c.y0 if args.y0 is None else args.y0
if args.seg == 'mosaic':
    npix = int(args.npix)
    log.info('Creating mosaic with width = %d pix.' % npix)
    segmask = seg.mosaic_segmentation((c.Ny, c.Nx), bin_size=npix)
elif args.seg == 'ring':
    log.info('Creating rings with step = %.1f pix.' % args.npix)
    segmask = seg.ring_segmentation((c.Ny, c.Nx), x0, y0, pa, ba)
elif args.seg == 'aperture':
    log.info('Creating apertures with step = %.1f pix.' % args.npix)
    segmask = seg.aperture_segmentation((c.Ny, c.Nx), c.x0, c.y0, c.pa, c.ba)
elif args.seg == 'voronoi':
    log.info('Creating voronoi zones with S/N = %.1f.' % args.sn)
    segmask = seg.voronoi_segmentation(c.flux_norm_window, c.noise_norm_window, args.sn)
else:
    log.info('Loading segmentation from file %s.' % args.seg)
    segmask = seg.read_segmentation_map(args.seg)

spatial_mask = c.getSpatialMask(flags.no_obs)
segmask = seg.prune_segmask(segmask, spatial_mask)

f_obs, f_err, f_flag = seg.sum_spectra(segmask, c.f_obs, c.f_err, c.f_flag)

cz = FitsCube()
cz._initFits(f_obs, f_err, f_flag, c._header, c._wcs, segmask)
if args.newName is None:
    cz.name = c.name
else:
    log.debug('Renamed cube to %s' % args.newName)
    cz.name = args.newName
print 'Saving cube to %s.' % args.cubeOut
cz.write(args.cubeOut, overwrite=args.overwrite)
