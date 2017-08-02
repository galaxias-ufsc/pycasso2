'''
Created on 15/06/2015

@author: andre

'''

from pycasso2.importer import read_type
from pycasso2.starlight.tables import read_wavelength_mask
from pycasso2.config import get_config, default_config_path, parse_slice
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
                        help='Import only a slice of the cube. Example: y1:y2,x1:x2. Default: full cube.')
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

log.info('Applying telluric lines masks (z = %f)' % g.redshift)
telluric_mask_file = cfg.get('tables', 'telluric_mask')
telluric_mask = read_wavelength_mask(
    telluric_mask_file, g.l_obs, g.redshift, dest='rest')
g.f_flag[telluric_mask] |= flags.telluric

log.info('Saving cube %s.' % args.cubeOut)
g.write(args.cubeOut, overwrite=args.overwrite)

