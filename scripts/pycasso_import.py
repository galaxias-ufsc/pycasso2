'''
Created on 15/06/2015

@author: andre

'''

from pycasso2.importer import read_califa, read_diving3d, read_manga, read_muse, read_gmos
from pycasso2.starlight.tables import read_wavelength_mask
from pycasso2 import flags
from pycasso2.config import get_config, default_config_path

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

def parse_slice(sl):
    if sl is None:
        return None
    try:
        yy, xx = sl.split(',')
        y1, y2 = yy.split(':')
        y1 = int(y1)
        y2 = int(y2)
        x1, x2 = xx.split(':')
        x1 = int(x1)
        x2 = int(x2)
        assert(x1 < x2)
        assert(y1 < y2)
    except:
        log.error('Error reading or bad slice definition: %s' % sl)
        sys.exit()
    return slice(y1, y2, 1), slice(x1, x2, 1)

log.setLevel('DEBUG')
args = parse_args()
cfg = get_config(args.configFile)

if args.name is None:
    name = path.basename(args.cubeIn[0])
else:
    name = args.name

sl = parse_slice(args.slice)

if args.cubeType == 'diving3d':
    g = read_diving3d(args.cubeIn[0], args.cubeIn[1], name, cfg, sl)

elif args.cubeType == 'califa':
    g = read_califa(args.cubeIn[0], name, cfg, sl)

elif args.cubeType == 'manga':
    g = read_manga(args.cubeIn[0], name, cfg, sl)

elif args.cubeType == 'muse':
    g = read_muse(args.cubeIn[0], name, cfg, sl)
    
elif args.cubeType == 'gmos':
    g = read_gmos(args.cubeIn[0], name, cfg, sl)
    
else:
    log.error('Unknown cube type %s' % args.cubeType)
    sys.exit()


print(('Applying telluric lines masks (z = %f)' % g.redshift))
telluric_mask_file = cfg.get('tables', 'telluric_mask')
telluric_mask = read_wavelength_mask(
    telluric_mask_file, g.l_obs, g.redshift, dest='rest')
g.f_flag[telluric_mask] |= flags.telluric

print(('Saving cube %s.' % args.cubeOut))
g.write(args.cubeOut, overwrite=args.overwrite)

