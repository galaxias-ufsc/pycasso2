'''
Created on 24/06/2015

@author: andre
'''

from diving3d.starlight import SynthesisAdapter
from diving3d.config import default_config_path, get_config
from pystarlight.util import starlight_runner as sr

import argparse
from os import path
from multiprocessing import cpu_count
from astropy import log


###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='Run starlight for a Diving3D cube.')
    
    parser.add_argument('galaxyId', type=str, nargs=1,
                        help='Cube. Ex.: T001')
    parser.add_argument('--config', dest='configFile', default=default_config_path,
                        help='Config file. Default: %s' % default_config_path)
    parser.add_argument('--nproc', dest='nproc', type=int, default=cpu_count()-1,
                        help='Number of worker processes.')
    parser.add_argument('--chunk-size', dest='chunkSize', type=int, default=5,
                        help='Grid chunk size, defaults to the same as --nproc.')
    parser.add_argument('--timeout', dest='timeout', type=int, default=30,
                        help='Timeout of starlight processes, in minutes.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output.')
    parser.add_argument('--use-error-flags', dest='useErrorsFlags', action='store_true',
                        help='Use errors and flags in this run.')
    parser.add_argument('--update-errors', dest='updateErrors', action='store_true',
                        help='Calculate errors from residual and update f_err extension.')
    parser.add_argument('--error-smooth-fwhm', dest='errorSmoothFwhm', type=float, default=15.0,
                        help='FWHM (in Angstroms) of the gaussian used to smooth and rectify the residual before estimating the error.')
    parser.add_argument('--error-box-width', dest='errorBoxWidth', type=float, default=100.0,
                        help='Running box width (in Angstroms) used to calculate the RMS of the residual, to estimate the error.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Debug mode. Fake starlight run.')

    return parser.parse_args()
###############################################################################

def get_pop_len(grids):
    for g in grids:
        if len(g.completed) == 0: continue
        ts = g.getTables()[0][2]
        return len(ts.population.popx)
    raise Exception('No output found in grids.')

log.setLevel('DEBUG')
args = parse_args()
cfg = get_config(args.configFile)
galaxy_id = args.galaxyId[0]
nproc = args.nproc if args.nproc > 1 else 1

if args.debug:
    from pystarlight import mock
    sr.starlight_exec_path = path.join(path.dirname(mock.__file__), 'mock_starlight.py')
else:
    sr.starlight_exec_path = cfg.get('starlight', 'exec_path')

cube_out_dir = cfg.get('path', 'cubes_out')
masterlist = cfg.get('tables', 'masterlist')
cube = path.join(cube_out_dir, '%s_resampled.fits' % galaxy_id)
temp_cube = path.join(cube_out_dir, '%s_resam_synth1.fits' % galaxy_id)
starlight_dir = cfg.get('starlight', 'starlight_dir')
grid_template = cfg.get('starlight', 'grid_template')

print 'Loading cube.'
sa = SynthesisAdapter(cube, starlight_dir, grid_template)

print 'Starting starlight runner.'
runner = sr.StarlightRunner(n_workers=nproc, timeout=args.timeout * 60.0, compress=True)
for grid in sa.gridIterator(chunk_size=args.chunkSize, use_errors_flags=args.useErrorsFlags):
    print 'Dispatching grid.'
    runner.addGrid(grid)

print 'Waiting jobs completion.'
runner.wait()

print 'Creating synthesis cubes.'
output_grids = runner.getOutputGrids()

sa.createSynthesisCubes(pop_len=get_pop_len(output_grids))

for grid in output_grids:
    print 'Reading results of grid %s.' % grid.name
    sa.updateSynthesis(grid)
    
if args.updateErrors:
    print 'Estimating errors from the starlight residual.'
    sa.updateErrorsFromResidual(args.errorSmoothFwhm, args.errorBoxWidth)
    
print 'Saving cube to %s' % temp_cube
sa.writeCube(temp_cube, args.overwrite)
