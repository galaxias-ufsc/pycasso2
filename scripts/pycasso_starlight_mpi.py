#!/usr/bin/env python
'''
Created on 01/10/2019

@author: andre
'''

from pycasso2.starlight import SynthesisAdapter
from pycasso2.starlight.runner import run_starlight_and_check
from pycasso2.config import default_config_path, get_config

import argparse
from mpi4py.futures import MPIPoolExecutor
from multiprocessing import cpu_count
from astropy import log
from itertools import islice, starmap
log.setLevel('DEBUG')


###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run starlight for a pycasso cube.')

    parser.add_argument('cubeIn', type=str, nargs=1,
                        help='Cube. Ex.: T001.fits')
    parser.add_argument('--out', dest='cubeOut', required=True,
                        help='Output cube.')
    parser.add_argument('--name', dest='newName',
                        help='Rename the output cube.')
    parser.add_argument('--config', dest='configFile', default=default_config_path,
                        help='Config file. Default: %s' % default_config_path)
    parser.add_argument('--max-workers', dest='maxWorkers', type=int, default=cpu_count() - 1,
                        help='Maximum number of worker processes. Defaults to the number of system processors - 1. Set to 1 to run on a single thread.')
    parser.add_argument('--queue-length', dest='queueLength', type=int, default=-1,
                        help='Worker queue length. Default: 10 * max workers.')
    parser.add_argument('--chunk-size', dest='chunkSize', type=int, default=2,
                        help='Grid chunk size. Default: 2.')
    parser.add_argument('--timeout', dest='timeout', type=int, default=30,
                        help='Timeout of starlight processes, in minutes.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output.')
    parser.add_argument('--synth-sn', dest='synthSN', type=float,
                        help='Synthetic spectra signal to noise.')
    parser.add_argument('--use-custom-masks', dest='useCustomMasks', action='store_true',
                        help='Use Custom per-spaxel emission line masks.')
    parser.add_argument('--no-error-flag', dest='noErrorFlag', action='store_true',
                        help='Don\' use errors and flags from the spectra.')
    parser.add_argument('--estimate-error', dest='estimateError', action='store_true',
                        help='Calculate errors from residual and update f_err extension.')
    parser.add_argument('--error-smooth-fwhm', dest='errorSmoothFwhm', type=float, default=15.0,
                        help='FWHM (in Angstroms) of the gaussian used to smooth and rectify the residual before estimating the error.')
    parser.add_argument('--error-box-width', dest='errorBoxWidth', type=float, default=100.0,
                        help='Running box width (in Angstroms) used to calculate the RMS of the residual, to estimate the error.')

    return parser.parse_args()
###############################################################################

def run_starlight(exec_path, grid, timeout, compress=True):
    g = run_starlight_and_check(exec_path, grid, timeout, compress)
    g.readTables()
    return g


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.configFile)
    nproc = args.maxWorkers if args.maxWorkers > 1 else 1
    
    log.info('Loading cube from %s.' % args.cubeIn[0])
    sa = SynthesisAdapter(args.cubeIn[0], cfg, new_name=args.newName)
    
    log.debug('Creating gridfile iterator.')
    gridfiles = sa.gridIterator(chunk_size=args.chunkSize, use_errors_flags=not args.noErrorFlag,
                                use_custom_masks=args.useCustomMasks, synth_sn=args.synthSN)
    
    exec_path = cfg.get('starlight', 'exec_path')
    timeout = args.timeout * 60
    queue_length = args.queueLength
    if queue_length <= 0:
        queue_length = args.maxWorkers * 10
        log.debug('Setting queue length to %d.' % queue_length)
    
    log.debug('Preparing worker args.')
    map_args = ((exec_path, g, timeout, True) for g in gridfiles)

    log.debug('Starting execution pool.')
    with MPIPoolExecutor(args.maxWorkers) as executor:
        while True:
            chunk = list(islice(map_args, queue_length))
            if len(chunk) == 0:
                log.debug('Exceution completed.')
                break
            log.debug('Dispatching %d runs.' % len(chunk))
            if args.maxWorkers == 1:
                log.warning('Running on a single thread.')
                output_grids = starmap(run_starlight, chunk)
            else:
                output_grids = executor.starmap(run_starlight, chunk, unordered=True)

            log.debug('Collecting runs.')
            for grid in output_grids:
                log.debug('Reading results of %s.' % grid.name)
                sa.updateSynthesis(grid)
    
    if args.estimateError:
        print('Estimating errors from the starlight residual. Will overwrite the previous error values.')
        sa.updateErrorsFromResidual(args.errorSmoothFwhm, args.errorBoxWidth)
    
    print('Saving cube to %s.' % args.cubeOut)
    sa.writeCube(args.cubeOut, args.overwrite)
