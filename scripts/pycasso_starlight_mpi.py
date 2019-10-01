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
                        help='Number of worker processes. Defaults to the number of system processors.')
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


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.configFile)
    nproc = args.maxWorkers if args.maxWorkers > 1 else 1
    
    log.info('Loading cube from %s.' % args.cubeIn[0])
    sa = SynthesisAdapter(args.cubeIn[0], cfg, new_name=args.newName)
    
    gridfiles = sa.gridIterator(chunk_size=args.chunkSize, use_errors_flags=not args.noErrorFlag,
                                use_custom_masks=args.useCustomMasks, synth_sn=args.synthSN)
    
    exec_path = cfg.get('starlight', 'exec_path')
    timeout = args.timeout * 60
    
    map_args = ((exec_path, g, timeout, True) for g in gridfiles)
    with MPIPoolExecutor(args.maxWorkers) as executor:
        output_grids = executor.starmap(run_starlight_and_check, map_args, unordered=True)

    for grid in output_grids:
        log.debug('Reading results of %s.' % grid.name)
        sa.updateSynthesis(grid)
    
    if args.estimateError:
        print('Estimating errors from the starlight residual. Will overwrite the previous error values.')
        sa.updateErrorsFromResidual(args.errorSmoothFwhm, args.errorBoxWidth)
    
    print('Saving cube to %s.' % args.cubeOut)
    sa.writeCube(args.cubeOut, args.overwrite)
