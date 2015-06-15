'''
Created on Jun 17, 2013

@author: andre
'''

from diving3d.masterlist import read_masterlist
from pystarlight.util.StarlightUtils import spec_resample
from pystarlight.util.gridfile import GridFile
from pystarlight.util import starlight_runner as sr

import pyfits
from astropy.io import ascii
import numpy as np
from os import path, makedirs as os_makedirs
import argparse
from multiprocessing import cpu_count



###############################################################################
def write_table(wl, flux, err, flags, filename):
    flags = np.where(flags, 1.0, 0.0)
    if flags is not None and err is not None:
        cols = [wl, flux, err, flags]
    else:
        cols = [wl, flux]
    ascii.write(cols, filename, Writer=ascii.NoHeader)
###############################################################################


###############################################################################
def get_galaxy_id(cube):
        return path.basename(cube).split('_')[0]
###############################################################################


###############################################################################
def makedirs(the_path):
    if not path.exists(the_path):
        os_makedirs(the_path)
###############################################################################


###############################################################################
def get_wl(head):
    dl = head['CD3_3']
    l_ini = head['CRVAL3'] + (1 - head['CRPIX3']) * dl
    l_fin = l_ini + (head['NAXIS3'] - 1) * dl
    return np.arange(l_ini, l_fin + dl, dl)
###############################################################################


###############################################################################
class GridManager(object):
    flux_unit = 1e-15
    _l_ini = 4300.0
    _l_fin = 6800.0
    _dl = 1.0
    
    def __init__(self, starlight_dir, cube, keywords=None):
        self.keywords = keywords
        self.starlightDir = starlight_dir
        self.galaxyId = get_galaxy_id(cube)
        self.l_obs, self.f_obs, self.f_err, self.f_flag = self.readData(cube)
        self._gridTemplate, self._runTemplate = self._getTemplates()
        self._createDirs()
    
        
    def readData(self, cube):
        f = pyfits.open(cube)

        head = f[0].header
        l_orig = get_wl(head)
        Nx = head['NAXIS1']
        Ny = head['NAXIS2']
        flux = f[0].data
        
        l_res = np.arange(self._l_ini, self._l_fin + self._dl, self._dl)
        flux_res = np.zeros((len(l_res), Ny, Nx))
        for j in xrange(Ny):
            for i in xrange(Nx):
                print 'Resampling spaxel %d,%d' % (i,j)
                flux_res[:, j, i] = spec_resample(l_orig, l_res, flux[:, j, i])
        
        assert np.allclose(np.trapz(flux, l_orig, axis=0), np.trapz(flux_res, l_res, axis=0), atol=0.1)
        return l_res, flux_res, None, None

        
    def _getTemplates(self):
        template_path = path.join(self.starlightDir, 'grid.template.in')
        grid = GridFile.fromFile(self.starlightDir, template_path)
        grid.setLogDir('log') 
        grid.fluxUnit = self.flux_unit
        run = grid.runs.pop()
        run.lumDistanceMpc = self.keywords['DL']
        grid.clearRuns()
        return grid, run

    
    def _createDirs(self):
        obs_dir = path.normpath(self._gridTemplate.obsDirAbs)
        self.obsDir = obs_dir
        out_dir = path.normpath(self._gridTemplate.outDirAbs)
        log_dir = path.normpath(self._gridTemplate.logDirAbs)

        makedirs(obs_dir)
        makedirs(out_dir)
        makedirs(log_dir)
        
        
    def _getGrid(self, y, x1, x2):
        grid = self._gridTemplate.copy()
        if x1 != x2:
            grid.name = 'grid_%s_%04d_%04d-%04d' % (self.galaxyId, y, x1, x2)
        else:
            grid.name = 'grid_%04d_%04d' % (self.galaxyId, y, x1)
        grid.randPhone = -958089828
        # grid.seed()
        
        for x in xrange(x1, x2):
            print 'Creating inputs for spaxel %d,%d' % (x, y)
            run = self._createRun(x, y)
            if run is not None:
                grid.runs.append(run)
            else:
                print 'Skipping spaxel %d,%d' % (x, y)
        return grid


    def _createRun(self, x, y):
        if self.f_flag is not None:
            n_good = (self.f_flag == 0.0).sum()
            if n_good <= 400:
                return None
        new_run = self._runTemplate.copy()
        new_run.inFile = '%s_%04d_%04d.in' % (self.galaxyId, y, x)
        new_run.outFile = '%s_%04d_%04d.out' % (self.galaxyId, y, x)
        if self.f_flag is not None and self.f_err is not None:
            write_table(self.l_obs, self.f_obs[:, y, x], self.f_err[:, y, x], self.f_flag[:, y, x],
                        path.join(self.obsDir, new_run.inFile))
        else:
            write_table(self.l_obs, self.f_obs[:, y, x], None, None,
                        path.join(self.obsDir, new_run.inFile))
        return new_run


    def gridIterator(self, chunk_size):
        Nx = self.f_obs.shape[2]
        Ny = self.f_obs.shape[1]
        for y in xrange(0, Ny, 1):
            for x in xrange(0, Nx, chunk_size):
                x2 = x + chunk_size
                if x2 > Nx: x2 = Nx
                yield self._getGrid(y, x, x2)

###############################################################################


sr.starlight_exec_path = '/Users/andre/astro/qalifa/pystarlight/src/pystarlight/mock/mock_starlight.py'

parser = argparse.ArgumentParser(description='Run starlight for a B/D decomposition.')

parser.add_argument('cube', type=str, nargs=1,
                    help='Cubes.')
parser.add_argument('--starlight-dir', dest='starlightDir', default='data/starlight',
                    help='HDF5 database path.')
parser.add_argument('--masterlist', dest='masterlist', default='data/masterlist_sampleT.txt',
                    help='Master list.')
parser.add_argument('--nproc', dest='nproc', type=int, default=cpu_count()-1,
                    help='Number of worker processes.')
parser.add_argument('--chunk-size', dest='chunkSize', type=int, default=61,
                    help='Grid chunk size, defaults to the same as --nproc.')
parser.add_argument('--timeout', dest='timeout', type=int, default=20,
                    help='Timeout of starlight processes, in minutes. Defaults to 20.')
args = parser.parse_args()
cube = args.cube[0]
nproc = args.nproc if args.nproc > 1 else 1

ml = read_masterlist(args.masterlist)
# FIXME: look up the master list, right now this only works for T001.
keywords = ml[0]

print 'Loading grid manager.'
gm = GridManager(args.starlightDir, cube, keywords)
print 'Starting starlight runner.'
runner = sr.StarlightRunner(n_workers=nproc, timeout=args.timeout * 60.0, compress=True)
for grid in gm.gridIterator(chunk_size=args.chunkSize):
    print 'Dispatching grid.'
    runner.addGrid(grid)

print 'Waiting jobs completion.'
runner.wait()

failed_grids = runner.getFailedGrids()
if len(failed_grids) > 0:
    print 'Failed to starlight:'
    for grid in failed_grids:
        print '\n'.join(r.outFile for r in grid.failed)

