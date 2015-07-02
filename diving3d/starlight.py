'''
Created on 26/06/2015

@author: andre
'''
from .tables import write_starlight_input
from .cube import D3DFitsCube
from . import flags

from astropy import log
from pystarlight.util.gridfile import GridRun, GridFile
from os import path, makedirs as os_makedirs
import numpy as np

__all__ = ['SynthesisAdapter', 'D3DGridFile', 'D3DGridRun']


def get_base_grid(popage_base, popZ_base):
    '''
    Get a grid description of the base consisting
    of a 2-d boolean array used convert 1-d population
    arrays from a starlight output file to 2-d
    population grids, and the grid coordinate values,
    metallicity and age.
    
    Parameters
    ----------
    popage_base : array
        Column ``popage_base`` from population table.
    
    popZbase : array
        Column ``popZ_base`` from population table.
    
    Returns
    -------
    base_mask : array
        Boolean 2-d array of defined elements in the base grid.
    
    Z : array
        Metallicities in the grid.
    
    ages : array
        Ages in the grid.
    
    Examples
    --------
    Calculate average stellar (log) age.
    
    >>> ts = atpy.TableSet('starlight.out.bz2', type='starlight')
    >>> pop = ts.population
    >>> base_mask, Z, age = get_base_grid(pop.popage_base, pop.popage_base)
    >>> popx_Zt = np.zeros(base_mask.shape)
    >>> popx_Zt[base_mask] = pop.popx
    >>> at_flux = (popx_Zt * np.log10(age)).sum() / popx_Zt.sum()
    '''
    age_base = np.unique(popage_base)
    Z_base = np.unique(popZ_base)

    shape = (len(Z_base), len(age_base))
    base_mask = np.zeros(shape, dtype='bool')
    for a, Z in zip(age_base, Z_base):
        i = np.where(Z_base == Z)[0]
        j = np.where(age_base == a)[0]
        base_mask[i, j] = True

    return base_mask, Z_base, age_base


class D3DGridFile(GridFile):
    def __init__(self, *args):
        GridFile.__init__(self, *args)
    
    
    def getTables(self):
        tables = []
        import atpy
        from pystarlight import io  # @UnusedImport
        for run in self.completed:
            outfile = path.join(self.outDirAbs, run.outFileCompressed)
            ts = atpy.TableSet(outfile, type='starlight')
            tables.append((run.x, run.y, ts))
        return tables
            

class D3DGridRun(GridRun):
    def __init__(self, x=None, y=None, *args):
        GridRun.__init__(self, *args)
        self.x = x
        self.y = y
    
    
    @classmethod
    def from_run(cls, run):
        d3drun = cls()
        d3drun.inFile = run.inFile
        d3drun.outFile = run.outFile
        d3drun.configFile = run.configFile
        d3drun.baseFile = run.baseFile
        d3drun.maskFile = run.maskFile
        d3drun.reddening = run.reddening
        d3drun.etcInfoFile = run.etcInfoFile
        d3drun.lumDistanceMpc = run.lumDistanceMpc
        d3drun.v0_Ini = run.v0_Ini
        d3drun.vd_Ini = run.vd_Ini
        return d3drun


def makedirs(the_path):
    if not path.exists(the_path):
        os_makedirs(the_path)


class SynthesisAdapter(object):
    
    def __init__(self, cube, starlight_dir, grid_template_path):
        self.starlightDir = starlight_dir
        self._d3d = D3DFitsCube(cube)
        self.galaxyId = self._d3d.id
        self._readData()
        self._gridTemplate, self._runTemplate = self._getTemplates(grid_template_path)
        self._createDirs()
        
    
    def _readData(self):
        self.l_obs = self._d3d.l_obs
        self.f_obs = self._d3d.f_obs
        self.f_err = None
        self.f_flag = self._d3d.f_flag
        self.spatialMask = self._d3d.getSpatialMask(0.5)
    
        
    def _getTemplates(self, template_path=None):
        if template_path is None:
            template_path = path.join(self.starlightDir, 'grid.template.in')
        grid = D3DGridFile.fromFile(self.starlightDir, template_path)
        grid.setObsDir(path.join(grid.obsDir, self.galaxyId))
        grid.setOutDir(path.join(grid.outDir, self.galaxyId))
        grid.setLogDir(path.join(grid.logDir, self.galaxyId))
        grid.fluxUnit = self._d3d.flux_unit
        grid.lLow_Syn = self.l_obs.min()
        grid.lUpp_Syn = self.l_obs.max()
        run = D3DGridRun.from_run(grid.runs.pop())
        run.lumDistanceMpc = self._d3d.masterlist['DL']
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
            log.debug('Creating inputs for spaxel (%d,%d)' % (x, y))
            run = self._createRun(x, y)
            if run is not None:
                grid.runs.append(run)
            else:
                log.debug('Skipping masked spaxel (%d,%d)' % (x, y))
        return grid


    def _createRun(self, x, y):
        if self.spatialMask[y, x]:
            self.f_flag[:, y, x] |= flags.starlight_masked_pix
            return None
        
        new_run = self._runTemplate.copy()
        new_run.inFile = '%s_%04d_%04d.in' % (self.galaxyId, y, x)
        new_run.outFile = '%s_%04d_%04d.out' % (self.galaxyId, y, x)
        new_run.x = x
        new_run.y = y
        if self.f_flag is not None and self.f_err is not None:
            write_starlight_input(self.l_obs, self.f_obs[:, y, x], self.f_err[:, y, x],
                                      self.f_flag[:, y, x], path.join(self.obsDir, new_run.inFile))
        else:
            write_starlight_input(self.l_obs, self.f_obs[:, y, x], None, None,
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


    def createSynthesisCubes(self, pop_len):
        self._d3d.createSynthesisCubes(pop_len)
        self.f_syn = self._d3d.f_syn
        self.f_wei = self._d3d.f_wei
        self.popage_base = self._d3d.popage_base
        self.popZ_base = self._d3d.popZ_base
        self.popx = self._d3d.popx
        self.popmu_ini = self._d3d.popmu_ini
        self.popmu_cor = self._d3d.popmu_cor
        
    
    def updateSynthesis(self, grid):
        for fr in grid.failed:
            self.f_flag[:, fr.y, fr.x] |= flags.starlight_failed_run            

        keyword_data = {}
        for k in self._d3d._ext_keyword_list:
            keyword_data[k] = self._d3d.getSynthExtension(k)
            
        for x, y, ts in grid.getTables():
            log.debug('Writing synthesis for spaxel (%d,%d)' %(x, y))
            f_obs_norm = ts.keywords['fobs_norm']
            self.f_syn[:, y, x] = ts.spectra.f_syn * f_obs_norm / grid.fluxUnit
            self.f_wei[:, y, x] = ts.spectra.f_wei
            # TODO: update flags with starlight clipped, etc.
            self.popZ_base[y, x, :] = ts.population.popZ_base
            self.popage_base[y, x, :] = ts.population.popage_base
            self.popx[y, x, :] = ts.population.popx
            self.popmu_ini[y, x, :] = ts.population.popmu_ini
            self.popmu_cor[y, x, :] = ts.population.popmu_cor
            for k in self._d3d._ext_keyword_list:
                keyword_data[k][y, x] = ts.keywords[k]
    
    def writeCube(self, filename, overwrite=False):
        self._d3d.write(filename, overwrite=overwrite)
        
