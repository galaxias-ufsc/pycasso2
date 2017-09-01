'''
Created on 26/06/2015

@author: andre
'''
from .tables import write_input, read_output_tables
from .gridfile import GridRun, GridFile
from ..resampling import get_subset_slices, find_nearest_index
from ..error import estimate_error
from .. import flags

from astropy import log
from os import path, makedirs as os_makedirs
import numpy as np

__all__ = ['SynthesisAdapter', 'PGridFile', 'PGridRun']


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
    '''
    age_base = np.unique(popage_base)
    Z_base = np.unique(popZ_base)

    shape = (len(Z_base), len(age_base))
    base_mask = np.zeros(shape, dtype='bool')
    for a, Z in zip(popage_base, popZ_base):
        i = np.where(Z_base == Z)[0]
        j = np.where(age_base == a)[0]
        base_mask[i, j] = True

    return base_mask, Z_base, age_base


class PGridFile(GridFile):

    def __init__(self, *args):
        GridFile.__init__(self, *args)

    def getTables(self):
        tables = []
        for run in self.completed:
            outfile = path.join(self.outDirAbs, run.outFileCompressed)
            ts = read_output_tables(outfile)
            tables.append((run.x, run.y, ts))
        return tables


class PGridRun(GridRun):

    def __init__(self, x=None, y=None, *args):
        GridRun.__init__(self, *args)
        self.x = x
        self.y = y


def makedirs(the_path):
    if not path.exists(the_path):
        os_makedirs(the_path)


class SynthesisAdapter(object):
    
    cfg_sec = 'starlight'

    def __init__(self, cube, cfg, new_name=None):
        from ..cube import FitsCube
        self.starlightDir = cfg.get(self.cfg_sec, 'starlight_dir')
        self._arqMaskFormat = cfg.get(self.cfg_sec, 'arq_mask_format')
        self._inFileFormat = cfg.get(self.cfg_sec, 'arq_obs_format')
        self._outFileFormat = cfg.get(self.cfg_sec, 'arq_out_format')
        self._cube = FitsCube(cube)
        if new_name is None:
            self.name = self._cube.name
        else:
            log.debug('Renamed cube to %s' % new_name)
            self.name = new_name
            self._cube.name = new_name
        self._gridTemplate, self._runTemplate = self._getTemplates(cfg)
        self._readData()
        self._createDirs()
        self._base_data_saved = False
        
    def getSpatialMask(self):
        mask = self._cube.getSpatialMask(flags.before_starlight)
        l1 = self._gridTemplate.lLow_SN
        l2 = self._gridTemplate.lUpp_SN
        i1, i2 = find_nearest_index(self.l_obs, [l1, l2])
        bad_snwin = (self.f_flag[i1:i2] & flags.before_starlight) > 0
        mask |= bad_snwin.all(axis=0)
        if self.isSegmented:
            mask = mask[np.newaxis, :]
        return mask

    def _readData(self):
        self.l_obs = self._cube.l_obs
        if self.isSegmented:
            self.f_obs = self._cube.f_obs[:, np.newaxis, :]
            self.f_err = self._cube.f_err[:, np.newaxis, :]
            self.f_flag = self._cube.f_flag[:, np.newaxis, :]
        else:
            self.f_obs = self._cube.f_obs
            self.f_err = self._cube.f_err
            self.f_flag = self._cube.f_flag
        self.spatialMask = self.getSpatialMask()

    def _getTemplates(self, cfg):
        grid = PGridFile(self.starlightDir)
        grid.fluxUnit = self._cube.flux_unit

        grid.setBasesDir(cfg.get(self.cfg_sec, 'base_dir'))
        obs_dir = path.join(cfg.get(self.cfg_sec, 'obs_dir'), self.name)
        grid.setObsDir(obs_dir)
        out_dir = path.join(cfg.get(self.cfg_sec, 'out_dir'), self.name)
        grid.setOutDir(out_dir)
        grid.setLogDir(path.join(grid.logDir, self.name))

        grid.setMaskDir(cfg.get(self.cfg_sec, 'mask_dir'))
        grid.setEtcDir(cfg.get(self.cfg_sec, 'etc_dir'))
        grid.randPhone = cfg.getint(self.cfg_sec, 'rand_seed')
        grid.lLow_SN = cfg.getfloat(self.cfg_sec, 'llow_SN')
        grid.lUpp_SN = cfg.getfloat(self.cfg_sec, 'lupp_SN')
        grid.lLow_Syn = cfg.getfloat(self.cfg_sec, 'Olsyn_ini')
        grid.lUpp_Syn = cfg.getfloat(self.cfg_sec, 'Olsyn_fin')
        grid.dLambda = self._cube.dl
        grid.fitFix = cfg.get(self.cfg_sec, 'fit_fix')
        grid.isPhoEnabled = cfg.getint(self.cfg_sec, 'IsPHOcOn')
        grid.isQHREnabled = cfg.getint(self.cfg_sec, 'IsQHRcOn')
        grid.isFIREnabled = cfg.getint(self.cfg_sec, 'IsFIRcOn')
        grid.EtcESM = cfg.get(self.cfg_sec, 'ETC_ESM')
        grid.EtcGamma = cfg.getfloat(self.cfg_sec, 'ETC_gamma')

        run = PGridRun()
        run.configFile = cfg.get(self.cfg_sec, 'arq_config')
        run.baseFile = cfg.get(self.cfg_sec, 'arq_base')
        run.maskFile = cfg.get(self.cfg_sec, 'arq_mask')
        run.reddening = cfg.get(self.cfg_sec, 'red_law')
        run.etcInfoFile = cfg.get(self.cfg_sec, 'arq_etc')
        run.v0_Ini = cfg.getfloat(self.cfg_sec, 'v0_ini')
        run.vd_Ini = cfg.getfloat(self.cfg_sec, 'vd_ini')
        run.lumDistanceMpc = self._cube.lumDistMpc

        return grid, run

    def _createDirs(self):
        obs_dir = path.normpath(self._gridTemplate.obsDirAbs)
        self.obsDir = obs_dir
        out_dir = path.normpath(self._gridTemplate.outDirAbs)
        log_dir = path.normpath(self._gridTemplate.logDirAbs)

        makedirs(obs_dir)
        makedirs(out_dir)
        makedirs(log_dir)

    def _getGrid(self, y, x1, x2, use_errors_flags, use_custom_masks, synth_sn):
        grid = self._gridTemplate.copy()
        if x1 != x2:
            grid.name = 'grid_%s_%04d_%04d-%04d' % (self.name, y, x1, x2)
        else:
            grid.name = 'grid_%04d_%04d' % (self.name, y, x1)
        grid.randPhone = -958089828
        # grid.seed()
        if use_errors_flags:
            grid.errSpecAvail = 1
            grid.flagSpecAvail = 1
        else:
            grid.errSpecAvail = 0
            grid.flagSpecAvail = 0

        for x in range(x1, x2):
            run = self._createRun(x, y, use_errors_flags, use_custom_masks, synth_sn)
            if run is not None:
                grid.runs.append(run)
            else:
                log.debug('Skipping masked spaxel (%d,%d)' % (x, y))
        return grid

    def _createRun(self, x, y, use_errors_flags, use_custom_masks, synth_sn):
        if self.spatialMask[y, x]:
            self.f_flag[:, y, x] |= flags.starlight_masked_pix
            return None

        log.debug('Creating inputs for spaxel (%d,%d)' % (x, y))
        new_run = self._runTemplate.copy()
        new_run.inFile = self._inFileFormat % (self.name, y, x)
        new_run.outFile = self._outFileFormat % (self.name, y, x)
        if use_custom_masks:
            new_run.maskFile = self._arqMaskFormat % (self.name, y, x)
        new_run.x = x
        new_run.y = y
        f_obs = self.f_obs[:, y, x]
        if use_errors_flags:
            f_err = self.f_err[:, y, x]
            f_flag = self.f_flag[:, y, x]
            if synth_sn is not None:
                f_err = np.sqrt(f_err**2 + (f_obs / synth_sn)**2)
        else:
            f_err = None
            f_flag = None
        write_input(self.l_obs, f_obs, f_err, f_flag, path.join(self.obsDir, new_run.inFile))
        return new_run

    def gridIterator(self, chunk_size, use_errors_flags=True,
                     use_custom_masks=False, synth_sn=None):
        Nx = self.f_obs.shape[2]
        Ny = self.f_obs.shape[1]
        for y in range(0, Ny, 1):
            for x1 in range(0, Nx, chunk_size):
                x2 = x1 + chunk_size
                if x2 > Nx:
                    x2 = Nx
                yield self._getGrid(y, x1, x2, use_errors_flags, use_custom_masks, synth_sn)

    def createSynthesisCubes(self, pop_len):
        self._cube.createSynthesisCubes(pop_len)
        if self.isSegmented:
            self.f_syn = self._cube.f_syn[:, np.newaxis, :]
            self.f_wei = self._cube.f_wei[:, np.newaxis, :]
            self.popx = self._cube.popx[:, np.newaxis, :]
            self.popmu_ini = self._cube.popmu_ini[:, np.newaxis, :]
            self.popmu_cor = self._cube.popmu_cor[:, np.newaxis, :]
        else:
            self.f_syn = self._cube.f_syn
            self.f_wei = self._cube.f_wei
            self.popx = self._cube.popx
            self.popmu_ini = self._cube.popmu_ini
            self.popmu_cor = self._cube.popmu_cor
        
        self.popage_base = self._cube.popage_base
        self.popZ_base = self._cube.popZ_base
        self.popaFe_base = self._cube.popaFe_base
        self.Mstars = self._cube.Mstars
        self.fbase_norm = self._cube.fbase_norm

    def updateSynthesis(self, grid):
        for fr in grid.failed:
            log.warn('Failed run for pixel (%d, %d)' % (fr.y, fr.x))
            self.f_flag[:, fr.y, fr.x] |= flags.starlight_failed_run

        keyword_data = {}
        for k in self._cube._ext_keyword_list:
            keyword_data[k] = self._cube._getSynthExtension(k)

        for x, y, ts in grid.getTables():
            population = ts['population']
            spectra = ts['spectra']
            keywords = ts['keywords']
            if not self._base_data_saved:
                self.Mstars[:] = population['popMstars']
                self.fbase_norm[:] = population['popfbase_norm']
                self.popZ_base[:] = population['popZ_base']
                self.popage_base[:] = population['popage_base']
                self.popaFe_base[:] = population['aFe']
                self._base_data_saved = True

            log.debug('Writing synthesis for spaxel (%d,%d)' % (x, y))
            f_obs_norm = keywords['fobs_norm']
            slice_d, slice_o = get_subset_slices(self.l_obs, spectra['l_obs'])
            self.f_syn[slice_d, y, x] = spectra['f_syn'][slice_o] * (f_obs_norm / grid.fluxUnit)
            self.f_wei[slice_d, y, x] = spectra['f_wei'][slice_o]
            # TODO: update flags with starlight clipped, etc.
            self.f_flag[:slice_d.start] |= flags.no_starlight
            self.f_flag[slice_d.stop:] |= flags.no_starlight

            self.popx[:, y, x] = population['popx']
            self.popmu_ini[:, y, x] = population['popmu_ini']
            self.popmu_cor[:, y, x] = population['popmu_cor']
            for k in self._cube._ext_keyword_list:
                if self.isSegmented:
                    keyword_data[k][x] = keywords[k]
                else:
                    keyword_data[k][y, x] = keywords[k]

    def updateErrorsFromResidual(self, smooth_fwhm=15.0, box_width=100.0):
        f_res = np.ma.array(self.f_obs - self.f_syn, mask=self.f_wei <= 0)
        self.f_err[...] = estimate_error(
            self.l_obs, f_res, self.spatialMask, smooth_fwhm, box_width)

    def writeCube(self, filename, overwrite=False):
        self._cube.write(filename, overwrite=overwrite)
        
    @property
    def isSegmented(self):
        return self._cube.hasSegmentationMask
