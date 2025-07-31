'''
Fitting emission lines in MaNGA galaxies.

Usage:

python3 dobby_fit_manga.py 7960-6101

For many galaxies:
for i in `tail +3 ~/data/MaNGA/jpgs_subsample20_elisa/subsample20_elisa.txt`
   do time /usr/local/anaconda/bin/python3 fit_manga.py $i
done

Natalia@UFSC - 20/Sep/2017
'''

from os import path, makedirs, unlink
import numpy as np
import argparse
from astropy import log
from astropy.table import Table
from mpi4py.futures import MPIPoolExecutor
from itertools import islice, starmap
from multiprocessing import cpu_count

from pycasso2 import FitsCube
from pycasso2.segmentation import sum_spectra
from pycasso2.config import default_config_path
from pycasso2.dobby.fitting import fit_strong_lines
from pycasso2.dobby.utils import plot_el, read_summary_from_file, summary_elines
from pycasso2.dobby import flags as dobby_flags
from pycasso2 import flags

log.setLevel('DEBUG')


###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run starlight for a pycasso cube.')

    parser.add_argument('cubeIn', type=str, nargs=1,
                        help='Cube. Ex.: T001.fits')
    parser.add_argument('--out', dest='cubeOut',
                        help='Output cube. If not set, will update the input.')
    parser.add_argument('--name', dest='newName',
                        help='Rename the output cube.')
    parser.add_argument('--config', dest='configFile', default=default_config_path,
                        help='Config file. Default: %s' % default_config_path)
    parser.add_argument('--max-workers', dest='maxWorkers', type=int, default=cpu_count() - 1,
                        help='Maximum mumber of worker processes.')
    parser.add_argument('--queue-length', dest='queueLength', type=int, default=-1,
                        help='Worker queue length. Default: 10 * max workers.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Save detailed fit plots.')
    parser.add_argument('--display-plots', dest='displayPlots', action='store_true',
                        help='Display detailed fit plots.')
    parser.add_argument('--tmp-dir', dest='tmpDir', default='./tmp_el',
                        help='Write temporary tables here. Default: ./tmp_el')
    parser.add_argument('--only-center', dest='onlyCenter', action='store_true',
                        help='Test fit, only central spaxel.')
    parser.add_argument('--model', dest='model', default='gaussian',
                        help='Line profile model: gaussian or resampled_gaussian. Default: gaussian')
    parser.add_argument('--enable-kin-ties', dest='enableKinTies', action='store_true',
                        help='Enable kinematic ties.')
    parser.add_argument('--enable-balmer-lim', dest='enableBalmerLim', action='store_true',
                        help='Do not allow Ha/Hb < 2.0.')
    parser.add_argument('--turnoff-Ha', dest='noHa', action='store_true',
                        help='Warn dobby not to use Ha.')
    parser.add_argument('--correct-good-frac', dest='correct_good_frac', action='store_true',
                        help='Correct spectra for the segmentation good frac spec prior to fitting.')
    parser.add_argument('--degree', dest='degree', type=int, default=16,
                        help='Degree for Legendre polynomial fits in the local continuum. Default: 16')
    parser.add_argument('--vd_max', dest='vd_max', type=np.float64, default=1000.,
                        help='Maximum vd to remove the pseudocontinuum (decrease to 100 to remove broad lines). Default: 1000 km/s')

    return parser.parse_args()
###############################################################################

###############################################################################
class DobbyAdapter(object):
    def __init__(self, filename, mode='readonly', correct_good_frac=False,
                 kin_ties=False, bal_lim=False, noHa=False, model='gaussian'):
        log.debug('Reading cube: %s' % filename)
        c = FitsCube(filename, mode=mode)
        self._c = c
        self.correct_good_frac = correct_good_frac
        self.kinTies = kin_ties
        self.balLim = bal_lim
        self.noHa = noHa
        self.model = model
        self.name = c.name
        self.ll = c.l_obs
        self.integ_mask = np.where(c.synthImageMask, 0, 1)[np.newaxis, ...]
        self.v_0 = c.v_0
        self.v_d = c.v_d
        self.integ_v_0 = c.synthIntegKeywords['v0']
        self.integ_v_d = c.synthIntegKeywords['vd']
        self.vd_max = args.vd_max
            
        if c.hasSegmentationMask:
            log.debug('Cube is segmented.')
            if correct_good_frac:
                log.debug('Will correct fractional segments.')
            self.f_obs = c.f_obs[..., np.newaxis]
            self.f_syn = c.f_syn[..., np.newaxis]
            self.f_err = c.f_err[..., np.newaxis]
            self.f_flag = c.f_flag[..., np.newaxis]
            self.Ny = c.Nzone
            self.Nx = 1
            self.y0 = 0
            self.x0 = 0
        else:
            log.debug('Cube is spatially resolved.')
            self.f_obs = c.f_obs
            self.f_syn = c.f_syn
            self.f_err = c.f_err
            self.f_flag = c.f_flag
            self.Ny = c.Ny
            self.Nx = c.Nx
            self.y0 = c.y0
            self.x0 = c.x0
                

    def save(self, output):
        if output is not None:
            log.info('Saving output to %s' % output)
            self._c.write(output, overwrite=True)
        else:
            log.info('Saving cube (in place).')
            self._c.flush()
            
        
    def createDobbyExtensions(self, el_info):
        if self._c.hasELines:
            log.warning('Deleting existing emission line data.')
            self._c.deleteELinesCubes()
        self._c.createELinesCubes(el_info)

        if self._c.hasSegmentationMask:
            self.El_F = self._c._EL_flux[:, np.newaxis, :]
            self.El_v0 = self._c._EL_v_0[:, np.newaxis, :]
            self.El_vd = self._c._EL_v_d[:, np.newaxis, :]
            self.El_flag = self._c._EL_flag[:, np.newaxis, :]
            self.El_EW = self._c._EL_EW[:, np.newaxis, :]
            self.El_vdins = self._c._EL_v_d_inst[:, np.newaxis, :]
            self.El_lcrms = self._c._EL_continuum_RMS[:, np.newaxis, :]
            self.El_lc = self._c.EL_continuum[:, np.newaxis, :]
            self.El_F_integ = self._c._El_F_integ[:, np.newaxis, :]
            self.El_F_imed = self._c._El_F_imed[:, np.newaxis, :]   
        else:
            self.El_F = self._c._EL_flux
            self.El_v0 = self._c._EL_v_0
            self.El_vd = self._c._EL_v_d
            self.El_flag = self._c._EL_flag
            self.El_EW = self._c._EL_EW
            self.El_vdins = self._c._EL_v_d_inst
            self.El_lcrms = self._c._EL_continuum_RMS
            self.El_lc = self._c.EL_continuum
            self.El_F_integ = self._c._El_F_integ
            self.El_F_imed = self._c._El_F_imed
        

    def updateELines(self, j, i, elines, spec):
        if not self._c.hasELines:
            log.info('Creating emission line extensions.')
            el_info = get_EL_info(elines, self.kinTies, self.balLim, self.model)
            self.createDobbyExtensions(el_info)

        if elines is None:
            log.debug('Spaxel [%d, %d] flagged as no_data.' % (j, i))
            self.El_flag[:, j, i] = dobby_flags.no_data
            return

        self.El_F[:, j, i] = elines['El_F']
        self.El_v0[:, j, i] = elines['El_v0']
        self.El_vd[:, j, i] = elines['El_vd']
        self.El_flag[:, j, i] = elines['El_flag']
        self.El_EW[:, j, i] = elines['El_EW']
        self.El_vdins[:, j, i] = elines['El_vdins']
        self.El_lcrms[:, j, i] = elines['El_lcrms']
        self.El_lc[:, j, i] = spec['total_lc']
                
    def getIntegratedSpectra(self):
        if not self.correct_good_frac:
            f_obs = self._c.integ_f_obs
            f_syn = self._c.integ_f_syn
            f_err =  self._c.integ_f_err
        else:
            log.warning('Recalculating integrated spectra to avoid segmentation overlaps.')
            f_obs, f_err, good_frac = sum_spectra(self.integ_mask, self.f_obs, self.f_err,
                                                  cov_factor_A=0.0, cov_factor_B=1.0)
            f_syn = np.tensordot(self.f_syn, self.integ_mask, axes=[[1, 2], [1, 2]])
            bad = (good_frac <= 0.5)
            f_obs = np.ma.masked_where(bad, f_obs, copy=False).squeeze()
            f_syn = np.ma.masked_where(bad, f_syn, copy=False).squeeze()
            f_err = np.ma.masked_where(bad, f_err, copy=False).squeeze()
        f_res = f_obs - f_syn
        return f_res, f_syn, f_err
           
    def getArgs(self, j, i):
        f_obs = self.f_obs[:, j, i]
        f_syn = self.f_syn[:, j, i]
        f_err = self.f_err[:, j, i]
        if self.correct_good_frac and self._c.seg_good_frac is not None:
            gf = self._c.seg_good_frac[:, j, i]
            f_obs *= gf
            f_syn *= gf
            f_err *= gf
        f_res = f_obs - f_syn
        f_flagged = ((flags.no_starlight & self.f_flag[:, j, i]) > 0)
        f_res[f_flagged] = np.ma.masked
        return j, i, f_res, f_syn, f_err, self.v_0[j, i], self.v_d[j, i]
    
    def __iter__(self):
        for j in range(self.Ny):
            for i in range(self.Nx):
                yield self.getArgs(j, i)
###############################################################################


###############################################################################
def get_EL_info(elines, kinTies, balLim, model):
    N_lines = len(elines)
    el_info = Table({'lambda': elines['lambda'],
                     'name': elines['line'],
                     'l0': elines['El_l0'],
                     'model': N_lines * [model],
                     'kinematic_ties_on': np.array(N_lines * [kinTies], dtype='int'),
                     'Balmer_dec_limit': np.array(N_lines * [balLim], dtype='int'),
                     })
    el_info.convert_unicode_to_bytestring()
    return el_info
###############################################################################


###############################################################################
def get_suffix(model, kinematic_ties_on, balmer_limit_on):
    _k = 1 * kinematic_ties_on
    _b = 1 * balmer_limit_on
    if model == 'gaussian':
        _m = 'GA'
    elif model == 'resampled_gaussian':
        _m = 'RG'
    else:
        raise Exception('Unknown model: %s' % model)
    suffix = 'El%sk%ib%i' % (_m, _k, _b)
    return suffix
###############################################################################


###############################################################################
# Multiprocessing functions
def fit_spaxel_kwargs(j, i, f_res, f_syn, f_err, stellar_v0, stellar_vd, kwargs):
    return fit_spaxel(j, i, f_res, f_syn, f_err, stellar_v0, stellar_vd, **kwargs)
###############################################################################


###############################################################################
def fit_spaxel(iy, ix, f_res, f_syn, f_err, stellar_v0, stellar_vd,
               suffix, name_template, tmpdir,
               ll, vd_inst, vd_max,
               kinematic_ties_on, balmer_limit_on, noHa, model,
               degree, debug, display_plot):
    
    Nmasked = np.ma.getmaskarray(f_res).sum()
    if (Nmasked / len(f_res)) > 0.5:
        log.debug('Skipping masked spaxel [%d, %d]' % (iy, ix))
        return iy, ix, None, None
    
    name = suffix + '.' + name_template % (iy, ix)
    outfile = path.join(tmpdir, '%s.hdf5' % name)

    if path.exists(outfile):
        try:
            elines, spec = read_summary_from_file(outfile)
            log.debug('Read results for spaxel [%d, %d] from %s.' % (iy, ix, outfile))
            return iy, ix, elines, spec
        except:
            log.debug('Corrupt result file (%s), removing and repeating the fit.')
            unlink(outfile)

    # Using stellar v0 and vd from the spaxel spectrum (even though from the integrated is usually safer)
    log.info(f'Fitting spaxel [{iy}, {ix}] with stellar_vd = {stellar_vd}')
    el = fit_strong_lines(ll, f_res, f_syn, f_err, vd_inst=vd_inst,
                          kinematic_ties_on=kinematic_ties_on, balmer_limit_on=balmer_limit_on, noHa=noHa, model=model,
                          degree=degree, saveAll=True, outname=name, outdir=tmpdir, overwrite=True,
                          vd_kms=True, stellar_v0=stellar_v0, stellar_vd=stellar_vd)
    elines, spec = summary_elines(el)
    if debug:
        fig = plot_el(ll, f_res, el, ifig = 0, display_plot = display_plot)
        fig.savefig(path.join(tmpdir, '%s.pdf' % name))

    return iy, ix, elines, spec
###############################################################################


###############################################################################
def fit_integrated(da, suffix, tmpdir, vd_inst,
                   kinematic_ties_on, balmer_limit_on, noHa, model,
                   degree, debug, vd_max, display_plot):
    name = suffix + '.' + 'integ'
    outfile = path.join(el_dir, '%s.hdf5' % name)

    if path.exists(outfile):
        try:
            elines, spec = read_summary_from_file(outfile)
            log.debug('Read results for integrated spectrum from %s.' % outfile)
            return elines, spec
        except:
            log.debug('Corrupt result file (%s), removing and repeating the fit.')
            unlink(outfile)

    f_res, f_syn, f_err = da.getIntegratedSpectra()

    Nmasked = np.ma.getmaskarray(f_res).sum()
    if (Nmasked / len(f_res)) > 0.5:
        log.debug('Skipping integrated spectrum fit, not enough data.')
        return None, None

    # Avoid broad lines
    # TO DO: fit broad lines...
    print('@@> Using vd_integ = ', min(da.integ_v_d, vd_max))
    
    log.info('Fitting integrated spectrum.')
    el = fit_strong_lines(da.ll, f_res, f_syn, f_err, vd_inst=vd_inst,
                          kinematic_ties_on=kinematic_ties_on, balmer_limit_on=balmer_limit_on, noHa=noHa, model=model,
                          degree=degree, saveAll=True, outname=name, outdir=tmpdir, overwrite=True,
                          vd_kms=True, stellar_v0=da.integ_v_0, stellar_vd=min(da.integ_v_d, vd_max))
    elines, spec = summary_elines(el)
    if debug:
        fig = plot_el(da.ll, f_res, el, ifig = 0, display_plot = display_plot)
        fig.savefig(path.join(tmpdir, '%s.pdf' % name))

    return elines, spec
###############################################################################


###############################################################################
def MUSE_vd_inst(ll):
    # FIXME: Hard-coded for now!
    vdi_R2000 = 64.
    vdi_R4000 = 32.
    llim = 6000.
    return np.where(ll <= llim, vdi_R2000, vdi_R4000)
###############################################################################


###############################################################################
if __name__ == '__main__':
    
    args = parse_args()
    cube_in = args.cubeIn[0]
    
    if args.cubeOut is None:
        log.warning('Output not set, this will update the input file!')
        mode = 'update'
    else:
        mode = 'readonly'
    
    # TODO: read from config
    name_template = 'p%04i-%04i'
    suffix = get_suffix(args.model, args.enableKinTies, args.enableBalmerLim)
    
    log.info('Loading cube %s.' % cube_in)
    da = DobbyAdapter(cube_in, mode=mode, correct_good_frac=args.correct_good_frac,
                      kin_ties=args.enableKinTies, bal_lim=args.enableBalmerLim, noHa=args.noHa, model=args.model)

    el_dir = path.join(args.tmpDir, da.name)
    if not path.exists(el_dir):
        log.debug('Creating directory %s.' % el_dir)
        makedirs(el_dir)

    vd_inst = MUSE_vd_inst(da.ll)

    integ_elines, integ_spec = fit_integrated(da, suffix=suffix, tmpdir=el_dir, vd_inst=vd_inst,
                                              kinematic_ties_on=args.enableKinTies,
                                              balmer_limit_on=args.enableBalmerLim, 
                                              noHa=args.noHa,
                                              model=args.model,
                                              degree=args.degree, debug=args.debug, vd_max=args.vd_max,
                                              display_plot=args.displayPlots)
    if integ_elines is not None:
        log.info('Creating emission line extensions.')
        el_info = get_EL_info(integ_elines, args.enableKinTies, args.enableBalmerLim, args.model)
        da.createDobbyExtensions(el_info)
        log.info('Saving integrated spectrum fit.')
        da._c._EL_integ[:] = integ_elines.as_array()
        da._c.EL_integ_continuum[:] = integ_spec['total_lc']
    
    
    kwargs = {'suffix' : suffix,
              'name_template' : name_template,
              'tmpdir' : el_dir,
              'll' : da.ll,
              'vd_inst' : vd_inst,
              'kinematic_ties_on' : args.enableKinTies,
              'balmer_limit_on' : args.enableBalmerLim,
              'noHa' : args.noHa,
              'model' : args.model,
              'degree' : args.degree,
              'debug' : args.debug,
              'vd_max': args.vd_max,
              'display_plot' : args.displayPlots,
             }
    
    queue_length = args.queueLength
    if queue_length <= 0:
        queue_length = args.maxWorkers * 10
    log.info('Setting queue length to %d.' % queue_length)
    

    log.debug('Starting execution pool.')
    map_args = ((*a , kwargs) for a in da)

    with MPIPoolExecutor(args.maxWorkers) as executor:
        while True:
            chunk = list(islice(map_args, queue_length))
            if len(chunk) == 0:
                log.info('Execution completed.')
                break
            log.info('Dispatching %d runs.' % len(chunk))
            if args.maxWorkers == 1:
                log.warning('Running on a single thread.')
                fit_result = starmap(fit_spaxel_kwargs, chunk)
            else:
                fit_result = executor.starmap(fit_spaxel_kwargs, chunk, unordered=True)
            
            for j, i, elines, spec in fit_result:
                log.debug('Saving fit for spaxel [%d, %d].' % (j, i))
                da.updateELines(j, i, elines, spec)
    
    da.save(args.cubeOut)
    
# EOF
###############################################################################
