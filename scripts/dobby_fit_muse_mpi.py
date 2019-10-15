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

from os import path, makedirs
import numpy as np
import argparse
from astropy import log
from contextlib import closing

from multiprocessing import cpu_count
import multiprocessing as mp

from pycasso2 import FitsCube
from pycasso2.segmentation import sum_spectra
from pycasso2.config import default_config_path
from pycasso2.dobby.fitting import fit_strong_lines
from pycasso2.dobby.utils import plot_el
from pycasso2.dobby.save_fits import dobby_save_fits_pixels


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
    parser.add_argument('--nproc', dest='nProc', type=int, default=cpu_count() - 1,
                        help='Number of worker processes.')
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
    parser.add_argument('--correct-good-frac', dest='correct_good_frac', action='store_true',
                        help='Correct spectra for the segmentation good frac spec prior to fitting.')
    parser.add_argument('--degree', dest='degree', type=int, default=16,
                        help='Degree for Legendre polynomial fits in the local continuum. Default: 16')

    return parser.parse_args()
###############################################################################

###############################################################################
class EmLineInput(object):
    def __init__(self, filename, correct_good_frac=False):
        log.debug('Reading cube: %s' % filename)
        with closing(FitsCube(filename, memmap=False)) as c:
            self.galname = c.name
            self.ll = c.l_obs
            self.integ_mask = np.where(c.synthImageMask, 0, 1)[np.newaxis, ...]
            self.v_0 = c.v_0
            self.v_d = c.v_d
            self.integ_v_0 = c.synthIntegKeywords['v0']
            self.integ_v_d = c.synthIntegKeywords['vd']
            
            if c.hasSegmentationMask:
                log.debug('Cube is segmented.')
                f_obs = c.f_obs[..., np.newaxis]
                self.f_syn = c.f_syn[..., np.newaxis]
                self.f_err = c.f_err[..., np.newaxis]
                self.Ny = c.Nzone
                self.Nx = 1
                self.y0 = 0
                self.x0 = 0
            else:
                log.debug('Cube is spatially resolved.')
                f_obs = c.f_obs
                self.f_syn = c.f_syn
                self.f_err = c.f_err
                self.Ny = c.Ny
                self.Nx = c.Nx
                self.y0 = c.y0
                self.x0 = c.x0
        
        if correct_good_frac:
            log.debug('Correcting fractional segments.')
            f_obs *= c.seg_good_frac
            self.f_syn *= c.seg_good_frac
            self.f_err *= c.seg_good_frac
        
        log.debug('Calculating integrated data.')
        self.f_res_i, self.f_syn_i, self.f_err_i = calc_integrated(f_obs, self.f_syn, self.f_err, self.integ_mask)
            
        self.f_res = (f_obs - self.f_syn)
        assert not (np.ma.getmaskarray(self.f_res) ^ (np.ma.getmaskarray(self.f_syn) | np.ma.getmaskarray(f_obs))).any()
    
    
    def getArgs(self, j, i):
        return j, i, self.f_res[:, j, i], self.f_syn[:, j, i], self.f_err[:, j, i], self.v_0[j, i], self.v_d[j, i]
    
    
    def __iter__(self):
        for j in range(self.Ny):
            for i in range(self.Nx):
                yield self.getArgs(j, i)
    
###############################################################################


###############################################################################
def calc_integrated(f_obs, f_syn, f_err, integ_mask):
    # Add all spaxels (to take into account the seg_good_frac if using)
    f_obs, _, _ = sum_spectra(integ_mask, f_obs, f_err, cov_factor_A=0.0, cov_factor_B=1.0)
    f_syn, f_err, good_frac = sum_spectra(integ_mask, f_syn, f_err, cov_factor_A=0.0, cov_factor_B=1.0)
    f_obs = np.ma.masked_where((good_frac <= 0.5), f_obs).squeeze()
    f_syn = np.ma.masked_where((good_frac <= 0.5), f_syn).squeeze()
    f_err = np.ma.masked_where((good_frac <= 0.5), f_err).squeeze()
    f_res = f_obs - f_syn
    return f_res, f_syn, f_err
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
def func_with_kwargs(j, i, f_res, f_syn, f_err, v_0, v_d, kwargs):
    return fit_spaxel(j, i, f_res, f_syn, f_err, v_0, v_d, **kwargs)

def iter_with_kwargs(args, **kwargs):
    for x in args:
        yield list(x) + [kwargs,]
###############################################################################

###############################################################################
# Fit emission lines in all pixels and save the results into one file per pixel
def fit_spaxel(iy, ix, f_res, f_syn, f_err, stellar_v0, stellar_vd,
               suffix, name_template, tmpdir,
               ll, vd_inst,
               kinematic_ties_on, balmer_limit_on, model,
               degree, debug, display_plot, legendre_stellar_mask=True):
    Nmasked = np.ma.getmaskarray(f_res).sum()
    if (Nmasked / len(f_res)) > 0.5:
        log.debug('Skipping masked spaxel [%d, %d]' % (iy, ix))
        return
    
    # Output name
    name = suffix + '.' + name_template % (iy, ix)
    outfile = path.join(tmpdir, '%s.hdf5' % name)

    if not (path.exists(outfile)):

        log.info('Fitting pixel [%d, %d]' % (iy, ix))
        # Modelling the gaussian
        el = fit_strong_lines(ll, f_res, f_syn, f_err, vd_inst = vd_inst,
                              kinematic_ties_on = kinematic_ties_on, balmer_limit_on = balmer_limit_on, model = model,
                              degree = degree,
                              saveAll = True, outname = name, outdir = tmpdir, overwrite = True,
                              vd_kms = True,
                              stellar_v0=stellar_v0, stellar_vd=stellar_vd, legendre_stellar_mask=legendre_stellar_mask)

        if debug:
            # Plot spectrum
            fig = plot_el(ll, f_res, el, ifig = 0, display_plot = display_plot)
            fig.savefig( path.join(tmpdir, '%s.pdf' % name) )
###############################################################################

###############################################################################
# Fit all data cube
def fit(cubefile, suffix):

    log.info('Loading cube %s.' % cubefile)
    data = EmLineInput(cubefile, correct_good_frac=args.correct_good_frac)

    el_dir = path.join(args.tmpDir, data.galname)
    if not path.exists(args.tmpDir):
        log.debug('Creating directory %s.' % el_dir)
        makedirs(el_dir)

    # Calc vd_inst. Hard-coded for now!
    vdi_R2000 = 64.
    vdi_R4000 = 32.
    llim = 6000.
    vd_inst = np.where(data.ll <= llim, vdi_R2000, vdi_R4000)

    
    # Pixels to fit
    if args.onlyCenter:
        data.x0=2
        data.y0=3
        log.warning('Fitting only central spaxel.')
        data_iter = [data.getArgs(data.y0, data.x0)]
    else:
        data_iter = data

    kwargs = {'suffix' : suffix,
              'name_template' : name_template,
              'tmpdir' : el_dir,
              'll' : data.ll,
              'vd_inst' : vd_inst,
              'kinematic_ties_on' : args.enableKinTies,
              'balmer_limit_on' : args.enableBalmerLim,
              'model' : args.model,
              'degree' : args.degree,
              'debug' : args.debug,
              'display_plot' : args.displayPlots,
             }
                
    # Fit spaxel by spaxel
    if args.nProc == 1:
        for a in data_iter:
            fit_spaxel(*a, **kwargs)
    else:
        log.info('Using %d processes.' % args.nProc)
        with mp.Pool(args.nProc) as pool:
            pool.starmap(func_with_kwargs, iter_with_kwargs(data_iter, **kwargs), chunksize=10)

        
    log.info('Fitting integrated spectrum...')
    name = suffix + '.' + 'integ'
    outfile = path.join(el_dir, '%s.hdf5' % name)
            
    if not path.exists(outfile):

        # Fit with dobby
        el = fit_strong_lines(data.ll, data.f_res_i, data.f_syn_i, data.f_err_i, vd_inst=vd_inst,
                              kinematic_ties_on=args.enableKinTies, balmer_limit_on=args.enableBalmerLim,
                              model=args.model, degree=args.degree, saveAll=True, outname=name, outdir=el_dir,
                              overwrite=True, vd_kms=True, stellar_v0=data.integ_v_0,
                              stellar_vd=data.integ_v_d, legendre_stellar_mask=True)
                               
        if args.debug:
            # Plot integrated spectrum
            fig = plot_el(data.ll, data.f_res_i, el, ifig=0, display_plot=args.displayPlots)
            fig.savefig(path.join(el_dir, '%s.pdf' % name))

    return el_dir
        
###############################################################################

###############################################################################
# Call function to fit all data cube

###############################################################################
if __name__ == '__main__':
    log.setLevel('DEBUG')
    
    args = parse_args()
    cube_in = args.cubeIn[0]
    
    if args.cubeOut is None:
        log.warning('Output not set, this will update the input file!')
    
    # TODO: read from config
    name_template = 'p%04i-%04i'
    suffix = get_suffix(args.model, args.enableKinTies, args.enableBalmerLim)
    
    log.info('Beginning fit routine.')
    el_dir = fit(cube_in, suffix)
    
    log.info('Reading fit results from %s' % el_dir)
    log.debug('Reloading cube %s' % cube_in)
    with closing(FitsCube(cube_in, mode='append')) as c:
        dobby_save_fits_pixels(c, args.cubeOut, el_dir, name_template, suffix,
                               kinTies=args.enableKinTies, balLim=args.enableBalmerLim, model=args.model)

# EOF
###############################################################################
