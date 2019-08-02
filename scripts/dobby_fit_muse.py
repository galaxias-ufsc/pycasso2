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

from multiprocessing import cpu_count
import multiprocessing as mp

from pycasso2 import FitsCube
from pycasso2 import flags
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
    parser.add_argument('--out', dest='cubeOut', required=True,
                        help='Output cube.')
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
log.setLevel('DEBUG')

args = parse_args()
log.info('Loading cube %s.' % args.cubeIn[0])
c = FitsCube(args.cubeIn[0])
galname = c.name

# Create output directory
tmpdir = path.join(args.tmpDir, galname)
if not path.exists(tmpdir):
    log.debug('Creating directory %s.' % tmpdir)
    makedirs(tmpdir)

# TODO: read from config
name_template = 'p%04i-%04i'
###############################################################################

###############################################################################
# Multiprocessing functions
def func_with_kwargs(iy, ix, kwargs):
    return fit_spaxel(iy, ix, **kwargs)

def iter_with_kwargs(args, **kwargs):
    for x in args:
        yield x.tolist() + [kwargs,]
###############################################################################

###############################################################################
# Fit emission lines in all pixels and save the results into one file per pixel
def fit_spaxel(iy, ix,
               hasSegmentationMask, synthImageMask,
               suffix, name_template, tmpdir,
               ll, f_res, f_syn, f_err, vd_inst,
               kinematic_ties_on, balmer_limit_on, model,
               degree, debug, display_plot):
    '''
    Fit only one spaxel
    '''
    # Only measure emission lines if STARLIGHT was run on that pixel
    if ~(not hasSegmentationMask and synthImageMask[iy, ix]):

        # Output name
        name = suffix + '.' + name_template % (iy, ix)
        outfile = path.join(tmpdir, '%s.hdf5' % name)
    
        if not (path.exists(outfile)):
    
            log.info('Fitting pixel [%d, %d]' % (iy, ix))
            # Modelling the gaussian
            el = fit_strong_lines(ll, f_res[..., iy, ix], f_syn[..., iy, ix], f_err[..., iy, ix], vd_inst = vd_inst,
                                  kinematic_ties_on = kinematic_ties_on, balmer_limit_on = balmer_limit_on, model = model,
                                  degree = degree,
                                  saveAll = True, outname = name, outdir = tmpdir, overwrite = True,
                                  vd_kms = True)
    
            if debug:
                # Plot spectrum
                fig = plot_el(ll, f_res[..., iy, ix], el, ifig = 0, display_plot = display_plot)
                fig.savefig( path.join(tmpdir, '%s.pdf' % name) )
            
    return None
###############################################################################

###############################################################################
# Fit all data cube
def fit(kinematic_ties_on, balmer_limit_on, model, correct_good_frac=False):

    _k = 1 * kinematic_ties_on
    _b = 1 * balmer_limit_on
    if model == 'gaussian':
        _m = 'GA'
    elif model == 'resampled_gaussian':
        _m = 'RG'
    else:
        raise Exception('Unknown model: %s' % model)
    suffix = 'El%sk%ib%i' % (_m, _k, _b)

    ll = c.l_obs
    f_res = (c.f_obs - c.f_syn)
    f_flagged = ((flags.no_starlight & c.f_flag) > 0)
    f_res[f_flagged] = np.ma.masked

    # Calc vd_inst. Hard-coded for now!
    vdi_R2000 = 64.
    vdi_R4000 = 32.
    llim = 6000.
    vd_inst = np.where(ll <= llim, vdi_R2000, vdi_R4000)

    if c.hasSegmentationMask:
        f_res = f_res[..., np.newaxis]
        f_syn = c.f_syn[..., np.newaxis]
        f_obs = c.f_obs[..., np.newaxis]
        f_err = c.f_err[..., np.newaxis]
        Ny = c.Nzone
        Nx = 1
        y0 = 0
        x0 = 0
    else:
        f_syn = c.f_syn
        f_obs = c.f_obs
        f_err = c.f_err
        Ny = c.Ny
        Nx = c.Nx
        y0 = c.y0
        x0 = c.x0

    if correct_good_frac:
        f_obs *= c.seg_good_frac
        f_syn *= c.seg_good_frac
        f_res *= c.seg_good_frac
        f_err *= c.seg_good_frac
        
    # Pixels to fit
    if args.onlyCenter:
        log.warn('Fitting only central spaxel.')
        iys, ixs = [y0,], [x0,]
    else:
        iys, ixs = np.arange(Ny), np.arange(Nx)

    kwargs = {'hasSegmentationMask' : c.hasSegmentationMask,
              'synthImageMask' : c.synthImageMask.copy(),
              'suffix' : suffix,
              'name_template' : name_template,
              'tmpdir' : tmpdir,
              'll' : ll.copy(),
              'f_res' : f_res.copy(),
              'f_syn' : f_syn.copy(),
              'f_err' : f_err.copy(),
              'vd_inst' : vd_inst,
              'kinematic_ties_on' : kinematic_ties_on,
              'balmer_limit_on' : balmer_limit_on,
              'model' : model,
              'degree' : args.degree,
              'debug' : args.debug,
              'display_plot' : args.displayPlots,
             }
                
    # Fit spaxel by spaxel
    if args.nProc == 1:

        np.random.shuffle(iys)
        np.random.shuffle(ixs)
        
        for iy in iys:
            for ix in ixs:
                fit_spaxel(iy, ix, **kwargs)
            
    else:

        log.info('Starting multithreading...')

        _ixs, _iys = np.meshgrid(ixs, iys)
        ixs_iys = np.vstack([_ixs.flatten(), _iys.flatten()]).T
        
        pool = mp.Pool(args.nProc)
        pool.starmap(func_with_kwargs, iter_with_kwargs(ixs_iys, **kwargs), chunksize=100)
        pool.close()
        
    log.debug('Fitting integrated spectrum...')
    name = suffix + '.' + 'integ'
    outfile = path.join(tmpdir, '%s.hdf5' % name)
            
    if not path.exists(outfile):

        # Add all spaxels (to take into account the seg_good_frac if using)
        cov_factor_A = 0.0
        cov_factor_B = 1.0
        segmask = np.where(c.synthImageMask, 0, 1)[np.newaxis, ...]
        f_flag = (c.f_flag & flags.no_obs) == 0
        f_syn, f_err, good_frac = sum_spectra(segmask, f_syn, f_err, f_flag, cov_factor_A, cov_factor_B)
        f_obs, _, _ = sum_spectra(segmask, f_obs, c.f_err, f_flag, cov_factor_A, cov_factor_B)
        f_obs = np.ma.masked_where((good_frac <= 0.5), f_obs).squeeze()
        f_syn = np.ma.masked_where((good_frac <= 0.5), f_syn).squeeze()
        f_err = np.ma.masked_where((good_frac <= 0.5), f_err).squeeze()
        f_res = f_obs - f_syn
        f_flag = np.where(good_frac <= 0.5, flags.no_obs, 0).squeeze()

        # Fit with dobby
        el = fit_strong_lines( ll, f_res, f_syn, f_err, vd_inst = vd_inst,
                               kinematic_ties_on = kinematic_ties_on, balmer_limit_on = balmer_limit_on, model = model,
                               degree = args.degree,
                               saveAll = True, outname = name, outdir = tmpdir, overwrite = True,
                               vd_kms = True)
        if args.debug:
            # Plot integrated spectrum
            fig = plot_el(ll, f_res, el, ifig = 0, display_plot = args.displayPlots)
            fig.savefig( path.join(tmpdir, '%s.pdf' % name) )

        
    # After pixel-by-pixel fitting, read all individual files and
    # save to a super-fits file (including the original STARLIGHT file).
    dobby_save_fits_pixels(c, args.cubeOut, tmpdir, name_template,
                           suffix, kinTies = kinematic_ties_on, balLim = balmer_limit_on, model = model)
###############################################################################

###############################################################################
# Call function to fit all data cube

fit(kinematic_ties_on=args.enableKinTies, balmer_limit_on=args.enableBalmerLim, model=args.model, correct_good_frac=args.correct_good_frac)

# EOF
###############################################################################
