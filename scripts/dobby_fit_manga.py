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
    parser.add_argument('--degree', dest='degree', type=int, default=16,
                        help='Degree for Legendre polynomial fits in the local continuum. Default: 16')

    return parser.parse_args()
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
    
########################################################################
# Fit emission lines in all pixels and save the results into one file per pixel
def fit_spaxel(iy, ix,
               suffix, name_template, tmpdir,
               ll, f_res, f_syn, f_err, f_disp,
               kinematic_ties_on, balmer_limit_on, model,
               degree, debug, display_plot):
    '''
    Fit only one spaxel
    '''
    # Output name
    name = suffix + '.' + name_template % (iy, ix)
    outfile = path.join(tmpdir, '%s.hdf5' % name)

    if not (path.exists(outfile)):

        log.info('Fitting pixel [%d, %d]' % (iy, ix))
        # Modelling the gaussian
        el = fit_strong_lines(ll, f_res[..., iy, ix], f_syn[..., iy, ix], f_err[..., iy, ix], vd_inst = f_disp[..., iy, ix],
                              kinematic_ties_on = kinematic_ties_on, balmer_limit_on = balmer_limit_on, model = model,
                              degree = degree,
                              saveAll = True, outname = name, outdir = tmpdir, overwrite = True,
                              vd_kms = False)

        if debug:
            # Plot spectrum
            fig = plot_el(ll, f_res[..., iy, ix], el, ifig = 0, display_plot = display_plot)
            fig.savefig( path.join(tmpdir, '%s.pdf' % name) )
            
    return None

def fit(kinematic_ties_on, balmer_limit_on, model):

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

    if c.hasSegmentationMask:
        f_res = f_res[..., np.newaxis]
        f_syn = c.f_syn[..., np.newaxis]
        f_err = c.f_err[..., np.newaxis]
        f_disp = c.f_disp[..., np.newaxis]
        Ny = c.Nzone
        Nx = 1
        y0 = 0
        x0 = 0
    else:
        f_syn = c.f_syn
        f_err = c.f_err
        f_disp = c.f_disp
        Ny = c.Ny
        Nx = c.Nx
        y0 = c.y0
        x0 = c.x0
        
    # Pixels to fit
    if args.onlyCenter:
        log.warn('Fitting only central spaxel.')
        iys, ixs = [y0,], [x0,]
    else:
        iys, ixs = range(Ny), range(Nx)

    kwargs = {'suffix' : suffix,
              'name_template' : name_template,
              'tmpdir' : tmpdir,
              'll' : ll,
              'f_res' : f_res,
              'f_syn' : f_syn,
              'f_err' : f_err,
              'f_disp' : f_disp,
              'kinematic_ties_on' : kinematic_ties_on,
              'balmer_limit_on' : balmer_limit_on,
              'model' : model,
              'degree' : args.degree,
              'debug' : args.debug,
              'display_plot' : args.displayPlots,
             }
                
    # Fit spaxel by spaxel
    if args.nProc == 1:
        
        for iy in iys:
            for ix in ixs:    
                # Only measure emission lines if STARLIGHT was run on that pixel
                if not c.hasSegmentationMask and c.synthImageMask[iy, ix]:
                    continue
                fit_spaxel(iy, ix, **kwargs)
            
    else:

        log.info('Starting multithreading...')
        processes = []

        for iy in iys:
            for ix in ixs:    
                # Only measure emission lines if STARLIGHT was run on that pixel
                if not c.hasSegmentationMask and c.synthImageMask[iy, ix]:
                    continue
                p = mp.Process(target=fit_spaxel, args=(ix, iy), kwargs = kwargs)
                p.start()
                processes.append(p)

        for p in processes:
            p.join()
        
    log.debug('Fitting integrated spectrum...')
    integ_f_res = (c.integ_f_obs - c.integ_f_syn)
    name = suffix + '.' + 'integ'
    outfile = path.join(tmpdir, '%s.hdf5' % name)
            
    if not path.exists(outfile):
        el = fit_strong_lines( ll, integ_f_res, c.integ_f_syn, c.integ_f_err, vd_inst = c.integ_f_disp,
                               kinematic_ties_on = kinematic_ties_on, balmer_limit_on = balmer_limit_on, model = model,
                               degree = args.degree,
                               saveAll = True, outname = name, outdir = tmpdir, overwrite = True,
                               vd_kms = False)
        if args.debug:
            # Plot integrate spectrum
            fig = plot_el(ll, integ_f_res, el, ifig = 0, display_plot = args.displayPlots)
            fig.savefig( path.join(tmpdir, '%s.pdf' % name) )
    

        
    # After pixel-by-pixel fitting, read all individual files and
    # save to a super-fits file (including the original STARLIGHT file).
    dobby_save_fits_pixels(c, args.cubeOut, tmpdir, name_template,
                           suffix, kinTies = kinematic_ties_on, balLim = balmer_limit_on, model = model)

fit(kinematic_ties_on=args.enableKinTies, balmer_limit_on=args.enableBalmerLim, model=args.model)


# EOF
