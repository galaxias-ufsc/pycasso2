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

import sys
from os import path, makedirs

import numpy as np

from astropy.table import Table

from pycasso2 import FitsCube
from pycasso2 import flags

from pycasso2.dobby.fitting import fit_strong_lines
from pycasso2.dobby.utils import plot_el


########################################################################
# Options
in_cassiopea = True
fit_all = True
debug = False


########################################################################
# Get galaxy name
galname = sys.argv[1]


########################################################################
# Defining input and output dirs

if in_cassiopea:
    in_dir = '/home/ASTRO/manga/dr14/starlight/dr14.bin2.cA.CB17_7x16/'
    el_dir = '/home/ASTRO/manga/dr14/starlight_el/dr14.bin2.cA.CB17_7x16/'    
else:
    in_dir = '/Users/natalia/data/MaNGA/dr14/starlight/dr14.bin2.cA.CB17_7x16/'
    el_dir = '/Users/natalia/data/MaNGA/dr14/starlight_el/dr14.bin2.cA.CB17_7x16/'    

# Create output directory
outdir = path.join(el_dir, 'el', galname)
if not path.exists(outdir):
    makedirs(outdir)

    
########################################################################
# Read STARLIGHT cube and get data
c = FitsCube(path.join(in_dir, 'manga-%s.dr14.bin2.cA.CB17_7x16.fits' % galname))

ll = c.l_obs

f_res = (c.f_obs - c.f_syn)
f_flagged = ((flags.before_starlight & c.f_flag) > 0)
f_res[f_flagged] = np.ma.masked

vd_inst = 70.

Nl, Ny, Nx = c.f_obs.shape
name_template = 'p%04i-%04i'


########################################################################
# Pixels to fit
if fit_all:
    iys, ixs = range(Ny), range(Nx)
else:
    iys, ixs = [c.y0,], [c.x0,]
    #iys, ixs = np.arange(11, 16), np.arange(11, 16)

    
########################################################################
# Fit emission lines in all pixels and save the results into one file per pixel

def fit(kinematic_ties_on, balmer_limit_on, model):

    _k = 1 * kinematic_ties_on
    _b = 1 * balmer_limit_on
    if model == 'gaussian': _m = 'GA'
    if model == 'resampled_gaussian': _m = 'RG'
    suffix = 'El%sk%ib%i' % (_m, _k, _b)
    
    for iy in iys:
      for ix in ixs:
    
            # Only measure emission lines if STARLIGHT was run on that pixel
            #if (not c.synthImageMask[iy, ix]):
            if True:
                # Output name
                name = suffix + '.' + name_template % (iy, ix)
                outfile = path.join(outdir, '%s.hdf5' % name)
    
                if not (path.exists(outfile)):
                
                    print ('Fitting pixel ', iy, ix)
    
                    # Modelling the gaussian
                    el = fit_strong_lines( ll, f_res[..., iy, ix], c.f_syn[..., iy, ix], c.f_err[..., iy, ix], vd_inst = vd_inst,
                                           kinematic_ties_on = kinematic_ties_on, balmer_limit_on = balmer_limit_on, model = model,
                                           saveAll = True, outname = name, outdir = outdir, overwrite = True)
    
                    if debug:
                        # Plot spectrum
                        fig = plot_el(ll, f_res[..., iy, ix], el, ifig = 0)
                        fig.savefig( path.join(outdir, '%s.pdf' % name) )
    
    
    # Fit integrated spectrum
    integ_f_res = (c.integ_f_obs - c.integ_f_syn)
    name = suffix + '.' + 'integ'
    el = fit_strong_lines( ll, integ_f_res, c.integ_f_syn, c.integ_f_err, vd_inst = vd_inst,
                           kinematic_ties_on = kinematic_ties_on, balmer_limit_on = balmer_limit_on, model = model,
                           saveAll = True, outname = name, outdir = outdir, overwrite = True)
    if debug:
        # Plot integrate spectrum
        fig = plot_el(ll, integ_f_res, el, ifig = 0)
        fig.savefig( path.join(outdir, '%s.pdf' % name) )
    

        
    # After pixel-by-pixel fitting, read all individual files and
    # save to a super-fits file (including the original STARLIGHT file).
    import dobby_save_fits
    dobby_save_fits.save_fits(c, galname, outdir, el_dir, name_template,
                              suffix, kinTies = kinematic_ties_on, balLim = balmer_limit_on, model = model)

    
# Fit!
#++for kin_ties in [True, False]:
#++    for balmer_lim in [True, False]:
#++        for model in ['gaussian', 'resampled_gaussian']:
#++            fit(kinematic_ties_on = kin_ties, balmer_limit_on = balmer_lim, model = model)

for kin_ties in [False, ]:
    for balmer_lim in [False, ]:
        for model in ['gaussian', ]:
            fit(kinematic_ties_on = kin_ties, balmer_limit_on = balmer_lim, model = model)

# EOF
