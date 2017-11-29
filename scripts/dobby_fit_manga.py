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

import h5py

import numpy as np

from astropy.table import Table
from astropy.io.fits import BinTableHDU

from pycasso2 import FitsCube
from pycasso2 import flags

from pycasso2.dobby.fitting import fit_strong_lines
from pycasso2.dobby.utils import plot_el


########################################################################
# Get galaxy name
galname = sys.argv[1]


########################################################################
# Defining input and output dirs
in_cassiopea = False

if in_cassiopea:
    in_dir = '/home/ASTRO/manga/dr14/starlight/dr14.bin2.cA.Ca0c_6x16/'
    el_dir = '/home/ASTRO/manga/dr14/starlight_el/dr14.bin2.cA.Ca0c_6x16/'    
else:
    in_dir = '/Users/natalia/data/MaNGA/dr14/starlight/dr14.bin2.cA.Ca0c_6x16/'
    el_dir = '/Users/natalia/data/MaNGA/dr14/starlight_el/dr14.bin2.cA.Ca0c_6x16/'    

# Create output directory
outdir = path.join(el_dir, 'el', galname)
if not path.exists(outdir):
    makedirs(outdir)

    
########################################################################
# Read STARLIGHT cube and get data
c = FitsCube(path.join(in_dir, 'manga-%s.dr14.bin2.cA.Ca0c_6x16.fits' % galname))

ll = c.l_obs

f_res = (c.f_obs - c.f_syn)
f_flagged = ((flags.before_starlight & c.f_flag) > 0)
f_res[f_flagged] = np.ma.masked
Nl, Ny, Nx = c.f_obs.shape

########################################################################
# Fit emission lines in all pixels and save the results into one file per pixel
fit_all = False

if fit_all:
    iys, ixs = range(Ny), range(Nx)
else:
    iys, ixs = [c.y0,], [c.x0,]

    
for iy in iys:
  for ix in ixs:

        # Only measure emission lines if STARLIGHT was run on that pixel
        if (not c.synthImageMask[iy, ix]):

            # Output name
            name = 'p%04i-%04i' % (iy, ix)
            outfile = path.join(outdir, '%s.hdf5' % name)
            
            if not (path.exists(outfile)):

            
                print ('Fitting pixel ', iy, ix)
                    
                # Modelling the gaussian
                el = fit_strong_lines( ll, f_res[..., iy, ix], c.f_syn[..., iy, ix], f_res[..., iy, ix],
                                       kinematic_ties_on = True, 
                                       saveAll = True, outname = name, outdir = outdir, overwrite = True)
                mod_fit_HbHaN2, mod_fit_O3, el_extra = el
                
                
                # Plot spectra
                fig = plot_el(ll, f_res[..., iy, ix], el, ifig = 2)
                fig.savefig( path.join(outdir, '%s.pdf' % name) )


########################################################################
# After pixel-by-pixel fitting, read all individual files and
# save to a super-fits file (including the original STARLIGHT file).

# Read the central pixel to find the emission lines fitted
iy, ix = c.y0, c.x0
name = 'p%04i-%04i' % (iy, ix)
filename = path.join(outdir, '%s.hdf5' % name)
    
with h5py.File(filename, 'r') as f:
    El_lambda = f['elines']['lambda']
    El_name   = f['elines']['line']
    El_l0     = f['elines']['El_l0']

Nl = len(El_lambda)

El_F  = np.zeros((Nl, Ny, Nx))
El_v0 = np.zeros((Nl, Ny, Nx))
El_vd = np.zeros((Nl, Ny, Nx))
El_EW = np.zeros((Nl, Ny, Nx))

# Reading hdf5 files
for iy in iys:
  for ix in ixs:

        if (c.SN_normwin[iy, ix] > 3):

            print ('Reading pixel ', iy, ix)

            name = 'p%04i-%04i' % (iy, ix)
            filename = path.join(outdir, '%s.hdf5' % name)
    
            with h5py.File(filename, 'r') as f:

                  for il, l in enumerate(El_lambda):
                        flag_line = (f['elines']['lambda'] == l)
                        El_F [il, iy, ix] = f['elines']['El_F' ][flag_line]
                        El_v0[il, iy, ix] = f['elines']['El_v0'][flag_line]
                        El_vd[il, iy, ix] = f['elines']['El_vd'][flag_line]
                        El_EW[il, iy, ix] = f['elines']['El_EW'][flag_line]
                        
# Save info about each emission line                  
# Central wavelength - TO ADD to this table ++
aux = { 'lambda': El_lambda,
        'name'  : El_name,
        'l0'    : El_l0,
         'model' : Nl*['GaussianIntELModel'],
        'kinematic_ties_on' : Nl*[True],
        }
    
El_info = Table(aux)
El_info.convert_unicode_to_bytestring()
El_info_HDU = BinTableHDU(data=El_info.as_array(), name='El_info')
c._HDUList.append(El_info_HDU)

# Save fluxes, EWs, etc ++
c._addExtension('El_F', data=El_F, wcstype='image', overwrite=True)
c._addExtension('El_v0', data=El_v0, wcstype='image', overwrite=True)
c._addExtension('El_vd', data=El_vd, wcstype='image', overwrite=True)
c._addExtension('El_EW', data=El_EW, wcstype='image', overwrite=True)

c.write( path.join(el_dir, 'manga-%s.dr14.bin2.cA.Ca0c_6x16.El.fits' % galname), overwrite=True )
