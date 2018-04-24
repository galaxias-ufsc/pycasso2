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
from os import path, makedirs, copy

import h5py

import numpy as np

from astropy.table import Table
from astropy.io.fits import BinTableHDU

from pycasso2.dobby.fitting import fit_strong_lines
from pycasso2.dobby.utils import plot_el

from read_pycasso_files import readPycasso

########################################################################
# Get galaxy name
galname = sys.argv[1]

########################################################################
# Create output directory

el_dir = ""

outdir = path.join(el_dir, 'el', galname)
if not path.exists(outdir):
    makedirs(outdir)

    
########################################################################
# Read STARLIGHT cube and get data
c = readPycasso(galname, cleverReader = True)

ll = c.l_obs
f_res = (c.f_obs - c.f_syn)
Nl, Nz = c.f_obs.shape

name_template = 'z%05i'

########################################################################
# Calculate vd_instrumental for CALIFA

FWHM_inst = 6. # angstroms
sigma_inst = FWHM_inst / np.sqrt(8. * np.log(2.))

########################################################################
# Fit emission lines in all pixels and save the results into one file per pixel
fit_all = False

if fit_all:
    izs = range(Nz)
else:
    #izs = np.linspace(1, Nz-1, 20)
    #izs = [0, 2700,]
    izs = [0,]
    
for iz in izs:

    # Output name
    name = name_template % (iz)
    outfile = path.join(outdir, '%s.hdf5' % name)

    #++
    if not (path.exists(outfile)):
    #if True:

            
        print ('Fitting zone ', iz)
                    
        # Modelling the gaussian
        el = fit_strong_lines( ll, f_res[..., iz], c.f_syn[..., iz], c.f_err[..., iz],
                               vd_inst = sigma_inst, vd_kms = False, debug = False,
                               kinematic_ties_on = True, model = 'gaussian',
                               lines_windows_file = 'lines_califa.dat',
                               saveAll = True, outname = name + '-1', outdir = outdir, overwrite = True)
        
        # Plot spectra
        fig = plot_el(ll, f_res[..., iz], el, ifig = 0)
        fig.savefig( path.join(outdir, '%s-1.pdf' % name) )

        if iz == 0:
            copy(outfile, path.join(outdir, 'integ.hdf5' % name))

########################################################################
# After pixel-by-pixel fitting, read all individual files and
# save to a super-fits file (including the original STARLIGHT file).

'''
# Read the central pixel to find the emission lines fitted
iz = 0
name = 'p%05i' % (iz)
filename = path.join(outdir, '%s-1.hdf5' % name)
    
with h5py.File(filename, 'r') as f:
    El_lambda = f['elines']['lambda']
    El_name   = f['elines']['line']
    El_l0     = f['elines']['El_l0']

Nl = len(El_lambda)

El_F     = np.zeros((Nl, Nz))
El_v0    = np.zeros((Nl, Nz))
El_vd    = np.zeros((Nl, Nz))
El_EW    = np.zeros((Nl, Nz))
El_lcrms = np.zeros((Nl, Nz))
El_lc    = np.zeros((len(ll), Nz))

# Reading hdf5 files
for iz in izs:

    print ('Reading pixel ', iz)

    name = 'p%05i' % (iz)
    filename = path.join(outdir, '%s-1.hdf5' % name)

    with h5py.File(filename, 'r') as f:

          for il, l in enumerate(El_lambda):
                flag_line = (f['elines']['lambda'] == l)
                El_F [il, iz] = f['elines']['El_F' ][flag_line]
                El_v0[il, iz] = f['elines']['El_v0'][flag_line]
                El_vd[il, iz] = f['elines']['El_vd'][flag_line]
                El_EW[il, iz] = f['elines']['El_EW'][flag_line]
                        
# Save info about each emission line                  
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

# Save fluxes, EWs
c._addExtension('El_F', data=El_F, wcstype='image', overwrite=True)
c._addExtension('El_v0', data=El_v0, wcstype='image', overwrite=True)
c._addExtension('El_vd', data=El_vd, wcstype='image', overwrite=True)
c._addExtension('El_EW', data=El_EW, wcstype='image', overwrite=True)

c.write( path.join(el_dir, '%s-1.el.fits' % galname), overwrite=True )



##################################
# Add up regions of low Ha, Hb S/N

flag_Ha = (El_info['lambda'] == 6563)
flag_Hb = (El_info['lambda'] == 4861)

# TO FIX
import astropy.constants as const
c_light = const.c.to('km/s').value
dl = 1.
El_vdl = El_vd * El_info['l0'][:, np.newaxis, np.newaxis] / c_light
El_N   = dl * El_lcrms * np.sqrt(6. * El_vdl / dl)
El_SN  = El_F / El_N

# Plot SN(Hb)
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.imshow(El_SN[flag_Hb, ...][0])
plt.title(r'$S/N_{\mathrm{H}\beta}$')
plt.colorbar()

# Sim! Inclusive tem uma função
# pycasso2.segmentation.integrate_spectra(). Nota: bin_size é o
# binning inicial na importação, pra somar os erros corretamente. Dá
# uma olhada em
# pycasso2.starlight.synthesis.SynthesisAdapter._readData() pra ver um
# exemplo.
'''


import dobby_save_fits

dobby_save_fits.save_fits(c, galname, outdir, el_dir, name_template)
