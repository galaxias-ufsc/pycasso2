'''

Created on 08/26/1016

@author: ariel

Provides an example of the reddening module.

Note: change datadir and plotdir to your local directories before running.

''' 

import numpy as np
from astroML.datasets import fetch_sdss_spectrum
import pycasso2.reddening as rd
import matplotlib.pyplot as plt
import seaborn as sns
ma = np.ma


#User-dependent parameters:
datadir = '/home/ariel/Workspace/GALEX-SDSS Match/Data/'
plotdir = '/home/ariel/Workspace/GALEX-SDSS Match/Plots/'


#Reading dust map
dust_map = rd.get_EBV_map('planck_dust_maps.fits', datadir)


#Fetching the SDSS spectra of an example galaxy using astroML:
plate,mjd,fiber = (2953, 54560, 516)
spec = fetch_sdss_spectrum(plate,mjd,fiber, data_home = datadir )


#Other parameters for this galaxy:
ra, dec = (231.13958444539463, 2.031068342199648)
objid   = 1237651736853086463


#Getting corrected spctra and other parameters:
corr_spec = rd.extinction_corr(spec.wavelength(),spec.spectrum
,ra,dec,EBV_map=dust_map)
A_lambdas, EBV = rd.calc_extinction(ra,dec,spec.wavelength(),dust_map)


#Plotting:

fig = plt.figure(figsize=(5,8))

plt.subplot(211)

plt.plot(spec.wavelength()
, ma.masked_array(data = spec.spectrum
, mask = spec.compute_mask()), '-k', linewidth=.3, label = 'Spectrum')

plt.plot(spec.wavelength()
, ma.masked_array(data= spec.spectrum
, mask = np.logical_not(spec.compute_mask()))
, color = sns.color_palette()[4], linewidth=.3, label = 'Mask')

plt.plot(spec.wavelength(), spec.error , '--r'
, linewidth = .5, label = 'Error Spectrum')

plt.plot(spec.wavelength(),corr_spec, color = sns.color_palette()[0]
, linewidth=.3, label = 'Extinction Corrected Spectrum')

plt.ylim(0, 1.3 * corr_spec[np.argmax(corr_spec)]) 
plt.xlabel(r'$\lambda \; [\mathrm{\AA}]$', fontsize = 15)
plt.ylabel(r'$F_{\lambda}[10^{-17} \mathrm{erg \; cm^{-2}  \AA^{-1}  s^{-1} }]$', fontsize = 15)

plt.legend(loc=2, fontsize = 10)

plt.title('RA = ' + str(ra)[0:5] + ', Dec = ' + str(dec)[0:5] + ', E(B-V) = ' + str(EBV)[0:6])

plt.subplot(212)

plt.plot( ( 1 / spec.wavelength() * 10000 ) ,A_lambdas, label = 'E(B-V)='+str(EBV)[0:6]
, color = sns.color_palette()[2])
plt.xlabel(r'$1 / \lambda \; [\mathrm{\mu}^{-1}]$', fontsize = 15)
plt.ylabel(r'$A_{\lambda} \; [\mathrm{mag}]$', fontsize = 15)

fig.subplots_adjust(hspace=0.3)

plt.savefig(plotdir + str(objid) + '_spec.png', dpi=500)
plt.show()
