'''
Created on 15/06/2015

@author: andre

Resample spectra in 1 \AA bins and change wavelength interval. 

'''

from pystarlight.util.StarlightUtils import spec_resample
import pyfits
import numpy as np

def get_wl(head):
    dl = head['CD3_3']
    l_ini = head['CRVAL3'] + (1 - head['CRPIX3']) * dl
    l_fin = l_ini + (head['NAXIS3'] - 1) * dl
    return np.arange(l_ini, l_fin + dl, dl)


cube = 'cubes/T001_AV_r_d_NIT_6_fft_x_0pt15_y_0pt15_n_eq_6_bg_rec_wav_rec_pca_dop_hel.fits'
flux_unit = 1e-15
l_ini = 4300.0
l_fin = 6800.0
dl = 1.0

f = pyfits.open(cube)

head = f[0].header
Nx = head['NAXIS1']
Ny = head['NAXIS2']
Nl = head['NAXIS3']
flux = f[0].data

l_orig = get_wl(head)
l_res = np.arange(l_ini, l_fin, dl)

flux_res = np.zeros((len(l_res), Ny, Nx))

for i in xrange(Nx):
    for j in xrange(Ny):
        print i, j
        flux_res[:, j, i] = spec_resample(l_orig, l_res, flux[:, j, i])

assert np.allclose(np.trapz(flux, l_orig, axis=0), np.trapz(flux_res, l_res, axis=0), atol=0.1)




