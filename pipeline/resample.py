'''
Created on 15/06/2015

@author: andre

Resample spectra in 1 \AA bins and change wavelength interval. 

'''

from diving3d.cube import D3DFitsCube
from diving3d.masterlist import get_galaxy_id

redcube = 'data/cubes/T001_AV_r_d_NIT_6_fft_x_0pt15_y_0pt15_n_eq_6_bg_rec_wav_rec_pca_dop_hel.fits'
resamcube = 'data/cubes/T001_resampled.fits'

kwargs = dict(l_ini=4000.0,
              l_fin=7200.0,
              dl=1.0,
              width=100,
              height=100)

d3d = D3DFitsCube.from_reduced(redcube, **kwargs)
galaxyId = get_galaxy_id()
d3d.write(resamcube)
