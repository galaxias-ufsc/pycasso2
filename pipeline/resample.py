'''
Created on 15/06/2015

@author: andre

Resample spectra in 1 \AA bins and change wavelength interval. 

'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id, read_masterlist, get_wavelength_mask
from diving3d import flags

def velocity_to_redshift(v):
    c = 299792458.0 # km/s
    return v / c

# TODO: read resampling setup from a config file
overwrite = False
redcube = 'data/cubes/T001_AV_r_d_NIT_6_fft_x_0pt15_y_0pt15_n_eq_6_bg_rec_wav_rec_pca_dop_hel.fits'
resamcube = 'data/cubes/T001_resampled.fits'
masterlist = 'data/masterlist_sampleT.txt'
gap_mask_template = 'data/Masks.gap.%s'
telluric_mask_file = 'data/Masks.telluric'

galaxyId = get_galaxy_id(redcube)
ml = read_masterlist(masterlist, galaxyId)

kwargs = dict(l_ini=4000.0,
              l_fin=7200.0,
              dl=1.0,
              width=100,
              height=100,
              ml=ml)

d3d = D3DFitsCube.from_reduced(redcube, **kwargs)

z = velocity_to_redshift(ml['V_hel'])
gap_mask_file = gap_mask_template % ml['grating']
gap_mask = get_wavelength_mask(gap_mask_file, d3d.l_obs, z, dest='rest')
telluric_mask = get_wavelength_mask(telluric_mask_file, d3d.l_obs, z, dest='rest')
d3d.f_flag[gap_mask] |= flags.ccd_gap
d3d.f_flag[telluric_mask] |= flags.telluric

d3d.write(resamcube, overwrite=overwrite)
