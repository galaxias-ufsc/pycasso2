'''
Created on 15/06/2015

@author: andre

Resample spectra in 1 \AA bins and change wavelength interval. 

'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import read_masterlist, get_wavelength_mask
from diving3d.resampling import velocity_to_redshift
from diving3d import flags
from os import path

# TODO: read resampling setup from a config file
overwrite = False
galaxy_id = 'T001'
cube_dir = 'data/cubes/'
cube_obs_dir = 'data/cubes_obs/'
cube_resam_dir = 'data/cubes_resam/'
masterlist = 'data/masterlist_sampleT.txt'
gap_mask_template = 'data/Masks.gap.%s'
telluric_mask_file = 'data/Masks.telluric'

ml = read_masterlist(masterlist, galaxy_id)
cube = path.join(cube_dir, ml['cube'])
cube_obs = path.join(cube_obs_dir, ml['cube_obs'])
cube_resam = path.join(cube_obs_dir, '%s_resampled.fits')
print ml
kwargs = dict(l_ini=4000.0,
              l_fin=7200.0,
              dl=1.0,
              width=100,
              height=100,
              ml=ml)

d3d = D3DFitsCube.from_reduced(cube, cube_obs, **kwargs)

z = velocity_to_redshift(ml['V_hel'])
gap_mask_file = gap_mask_template % ml['grating']
gap_mask = get_wavelength_mask(gap_mask_file, d3d.l_obs, z, dest='rest')
telluric_mask = get_wavelength_mask(telluric_mask_file, d3d.l_obs, z, dest='rest')
d3d.f_flag[gap_mask] |= flags.ccd_gap
d3d.f_flag[telluric_mask] |= flags.telluric

d3d.write(cube_resam, overwrite=overwrite)
