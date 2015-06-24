'''
Created on 15/06/2015

@author: andre

Resample spectra in 1 \AA bins and change wavelength interval. 

'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import read_masterlist, get_wavelength_mask
from diving3d.resampling import velocity_to_redshift
from diving3d import flags
from diving3d.config import get_config
from os import path

overwrite = False
galaxy_id = 'T001'

cfg = get_config('data/diving3d.cfg')
cube_dir = cfg.get('path', 'cubes')
cube_obs_dir = cfg.get('path', 'cubes_obs')
cube_out_dir = cfg.get('path', 'cubes_out')
masterlist = cfg.get('tables', 'masterlist')
gap_mask_template = cfg.get('tables', 'gap_mask_template')
telluric_mask_file = cfg.get('tables', 'telluric_mask')

ml = read_masterlist(masterlist, galaxy_id)
cube = path.join(cube_dir, ml['cube'])
cube_obs = path.join(cube_obs_dir, ml['cube_obs'])
cube_out = path.join(cube_out_dir, '%s_resampled.fits' % galaxy_id)

kwargs = dict(l_ini=cfg.getfloat('dimensions', 'l_ini'),
              l_fin=cfg.getfloat('dimensions', 'l_fin'),
              dl=cfg.getfloat('dimensions', 'dl'),
              width=cfg.getint('dimensions', 'Nx'),
              height=cfg.getint('dimensions', 'Ny'),
              ml=ml)

d3d = D3DFitsCube.from_reduced(cube, cube_obs, **kwargs)

z = velocity_to_redshift(ml['V_hel'])
gap_mask_file = gap_mask_template % ml['grating']
gap_mask = get_wavelength_mask(gap_mask_file, d3d.l_obs, z, dest='rest')
telluric_mask = get_wavelength_mask(telluric_mask_file, d3d.l_obs, z, dest='rest')
d3d.f_flag[gap_mask] |= flags.ccd_gap
d3d.f_flag[telluric_mask] |= flags.telluric

d3d.write(cube_out, overwrite=overwrite)
