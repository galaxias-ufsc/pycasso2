'''
Created on 24/06/2015

@author: andre
'''
from diving3d.tables import read_masterlist, get_galaxy_id
from diving3d.cube import get_cube_limits
from diving3d.config import get_config, default_config_path, save_config
import numpy as np
from os import path


cfg = get_config(default_config_path)
cube_dir = cfg.get('path', 'cubes')
masterlist = cfg.get('tables', 'masterlist')
ml = read_masterlist(masterlist)

l_ini = np.empty(len(ml), dtype='float64')
l_fin = np.empty(len(ml), dtype='float64')
Ny = np.empty(len(ml), dtype='int')
Nx = np.empty(len(ml), dtype='int')

for i, g in enumerate(ml):
    cube = path.join(cube_dir, g['cube'])
    galaxy_id = get_galaxy_id(cube)
    l_ini[i], l_fin[i], Ny[i], Nx[i] = get_cube_limits(cube, 0)
    print galaxy_id, l_ini[i], l_fin[i], Ny[i], Nx[i]

print 'Limits:', l_ini.min(), l_fin.max(), Ny.max(), Nx.max()

cfg.set('dimensions', 'l_ini', l_ini.min())
cfg.set('dimensions', 'l_fin', l_fin.max())
cfg.set('dimensions', 'Ny', Ny.max())
cfg.set('dimensions', 'Nx', Nx.max())
new_config_file = default_config_path + '.found'
print 'Saving to %s.' % new_config_file
save_config(cfg, new_config_file)

