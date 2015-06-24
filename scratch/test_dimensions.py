'''
Created on 24/06/2015

@author: andre
'''
from diving3d.tables import read_masterlist, get_galaxy_id
from diving3d.cube import get_axis_coordinates
import numpy as np
from os import path

def get_limits(cube):
    from astropy.io import fits
    with fits.open(cube) as hl:
        header = hl[0].header
        l_obs = get_axis_coordinates(header, ax=3)
        yy = get_axis_coordinates(header, ax=2)
        xx = get_axis_coordinates(header, ax=1)
        return l_obs.min(), l_obs.max(), len(yy), len(xx) 

cube_dir = 'data/cubes/'
masterlist = 'data/masterlist_sampleT.txt'


ml = read_masterlist(masterlist)

l_ini = np.empty(len(ml))
l_fin = np.empty(len(ml))
Ny = np.empty(len(ml))
Nx = np.empty(len(ml))

for i, g in enumerate(ml):
    cube = path.join(cube_dir, g['cube'])
    galaxy_id = get_galaxy_id(cube)
    l_ini[i], l_fin[i], Ny[i], Nx[i] = get_limits(cube)
    print galaxy_id, l_ini[i], l_fin[i], Ny[i], Nx[i]
    
print 'Limits:', l_ini.min(), l_fin.max(), Ny.max(), Nx.max()
