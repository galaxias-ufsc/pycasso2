'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id

def plot_example(d3d, galaxyId, y_slice):
    import matplotlib.pyplot as plt
    plt.ioff()
    xx = d3d.x_coords
    yy = d3d.y_coords
    ll = d3d.l_obs
    plt.pcolormesh(xx, ll, d3d.f_flag[:, y_slice, :], cmap='cubehelix_r')
    plt.xlabel(r'R.A. [arcsec]')
    plt.ylabel(r'Wavelength [$\AA$]')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(ll.min(), ll.max())
    plt.title('%s - flags example @ dec = %.02f "' % (galaxyId, yy[y_slice]))
    plt.colorbar()
    plt.show()


cube = 'data/cubes_out/T001_resampled.fits'
galaxy_id = get_galaxy_id(cube)
d3d = D3DFitsCube(cube)

plot_example(d3d, galaxy_id, y_slice=10)
