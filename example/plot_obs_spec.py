'''
Created on 23/06/2015

@author: andre
'''

from pycasso2 import FitsCube
import sys

def plot_example(c, cube_name, x_slice):
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.figure()
    xx = c.x_coords
    yy = c.y_coords
    ll = c.l_obs
    
    plt.subplot(211)
    plt.pcolormesh(ll, yy, c.f_obs[:, :, x_slice].T * c.flux_unit, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'%s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (cube_name, xx[x_slice]))
    plt.colorbar()
    
    plt.subplot(212)
    plt.pcolormesh(ll, yy, c.f_flag[:, :, x_slice].T, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.xlim(ll.min(), ll.max())
    plt.title('flags')
    plt.colorbar()
    
    plt.show()


cube = sys.argv[1]
c = FitsCube(cube)

plot_example(c, cube, x_slice=10)
