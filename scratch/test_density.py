'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id

import numpy as np

def plot_image(im, yy, xx, galaxy_id):
    import matplotlib.pyplot as plt
        
    plotpars = {'legend.fontsize': 8,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.fontsize': 10,
                'axes.titlesize': 12,
                'lines.linewidth': 0.5,
                'font.family': 'Times New Roman',
    #             'figure.subplot.left': 0.08,
    #             'figure.subplot.bottom': 0.08,
    #             'figure.subplot.right': 0.97,
    #             'figure.subplot.top': 0.95,
    #             'figure.subplot.wspace': 0.42,
    #             'figure.subplot.hspace': 0.1,
                'image.cmap': 'GnBu',
                }
    plt.rcParams.update(plotpars)
    plt.ioff()
    plt.figure()
    
    plt.subplot(111)
    plt.pcolormesh(xx, yy, im, cmap='cubehelix_r')
    plt.gca().set_aspect('equal')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    #plt.title(r'%s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (galaxy_id, xx[x_slice]))
    plt.colorbar()
    plt.show()


cube = '/Volumes/data/diving3d/cubes_out/T001_resam_synth2.fits'

galaxy_id = get_galaxy_id(cube)
d3d = D3DFitsCube(cube)
xx = d3d.x_coords
yy = d3d.y_coords

print 'pixel area: %f pc^2' % d3d.pixelArea_pc2
print 'pixel length: %f pc' % d3d.pixelLength_pc
print 'pixel length: %f arcsec' % np.abs(xx[0] - xx[1])

plot_image(d3d.McorSD.sum(axis=0), yy, xx, galaxy_id)

assert np.allclose(d3d.McorSD.sum(axis=0), d3d.Mcor_tot / d3d.pixelArea_pc2)
