'''
Created on 23/06/2015

@author: andre
'''

from pycasso2 import FitsCube

import numpy as np
import sys

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
    plt.colorbar()
    plt.show()


cube = sys.argv[1]

c = FitsCube(cube)
xx = c.x_coords
yy = c.y_coords

print 'pixel area: %f pc^2' % c.pixelArea_pc2
print 'pixel length: %f pc' % c.pixelLength_pc
print 'pixel length: %f arcsec' % np.abs(xx[0] - xx[1])

plot_image(c.McorSD.sum(axis=0), yy, xx, c.objectName)

assert np.allclose(c.McorSD.sum(axis=0), c.Mcor_tot / c.pixelArea_pc2)
