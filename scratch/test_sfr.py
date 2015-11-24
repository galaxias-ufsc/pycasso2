'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
import numpy as np
import sys

def plot_sfr(sfr, t, center, galaxy_id, suffix):
    import matplotlib.pyplot as plt
        
    plotpars = {'legend.fontsize': 8,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'text.fontsize': 11,
                'axes.titlesize': 12,
                'lines.linewidth': 1.0,
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
        
    plt.close('all')
    plt.figure(1, figsize=(5, 3))
    plt.subplot(111)
    plt.plot(t, sfr[:, center[1], center[2]], 'k-', label='sfr')
    plt.ylabel(r'SFR')
    plt.xlabel(r'Time [yr]')
    plt.xlim(0, 2e10)
    plt.legend(loc='center right')
    plt.title('%s %s - center spaxel' % (galaxy_id, suffix))
    plt.show()
    
galaxy_id = sys.argv[1]
suffix = sys.argv[2]
cube = 'data/cubes_out/%s_%s.fits' % (galaxy_id, suffix)
d3d = D3DFitsCube(cube)
center = d3d.center

sfr, t = d3d.SFRSD(dt=0.1e9)
assert np.allclose(np.trapz(sfr, t, axis=0), d3d.MiniSD.sum(axis=0))

plot_sfr(sfr, t, center, galaxy_id, suffix)
