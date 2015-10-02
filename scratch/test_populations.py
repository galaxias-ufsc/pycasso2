'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id
from diving3d import flags

import numpy as np

def plot_age(at_flux, yy, xx, galaxy_id):
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
    
    plt.subplot(111)
    plt.pcolormesh(xx, yy, at_flux, cmap='cubehelix_r')
    plt.gca().set_aspect('equal')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    #plt.title(r'%s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (galaxy_id, xx[x_slice]))
    plt.colorbar()
    plt.show()


cube = 'data/cubes_out/T001_resam_synth1.fits'

galaxy_id = get_galaxy_id(cube)
d3d = D3DFitsCube(cube)
xx = d3d.x_coords
yy = d3d.y_coords

# Get flagged x,y where there's no good starlight data.
no_starlight_flags = flags.no_data | flags.starlight_masked_pix | flags.starlight_failed_run
flagged = (d3d.f_flag & no_starlight_flags) > 0
im_flagged = im_fl = flagged.sum(axis=0) > (0.5 * flagged.shape[0])

# light fractions
popx = np.ma.array(d3d.popx)
popx[im_flagged] = np.ma.masked

# ages (constant)
popage = np.ma.array(d3d.popage_base)
popage[im_fl] = np.ma.masked

at_flux = (popx * np.log10(popage)).sum(axis=2) / popx.sum(axis=2)


plot_age(at_flux, yy, xx, galaxy_id)
