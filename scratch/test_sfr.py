'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
import numpy as np
import sys

def plot_sfr(sfr, t, sfr_sm, t_sm, center, yy, xx, galaxy_id, suffix):
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
    fig = plt.figure(1, figsize=(5, 3))
    plt.subplot(111)
    plt.plot(t / 1e9, sfr[:, center[1], center[2]], 'k-', label='sfr')
    plt.plot(t_sm/ 1e9, sfr_sm[:, center[1], center[2]], 'r:', label='sfr (smooth)')
    plt.ylabel(r'SFR')
    plt.xlabel(r'Time [Gyr]')
    plt.xlim(0, 20)
    plt.legend(loc='upper left')
    plt.title('%s %s - center spaxel' % (galaxy_id, suffix))
    fig.tight_layout()

    x_slice = center[2]    
    fig = plt.figure(2, figsize=(5, 5))
    plt.subplot(111)
    plt.pcolormesh(t_sm / 1e9, yy, np.log10(sfr_sm[:, :, x_slice].T), vmin=-7, vmax=-3, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'time [Gyr]')
    plt.xlim(t_sm.min() / 1e9, t_sm.max() / 1e9)
    plt.title(r'%s %s - $\log SFR\ [\mathrm{M}_\odot\ \mathrm{yr}^{-1} \mathrm{pc}^{-2}]$ @ R.A. = %.02f' % (galaxy_id, suffix, xx[x_slice]))
    plt.colorbar()
    fig.tight_layout()

    plt.show()
    
galaxy_id = sys.argv[1]
suffix = sys.argv[2]
cube = 'data/cubes_out/%s_%s.fits' % (galaxy_id, suffix)
d3d = D3DFitsCube(cube)
center = d3d.center
xx = d3d.x_coords
yy = d3d.y_coords

sfr, t = d3d.SFRSD(dt=0.1e9)
sfr_sm, t_sm = d3d.SFRSD_smooth(dt=0.1e9, logtc_FWHM=0.25)

print 'total mass density: %.2e M_\odot / pc^2' % d3d.MiniSD.sum()
print 'total mass density (integral of SFR): %.2e M_\odot / pc^2' % np.trapz(sfr, t, axis=0).sum()
print 'total mass density (integral of smooth SFR): %.2e M_\odot / pc^2' % np.trapz(sfr_sm, t_sm, axis=0).sum()

plot_sfr(sfr, t, sfr_sm, t_sm, center, yy, xx, galaxy_id, suffix)


