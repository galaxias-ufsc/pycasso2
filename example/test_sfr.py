'''
Created on 23/06/2015

@author: andre
'''

from pycasso2 import FitsCube
import numpy as np
import sys

def plot_sfr(sfr, t, sfr_sm, t_sm, center, yy, xx, object_name):
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
    plt.title('%s - center spaxel' % object_name)
    fig.tight_layout()

    x_slice = center[2]    
    fig = plt.figure(2, figsize=(5, 5))
    plt.subplot(111)
    plt.pcolormesh(t_sm / 1e9, yy, np.log10(sfr_sm[:, :, x_slice].T), vmin=-7, vmax=-3, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'time [Gyr]')
    plt.xlim(t_sm.min() / 1e9, t_sm.max() / 1e9)
    plt.title(r'%s - $\log SFR\ [\mathrm{M}_\odot\ \mathrm{yr}^{-1} \mathrm{pc}^{-2}]$ @ R.A. = %.02f' % (object_name, xx[x_slice]))
    plt.colorbar()
    fig.tight_layout()

    plt.show()
    
cube = sys.argv[1]
c = FitsCube(cube)
center = c.center
xx = c.x_coords
yy = c.y_coords

sfr, t = c.SFRSD(dt=0.1e9)
sfr_sm, t_sm = c.SFRSD_smooth(dt=0.1e9, logtc_FWHM=0.25)

print 'total mass density: %.2e M_\odot / pc^2' % c.MiniSD.sum()
print 'total mass density (integral of SFR): %.2e M_\odot / pc^2' % np.trapz(sfr, t, axis=0).sum()
print 'total mass density (integral of smooth SFR): %.2e M_\odot / pc^2' % np.trapz(sfr_sm, t_sm, axis=0).sum()

plot_sfr(sfr, t, sfr_sm, t_sm, center, yy, xx, c.objectName)


