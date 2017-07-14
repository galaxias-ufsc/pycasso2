'''
Created on 23/06/2015

@author: andre
'''

from pycasso2 import FitsCube

import numpy as np
import sys
from matplotlib.ticker import MultipleLocator
from pycasso2.resampling import find_nearest_index

def plot_spectra(f_obs, f_syn, f_res, f_err, ll, yy, xx, center, cube):
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
    
    vmax = np.median(f_obs[:, center[1], center[2]]) * 1.5
    err_lim = 5.0
    x_slice = center[2]
    
    plt.close('all')
    plt.figure(1, figsize=(5, 5))
    plt.subplot(311)
    plt.pcolormesh(ll, yy, f_obs[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'%s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (cube, xx[x_slice]))
    plt.colorbar(ticks=[0.0, 0.5e-16, 1.0e-16, 1.5e-16, 2.0e-16])
    
    if f_syn is not None:
        plt.subplot(312)
        plt.pcolormesh(ll, yy, f_syn[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='cubehelix_r')
        plt.ylabel(r'dec. [arcsec]')
        plt.ylim(yy.min(), yy.max())
        plt.gca().xaxis.set_ticklabels([])
        plt.xlim(ll.min(), ll.max())
        plt.title(r'Synthetic flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$]')
        plt.colorbar(ticks=[0.0, 0.5e-16, 1.0e-16, 1.5e-16, 2.0e-16])
    
    if f_res is not None and f_syn is not None:
        plt.subplot(313)
        plt.pcolormesh(ll, yy, (f_res[:, :, x_slice] * 100 / f_syn[:, :, x_slice]).T, vmin=-err_lim, vmax=err_lim, cmap='RdBu')
        plt.ylabel(r'dec. [arcsec]')
        plt.ylim(yy.min(), yy.max())
        plt.xlabel(r'Wavelength [$\AA$]')
        plt.xlim(ll.min(), ll.max())
        plt.title('residual [%]')
        plt.colorbar(ticks=[-5, -2.5, 0, 2.5, 5])
        plt.gcf().set_tight_layout(True)
        #plt.savefig('plots/%s_slice.png' % cube, dpi=300)
    
    plt.figure(2, figsize=(5, 5))
    plt.subplot(211)
    f = f_obs[:, center[1], center[2]]
    plt.plot(ll, f, 'k-', label='observed')
    if f_syn is not None:
        s = f_syn[:, center[1], center[2]]
        plt.plot(ll, s, 'r-', label='synthetic')
    plt.ylabel(r'$F_\lambda$ [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$]')
    plt.xlim(ll.min(), ll.max())
    plt.ylim(0.0, vmax)
    plt.gca().xaxis.set_ticklabels([])
    plt.legend(loc='center right')
    plt.title('%s - center spaxel' % cube)

    plt.subplot(212)
    if f_res is not None:
        r = f_res[:, center[1], center[2]] / f_syn[:, center[1], center[2]]
        plt.plot(ll, r, 'm-', label='residual')
        err = f_err[:, center[1], center[2]] / f_syn[:, center[1], center[2]]
        plt.ylabel(r'residual flux (normalized to $F_\lambda^{syn}$)')
    else:
        err = f_err[:, center[1], center[2]] / f_obs[:, center[1], center[2]]
        plt.ylabel(r'residual flux (normalized to $F_\lambda^{obs}$)')
    plt.plot(ll, err, 'b-', label='error (estimated)')
    plt.plot(ll, np.zeros_like(ll), 'k:')
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.ylim(-0.1, 0.1)
    plt.xlim(ll.min(), ll.max())
    plt.legend(loc='lower right')
    plt.gcf().set_tight_layout(True)
    #plt.savefig('plots/%s_center.pdf' % cube)
    
    plt.figure(3, figsize=(4, 5))
    l_norm = 5635.0 #AA
    i_norm = find_nearest_index(ll, l_norm)
    signal_image = np.median(f_obs[i_norm - 50:i_norm + 50], axis=0)
    noise_image = np.median(f_err[i_norm - 50:i_norm + 50], axis=0)
    plt.pcolormesh(-xx, yy, signal_image / noise_image, cmap='cubehelix_r')
    plt.colorbar(ticks=MultipleLocator(10))
    plt.gca().set_aspect('equal')
    plt.xlabel(r'R. A. [arcsec]')
    plt.ylabel(r'dec. [arcsec]')
    plt.title(r'%s - S/N at $5635\,\AA$' % cube)
    #plt.savefig('plots/%s_sn.pdf' % cube)

    plt.show()
    
cube = sys.argv[1]
c = FitsCube(cube)
xx, yy = c.celestial_coords
ll = c.l_obs
center = c.center

f_obs = c.f_obs * c.flux_unit
f_err = c.f_err * c.flux_unit
try:
    f_syn = c.f_syn * c.flux_unit
    f_res = f_obs - f_syn
except:
    print('No synthetic spectra found.')
    f_syn = None
    f_res = None

plot_spectra(f_obs, f_syn, f_res, f_err, ll, yy, xx, center, cube)
