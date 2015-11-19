'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
from diving3d.wcs import find_nearest_index
from diving3d import flags

import numpy as np
import sys
from matplotlib.ticker import MultipleLocator

def plot_spectra(f_obs, f_syn, f_res, f_err, at_flux, ll, yy, xx, center, galaxy_id, suffix):
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
    
    vmax = f_syn.max() * 1.1
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
    plt.title(r'%s %s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (galaxy_id, suffix, xx[x_slice]))
    plt.colorbar(ticks=[0.0, 0.5e-16, 1.0e-16, 1.5e-16, 2.0e-16])
    
    plt.subplot(312)
    plt.pcolormesh(ll, yy, f_syn[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'Synthetic flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$]')
    plt.colorbar(ticks=[0.0, 0.5e-16, 1.0e-16, 1.5e-16, 2.0e-16])
    
    plt.subplot(313)
    plt.pcolormesh(ll, yy, (f_res[:, :, x_slice] * 100 / f_syn[:, :, x_slice]).T, vmin=-err_lim, vmax=err_lim, cmap='RdBu')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.xlim(ll.min(), ll.max())
    plt.title('residual [%]')
    plt.colorbar(ticks=[-5, -2.5, 0, 2.5, 5])
    plt.gcf().set_tight_layout(True)
    plt.savefig('data/plots/%s_%s_slice.png' % (galaxy_id, suffix), dpi=300)
    
    plt.figure(2, figsize=(5, 5))
    plt.subplot(211)
    err = f_err[:, center[1], center[2]]
    f = f_obs[:, center[1], center[2]]
    s = f_syn[:, center[1], center[2]]
    plt.plot(ll, np.log10(f), 'k-', label='observed')
    plt.plot(ll, np.log10(s), 'r-', label='synthetic')
    plt.plot(ll, np.log10(err), 'b-', label='error')
    plt.ylabel(r'$\log F_\lambda$ [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$]')
    plt.xlim(ll.min(), ll.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.legend(loc='center right')
    plt.title('%s %s - center spaxel' % (galaxy_id, suffix))

    plt.subplot(212)
    r = f_res[:, center[1], center[2]]
    err = f_err[:, center[1], center[2]]
    plt.plot(ll, r, 'm-', label='residual')
    plt.plot(ll, err, 'b-', label='error (estimated)')
    plt.plot(ll, np.zeros_like(ll), 'k:')
    plt.ylabel(r'error / residual flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$]')
    plt.xlabel(r'Wavelength [$\AA$]')
    #plt.ylim(-err_lim, err_lim)
    plt.xlim(ll.min(), ll.max())
    plt.legend(loc='lower right')
    plt.gcf().set_tight_layout(True)
    plt.savefig('data/plots/%s_%s_center.pdf' % (galaxy_id, suffix))
    
    plt.figure(3, figsize=(4, 5))
    l_norm = 5635.0 #AA
    i_norm = find_nearest_index(ll, l_norm)
    signal_image = np.median(f_obs[i_norm - 50:i_norm + 50], axis=0)
    noise_image = np.median(f_err[i_norm - 50:i_norm + 50], axis=0)
    plt.pcolormesh(xx, yy, signal_image / noise_image, cmap='cubehelix_r')
    plt.colorbar(ticks=MultipleLocator(10))
    plt.gca().set_aspect('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel(r'R. A. [arcsec]')
    plt.ylabel(r'dec. [arcsec]')
    plt.title(r'%s %s - S/N at $5635\,\AA$' % (galaxy_id, suffix))
    plt.savefig('data/plots/%s_%s_sn.pdf' % (galaxy_id, suffix))

    plt.figure(5, figsize=(4, 5))
    plt.pcolormesh(xx, yy, at_flux, cmap='cubehelix_r')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel(r'R. A. [arcsec]')
    plt.ylabel(r'dec. [arcsec]')
    plt.title(r'%s %s - $\langle \log t \rangle_\mathrm{flux}\ [\mathrm{Gyr}]$' % (galaxy_id, suffix))
    plt.savefig('data/plots/%s_%s_atflux.pdf' % (galaxy_id, suffix))
    #plt.show()
    
galaxy_id = sys.argv[1]
suffix = sys.argv[2]
cube = '/Volumes/data/diving3d/cubes_out/%s_%s.fits' % (galaxy_id, suffix)
d3d = D3DFitsCube(cube)
xx = d3d.x_coords
yy = d3d.y_coords
ll = d3d.l_obs
center = d3d.center

f_obs = d3d.f_obs * d3d.flux_unit
f_syn = d3d.f_syn * d3d.flux_unit
f_err = d3d.f_err * d3d.flux_unit
f_res = f_obs - f_syn

plot_spectra(f_obs, f_syn, f_res, f_err, d3d.at_flux, ll, yy, xx, center, galaxy_id, suffix)
