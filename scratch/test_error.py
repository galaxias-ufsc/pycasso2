'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id, get_wavelength_mask
from diving3d.wcs import find_nearest_index

import numpy as np
from diving3d.error import estimate_error

def plot_spectra(f_obs, f_syn, f_res, f_err, at_flux, ll, yy, xx, galaxy_id, center):
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
    plt.title(r'%s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (galaxy_id, xx[x_slice]))
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
    #plt.savefig('posdoc/T001_slice.png', dpi=300)
    
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
    plt.title('%s - center spaxel' % galaxy_id)

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
    #plt.savefig('posdoc/T001_center.pdf')
    
    plt.figure(3, figsize=(4, 5))
    l_norm = 5635.0 #AA
    i_norm = find_nearest_index(ll, l_norm)
    signal_image = np.median(f_obs[i_norm - 50:i_norm + 50], axis=0)
    noise_image = np.median(f_err[i_norm - 50:i_norm + 50], axis=0)
    plt.pcolormesh(xx, yy, signal_image / noise_image, cmap='cubehelix_r')
    plt.colorbar(ticks=[30, 40, 50, 60, 70, 80, 90, 100])
    plt.gca().set_aspect('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel(r'R. A. [arcsec]')
    plt.ylabel(r'dec. [arcsec]')
    plt.title(r'%s - S/N at $5635\,\AA$' % galaxy_id)
    #plt.savefig('posdoc/T001_sn.pdf')

    plt.figure(5, figsize=(4, 5))
    plt.pcolormesh(xx, yy, at_flux, cmap='cubehelix_r')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel(r'R. A. [arcsec]')
    plt.ylabel(r'dec. [arcsec]')
    plt.title(r'%s - $\langle \log t \rangle_\mathrm{flux}\ [\mathrm{Gyr}]$' % galaxy_id)
    #plt.savefig('posdoc/T001_atflux.pdf')
    plt.show()
    


cube = 'data/cubes_out/T001_resam_synth1.fits'
maskfile = 'data/starlight/masks/Masks.EmLines.SDSS.gm'

galaxy_id = get_galaxy_id(cube)
d3d = D3DFitsCube(cube)
xx = d3d.x_coords
yy = d3d.y_coords
ll = d3d.l_obs
center = d3d.center

is_eline = get_wavelength_mask(maskfile, ll)
flagged = d3d.f_flag > 0
flagged[is_eline] = True
spatial_mask = d3d.getSpatialMask()

f_obs = np.ma.array(d3d.f_obs * d3d.flux_unit, mask=flagged)
f_syn = np.ma.array(d3d.f_syn * d3d.flux_unit, mask=flagged)
f_res = f_obs - f_syn
f_err = estimate_error(ll, f_res, spatial_mask, smooth_fwhm=15.0, box_width=100.0)

at_flux = np.ma.array(d3d.at_flux, mask=spatial_mask)

plot_spectra(f_obs, f_syn, f_res, f_err.data, at_flux, ll, yy, xx, galaxy_id, center)
