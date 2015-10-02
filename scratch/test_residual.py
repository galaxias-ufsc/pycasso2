'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id, get_wavelength_mask
from diving3d.resampling import gaussian1d_spectra, interp1d_spectra
from diving3d.wcs import find_nearest_index

import numpy as np

def plot_spectra(f_obs, f_syn, f_res, f_res_filt, f_err, ll, yy, xx, galaxy_id, center):
    import matplotlib.pyplot as plt
        
    plotpars = {'legend.fontsize': 8,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.fontsize': 10,
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
    plt.figure(1, figsize=(8, 7))
    plt.subplot(311)
    plt.pcolormesh(ll, yy, f_obs[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'%s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (galaxy_id, xx[x_slice]))
    plt.colorbar()
    
    plt.subplot(312)
    plt.pcolormesh(ll, yy, f_syn[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'Synthetic flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$]')
    plt.colorbar()
    
    plt.subplot(313)
    plt.pcolormesh(ll, yy, (f_res_filt[:, :, x_slice] * 100 / f_syn[:, :, x_slice]).T, vmin=-err_lim, vmax=err_lim, cmap='RdBu')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.xlim(ll.min(), ll.max())
    plt.title('residual [%]')
    plt.colorbar()
    
    plt.figure(2, figsize=(8, 7))
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
    r = f_res[:, center[1], center[2]] * 100.0 / f
    r_filt = f_res_filt[:, center[1], center[2]] * 100.0 / f
    err = f_err[:, center[1], center[2]] * 100.0 / f
    plt.plot(ll, r, 'm-', label='residual')
    plt.plot(ll, r_filt, 'g-', label='residual (rectified)')
    plt.plot(ll, err, 'b-', label='error (estimated)')
    plt.plot(ll, np.zeros_like(ll), 'k:')
    plt.ylabel(r'residual flux [%]')
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.ylim(-err_lim, err_lim)
    plt.xlim(ll.min(), ll.max())
    plt.legend(loc='lower right')
    
    plt.figure(3, figsize=(6, 7))
    l_norm = 5635.0 #AA
    i_norm = find_nearest_index(ll, l_norm)
    signal_image = np.median(f_obs[i_norm - 50:i_norm + 50], axis=0)
    noise_image = np.median(f_err[i_norm - 50:i_norm + 50], axis=0)
    plt.pcolormesh(xx, yy, signal_image / noise_image, cmap='cubehelix_r')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel(r'R. A. [arcsec]')
    plt.ylabel(r'dec. [arcsec]')
    plt.title(r'%s - S/N at $5635\,\AA$' % galaxy_id)
    plt.show()
    


def filter_spectra(fwhm, ll, flux):
    flux_filt = np.zeros_like(flux)
    Nl, Ny, Nx = flux.shape
    for j in xrange(Ny):
        for i in xrange(Nx):
            f = flux[:, j, i]
            N_flagged = f.mask.astype('int').sum()
            if N_flagged / Nl > 0.1:
                flux_filt[:, j, i] = np.ma.masked
            else:
                print 'Smoothing (%d, %d)' % (i, j)
                flux_filt[:, j, i] = gaussian1d_spectra(fwhm, ll, f)
    return flux - flux_filt


def interp_spectra(ll, flux):
    flux_filt = np.zeros_like(flux)
    Nl, Ny, Nx = flux.shape
    for j in xrange(Ny):
        for i in xrange(Nx):
            f = flux[:, j, i]
            N_flagged = f.mask.astype('int').sum()
            if N_flagged / Nl > 0.1:
                flux_filt[:, j, i] = np.ma.masked
            else:
                print 'Interpolating (%d, %d)' % (i, j)
                flux_filt[:, j, i] = interp1d_spectra(ll, f)
    return flux_filt


def rms_box(width, ll, flux, threshold=0.5):
    dl = ll[1] - ll[0]
    r = np.ceil(width / dl / 2.0) 
    Nl = flux.shape[0]
    rms = np.ma.masked_all(flux.shape)
    for l in xrange(Nl):
        l1 = l - r
        if l1 < 0:
            l1 = 0
        l2 = l + r
        if l2 >= Nl:
            l2 = Nl - 1
        Nbox = l2 - l1
        f = flux[l1:l2]
        Ngood = (~f.mask).astype('int').sum(axis=0)
        print 'box l=%.1f AA (%d fluxes)' % (ll[l], Nbox)
        rms[l] = np.sqrt((f * f).sum(axis=0) / (Ngood - 1)) 
        rms[l][Ngood < (threshold * Nbox)] = np.ma.masked
    return interp_spectra(ll, rms)


cube = 'data/cubes_out/T002_resam_synth2.fits'
maskfile = 'data/starlight/masks/Masks.EmLines.SDSS.gm'

galaxy_id = get_galaxy_id(cube)
d3d = D3DFitsCube(cube)
xx = d3d.x_coords
yy = d3d.y_coords
ll = d3d.l_obs

is_eline = get_wavelength_mask(maskfile, ll)

flagged = d3d.f_flag > 0
flagged[is_eline] = True

f_obs = np.ma.array(d3d.f_obs * d3d.flux_unit, mask=flagged)
f_syn = np.ma.array(d3d.f_syn * d3d.flux_unit, mask=flagged)
f_res = f_obs - f_syn
f_res_filt = filter_spectra(15.0, ll, f_res)
f_err = rms_box(100.0, ll, f_res_filt, threshold=0.5)

center = d3d.center

plot_spectra(f_obs, f_syn, f_res, f_res_filt, f_err, ll, yy, xx, galaxy_id, center)
