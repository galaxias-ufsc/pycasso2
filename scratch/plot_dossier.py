'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
import numpy as np
import sys
from diving3d.wcs import find_nearest_index
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def plot_maps(d3d, pdf):
        
    plotpars = {'legend.fontsize': 8,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'font.size': 10,
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
    
    xx = d3d.x_coords
    yy = d3d.y_coords
    
    fig = plt.figure(figsize=(8, 10))
    
    plt.subplot(331)
    m = plt.pcolormesh(xx, yy, np.log10(d3d.LobnSD.sum(axis=0)), cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.1))
    plt.gca().set_aspect('equal')
    #plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    #plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$L^{\mathrm{obs}}_{5635\AA}\ [\mathrm{L}_\odot\ \AA^{-1}\ pc^{-2}]$')

    plt.subplot(332)
    m = plt.pcolormesh(xx, yy, np.log10(d3d.McorSD.sum(axis=0)), cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.1))
    plt.gca().set_aspect('equal')
    #plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    #plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\log M_\star\ [\mathrm{M}_\odot\ pc^{-2}]$')

    plt.subplot(333)
    l_norm = 5635.0 #AA
    i_norm = find_nearest_index(d3d.l_obs, l_norm)
    f_obs = d3d.f_obs
    f_err = d3d.f_err
    signal_image = np.median(f_obs[i_norm - 50:i_norm + 50], axis=0)
    noise_image = np.median(f_err[i_norm - 50:i_norm + 50], axis=0)
    m = plt.pcolormesh(xx, yy, signal_image / noise_image, cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(10))
    plt.gca().set_aspect('equal')
    #plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    #plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$S/N\ \mathrm{at}\ 5635\,\AA$')
    
    plt.subplot(334)
    m = plt.pcolormesh(xx, yy, d3d.at_flux, cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.gca().set_aspect('equal')
    plt.colorbar(ticks=MultipleLocator(0.1))
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    #plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\langle \log t \rangle_L [\mathrm{yr}]$')

    plt.subplot(335)
    m = plt.pcolormesh(xx, yy, d3d.alogZ_mass, cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.1))
    plt.gca().set_aspect('equal')
    #plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    #plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\langle \log Z/\mathrm{Z}_\odot \rangle_M$')

    plt.subplot(336)
    m = plt.pcolormesh(xx, yy, d3d.tau_V, cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.1))
    plt.gca().set_aspect('equal')
    #plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    #plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\tau_V$')

    plt.subplot(337)
    m = plt.pcolormesh(xx, yy, d3d.v_0, vmin=-200, vmax=200, cmap='RdBu')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(100))
    plt.gca().set_aspect('equal')
    #plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    #plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.title(r'$v_0\ [\mathrm{km}\ \mathrm{s}^{-1}]$')

    plt.subplot(338)
    m = plt.pcolormesh(xx, yy, d3d.v_d, cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(50))
    plt.gca().set_aspect('equal')
    #plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.title(r'$v_d\ [\mathrm{km}\ \mathrm{s}^{-1}]$')
    
    plt.subplot(339)
    m = plt.pcolormesh(xx, yy, d3d.chi2, cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.5))
    plt.gca().set_aspect('equal')
    #plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    #plt.xlabel(r'r.a. [arcsec]')
    plt.xlim(xx.min(), xx.max())
    plt.title(r'$\chi^2 / N_\lambda$')
    
    plt.suptitle('%s - %s' % (d3d.object_name, d3d.id))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    pdf.savefig()


def plot_spectra(d3d, pdf):
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
    
    xx = d3d.x_coords
    yy = d3d.y_coords
    ll = d3d.l_obs
    center = d3d.center

    f_obs = d3d.f_obs * d3d.flux_unit
    f_syn = d3d.f_syn * d3d.flux_unit
    f_err = d3d.f_err * d3d.flux_unit
    f_res = f_obs - f_syn

    vmax = f_syn.max() * 1.1
    err_lim = 5.0
    x_slice = center[2]
    
    plt.close('all')
    plt.figure(1, figsize=(5, 5))
    plt.subplot(311)
    m = plt.pcolormesh(ll, yy, f_obs[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'%s %s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (d3d.object_name, d3d.id, xx[x_slice]))
    plt.colorbar(ticks=[0.0, 0.5e-16, 1.0e-16, 1.5e-16, 2.0e-16])
    
    plt.subplot(312)
    m = plt.pcolormesh(ll, yy, f_syn[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='cubehelix_r')
    m.set_rasterized(True)
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'Synthetic flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$]')
    plt.colorbar(ticks=[0.0, 0.5e-16, 1.0e-16, 1.5e-16, 2.0e-16])
    
    plt.subplot(313)
    m = plt.pcolormesh(ll, yy, (f_res[:, :, x_slice] * 100 / f_syn[:, :, x_slice]).T, vmin=-err_lim, vmax=err_lim, cmap='RdBu')
    m.set_rasterized(True)
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.xlim(ll.min(), ll.max())
    plt.title('residual [%]')
    plt.colorbar(ticks=[-5, -2.5, 0, 2.5, 5])
    plt.gcf().set_tight_layout(True)
    pdf.savefig()
    
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
    plt.title('%s %s - center spaxel' % (d3d.object_name, d3d.id))

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
    pdf.savefig()
    

galaxy_id = sys.argv[1]
suffix = sys.argv[2]
cube = 'data/cubes_out/%s_%s.fits' % (galaxy_id, suffix)
d3d = D3DFitsCube(cube)

pdf = PdfPages('data/plots/%s_%s.pdf' % (galaxy_id, suffix))

plot_spectra(d3d, pdf)
plot_maps(d3d, pdf)

pdf.close()

#plt.show()
