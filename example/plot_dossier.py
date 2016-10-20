'''
Created on 23/06/2015

@author: andre
'''

from pycasso2 import FitsCube
import numpy as np
import sys
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def plot_setup():
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
    

def plot_mapsf(c, pdf):
    xx, yy = c.celestial_coords
    xx = -xx
   
    fig = plt.figure(figsize=(8, 7))
    
    plt.subplot(331)
    m = plt.pcolormesh(xx, yy, np.log10(c.LobnSD.sum(axis=0)), cmap='viridis_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.5))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$L^{\mathrm{obs}}_{5635\AA}\ [\mathrm{L}_\odot\ \AA^{-1}\ pc^{-2}]$')

    plt.subplot(332)
    m = plt.pcolormesh(xx, yy, np.log10(c.McorSD.sum(axis=0)), cmap='viridis_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.5))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\log M_\star\ [\mathrm{M}_\odot\ pc^{-2}]$')


    plt.subplot(334)
    m = plt.pcolormesh(xx, yy, c.at_flux, cmap='viridis_r')
    m.set_rasterized(True)
    plt.gca().set_aspect('equal')
    plt.colorbar(ticks=MultipleLocator(0.5))
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\langle \log t \rangle_L [\mathrm{yr}]$')

    plt.subplot(335)
    m = plt.pcolormesh(xx, yy, c.alogZ_mass, cmap='viridis_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.5))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\langle \log Z/\mathrm{Z}_\odot \rangle_M$')

    plt.subplot(333)
    m = plt.pcolormesh(xx, yy, c.tau_V, cmap='viridis_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(0.5))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\tau_V$')

    plt.subplot(337)
    m = plt.pcolormesh(xx, yy, c.v_0, vmin=-200, vmax=200, cmap='RdBu_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(50))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.xlim(xx.min(), xx.max())
    plt.title(r'$v_0\ [\mathrm{km}\ \mathrm{s}^{-1}]$')

    plt.subplot(338)
    m = plt.pcolormesh(xx, yy, c.v_d, cmap='viridis_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(50))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$v_d\ [\mathrm{km}\ \mathrm{s}^{-1}]$')
  
  
    plt.suptitle('%s' % c.name)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    pdf.savefig()



def plot_maps(c, pdf):
    xx, yy = c.celestial_coords
    xx = -xx
          
    fig = plt.figure(figsize=(6, 5))
  
  
    plt.subplot(221)
    m = plt.pcolormesh(xx, yy, c.SN_normwin, cmap='viridis_r')
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(10))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$S/N\ \mathrm{at}\ 5635\,\AA$')
  
  
    plt.subplot(222)
    m = plt.pcolormesh(xx, yy, c.adev, cmap='viridis_r', vmin = 0, vmax = 10)
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(1.0))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\Delta\ [\%]$')

  
  
    plt.subplot(223)
    m = plt.pcolormesh(xx, yy, c.Nclipped / c.Nwave * 100.0, cmap='viridis_r', vmin = 0, vmax = 10)
    m.set_rasterized(True)
    plt.colorbar(ticks=MultipleLocator(1.0))
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.xlim(xx.min(), xx.max())
    plt.title(r'$N_\mathrm{clipped} / N_\lambda\ [\%]$')
    
    
    plt.subplot(224)
    m = plt.pcolormesh(xx, yy, c.chi2, cmap='viridis_r')
    m.set_rasterized(True)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.ylim(yy.min(), yy.max())
    plt.gca().yaxis.set_ticklabels([])
    plt.xlim(xx.min(), xx.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.title(r'$\chi^2$')


    plt.suptitle('%s' % c.name)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    pdf.savefig()

def plot_spectra(c, pdf):
    plotpars = {'legend.fontsize': 8,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'text.fontsize': 11,
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
    
    xx, yy = c.celestial_coords
    ll = c.l_obs
    center = c.center

    f_obs = c.f_obs / c.flux_norm_window
    f_syn = c.f_syn / c.flux_norm_window
    f_err = c.f_err / c.flux_norm_window
    f_res = (f_obs - f_syn) / f_obs * 100
    f_wei = c.f_wei

    vmax = f_syn.max() * 1.1
    err_lim = 10.0
    x_slice = center[2]
    
    plt.close('all')
    plt.figure(1, figsize=(5, 5))
    plt.subplot(311)
    m = plt.pcolormesh(ll, yy, f_obs[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='viridis_r')
    m.set_rasterized(True)
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'%s - flux (normalized @ $5635\,\AA$) - R.A. = %.02f' % (c.name, xx[x_slice]))
    plt.colorbar(ticks=MultipleLocator(0.5))
    
    plt.subplot(312)
    m = plt.pcolormesh(ll, yy, f_syn[:, :, x_slice].T, vmin=0.0, vmax=vmax, cmap='viridis_r')
    m.set_rasterized(True)
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'synthetic flux (normalized @ $5635\,\AA$)')
    plt.colorbar(ticks=MultipleLocator(0.5))
    
    plt.subplot(313)
    m = plt.pcolormesh(ll, yy, (f_res[:, :, x_slice]).T, vmin=-err_lim, vmax=err_lim, cmap='RdBu')
    m.set_rasterized(True)
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'wavelength [$\AA$]')
    plt.xlim(ll.min(), ll.max())
    plt.title('Residual [%]')
    plt.colorbar(ticks=MultipleLocator(5))
    plt.gcf().set_tight_layout(True)
    pdf.savefig()
    

    
    plt.figure(2, figsize=(5, 4))
    plt.subplot(211)
    err = f_err[:, center[1], center[2]]
    f = f_obs[:, center[1], center[2]]
    err_scale = int(0.2 * f.mean() / err.mean())
    s = f_syn[:, center[1], center[2]]
    plt.plot(ll, f, 'b-', label='observed')
    plt.plot(ll, s, 'r-', label='synthetic')
    plt.plot(ll, err * err_scale, 'k-', label='error (x %d)' % err_scale)
    plt.ylabel(r'$F_\lambda$ (normalized @ $5635\,\AA$)')
    plt.xlim(ll.min(), ll.max())
    plt.ylim(0, 2.0*np.median(f))
    plt.gca().xaxis.set_ticklabels([])
    plt.legend(loc='upper left', frameon=False)
    plt.title('%s - center spaxel, S/N = %.1f' % (c.name, c.SN_normwin[center[1], center[2]]))

    plt.subplot(212)
    r = f_res[:, center[1], center[2]]
    w = f_wei[:, center[1], center[2]]
    err = f_err[:, center[1], center[2]]
    plt.plot(ll, r, 'k-', label='residual')
    plt.plot(ll, np.zeros_like(ll), 'k:')
    
    fitted = np.ma.masked_where(w < 0, r)
    plt.plot(ll, fitted, 'b-')
    
    masked = np.ma.masked_where(w != 0, r)
    plt.plot(ll, masked, '-', color='magenta')
    
    clipped = np.ma.masked_where(w != -1, r)
    plt.plot(ll, clipped, 'x', color='red')
    
    flagged = np.ma.masked_where(w != -2, r)
    plt.plot(ll, flagged, 'o', color='green')
    
    plt.ylabel(r'residual flux [%]')
    plt.xlabel(r'wavelength [$\AA$]')
    plt.ylim(-20, 20)
    plt.xlim(ll.min(), ll.max())
    plt.legend(loc='upper left', frameon=False)
    
    plt.gcf().set_tight_layout(True)
    pdf.savefig()
    


    plt.figure(3, figsize=(5, 4))
    plt.subplot(211)
    err = f_err[:, center[1], center[2]+10]
    f = f_obs[:, center[1], center[2]+10]
    err_scale = int(0.2 * f.mean() / err.mean())
    s = f_syn[:, center[1], center[2]+10]
    plt.plot(ll, f, 'b-', label='observed')
    plt.plot(ll, s, 'r-', label='synthetic')
    plt.plot(ll, err * err_scale, 'k-', label='error (x %d)' % err_scale)
    plt.ylabel(r'$F_\lambda$ (normalized @ $5635\,\AA$)')
    plt.xlim(ll.min(), ll.max())
    plt.ylim(0, 2.0*np.median(f))
    plt.gca().xaxis.set_ticklabels([])
    plt.legend(loc='upper left', frameon=False)
    plt.title('%s - center spaxel + 10\", S/N = %.1f' % (c.name, c.SN_normwin[center[1], center[2]+10]))
    
    plt.subplot(212)
    r = f_res[:, center[1], center[2]+10]
    w = f_wei[:, center[1], center[2]+10]
    err = f_err[:, center[1], center[2]+10]
    plt.plot(ll, r, 'k-', label='residual')
    plt.plot(ll, np.zeros_like(ll), 'k:')
    
    fitted = np.ma.masked_where(w < 0, r)
    plt.plot(ll, fitted, 'b-')

    masked = np.ma.masked_where(w != 0, r)
    plt.plot(ll, masked, '-', color='magenta')
    
    clipped = np.ma.masked_where(w != -1, r)
    plt.plot(ll, clipped, 'x', color='red')
    
    flagged = np.ma.masked_where(w != -2, r)
    plt.plot(ll, flagged, 'o', color='green')
        
    plt.ylabel(r'residual flux [%]')
    plt.xlabel(r'wavelength [$\AA$]')
    plt.ylim(-20, 20)
    plt.xlim(ll.min(), ll.max())
    plt.legend(loc='upper left', frameon=False)

    plt.gcf().set_tight_layout(True)
    pdf.savefig()
    

def plot_metal_poor(c, pdf):
    at_flux = c.at_flux.compressed()
    alogZ_mass = c.alogZ_mass.compressed()
    mask = c.popx.sum(axis=0) == 0.0
    tau_V = c.tau_V.copy()
    tau_V[mask] = np.ma.masked
    tau_V = tau_V.compressed()
    colors = np.log10(c.McorSD.sum(axis=0))
    colors[mask] = np.ma.masked
    colors = colors.compressed()
    
    plt.figure(figsize=(6, 4))
    plt.subplot(221)
    plt.scatter(at_flux, alogZ_mass, c=colors, cmap='viridis_r', s=3, alpha=0.5, edgecolor='none')
    h, xe, ye = np.histogram2d(at_flux, alogZ_mass, bins = [20, 20])
    h /= h.max()
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    plt.contour(h.T, extent=extent, levels=[0.25, 0.5], colors='k')
    plt.gca().set_xticks(np.log10(c.age_base), minor=True)
    plt.gca().set_xticklabels([])
    plt.gca().xaxis.grid(True, which='minor')
    plt.xlim(at_flux.min() - 0.2, at_flux.max() + 0.2)
    plt.ylabel(r'$\langle \log Z/\mathrm{Z}_\odot \rangle_M$')
    plt.gca().set_yticks(np.log10(c.Z_base / 0.019), minor=True)
    plt.gca().yaxis.grid(True, which='minor')
    plt.ylim(alogZ_mass.min() - 0.2, alogZ_mass.max() + 0.2)

    plt.subplot(223)
    sc = plt.scatter(at_flux, tau_V, c=colors, cmap='viridis_r', s=3, alpha=0.5, edgecolor='none')
    h, xe, ye = np.histogram2d(at_flux, tau_V, bins = [20, 20])
    h /= h.max()
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    plt.contour(h.T, extent=extent, levels=[0.25, 0.5], colors='k')
    plt.ylabel(r'$\tau_V$')
    plt.xlabel(r'$\langle \log t \rangle_L [\mathrm{yr}]$')
    plt.gca().set_xticks(np.log10(c.age_base), minor=True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.xlim(at_flux.min() - 0.2, at_flux.max() + 0.2)
    
    ax = plt.subplot(224)
    ax.axis('off')
    cb = plt.colorbar(sc, ax=ax, orientation='horizontal')
    cb.set_label(r'$\log M_\star\ [\mathrm{M}_\odot\ pc^{-2}]$')

    plt.subplot(222)
    plt.scatter(tau_V, alogZ_mass, c=colors, cmap='viridis_r', s=3, alpha=0.5, edgecolor='none')
    h, xe, ye = np.histogram2d(tau_V, alogZ_mass, bins = [20, 20])
    h /= h.max()
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    plt.contour(h.T, extent=extent, levels=[0.25, 0.5], colors='k')
    plt.gca().set_yticks(np.log10(c.Z_base / 0.019), minor=True)
    plt.gca().set_yticklabels([])
    plt.gca().yaxis.grid(True, which='minor')
    plt.ylim(alogZ_mass.min() - 0.2, alogZ_mass.max() + 0.2)
    plt.xlabel(r'$\tau_V$')
    plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.15)
    pdf.savefig()


cube = sys.argv[1]
c = FitsCube(cube)
c.synthImageMask[c.SN_normwin < 5.0] = True
plt.close('all')
dossier_plot = sys.argv[2]
pdf = PdfPages(dossier_plot)

plot_setup()
plot_spectra(c, pdf)
plot_mapsf(c, pdf)
plot_maps(c, pdf)
plot_metal_poor(c, pdf)

pdf.close()

#plt.show()
