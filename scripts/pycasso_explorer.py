#!/usr/bin/env python
from pycasso2 import FitsCube, flags
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
from pycasso2.segmentation import spatialize

##########################################################################


class PycassoExplorer:

    def __init__(self, cube, figsize=(8, 8)):
        self.c = FitsCube(cube)
        self.createUI(figsize)
        self.raiseImage('1')
        if self.c.x0 >= self.c.Nx or self.c.x0 < 0:
            x0 = int(self.c.Nx / 2)
        else:
            x0 = self.c.x0
        if self.c.y0 >= self.c.Ny or self.c.y0 < 0:
            y0 = int(self.c.Ny / 2)
        else:
            y0 = self.c.y0
        self.selectPixel(x0, y0)
        self.redraw()

    def createUI(self, figsize):
        plotpars = {'legend.fontsize': 8,
                    'xtick.labelsize': 11,
                    'ytick.labelsize': 11,
                    'font.size': 11,
                    'axes.titlesize': 12,
                    'lines.linewidth': 0.5,
                    'font.family': 'Times New Roman',
                    'image.cmap': 'GnBu',
                    }
        plt.rcParams.update(plotpars)
        plt.ioff()
        self.fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2)
        self.ax_im = self.fig.add_subplot(gs[0, 0], projection=self.c._wcs.celestial)
        self.ax_sp = self.fig.add_subplot(gs[1, :])
        self.ax_res = self.fig.add_subplot(gs[2, :], sharex=self.ax_sp)

        self.ax_sp.set_ylabel(r'$F_\lambda [\mathrm{normalized}]$')
        self.ax_res.set_xlabel(r'$\lambda [\AA]$')
        self.ax_res.set_ylabel(r'$O_\lambda - M_\lambda$')
        plt.setp(self.ax_sp.get_xticklabels(), visible=False)
        
        c = self.c
        if c.hasSynthesis:
            images = {'light': c.flux_norm_window,
                      'mass': c.McorSD.sum(axis=0),
                      'sfr': c.recentSFRSD(),
                      'tau_V': self.c.tau_V,
                      'd4000': self.c.LickIndex('D4000'),
                      'age': self.c.at_flux,
                      'met': self.c.alogZ_mass,
                      'v_0': self.c.v_0,
                      'v_d': self.c.v_d,
                      }
        else:
            c.l_norm = 5635.0
            c.dl_norm = 90.0
            images = {'light': c.flux_norm_window,
                      'mass': np.zeros_like(c.flux_norm_window),
                      'sfr':  np.zeros_like(c.flux_norm_window),
                      'tau_V': np.zeros_like(c.flux_norm_window),
                      'd4000': self.c.LickIndex('D4000'),
                      'age':  np.zeros_like(c.flux_norm_window),
                      'met':  np.zeros_like(c.flux_norm_window),
                      'v_0':  np.zeros_like(c.flux_norm_window),
                      'v_d':  np.zeros_like(c.flux_norm_window),
                      }
        
        label = {'light': r'Image @ $5635 \AA$',
                 'mass': r'$\Sigma_\star$',
                 'sfr': r'$\Sigma_\mathrm{SFR}$',
                 'tau_V': r'$\tau_V$',
                 'age': r'$\langle \log\,t \rangle_L$',
                 'met': r'$\langle \log\,Z/Z_\odot \rangle_M$',
                 'd4000': r'$D(4000)$',
                 'v_0': r'$v_\star\ [km\,s_{-1}]$',
                 'v_d': r'$\sigma_\star\ [km\,s_{-1}]$',
                 }

        is_ext = {'light': True,
                  'mass': True,
                  'sfr': True,
                  'tau_V': False,
                  'age': False,
                  'met': False,
                  'd4000': False,
                  'v_0': False,
                  'v_d': False,
                  }

        op = {'light': np.log10,
              'mass': np.log10,
              'sfr': np.log10,
              'tau_V': lambda x: x,
              'age': lambda x: x,
              'met': lambda x: x,
              'd4000': lambda x: x,
              'v_0': lambda x: x,
              'v_d': lambda x: x,
              }
        
        vmin = {'light': None,
                'mass': None,
                'sfr': None,
                'tau_V': 0.0,
                'age': 7.0,
                'met': -0.7,
                'd4000': 0.9,
                'v_0': -300.0,
                'v_d': 0.0,
                }
        
        vmax = {'light': None,
                'mass': None,
                'sfr': None,
                'tau_V': 1.5,
                'age': 10.3,
                'met': 0.4,
                'd4000': 2.5,
                'v_0': 300.0,
                'v_d': 500.0,
                }
        
        cmap = {'light': 'viridis_r',
                'mass': 'viridis_r',
                'sfr': 'viridis_r',
                'tau_V': 'viridis_r',
                'age': 'viridis_r',
                'met': 'viridis_r',
                'd4000': 'viridis_r',
                'v_0': 'RdBu',
                'v_d': 'viridis_r',
                }
        
        image_order = ['light',
                       'mass',
                       'sfr',
                       'age',
                       'met',
                       'tau_V',
                       'v_0',
                       'v_d',
                       ]
        
        for k in image_order:
            print(k)
            im = images[k]
            if self.c.hasSegmentationMask:
                im = spatialize(im, self.c.segmentationMask, is_ext[k])
            im = op[k](im)
            self.ax_im.imshow(im, cmap=cmap[k], vmin=vmin[k], vmax=vmax[k], label=label[k])

        self.cb = plt.colorbar(self.ax_im.images[0], ax=self.ax_im)

        self.cursor = Circle(
            (0, 0), radius=1.5, lw=1.0, facecolor='none', edgecolor='r', figure=self.fig)
        self.ax_im.add_patch(self.cursor)

    def redraw(self):
        self.fig.canvas.draw()

    def run(self):
        self.fig.canvas.mpl_connect('button_press_event', self.onButtonPress)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        plt.show()

    def onButtonPress(self, ev):
        if (ev.button == 1) and (ev.inaxes is self.ax_im):
            x, y = np.rint(ev.xdata), np.rint(ev.ydata)
            self.selectPixel(x, y)
            self.redraw()

    def onKeyPress(self, ev):
        if ev.key in ['1', '2', '3', '4', '5', '6', '7', '8']:
            self.raiseImage(ev.key)
        elif ev.key in ['up', 'down', 'left', 'right']:
            self.displaceCursor(ev.key)
        elif ev.key == 'z':
            self.changeCLim(dmin=-0.05)
        elif ev.key == 'x':
            self.changeCLim(dmin=0.05)
        elif ev.key == 'c':
            self.changeCLim(dmax=-0.05)
        elif ev.key == 'v':
            self.changeCLim(dmax=0.05)
        elif ev.key == ' ':
            vmin, vmax = self.curImage.get_clim()
            print('vmin=%.2f, vmax=%.2f' % (vmin, vmax))
        self.redraw()

    def changeCLim(self, dmin=0.0, dmax=0.0):
        vmin, vmax = self.curImage.get_clim()
        rng = vmax - vmin
        vmin += rng * dmin
        vmax += rng * dmax
        self.curImage.set_clim(vmin, vmax)

    def updateColorbar(self, im):
        self.cb.mappable = im
        cid = im.callbacksSM.connect('changed', self.cb.on_mappable_changed)
        im.colorbar = self.cb
        im.colorbar_cid = cid
        self.cb.set_norm(im.norm)
        self.cb.on_mappable_changed(im)

    def raiseImage(self, key):
        try:
            ev_id = int(key) - 1
        except:
            pass
        for i in range(len(self.ax_im.images)):
            im = self.ax_im.images[i]
            if i == ev_id:
                im.set_alpha(1.0)
                self.curImage = im
                self.ax_im.set_title(im.get_label())
                self.updateColorbar(im)
            else:
                im.set_alpha(0.0)

    def displaceCursor(self, key):
        x, y = self.cursor.center
        if key == 'up':
            y += 1
        elif key == 'down':
            y -= 1
        elif key == 'right':
            x += 1
        elif key == 'left':
            x -= 1
        else:
            print('Cant displace cursor with key %s' % key)
            return
        self.selectPixel(x, y)

    def selectPixel(self, x, y):
        self.cursor.center = (x, y)
        x = int(x)
        y = int(y)
        c = self.c
        self.ax_sp.lines = []
        self.ax_res.lines = []
        self.fig.texts = []
        textsize = 'medium'
       
        self.fig.text(.5, .95, r'%s' % c.name,
                      size='larger', ha='center')

        if c.hasSegmentationMask:
            z = np.where(c.segmentationMask[:, y, x])[0]
            if len(z) == 0:
                self.fig.text(.6, .92, r'$(y, x) = (%d, %d)$ - no data' % (y, x),
                              size=textsize)
                return
            z = np.asscalar(z)
            f_norm = c.flux_norm_window[z]
            e_norm = c.noise_norm_window[z]
            f = c.f_obs[:,z] / f_norm
            e = c.f_err[:, z] / f_norm
            fl = c.f_flag[:,z]
            SN = f_norm / e_norm
            if c.hasSynthesis:
                s = c.f_syn[:, z] / f_norm
                chi2 = c.chi2[z]
                adev = c.adev[z]
                A_V = c.A_V[z]
                v_0 = c.v_0[z]
                v_d = c.v_d[z]
                Nclip = c.Nclipped[z]
        else:
            f_norm = c.flux_norm_window[y, x]
            e_norm = c.noise_norm_window[y, x]
            if f_norm is np.ma.masked or not np.isfinite(f_norm):
                f_norm = np.ma.median(c.f_obs[:, y, x])
            f = c.f_obs[:, y, x] / f_norm
            e = c.f_err[:, y, x] / f_norm
            fl = c.f_flag[:,y, x]
            SN = f_norm / e_norm
            if c.hasSynthesis:
                s = c.f_syn[:, y, x] / f_norm
                chi2 = c.chi2[y, x]
                adev = c.adev[y, x]
                A_V = c.A_V[y, x]
                v_0 = c.v_0[y, x]
                v_d = c.v_d[y, x]
                Nclip = c.Nclipped[y, x]

        ax = self.ax_sp
        ax.set_ylim(0, 2.5)
        ax.set_xlim(c.l_obs[0], c.l_obs[-1])
        ax.plot(c.l_obs, np.ma.masked_where(fl & (flags.telluric | flags.seg_has_badpixels) > 0, f),
                '-', color='blue', label='observed')
        err_scale = np.ceil(0.2 * f.mean() / e.mean())
        if not np.isfinite(err_scale):
            print('Error scale is non-finite. Setting it to zero.')
            err_scale = 0.0
        ax.plot(c.l_obs, e * err_scale, '-', color='k', label='error (x%d)' % err_scale)

        ax.plot(c.l_obs, np.ma.masked_where((fl & flags.telluric) == 0, f),
                '-', color='brown', label='telluric')

        ax.plot(c.l_obs, np.ma.masked_where((fl & flags.seg_has_badpixels) == 0, f),
                '-', color='purple', label='incomplete')

        masked_pix = (fl & flags.no_starlight > 0).all()

        self.fig.text(.6, .92, r'$(y, x) = (%d, %d)$' % (y, x), size=textsize)
        self.fig.text(.6, .88, r'$\mathrm{S/N (norm. window)} = %.1f$' % SN, size=textsize)

        if not c.hasSynthesis or masked_pix:
            self.fig.text(.6, .84, r'No synthesis', size=textsize)
            ax.legend(frameon=False)
            return

        ax.plot(c.l_obs, s, '-', color='red', label='model')
        ax.legend(frameon=False)

        ax = self.ax_res
        r = f - s
        ax.set_ylim(-1.0, 1.0)
        ax.plot(c.l_obs, e, '-', color='k', label='error')
        ax.plot(c.l_obs, np.zeros_like(c.l_obs), 'k:')
        fitted = np.ma.masked_where(fl & (flags.starlight_clipped | flags.starlight_masked | flags.before_starlight) > 0, r)
        ax.plot(c.l_obs, fitted, 'b-', label='fitted')

        masked = np.ma.masked_where(fl & flags.starlight_masked == 0, r)
        ax.plot(c.l_obs, masked, '-', color='magenta', label='masked')

        clipped = np.ma.masked_where(fl & flags.starlight_clipped == 0, r)
        ax.plot(c.l_obs, clipped, 'x-', color='red', label='clipped')

        flagged = np.ma.masked_where(fl & flags.before_starlight == 0, r)
        ax.plot(c.l_obs, flagged, 'o-', mec='green', mfc='none', label='flagged')
        ax.legend(frameon=False)

        self.fig.text(.6, .84, r'$\chi^2 = %3.2f$' % chi2, size=textsize)
        self.fig.text(.6, .80, r'$\mathrm{adev} = %3.2f$' % adev, size=textsize)
        self.fig.text(.6, .76, r'$N_\mathrm{clip} = %d$' % Nclip, size=textsize)
        self.fig.text(.6, .72, r'$A_V = %3.2f$' % A_V, size=textsize)
        self.fig.text(.6, .68,
                      r'$\sigma_\star = %3.2f\,\mathrm{km/s}\ |\ v_\star = %3.2f\,\mathrm{km/s}$' \
                      % (v_d, v_0), size=textsize)
        self.fig.text(.6, .64, r'$z = %3.4f$' % c.redshift, size=textsize)

##########################################################################

parser = argparse.ArgumentParser(description='pycasso cube explorer.')
parser.add_argument('cube', type=str, nargs=1,
                    help='pycasso cube.')

args = parser.parse_args()

print('Opening file %s' % args.cube[0])
pe = PycassoExplorer(args.cube[0])
print('''Use the keys 1-5 to cycle between the images.
Left click plots starlight results for the selected pixel.

The keys z, x decrease or increase the vmin of the current image.
The keys c, v decrease or increase the vmax of the current image.
 
Press <space> to print vmin & vmax of the current image.



''')
pe.run()
