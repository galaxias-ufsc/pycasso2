#!/usr/bin/env python
from pycasso2 import FitsCube, flags
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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
        gs = GridSpec(3, 6)
        self.ax_im = self.fig.add_subplot(gs[0, :3], projection=self.c._wcs.celestial)
        self.ax_sp = self.fig.add_subplot(gs[1, :])
        self.ax_res1 = self.fig.add_subplot(gs[2, :2])
        self.ax_res2 = self.fig.add_subplot(gs[2, 2:4])
        self.ax_res3 = self.fig.add_subplot(gs[2, 4:])

        self.ax_sp.set_ylabel(r'$F_\lambda [\mathrm{normalized}]$')
        self.ax_res2.set_xlabel(r'$\lambda [\AA]$')
        self.ax_res1.set_ylabel(r'$O_\lambda - M_\lambda$')
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
                      'Ha':  np.zeros_like(c.flux_norm_window),
                      }
        if c.hasELines:
            images.update({'Ha': c.EL_flux(6563),
                      })

        label = {'light': r'Image @ $5635 \AA$',
                 'mass': r'$\Sigma_\star$',
                 'sfr': r'$\Sigma_\mathrm{SFR}$',
                 'tau_V': r'$\tau_V$',
                 'age': r'$\langle \log\,t \rangle_L$',
                 'met': r'$\langle \log\,Z/Z_\odot \rangle_M$',
                 'd4000': r'$D(4000)$',
                 'v_0': r'$v_\star\ [km\,s_{-1}]$',
                 'v_d': r'$\sigma_\star\ [km\,s_{-1}]$',
                 'Ha': r'$\log F(\mathrm{H\alpha})$',
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
                  'Ha': False,
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
              'Ha': np.log10,
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
                'Ha': 2.0,
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
                'Ha': 5.0,
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
                'Ha': 'viridis',
                }
        
        image_order = ['light',
                       'mass',
                       'sfr',
                       'age',
                       'met',
                       'tau_V',
                       'v_0',
                       'v_d',
                       'Ha'
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
        if ev.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
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
        cid = im.callbacks.connect('changed', self.cb.update_normal)
        im.colorbar = self.cb
        im.colorbar_cid = cid
        self.cb.mappable.set_norm(im.norm)
        self.cb.update_normal(im)

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
        self.ax_sp.clear()
        self.ax_res1.clear()
        self.ax_res2.clear()
        self.ax_res3.clear()
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
            if c.hasELines:
                fp = c.EL_continuum[:, z] / f_norm
                fe = (c.EL_continuum[:, z] + c.EL_total_flux(z)) / f_norm
                vlines = c.EL_info['l0']
            else:
                fe = None
                vlines = None            
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
            if c.hasELines:
                fp = c.EL_continuum[:, y, x] / f_norm
                fe = (c.EL_continuum[:, y, x] + c.EL_total_flux(y, x)) / f_norm
                vlines = c.EL_info['l0']
            else:
                fe = None
                vlines = None      
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
        r = f - s
        # ax.set_ylim(-0.1, 3.0)
        ax.plot(c.l_obs, r, '-', color='k', label='residual', drawstyle = 'steps-mid')
        ax.plot(c.l_obs, fe, 'r-', label='dobby fitted', drawstyle = 'steps-mid')
        ax.plot(c.l_obs, fp, 'grey', label='pseudocontinuum', drawstyle = 'steps-mid')
        ax.legend(frameon=False)

        self.fig.text(.6, .84, r'$\chi^2 = %3.2f$' % chi2, size=textsize)
        self.fig.text(.6, .80, r'$\mathrm{adev} = %3.2f$' % adev, size=textsize)
        self.fig.text(.6, .76, r'$N_\mathrm{clip} = %d$' % Nclip, size=textsize)
        self.fig.text(.6, .72, r'$A_V = %3.2f$' % A_V, size=textsize)
        self.fig.text(.6, .68,
                      r'$\sigma_\star = %3.2f\,\mathrm{km/s}\ |\ v_\star = %3.2f\,\mathrm{km/s}$' \
                      % (v_d, v_0), size=textsize)
        self.fig.text(.6, .64, r'$z = %3.4f$' % c.redshift, size=textsize)

        # Plot [OIII]
        ax = self.ax_res1
        ff = (c.l_obs >= 4909) & (c.l_obs <= 5057)
        ax.plot(c.l_obs[ff], r[ff], '-', color='k', label='residual')
        ax.plot(c.l_obs[ff], fe[ff], 'r-', label='dobby fitted')
        ax.plot(c.l_obs[ff], fp[ff], 'grey', label='pseudocontinuum')

        # Plot Hb
        ax = self.ax_res2
        ff = (c.l_obs >= 4811) & (c.l_obs <= 4911)
        ax.plot(c.l_obs[ff], r[ff], '-', color='k', label='residual')
        ax.plot(c.l_obs[ff], fe[ff], 'r-', label='dobby fitted')
        ax.plot(c.l_obs[ff], fp[ff], 'grey', label='pseudocontinuum')

        # Plot Ha & [NII]
        ax = self.ax_res3
        ff = (c.l_obs >= 6498) & (c.l_obs <= 6634)
        ax.plot(c.l_obs[ff], r[ff], '-', color='k', label='residual')
        ax.plot(c.l_obs[ff], fe[ff], 'r-', label='dobby fitted')
        ax.plot(c.l_obs[ff], fp[ff], 'grey', label='pseudocontinuum')
##########################################################################

parser = argparse.ArgumentParser(description='pycasso cube explorer.')
parser.add_argument('cube', type=str, nargs=1,
                    help='pycasso cube.')

args = parser.parse_args()

print('Opening file %s' % args.cube[0])
pe = PycassoExplorer(args.cube[0])
print('''Use the keys 1-9 to cycle between the images.
Left click plots starlight results for the selected pixel.

The keys z, x decrease or increase the vmin of the current image.
The keys c, v decrease or increase the vmax of the current image.
 
Press <space> to print vmin & vmax of the current image.



''')
pe.run()
