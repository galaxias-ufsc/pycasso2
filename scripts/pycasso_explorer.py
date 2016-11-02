from pycasso2 import FitsCube
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec

##########################################################################


class PycassoExplorer:

    def __init__(self, cube, figsize=(7, 7)):
        self.c = FitsCube(cube)
        self.createUI()
        self.raiseImage('1')
        self.selectPixel(self.c.x0, self.c.y0)
        self.redraw()

    def createUI(self):
        plt.ioff()
        self.fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(3, 2)
        self.ax_im = self.fig.add_subplot(gs[0, 0])
        self.ax_sp = self.fig.add_subplot(gs[1, :])
        self.ax_res = self.fig.add_subplot(gs[2, :], sharex=self.ax_sp)

        self.ax_sp.set_ylabel('$F_\lambda [normalized]$')
        self.ax_res.set_xlabel('$\lambda [\AA]$')
        self.ax_res.set_ylabel('Residual [%]')
        plt.setp(self.ax_sp.get_xticklabels(), visible=False)

        self.ax_im.imshow(np.log10(self.c.LobnSD.sum(axis=0)),
                          cmap='viridis_r', label=r'Image @ $5635 \AA$')
        self.ax_im.imshow(
            self.c.A_V, cmap='viridis_r', vmin=0.0, vmax=1.7, label=r'$A_V$')
        self.ax_im.imshow(
            self.c.at_flux, cmap='viridis_r', label=r'$\langle \log\,t \rangle_L$')
        self.ax_im.imshow(self.c.alogZ_flux, cmap='viridis_r',
                          label=r'$\langle \log\,Z/Z_\odot \rangle_M$')
        self.ax_im.imshow(self.c.LickIndex(
            'D4000'), cmap='viridis_r', vmin=0.9, vmax=2.5, label=r'$D(4000)$')
        self.ax_im.imshow(self.c.v_0, cmap='RdBu_r', vmin=-
                          300, vmax=300, label=r'$v_\star\ [km\,s_{-1}]$')
        self.ax_im.imshow(self.c.v_d, cmap='viridis_r', vmin=0,
                          vmax=500, label=r'$\sigma_\star\ [km\,s_{-1}]$')
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
        if ev.key in ['1', '2', '3', '4', '5', '6', '7']:
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
            print 'vmin=%.2f, vmax=%.2f' % (vmin, vmax)
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
        for i in xrange(len(self.ax_im.images)):
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
            print 'Cant displace cursor with key %s' % key
            return
        self.selectPixel(x, y)

    def selectPixel(self, x, y):
        self.cursor.center = (x, y)
        ax = self.ax_sp
        c = self.c
        ax.lines = []
        f = c.f_obs[:, y, x] / c.flux_norm_window[y, x]
        s = c.f_syn[:, y, x] / c.flux_norm_window[y, x]
        w = c.f_wei[:, y, x]
        r = (f - s) / f * 100.0
        ax.plot(c.l_obs, f, '-', color='blue')
        ax.plot(c.l_obs, s, '-', color='red')
        ax.set_ylim(0, 2.5)

        ax = self.ax_res
        ax.lines = []
        ax.set_ylim(-20, 20)
        ax.plot(c.l_obs, np.zeros_like(c.l_obs), 'k:')
        fitted = np.ma.masked_where(w < 0, r)
        ax.plot(c.l_obs, fitted, 'b-')

        masked = np.ma.masked_where(w != 0, r)
        ax.plot(c.l_obs, masked, '-', color='magenta')

        clipped = np.ma.masked_where(w != -1, r)
        ax.plot(c.l_obs, clipped, 'x', color='red')

        flagged = np.ma.masked_where(w != -2, r)
        plt.plot(c.l_obs, flagged, 'o', color='green')

        self.fig.texts = []
        textsize = 'large'
        self.fig.text(.6, .92, '(y, x) = (%d, %d)' % (y, x), size=textsize)
        self.fig.text(.6, .88, '$\chi^2 =\ $' + ('%3.2f' %
                                                 self.c.chi2[y, x]), size=textsize)
        self.fig.text(.6, .84, 'adev = ' + ('%3.2f' %
                                            self.c.adev[y, x]), size=textsize)
        self.fig.text(.6, .80, 'S/N (norm. window) = %.1f' %
                      self.c.SN_normwin[y, x], size=textsize)
        self.fig.text(.6, .76, '$A_V =\ $' + ('%3.2f' %
                                              self.c.A_V[y, x]), size=textsize)
        self.fig.text(.6, .72, '$\sigma_\star =\ ' + ('%3.2f' % self.c.v_d[
                      y, x]) + '\ $km/s\t$v_\star =\ ' + ('%3.2f' % self.c.v_0[y, x]) + '\ $km/s', size=textsize)

##########################################################################

parser = argparse.ArgumentParser(description='pycasso cube explorer.')
parser.add_argument('cube', type=str, nargs=1,
                    help='pycasso cube.')

args = parser.parse_args()

print 'Opening file %s' % args.cube[0]
pe = PycassoExplorer(args.cube[0])
print '''Use the keys 1-5 to cycle between the images.
Left click plots starlight results for the selected pixel.

The keys z, x decrease or increase the vmin of the current image.
The keys c, v decrease or increase the vmax of the current image.
 
Press <space> to print vmin & vmax of the current image.



'''
pe.run()
