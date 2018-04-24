'''
Checking emission lines fitted with dobby in MaNGA cubes.

Usage:

python3 dobby_plots_examples.py 7960-6101

Natalia@UFSC - 29/Nov/2017
'''

import sys
from os import path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import astropy.constants as const

from pycasso2 import FitsCube

c_light = const.c.to('km/s').value

########################################################################
# Get galaxy name
galname = sys.argv[1]


########################################################################
# Defining dirs
el_dir = '/Users/natalia/data/MaNGA/dr14/starlight_el/dr14.bin2.cA.CB17_7x16/'
    
########################################################################
# Reading file
c = FitsCube(path.join(el_dir, 'manga-%s.dr14.bin2.cA.CB17_7x16.ElGAk1b1.fits') % galname)

# Check cube
#print(c._HDUList.info())

# Get fluxes and EW
El_info  = c._getSynthExtension('El_info')
El_F     = c._getSynthExtension('El_F')
El_vd    = c._getSynthExtension('El_vd')
El_EW    = c._getSynthExtension('El_EW')
El_lcrms = c._getSynthExtension('El_lcrms')

# TO FIX
dl = 1.
El_vdl = El_vd * El_info['l0'][:, np.newaxis, np.newaxis] / c_light
El_N   = dl * El_lcrms * np.sqrt(6. * El_vdl / dl)
El_SN  = El_F / El_N

# Get position of Ha in the cubes
flag_Ha = (El_info['lambda'] == 6563)
flag_Hb = (El_info['lambda'] == 4861)

# Plot EW(Ha)
plt.figure(1)
plt.clf()
plt.imshow(El_EW[flag_Ha, ...][0])
plt.title(r'$\log \mathrm{EW}_{\mathrm{H}\alpha}$')
plt.colorbar()

# Plots L(Ha)
plt.figure(2)
plt.clf()
plt.imshow(El_F[flag_Ha, ...][0])
plt.title(r'$\log \Sigma_{\mathrm{H}\alpha}$')
plt.colorbar()

# Plot Ha/Hb
plt.figure(3)
plt.clf()
plt.scatter(El_SN[flag_Hb][0], El_F[flag_Ha][0]/El_F[flag_Hb][0])

# Plot emission lines
plt.figure(3)
plt.clf()
plt.scatter(El_SN[flag_Hb][0], El_F[flag_Ha][0]/El_F[flag_Hb][0])

# Plot emission lines
plt.figure(4)
plt.clf()
plt.scatter(El_EW[flag_Ha][0], El_F[flag_Ha][0]/El_F[flag_Hb][0])

# Plot emission lines
plt.figure(5)
plt.clf()
plt.scatter(El_EW[flag_Hb][0], El_F[flag_Ha][0]/El_F[flag_Hb][0])


def plot_fits(c, iy = None, ix = None, ifig = 1, integrated = False, model = 'resampled_gaussian'):

    if model == 'resampled_gaussian':
        from .models.resampled_gaussian import MultiResampledGaussian
        elModel = MultiResampledGaussian
    elif model == 'gaussian':
        from .models.gaussian import MultiGaussian
        elModel = MultiGaussian
    else:
        raise Exception('@@> No model found. Giving up.')
    
    if iy is None:
        iy = c.y0
    if ix is None:
        ix = c.x0
    
    # Get all emission lines which have been fitted
    lines = c._getSynthExtension('El_info')['lambda']

    fig = plt.figure(ifig, figsize=(12,6))
    gs = gridspec.GridSpec(2, 3)

    # Start full spectrum
    ll = c.l_obs
    f_res = (c.f_obs - c.f_syn)

    # Get fluxes and EW
    El_info  = c._getSynthExtension('El_info')
    El_F     = c._getSynthExtension('El_F')
    El_v0    = c._getSynthExtension('El_v0')
    El_vd    = c._getSynthExtension('El_vd')
    El_lc    = c._getSynthExtension('El_lc')
    El_vdins = c._getSynthExtension('El_vdins')

    El_EW    = c._getSynthExtension('El_EW')
    El_lcrms = c._getSynthExtension('El_lcrms')
    
    # Get position of Ha in the cubes
    flag_Ha = (El_info['lambda'] == 6563)
    
    # Write fluxes in FHa units
    FHa = El_F[flag_Ha][0]

    import astropy.constants as const
    c_light = const.c.to('km/s').value

    def gauss(l, l0, flux, v0, vd, vd_inst):
        v = c_light * (l - l0) / l0
        vd_obs2 = vd ** 2 + vd_inst ** 2
        sigma_l = np.sqrt(vd_obs2) * l0 / c_light
        ampl = flux / (sigma_l * np.sqrt(2. * np.pi))
        return ampl * np.exp( -0.5 * (v - v0)** 2 / vd_obs2 )
        
    # Get all emission lines
    El_model = El_lc.copy()

    for il, line in enumerate(lines):
        flag_line = (El_info['lambda'] == line)
        El_model += gauss(ll[:, np.newaxis, np.newaxis], El_info['l0'][il], El_F[flag_line], El_v0[flag_line], El_vd[flag_line], El_vdins[flag_line])


    def El_info_label(ax, line, xlab, ylab):
        flag_line = (El_info['lambda'] == line)
        ax.text(xlab, ylab, '$\lambda_0 = %s$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$' 
                % (line, El_v0[flag_line, iy, ix], El_vd[flag_line, iy, ix], El_F[flag_line, iy, ix] / FHa[iy, ix]),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

    # Plot integrated spectrum
    if integrated:
        pass

    # Plot a spaxel
    else:

        f_res_px = f_res[..., iy, ix]
        El_model_px = np.ma.array(El_model[..., iy, ix], mask=(El_lc == 0)[..., iy, ix])
        
        # Plot the full spectrum
        ax1 = plt.subplot(gs[0, :])
        plt.plot(ll, f_res_px, 'k', label = 'Residual')
        plt.plot(ll, El_model_px, 'r', label = 'Fit')
        plt.legend(loc = 'upper right')

        # Plot [OIII]
        ax2 = plt.subplot(gs[1, 0])
        flag = (ll >= 4900) & (ll < 5100)
        plt.plot(ll[flag], f_res_px[flag], 'k', label = 'Residual')
        plt.plot(ll[flag], El_model_px[flag], 'r', label = 'Fit')
        for i, line in enumerate([4959, 5007]):
            El_info_label(ax2, line, 0.99, 0.95 - i*0.45)
            
        # Plot Hbeta
        ax3 = plt.subplot(gs[1, 1])
        flag = (ll >= 4750) & (ll < 4950)
        plt.plot(ll[flag], f_res_px[flag], 'k', label = 'Residual')
        plt.plot(ll[flag], El_model_px[flag], 'r', label = 'Fit')
        for i, line in enumerate([4861]):
            El_info_label(ax3, line, 0.99, 0.95 - i*0.45)
                    
        # Plot [NII, Halpha]
        ax4 = plt.subplot(gs[1, 2])
        flag = (ll >= 6500) & (ll < 6600)
        plt.plot(ll[flag], f_res_px[flag], 'k', label = 'Residual')
        plt.plot(ll[flag], El_model_px[flag], 'r', label = 'Fit')
        for i, line in enumerate([6563, 6548, 6584]):
            xlab, ylab = 0.99, 0.55 + (i-1)*0.40
            if (line == 6563):
                xlab, ylab = 0.50, 0.95
            El_info_label(ax4, line, xlab, ylab)


            
for suf in ['ElGAk0b0', 'ElGAk0b1', 'ElGAk1b0', 'ElGAk1b1', 'ElRGk0b0', 'ElRGk0b1', 'ElRGk1b0', 'ElRGk1b1']:
    c = FitsCube(path.join(el_dir, 'manga-%s.dr14.bin2.cA.CB17_7x16.%s.fits') % (galname, suf))            
    #++plot_fits(c, ifig=suf)
    #++plt.suptitle('%s.%s.pdf' % (galname, suf))
    #++plt.savefig('%s.%s.pdf' % (galname, suf))
    #++plt.close()
