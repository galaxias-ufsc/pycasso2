'''
Created on 25 de mai de 2018

@author: andre
'''

from .. import flags
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_vlines(vlines, ax):
    if vlines is not None:
        for l in vlines:
            ax.axvline(l, ls='--', color='k')


def plot_spec(l_obs, f_obs, f_err, f_syn, f_flag, f_emline=None, vlines=None, fig=None):

    if fig is None:
        fig = plt.figure(figsize=(12, 6))
        
    gs = GridSpec(2, 1)
    ax_sp = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_sp)

    ax_sp.set_ylabel(r'$F_\lambda [\mathrm{normalized}]$')
    ax_res.set_xlabel(r'$\lambda [\AA]$')
    ax_res.set_ylabel(r'$O_\lambda - M_\lambda$')
    plt.setp(ax_sp.get_xticklabels(), visible=False)
   
    ax = ax_sp
    #ax.set_ylim(0, 2.5)
    ax.set_xlim(l_obs[0], l_obs[-1])
    ax.plot(l_obs, np.ma.masked_where(f_flag & (flags.telluric | flags.seg_has_badpixels) > 0, f_obs),
            '-', color='blue', label='observed')
    err_scale = np.ceil(0.2 * f_obs.mean() / f_err.mean())
    if not np.isfinite(err_scale):
        print('Error scale is non-finite. Setting it to zero.')
        err_scale = 0.0
    ax.plot(l_obs, f_err * err_scale, '-', color='k', label='error (x%d)' % err_scale)

    ax.plot(l_obs, np.ma.masked_where((f_flag & flags.telluric) == 0, f_obs),
            '-', color='brown', label='telluric')

    ax.plot(l_obs, np.ma.masked_where((f_flag & flags.seg_has_badpixels) == 0, f_obs),
            '-', color='purple', label='incomplete')

    if f_emline is not None:
        ax.plot(l_obs, f_syn + f_emline, '-', color='g', label='emission lines')

    ax.plot(l_obs, f_syn, '-', color='red', label='model')
    
    plot_vlines(vlines, ax)
    ax.legend(frameon=False)

    ax = ax_res
    f_res = f_obs - f_syn
    #ax.set_ylim(-1.0, 1.0)
    ax.plot(l_obs, f_err, '-', color='k', label='error')
    ax.plot(l_obs, np.zeros_like(l_obs), 'k:')
    fitted = np.ma.masked_where(f_flag & (flags.starlight_clipped | flags.starlight_masked | flags.before_starlight) > 0, f_res)
    ax.plot(l_obs, fitted, 'b-', label='residual')

    masked = np.ma.masked_where(f_flag & flags.starlight_masked == 0, f_res)
    ax.plot(l_obs, masked, '-', color='magenta', label='masked')

    clipped = np.ma.masked_where(f_flag & flags.starlight_clipped == 0, f_res)
    ax.plot(l_obs, clipped, 'x-', color='red', label='clipped')

    flagged = np.ma.masked_where(f_flag & flags.before_starlight == 0, f_res)
    ax.plot(l_obs, flagged, 'o-', mec='red', mfc='none', label='flagged')
    
    if f_emline is not None:
        ax.plot(l_obs, f_emline, '-', color='g', label='emission line')
    
    plot_vlines(vlines, ax)
    ax.legend(frameon=False)
    
    return fig


def plot_sfh(popx, popmu_ini, age_base, Z_base, ax):
    
    agevec = np.unique(age_base)
    sfh_x = np.array([np.sum(popx[age_base == agevec[i]]) for i in range(len(agevec))])
    sfh_x /= popx.sum()
    csfh_x = np.cumsum(sfh_x[::-1])    

    sfh_mu = np.array([np.sum(popmu_ini[age_base == agevec[i]]) for i in range(len(agevec))])
    sfh_mu /= popmu_ini.sum()
    csfh_mu = np.cumsum(sfh_mu[::-1])    
    
    
    ax.plot(np.log10(agevec), csfh_x, color='b', label=r'$x$')
    ax.plot(np.log10(agevec), csfh_x, '.b')

    ax.plot(np.log10(agevec), csfh_mu, color='r', label=r'$\mu$')
    ax.plot(np.log10(agevec), csfh_mu, '.r')
    
    ax.set_xlabel(r'$\log t_*$', fontsize=10)
    ax.set_ylabel('Cumulative Fraction [%]', fontsize=10)

