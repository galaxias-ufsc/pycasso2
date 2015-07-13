#! /usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt
import pystarlight.io
import atpy


def plot_fit( data_file, savefig=False ):

    ''' Plot results from the STARLIGHT output file - ver. PANcMEx_StarlightChains_v03b400.for.

    Top-left:      Plot containing the observed (in black), synthetic (red), error (green) spectra
                   and the masked pixels (dots).

    Bottom-left:   Residual spectrum in black, flagged pixels in green, masked pixels in magenta
                   and clipped pixels in red 'x's, if there's any.

    Top-right:     Info panel showing some useful information about the fit and some results.

    Bottom-right:  Population fractions in popx vectors (light, top subplot) and in popmu_cor
                   (mass, bottom subplot) by age.

    Parameters:
    -----------
    data_file: Your STARLIGHT output file.
    savefig: If True it your brand new figure as 'data_file_full_plot.png in your working dir. Image dimensions = 24in x 18in
    '''
    # To-Do: "Fix" the layout. The distribution plots are too small.


    t = atpy.TableSet( data_file, type='starlight' )

    popage_base = t.population.popage_base
    popx        = t.population.popx
    popmu_cor   = t.population.popmu_cor

    f_obs = t.spectra.f_obs
    f_wei = t.spectra.f_wei
    f_syn = t.spectra.f_syn
    l_obs = t.spectra.l_obs

    f_res = f_obs - f_syn
    flag_mask = ((f_wei < 0) & (f_wei != -1.0))

    # Creating masked arrays for everything we need
    f_res_clip_mask         = np.ma.array(f_res)
    f_res_flag_mask         = np.ma.array(f_res)
    f_res_masked_mask       = np.ma.array(f_res)
    f_res_no_bad_data_mask  = np.ma.array(f_res)
    f_wei_no_bad_data_mask  = np.ma.array(f_wei)

    # Here we mask all pixels to identify which ones where masked (w = 0), flagged (w < 0 and w != -1) or clipped (w=1).
    # Note that we're interested in the flagged/clipped/masked pixels themselves, so we need to mask everything that aren't them.
    f_res_clip_mask[ f_wei != -1.0 ]      = np.ma.masked
    f_res_flag_mask[ -flag_mask  ]        = np.ma.masked
    f_res_masked_mask[ f_wei != 0  ]      = np.ma.masked
    f_res_no_bad_data_mask[ f_wei <= 0 ]  = np.ma.masked
    f_wei_no_bad_data_mask[ f_wei <= 0 ]  = np.ma.masked

    # Computing some stuff
    logt_L      = np.dot( t.population.popx,      np.log10( t.population.popage_base ) ) / t.population.popx.sum()
    logt_M      = np.dot( t.population.popmu_cor, np.log10( t.population.popage_base ) ) / t.population.popmu_cor.sum()
    Z_L         = np.dot( t.population.popx,      t.population.popZ_base ) / t.population.popx.sum()
    Z_M         = np.dot( t.population.popmu_cor, t.population.popZ_base ) / t.population.popmu_cor.sum()

    # Creating the figure
    fig = plt.figure(figsize=(24,18))

    # Fit subplot
    fig_fit = plt.subplot2grid( (4,3), (0,0), rowspan=2, colspan=2 )
    fig_fit.grid( b='on' )
    plt.plot( l_obs, f_obs, 'black', alpha=0.7, label='Observed spectrum' )
    plt.plot( l_obs, f_syn, 'red', label='Synthetic spectrum' )
    plt.plot( l_obs, 1/f_wei_no_bad_data_mask, 'green', label='Error spectrum' )
    plt.plot( l_obs[flag_mask], flag_mask[flag_mask] - 0.9, 'yo', markersize=1.5 )
    plt.ylabel( r'$F_{\lambda}$ [normalized]' )
    xticks = fig_fit.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)                # These remove the leftmost and rightmost ticks

    # Residual spectrum subplot
    fig_residual = plt.subplot2grid( (4,3), (2,0), rowspan=2, colspan=2, sharex=fig_fit )
    fig_residual.grid( b='on' )
    plt.plot( l_obs, f_res, 'black', label='Residual spectrum' )
    plt.plot( l_obs, f_res_flag_mask, 'green', label='Flagged pixels' )
    plt.plot( l_obs, f_res_masked_mask, 'magenta', label='Masked pixels' )
    plt.plot( l_obs, f_res_clip_mask, 'red', marker='x', linestyle=' ', label='Clipped pixels' )
    plt.suptitle( data_file )
    plt.xlabel( r'$\lambda [\AA]$' )
    plt.ylabel( r'Residual spectrum' )
    yticks = fig_residual.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)                # This remove the overlaping tick

    # Info subplot
    fig_info = plt.subplot2grid( (4,3), (0,2), rowspan=2 )
    fig_info.axes.get_xaxis().set_visible(False)
    fig_info.axes.get_yaxis().set_visible(False)    # These remove the axes
    fig_info.text( 0.05, 0.92 - 0.00, r'$\chi^2 =$ '                         + str( t.keywords['chi2'] ) )
    fig_info.text( 0.05, 0.92 - 0.05, r'adev $=$ '                           + str( t.keywords['adev'] ) )
    fig_info.text( 0.05, 0.92 - 0.10, r'S/N  $=$ '                           + str( t.keywords['SN_normwin'] ) )
    fig_info.text( 0.05, 0.92 - 0.15, r'$A_V$ $=$ '                          + str( t.keywords['A_V'] ) )
    fig_info.text( 0.05, 0.92 - 0.20, r'$\sigma_*$  $=$ '                    + str( t.keywords['v_d'] ) )
    fig_info.text( 0.05, 0.92 - 0.25, r'$v_*$  $=$ '                         + str( t.keywords['v_0'] ) )
    fig_info.text( 0.05, 0.92 - 0.30, r'$N_{clipped}$ $=$ '                  + str( t.keywords['Ntot_clipped'] ) )
    fig_info.text( 0.05, 0.92 - 0.35, r'$\langle \log t\rangle_L$ $=$ '      + "%.4f" % logt_L )
    fig_info.text( 0.05, 0.92 - 0.40, r'$\langle \log t\rangle_M$ $=$ '      + "%.4f" % logt_M )
    fig_info.text( 0.05, 0.92 - 0.45, r'$\langle Z\rangle_L$ $=$ '           + "%.5f" % Z_L )
    fig_info.text( 0.05, 0.92 - 0.50, r'$\langle Z\rangle_M$ $=$ '           + "%.5f" % Z_M )

    # Population fraction in light
    fig_pop = plt.subplot2grid( (4,3), (2,2), rowspan=1 )
    fig_pop.axes.get_xaxis().set_visible(False)
    fig_pop.grid( b='off' )
    fig_pop.semilogx()
    plt.vlines( popage_base, 0, popx, linewidths=5, color='green' )
    plt.xlabel( r'$t_* \ [yr]$', fontsize=16 )
    plt.ylabel( r'$x_j \ [\%]$ @ $\lambda = 4020 \AA$', fontsize=16 )
    plt.ylim( [0.0, 100.0] )

    # Population fraction in mass
    fig_mu = plt.subplot2grid( (4,3), (3,2), rowspan=1 )
    fig_mu.grid( b='off' )
    fig_mu.semilogx()
    plt.vlines( popage_base, 0, popmu_cor, linewidths=5, color='green' )
    plt.xlabel( r'$t_* \ [yr]$', fontsize=16 )
    plt.ylabel( r'$\mu_j \ [\%]$', fontsize=16 )
    plt.ylim( [0.0, 100.0] )

    fig.subplots_adjust( hspace=0, wspace=0.35 )

    if savefig == True:
        plt.savefig( data_file + '_full_plot.png' )
        print "Figure " + data_file + "_full_plot.png saved!"
        plt.close()
    else:
        plt.show()
