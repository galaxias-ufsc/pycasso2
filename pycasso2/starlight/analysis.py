'''
Created on 08/12/2015

@author: andre
'''

from ..resampling import age_smoothing_kernel, light2mass_ini, interp_age, bin_edges, hist_resample
import numpy as np

__all__ = ['smooth_Mini', 'SFR']


def smooth_Mini(popx, fbase_norm, Lobs_norm, q_norm, A_V, logtb, logtc, logtc_FWHM):
    '''
    Calculate the initial mass from the light fractions,
    using a smoothed logarithmic (base 10) time base. The smoothed
    log. time base ``logtc`` must be evenly spaced.

    Parameters
    ----------
    popx: array
        The light fractions (in percents).

    fbase_norm: array
        Light to mass ratio for each population.

    Lobs_norm : array
        Luminosity norm of ``popx``.

    q_norm : float 
        Ratio between the extinction in l_norm (where Lobs_norm
        is calculated) and ``A_V``.

    A_V : array 
        Extinction in V band.

    logtb : array 
        Logarithm (base 10) of original time base.

    logtc : array 
        Logarithm (base 10) of resampled time base.
        Must be evenly spaced.

    logtc_FWHM : float
        Width of the age smoothing kernel used to resample ``popx``.

    Returns
    -------
    Mini_sm : array
        The initial mass rate, smoothed.

    '''
    smoothKernel = age_smoothing_kernel(logtb, logtc, logtc_FWHM)
    popx_sm = np.tensordot(smoothKernel, popx, (0, 0))

    fbase_norm_interp = interp_age(fbase_norm, logtb, logtc)
    Mini_sm = light2mass_ini(
        popx_sm, fbase_norm_interp, Lobs_norm, q_norm, A_V)
    return Mini_sm


def SFR(Mini, tb, dt=0.5e9):
    '''
    Calculate the star formation rate (SFR) resampling the initial mass.

    Parameters
    ----------
    Mini: array
        The initial mass. The age axis must be the leftmost (axis=0).

    tb : array 
        Original time base.

    dt : float 
        Sampling size of the SFR output.
    Returns
    -------
    SFR : array
        The star formation rate.

    t : array
        Time.
        Note: ``SFR.shape[0] == len(t)`` 

    '''
    is_masked = isinstance(Mini, np.ma.MaskedArray)
    logtb = np.log10(tb)
    logtb_bins = bin_edges(logtb)
    tb_bins = 10**logtb_bins
    tl = np.arange(tb_bins.min(), tb_bins.max() + dt, dt)
    tl_bins = bin_edges(tl)
    spatial_shape = Mini.shape[1:]
    Mini = Mini.reshape(Mini.shape[0], -1)
    sfr_shape = (len(tl) + 2, Mini.shape[1])
    sfr = np.zeros(sfr_shape)
    for i in range(sfr_shape[1]):
        if is_masked and np.ma.getmaskarray(Mini[:, i]).all():
            continue
        Mini_resam = hist_resample(tb_bins, tl_bins, Mini[:, i])
        sfr[1:-1, i] = Mini_resam / dt

    tl = np.hstack((tl[0] - dt, tl, tl[-1] + dt))
    sfr.shape = (len(tl),) + spatial_shape
    return sfr, tl
