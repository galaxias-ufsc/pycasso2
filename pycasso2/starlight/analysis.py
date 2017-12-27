'''
Created on 08/12/2015

@author: andre
'''

from ..resampling import age_smoothing_kernel, light2mass_ini, interp_age, bin_edges, hist_resample
import numpy as np
from scipy.interpolate import interp1d

__all__ = ['smooth_Mini', 'SFR', 'MStarsEvolution']


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


def SFR(Mini, tb1, tb2=None, dt=0.5e9):
    '''
    Calculate the star formation rate (SFR) resampling the initial mass.
    If tb1 is the same as tb2, the base is SSP. The base is expanded
    into a CSP with bins of constant SFR, assuming equally spaced bins
    in log space.

    Parameters
    ----------
    Mini: array
        The initial mass. The age axis must be the leftmost (axis=0).

    tb1 : array 
        Original age bin left edge.

    tb2 : array 
        Original age bin right edge.

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
    if tb2 is None or np.allclose(tb1, tb2):
        logtb = np.log10(tb1)
        logtb_bins = bin_edges(logtb)
        tb_bins = 10**logtb_bins
    else:
        tb_bins = np.zeros(len(tb1) + 1)
        tb_bins[0] = tb1[0]
        tb_bins[1:] = tb2[:]
    tl = np.arange(tb_bins.min(), tb_bins.max() + dt, dt)
    tl_bins = bin_edges(tl)
    spatial_shape = Mini.shape[1:]
    Mini = Mini.reshape(Mini.shape[0], -1)
    sfr_shape = (len(tl) + 2, Mini.shape[1])
    sfr = np.ma.zeros(sfr_shape)
    for i in range(sfr_shape[1]):
        if is_masked and np.ma.getmaskarray(Mini[:, i]).all():
            sfr[1:-1, i] = np.ma.masked
        else:
            Mini_resam = hist_resample(tb_bins, tl_bins, Mini[:, i])
            sfr[1:-1, i] = Mini_resam / dt

    tl = np.hstack((tl[0] - dt, tl, tl[-1] + dt))
    sfr.shape = (len(tl),) + spatial_shape
    return sfr, tl

def light_to_mass_ini(popx, fbase_norm, Lobs_norm, q_norm, A_V):

    '''
    Compute the initial mass from popx (and other parameters).
    The most important thing to remember is that popx (actually luminosity)
    must be "dereddened" before converting it to mass using fbase_norm
    (popmu_ini).
    
    Based on the subroutine ConvertLight2Mass (starlight source code).
    
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
    
    Returns
    -------
    Mini : array
        The initial mass in each population.

    '''
    Lobs = popx / 100.0
    Lobs *= Lobs_norm
    Lobs *= 10.0**(0.4 * q_norm * A_V)
    
    Mini = Lobs / fbase_norm[..., np.newaxis]
    return Mini


def calc_popmu(popx, fbase_norm, Lobs_norm, Mstars, q_norm, A_V):
    '''
    Compute popmu_ini and popmu_cor from popx (and other parameters).
    The most important thing to remember is that popx (actually luminosity)
    must be "dereddened" before converting it to mass using fbase_norm
    (popmu_ini). Also note the role of Mstars when computing the mass
    currently trapped inside stars (popmu_cor).
    
    Based on the subroutine ConvertLight2Mass (starlight source code).
    
    Parameters
    ----------
    popx: array
        The light fractions (in percents).
        
    fbase_norm: array
        Light to mass ratio for each population.
        
    Lobs_norm : array
        Luminosity norm of ``popx``.
        
    Mstars : array 
        Fraction of the initial stellar mass for a given
        population that is still trapped inside stars.
        
    q_norm : float 
        Ratio between the extinction in l_norm (where Lobs_norm
        is calculated) and ``A_V``.
        
    A_V : array 
        Extinction in V band.
    
    Returns
    -------
    popmu_ini : array
        The initial mass fractions (in percents).

    popmu_cor : array
        The current mass fractions (in percents).

    '''
    Mini = light_to_mass_ini(popx, fbase_norm, Lobs_norm, q_norm, A_V)
    Mcor = Mini * Mstars[..., np.newaxis]
    
    popmu_ini = Mini / Mini.sum(axis=1).sum(axis=0) * 100.0
    popmu_cor = Mcor / Mcor.sum(axis=1).sum(axis=0) * 100.0
    return popmu_ini, popmu_cor 


class MStarsEvolution(object):
    '''
    Handle Mstars interpolation for population mass evolution.
    '''
    
    def __init__(self, ageBase, metBase, Mstars):
        self._interpMstars = self._getInterpMstars(ageBase, metBase, Mstars)
        self._metBase = metBase

        
    def _getInterpMstars(self, ageBase, metBase, Mstars):
        '''
        Find the interpolation function for Mstars in the base
        ageBase and metBase. Points are added in age==0.0 and
        age==1e20, so one can find Mstars(today) and Mstars(at big bang).
        
        Returns a list of interpolation functions, one for each metallicity.
        '''
        interpMstars = [interp1d(ageBase, Mstars[:,Zi], fill_value='extrapolate') for Zi in range(len(metBase))]
        return interpMstars 


    def forTime(self, ageBase, Tc=None):
        '''
        MStars is the fraction of the initial stellar mass for a given
        population that is still trapped inside stars.
        This method calculates Mstars for populations given by ageBase,
        in evolutionary times given by Tc.
        
        If Tc is None, use the same times as ageBase.
        
        Returns a ndarray with shape (len(Tc), len(ageBase), len(metBase)),
        where metBase is the one used in constructor.
        '''
        if Tc is None:
            Tc = ageBase
        if not isinstance(Tc, np.ndarray):
            Tc = np.array([Tc])
            
        _f = np.zeros((len(Tc), len(ageBase), len(self._metBase)))
        for Ti in range(len(Tc)):
            mask = ageBase >= Tc[Ti]
            for Zi in range(len(self._metBase)):
                _f[Ti][mask,Zi] = self._interpMstars[Zi](ageBase[mask] - Tc[Ti] + ageBase[0])
    
        return _f
    

