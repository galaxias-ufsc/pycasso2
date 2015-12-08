'''
Created on 08/12/2015

@author: andre
'''

from ..resampling import age_smoothing_kernel, light2mass_ini, interp_age
import numpy as np

__all__ = ['smooth_Mini']

def smooth_Mini(popx, fbase_norm, Lobs_norm, q_norm, A_V, logtb, logtc, logtc_FWHM):
    '''
    Calculate the star formation rate (SFR) from the light fractions,
    using a smoothed logarithmic (base 10) time base. The smoothed
    log. time base ``logtc`` must be evenly spaced.
    
    This code is is basen on the equation (5) from Asari (2007)
    <http://adsabs.harvard.edu/abs/2007MNRAS.381..263A> 

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
    SFR_sm : array
        The star formation rate, smoothed.
        Note: ``len(SFR_sm) == len(logtc)`` 

    '''
    smoothKernel = age_smoothing_kernel(logtb, logtc, logtc_FWHM)
    popx_sm = np.tensordot(smoothKernel, popx, (0,0))
    
    fbase_norm_interp = interp_age(fbase_norm, logtb, logtc)
    Mini_sm = light2mass_ini(popx_sm, fbase_norm_interp, Lobs_norm, q_norm, A_V)
    return Mini_sm

