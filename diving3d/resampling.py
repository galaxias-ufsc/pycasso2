'''
Created on 22/06/2015

@author: andre
'''

import pystarlight.util.StarlightUtils as stutil
import numpy as np

from . import flags

__all__ = ['resample_spectra', 'reshape_spectra', 'apply_redshift', 'velocity2redshift',
           'interp1d_spectra', 'gaussian1d_spectra']



def resample_spectra(spectra, l_orig, l_resam):
    '''
    FIXME: document reshape_image
    '''
    R = stutil.ReSamplingMatrixNonUniform(l_orig, l_resam)
    spectra = np.tensordot(R, spectra, (1,0))
    flagged = np.zeros_like(spectra, dtype='int32') + flags.no_data
    good = (l_resam >= l_orig[0]) & (l_resam <= l_orig[-1])
    flagged[good] = 0
    return spectra, flagged


def reshape_spectra(f_obs, f_flag, center, new_shape):
    '''
    FIXME: document reshape_image
    '''
    shape = f_obs.shape
    y_axis = 1
    x_axis = 2
    y0 = new_shape[y_axis] / 2 - center[y_axis]
    yf = y0 + shape[y_axis]
    x0 = new_shape[x_axis] / 2 - center[x_axis]
    xf = x0 + shape[x_axis]
    
    res_f_obs = np.zeros((new_shape))
    res_f_obs[:, y0:yf, x0:xf] = f_obs
    res_f_flag = np.zeros_like(res_f_obs, dtype='int32') + flags.no_data
    res_f_flag[:, y0:yf, x0:xf] = f_flag
    
    res_center = (center[0], new_shape[1] / 2, new_shape[2] / 2)
    
    return res_f_obs, res_f_flag, res_center
    

def apply_redshift(l, z, dest='rest'):
    '''
    Apply redshift correction to wavelength from to rest or observed frames.
    
    Parameters
    ----------
    l : array
        Wavelength array.
        
    z : array or float.
        Redshift.
        
    dest : string, optional
        Destination frame. Either ``'rest'`` or ``'observed'``.
        Default: ``'rest'``.
    
    Returns
    -------
    l_red : array
        Redshifted wavelength array. If ``z`` is an array,
        ``l_red`` will have the shape ``l.shape + z.shape``.
    '''
    if dest == 'rest':
        op = lambda x, y: x * y
    elif dest == 'observed':
        op = lambda x, y: x / y
        
    if np.array(z).shape == ():
        return op(l, 1. + z)
    else:
        return op(l[:, np.newaxis], 1. + z[np.newaxis, :])


def velocity2redshift(v):
    c = 299792458.0 # km/s
    return v / c


def fwhm2sigma(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def interp1d_spectra(l, flux, flags=None):
    '''
    Interpolate linearly 1-d spectra to fill all the gaps and
    extend the limits (copies the boundaries).
    
    Parameters
    ----------
    l : array
        Wavelength array.
    
    flux : array
        [Masked] array with gaps.
    
    flags : array, optional
        Array with flags as integers. Bad values
        are greater than zero. Must be set if
        ``flux`` is not a masked array. Default: ``None``.
    
    Returns
    -------
    flux_interp : array
        The same as ``flux``, wit gaps replaced by linear interpolation.
    '''
    if not isinstance(flux, np.ma.MaskedArray):
        if flags is None:
            raise Exception('flux must be a masked array if flags is not set.')
        flux = np.ma.array(flux, mask=flags > 0)
    lc = l[~flux.mask]
    fc = flux.compressed()
    return np.interp(l, lc, fc)


def gaussian1d_spectra(fwhm, l, flux, flags=None):
    '''
    Filter 1-d spectra using a Gaussian kernel. Interpolate linearly
    the flagged wavelengths before applying the filter.
    
    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the gaussian.
    
    l : array
        Wavelength array.
    
    flux : array
        [Masked] array with gaps.
    
    flags : array, optional
        Array with flags as integers. Bad values
        are greater than zero. Must be set if
        ``flux`` is not a masked array. Default: ``None``.
    
    Returns
    -------
    flux_gauss : array
        The same as ``flux``, interpolated by a gaussian kernel.
    '''
    flux_i = interp1d_spectra(l, flux, flags)
    dl = (l[1] - l[0])
    sig = fwhm2sigma(fwhm)
    l_gauss = np.arange(min(-2 * dl, -5 * sig), max(2 * dl + 1, 5 * sig + 1), dl)
    amp = 1.0 / (sig * np.sqrt(2.0 * np.pi))
    gauss = dl * amp * np.exp( -0.5 * (l_gauss / sig)**2 )
    return np.convolve(flux_i,  gauss, 'same')
    

