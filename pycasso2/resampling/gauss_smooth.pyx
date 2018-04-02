'''
Created on Jun 13, 2013

@author: andre
'''
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

__all__ = ['gauss_velocity_smooth', 'gauss_velocity_smooth_error']

cdef extern from 'math.h':
    double exp(double x)
    double sqrt(double x)
    int floor(double x)


cdef inline double _interp(double fo_a, double fo_b, double lo_a, double lo_b, double lo_x):
                cdef double a, b, ff
                a = (fo_b - fo_a) / (lo_b - lo_a)
                b = fo_a - a * lo_a
                ff = a * lo_x + b
                return ff


# The same as above using less operations.
cdef inline double _interp2(double fo_a, double fo_b, double lo_a, double lo_b, double lo_x):
                return fo_a + (fo_b - fo_a) * (lo_x - lo_a) / (lo_b - lo_a)


cdef double c = 2.997925e5  # km/s
cdef double sqrt_2pi = sqrt(2.0 * np.pi)
cdef double sigma2fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))

@cython.boundscheck(False)
@cython.wraparound(False)
def gauss_velocity_smooth(np.ndarray[np.double_t, ndim=1, mode='c'] lo not None,
                          np.ndarray[np.double_t, ndim=1, mode='c'] fo not None,
                          double v0, double sig,
                          np.ndarray[np.double_t, ndim=1, mode='c'] ls=None,
                          int n_u=51, int n_sig=6, fill='nearest', double fill_val=0):
    '''
    Apply a gaussian velocity dispersion and displacement filter to a spectrum.
    Implements the integration presented in the page 27 of the 
    `STARLIGHT manual <http://www.starlight.ufsc.br/papers/Manual_StCv04.pdf>`_.
    
    Parameters
    ----------
    
    lo : array
        Original wavelength array.
        
    fo : array
        Original flux array, must be of same length as ``lo``.
    
    v0 : float
        Systemic velocity to apply to input spectrum.
        
    sig : float
        Velocity dispersion (sigma of the gaussian).
        
    ls : array, optional
        Wavelengths in which calculate the output spectrum.
        If not set, ``lo`` will be used.
        
    n_u : int, optional
        Number of points to sample the velocity gaussian.
        Defaults to 51 (also the maximum). Use an odd number to guarantee the center
        of the gaussian enters the calculation.
        
    n_sig : int, optional
        Width of the integration kernel, in units of sigma.
        
    fill : string, optional.
        Filling mode: ``'nearest'`` to use the first and last values,
        ``'constant'`` to use value specified by ``fill_value``.
        Defaults to ``'repeat'``.
    
    fill_val : int, optional
        Value to fill beyond the boundaries. Default: ``0``.
        
    Returns
    -------
    
    fs : array
        Resampled and smoothed flux array, must be of same length as ``ls``
        (or ``lo`` if ``ls`` is not set).
    
   '''
    if n_u < 5:
        raise ValueError('n_u=%d too small to integrate properly.' % n_u)

    if n_u > 51:
        raise ValueError('n_u=%d too large.' % n_u)
    
    if ls is None:
        ls = lo
        
    if sig < 0.0:
        raise ValueError('sig must be positive.')

    if n_sig < 0.0:
        raise ValueError('n_sig must be positive.')
        
# Parameters for the wavelength iteration
    cdef unsigned int i_s
    cdef unsigned int Ns = ls.shape[0]
    cdef unsigned int No = lo.shape[0]
    cdef double lo_0 = lo[0]
    cdef double d_lo = lo[1] - lo[0]    
    cdef double ll, lss
    cdef double fo_0
    cdef double fo_last
    
    if fill == 'nearest':
        fo_0 = fo[0]
        fo_last = fo[No - 1]
    elif fill == 'constant':
        fo_0 = fill_val
        fo_last = fill_val
    else:
        raise ValueError('Unknown filling mode: %s' % fill)


# Variables for the velocity integration
    cdef unsigned int i_u
    cdef int ind1, ind2
    cdef double u_low = -n_sig
    cdef double u_upp = n_sig
    cdef double du = (u_upp - u_low) / (n_u - 1)
    cdef double du_sqrt_2pi = du / sqrt_2pi

# Avoid calculating these values on every velocity loop
    cdef double[51] u, v, exp_u2
    cdef double uu
    for i_u from 0 <= i_u < n_u:
        uu = u_low + du * i_u
        u[i_u] = uu
        exp_u2[i_u] = exp(uu*uu / -2.0)
        v[i_u] = v0 + sig * uu
        
# Smoothed spectrum to be returned.
    cdef double sum_fg
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] fs = np.empty(Ns, dtype=np.float64)

# loop over ls, convolving fo(lo) with a gaussian
    for i_s from 0 <= i_s < Ns:
        lss = ls[i_s]
# reset integral of {fo[ll = ls/(1+v/c)] * Gaussian} & start integration
        sum_fg = 0.
        for i_u from 0 <= i_u < n_u:
# define lambda corresponding to u
            ll = lss / (1.0 + v[i_u] / c)

# find fo flux for lambda = ll
            ind1 = int((ll - lo_0) / d_lo)
            if ind1 < 0: 
                ff = fo_0
            elif ind1 >= (No - 1):
                ff = fo_last
            else:
                ind2 = ind1 + 1
                ff = _interp2(fo[ind2], fo[ind1], lo[ind2], lo[ind1], ll)

# smoothed spectrum
            sum_fg += ff * exp_u2[i_u]
        fs[i_s] = sum_fg * du_sqrt_2pi
    return fs


@cython.boundscheck(False)
@cython.wraparound(False)
def gauss_velocity_smooth_error(np.ndarray[np.double_t, ndim=1, mode='c'] lo not None,
                                np.ndarray[np.double_t, ndim=1, mode='c'] flux_o not None,
                                np.ndarray[np.double_t, ndim=1, mode='c'] err_o not None,
                                np.ndarray[np.uint8_t, cast=True, ndim=1, mode='c'] badpix_o not None,
                                double v0, double sig,
                                np.ndarray[np.double_t, ndim=1, mode='c'] ls=None,
                                int n_u=51, int n_sig=6, fill='nearest', double fill_val=0,
                                double badpix_threshold=0.8, double l_cov_FWHM=-1.0):
    '''
    Apply a gaussian velocity dispersion and displacement filter to a spectrum.
    Implements the integration presented in the page 27 of the 
    `STARLIGHT manual <http://www.starlight.ufsc.br/papers/Manual_StCv04.pdf>`_.
    Apllies the smoothing to the flux, error and badpix, optionally taking the
    error covariance into account. 
    
    Parameters
    ----------
    
    lo : array
        Original wavelength array.
        
    flux_o : array
        Original flux array, must be of same length as ``lo``.
    
    err_o : array
        Original error array, must be of same length as ``lo``.
    
    badpix_o : array of booleans
        Original flag array, must be of same length as ``lo``.
    
    v0 : float
        Systemic velocity to apply to input spectrum.
        
    sig : float
        Velocity dispersion (sigma of the gaussian).
        
    ls : array, optional
        Wavelengths in which calculate the output spectrum.
        If not set, ``lo`` will be used.
        
    n_u : int, optional
        Number of points to sample the velocity gaussian.
        Defaults to 51 (also the maximum). Use an odd number to guarantee the center
        of the gaussian enters the calculation.
        
    n_sig : int, optional
        Width of the integration kernel, in units of sigma.
        
    fill : string, optional.
        Filling mode: ``'nearest'`` to use the first and last values,
        ``'constant'`` to use value specified by ``fill_value``.
        Defaults to ``'repeat'``.
    
    fill_val : int, optional
        Value to fill beyond the boundaries. Default: ``0``.
        
    badpix_threshold : float, optional
        Flag output if the fraction of flagged input pixels contributing
        to an output pixel is less than ``badpix_threshold``. Default: ``0.8``.
        
    l_cov_FWHM : float, optional
        The FWHM of the spectral resolution, in Angstroms. A value less than zero
        disables covariance calculation. Default: ``-1.0``, do not compute covariance.
        
    Returns
    -------
    
    flux_s, err_s, badpix_s : array
        Resampled and smoothed flux array, must be of same length as ``ls``
        (or ``lo`` if ``ls`` is not set).
    
   '''
    if n_u < 5:
        raise ValueError('n_u=%d too small to integrate properly.' % n_u)

    if n_u > 51:
        raise ValueError('n_u=%d too large.' % n_u)
    
    if ls is None:
        ls = lo
        
    if sig < 0.0:
        raise ValueError('sig must be positive.')

    if n_sig < 0.0:
        raise ValueError('n_sig must be positive.')
        
# Parameters for the wavelength iteration
    cdef unsigned int Ns = ls.shape[0]
    cdef unsigned int No = lo.shape[0]
    cdef double lo_0 = lo[0]
    cdef double flux_0
    cdef double flux_last
    cdef double err_0
    cdef double err_last
    
    if fill == 'nearest':
        flux_0 = flux_o[0]
        flux_last = flux_o[No - 1]
        err_0 = err_o[0]
        err_last = err_o[No - 1]
    elif fill == 'constant':
        flux_0 = fill_val
        flux_last = fill_val
        err_0 = fill_val
        err_last = fill_val
    else:
        raise ValueError('Unknown filling mode: %s' % fill)


    cdef int do_covariance = False
    cdef double theta
    if l_cov_FWHM > 0.0:
        l_cov_FWHM /= sigma2fwhm
        theta = -0.5 / (l_cov_FWHM * l_cov_FWHM)
        do_covariance = True
    
# Variables for the velocity integration
    cdef unsigned int i_u, j_u
    cdef int ind1, ind2
    cdef double u_low = -n_sig
    cdef double u_upp = n_sig
    cdef double du = (u_upp - u_low) / (n_u - 1)
    #cdef double du_sqrt_2pi = du / sqrt_2pi
    cdef double tmp

# Avoid calculating these values on every velocity loop
    cdef double[51] u, z, w, w2
    cdef double uu
    for i_u from 0 <= i_u < n_u:
        uu = u_low + du * i_u
        u[i_u] = uu
        tmp = exp(uu*uu / -2.0) * du / sqrt_2pi
        w[i_u] = tmp
        w2[i_u] = tmp * tmp
        z[i_u] = 1.0 + (v0 + sig * uu) / c
        
# Smoothed spectrum to be returned.
    cdef unsigned int i_s
    cdef double ll_i, ll_j, dll, lss, lo1, lo2
    cdef double zz
    cdef double d_lo = lo[1] - lo_0
    cdef double sum_f, sum_e2, ff, ee2, norm, norm2, cov
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] flux_s = np.empty(Ns, dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] err_s = np.empty(Ns, dtype=np.float64)
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] badpix_s = np.empty(Ns, dtype=np.uint8)

# loop over ls, convolving fo(lo) with a gaussian
    for i_s from 0 <= i_s < Ns:
        lss = ls[i_s]
# reset integral of {fo[ll = ls/(1+v/c)] * Gaussian} & start integration
        sum_f = 0.0
        sum_e2 = 0.0
        norm = 0.0
        norm2 = 0.0
        cov = 0.0
        for i_u from 0 <= i_u < n_u:
# define lambda corresponding to u
            ll_i = lss / z[i_u]

# find fo flux for lambda = ll
            ind1 = floor((ll_i - lo_0) / d_lo)
            if ind1 < 0:
                continue
            elif ind1 >= (No - 1):
                continue
            ind2 = ind1 + 1
            if badpix_o[ind1] or badpix_o[ind2]:
                continue
            lo1 = lo[ind1]
            lo2 = lo[ind2]
            ff = _interp2(flux_o[ind2], flux_o[ind1], lo2, lo1, ll_i)
            
            e_i = _interp2(err_o[ind2], err_o[ind1], lo2, lo1, ll_i)
            ee2 = e_i * e_i

            norm +=w[i_u]
            norm2 += w2[i_u]
            sum_f += ff * w[i_u]
            sum_e2 += ee2 * w2[i_u]

            # Compute covariance between err[i_u] and err[j_u]
            if not do_covariance:
                continue
            for j_u from 0 <= j_u < n_u:
                if i_u == j_u:
                    continue
                ll_j = lss / z[j_u]
                ind1 = int((ll_j - lo_0) / d_lo)
                if ind1 < 0:
                    continue 
                if ind1 >= (No - 1):
                    continue
                ind2 = ind1 + 1
                if badpix_o[ind1] or badpix_o[ind2]:
                    continue
                lo1 = lo[ind1]
                lo2 = lo[ind2]
                e_j = _interp2(err_o[ind2], err_o[ind1], lo2, lo1, ll_j)
                dll = ll_j - ll_i
                cov += e_i * e_j * w[i_u] * w[j_u] * exp(theta * dll * dll) * abs(dll / (i_u - j_u))

# smoothed spectrum
        badpix_s[i_s] = norm < badpix_threshold
        if norm > 0:
            flux_s[i_s] = sum_f / norm
        else:
            flux_s[i_s] = 0.0
        if norm2 > 0.0:
            sum_e2 += cov / (sqrt_2pi * l_cov_FWHM)
            err_s[i_s] = sqrt(sum_e2 / norm2)
        else:
            err_s[i_s] = 0.0
    return flux_s, err_s, badpix_s

