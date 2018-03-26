from .resampling import bin_edges
from astropy import log
import numpy as np
cimport numpy as np
cimport cython

cdef extern from 'math.h':
    double exp(double x)
    double sqrt(double x)
    int floor(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
def hist_resample(np.ndarray[np.double_t, ndim=1, mode='c'] bins_o not None,
                  np.ndarray[np.double_t, ndim=1, mode='c'] bins_r not None, 
                  np.ndarray[np.double_t, ndim=1, mode='c'] v not None,
                  np.ndarray[np.double_t, ndim=1, mode='c'] v_r not None,
                  int density=0):
    '''
    Resample an histogram into another set of bins.
    
    Parameters
    ----------
    bins_o : array
        The original bin edges.
    
    bins_r : array
        Bin edges to resample.
        
    v : array
        The heights of the histogram.
        Must be of length ``len(bins_o) - 1``.

    v_r : array, out
        The heights of the resampled histogram
        (or resampled densities if ``density=True``),
        with length ``len(bins_r) - 1``.
        If ``bins_r`` completely overlaps ``bins_o``,
        then the sums ``v.sum()`` and ``v_r.sum()``
        will be equal (within numeric error).
        Must be allocated by the caller.

    density : boolean, optional
        ``v`` is a density, not a histogram.
        Default: ``False``.
    '''
    cdef int n_v_r = len(bins_r) - 1
    if len(v_r) != n_v_r:
        raise Exception('Resampled array size must be len(bins_r) - 1.')

    cdef int i_0 = 0
    cdef int j = 0
    cdef int i
    cdef double lbo = bins_o[i_0]
    cdef double rbo = bins_o[i_0+1]
    cdef double lbr = bins_r[j]
    cdef double rbr = bins_r[j+1]
    cdef double dbr = rbr - lbr
    
    cdef double last_edge, frac, v_r_j, v_i
    
    v_r_j = 0.0
    
    if lbo < lbr:
        # Skip leading original bins that do not overlap the resampled bins.
        while lbr > rbo:
            i_0 += 1
            lbo = bins_o[i_0]
            rbo = bins_o[i_0+1]
            dbo = rbo - lbo
            #print 'dbo', dbo
        frac = (rbo - lbr) / dbo
        if density:
            v_r_j += frac * v[i_0] * dbo
        else:
            v_r_j += frac * v[i_0]
        i_0 += 1
        last_edge = rbo
        v_r[j] = v_r_j / dbr
    else:
        # Skip leading resampled bins that do not overlap the original bins.
        while rbr < lbo:
            j += 1
            v_r[j] = 0.0
            lbr = bins_r[j]
            rbr = bins_r[j+1]
        last_edge = rbr
        
    for i from i_0 <= i < len(v):
        rbo = bins_o[i+1]
        lbo = bins_o[i]
        dbo = rbo - lbo
        v_i = v[i]
        if rbo < rbr:
            # This original bin is entirely contained in the resampled bin.
            if density:
                v_r_j += v_i * dbo
            else:
                v_r_j += v_i
            last_edge = rbo
            continue
        #print 'dbo', dbo
        while rbr < rbo:
            # Compute the slices of the original bin going to the resampled bins.
            frac = (rbr - last_edge) / dbo
            if density:
                v_r_j += frac * v_i * dbo
                v_r_j /= dbr
            else:
                v_r_j += frac * v_i
            last_edge = rbr
            v_r[j] = v_r_j
            j += 1
            v_r_j = 0.0
            if j >= n_v_r:
                break
            lbr = bins_r[j]
            rbr = bins_r[j+1]
            dbr = rbr - lbr
        if j >= n_v_r:
            break
        # Compute the last slice of the original bin.
        frac = (rbo - last_edge) / dbo
        if density:
            #print 'dbr', dbr
            v_r_j += frac * v_i * dbo
        else:
            v_r_j += frac * v_i
        last_edge = rbo

    while j < n_v_r:
        v_r[j] = v_r_j
        v_r_j = 0.0
        j += 1


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
                          int n_u=31, int n_sig=6, fill='nearest', double fill_val=0):
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
        Defaults to 31 (also the maximum). Use an odd number to guarantee the center
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

    if n_u > 31:
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
    cdef double[31] u, v, exp_u2
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
