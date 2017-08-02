from .resampling import bin_edges
from astropy import log
import numpy as np
cimport numpy as np
cimport cython

def resample_cube(l_orig, l_resam, f):
    Nlo = len(l_orig)
    Nlr = len(l_resam)
    spatial_shape = f.shape[1:]
    f = f.reshape(Nlo, -1)
    Nspec = f.shape[1]
    res_shape = (Nlr, Nspec)
    fr = np.zeros(res_shape, dtype=f.dtype)
    bins_orig = bin_edges(l_orig)
    bins_resam = bin_edges(l_resam)
    buf_in = np.empty(Nlo)
    buf_out = np.zeros(Nlr)
    for i in range(Nspec):
        if (i % 200) == 0:
            pass
            log.debug('    Resampled %d of %d' % (i, Nspec))
        buf_in[:] = f[:, i]
        _hist_resample(bins_orig, bins_resam, buf_in, buf_out, density=True)
        fr[:, i] = buf_out[:]
    new_shape = (Nlr,) + spatial_shape
    fr.shape = new_shape
    return fr
    

@cython.boundscheck(False)
@cython.wraparound(False)
def _hist_resample(np.ndarray[np.double_t, ndim=1, mode='c'] bins_o not None,
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

    v_r[j] = v_r_j
    while j < n_v_r:
        j += 1
        v_r[j] = 0.0


