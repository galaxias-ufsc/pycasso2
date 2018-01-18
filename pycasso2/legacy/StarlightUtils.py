'''
Created on Mar 13, 2012

@author: Andre Luiz de Amorim,
         Robeto Cid Fernandes

'''

import numpy as np
from scipy.interpolate import interp1d

###############################################################################
## Spectral resampling utility
###############################################################################
def ReSamplingMatrix(lorig , lresam, norm='True', extrap = False):
    '''
    Compute resampling matrix R_o2r, useful to convert a spectrum sampled at 
    wavelengths lorig to a new grid lresamp. Both are assumed to be uniform grids (ie, constant step).
    Input arrays lorig and lresamp are the bin centres of the original and final lambda-grids.
    ResampMat is a Nlresamp x Nlorig matrix, which applied to a vector F_o (with Nlorig entries) returns
    a Nlresamp elements long vector F_r (the resampled spectrum):

        [[ResampMat]] [F_o] = [F_r]

    Based on (but less general than) STARLIGHT's routine RebinSpec. Conserves flux, except for 
    possible loss at the blue & red edges of lorig (1st and last pixels).

    ElCid@Sanchica - 09/Feb/2012
    
    
    Parameters
    ----------
    lorig : array_like
            Original spectrum lambda array.
    
    lresam : array_like
             Spectrum lambda array in which the spectrum should be sampled.        
        
    Returns
    -------
    ResampMat : array_like
                Resample matrix. 
    
    Examples
    --------
    >>> lorig = np.linspace(3400, 8900, 4623)
    >>> lresam = np.linspace(3400, 8900, 9000)
    >>> forig = np.random.normal(size=len(lorig))**2
    >>> matrix = slut.ReSamplingMatrix(lorig, lresam)
    >>> fresam = np.dot(matrix, forig)
    >>> np.trapz(fresam, lresam)
    5588.7178984840939
    >>> np.trapz(forig, lorig)
    5588.7178984824877
    '''
    
    # Defs: steps & numbers
    dlorig  = lorig[1]  - lorig[0]
    dlresam = lresam[1] - lresam[0]
    Nlorig  = len(lorig)
    Nlresam = len(lresam)

    # Defs: lower & upper bin borders for lorig & lresam
    lorig_low  = lorig - dlorig/2
    lorig_upp  = lorig + dlorig/2
    lresam_low = lresam - dlresam/2
    lresam_upp = lresam + dlresam/2

    # Reset & fill resampling matrix
    ResampMat = np.zeros((Nlresam,Nlorig))

    # Find out if the original array is simply a subset of the resampled.
    # If it is, this routine already returns the correct resampling matrix
    subset = lresam_subset_lorig(lresam, lorig, ResampMat)

    if (not subset):

        for i_r in range(Nlresam):
            # inferior & superior lambdas representing the contribution of each lorig bin to current lresam bin.
            l_inf = np.where(lorig_low > lresam_low[i_r] , lorig_low , lresam_low[i_r])
            l_sup = np.where(lorig_upp < lresam_upp[i_r] , lorig_upp , lresam_upp[i_r])
    
            # lambda interval overlap of each lorig bin for current lresam bin. Negatives cliped to zero.
            dl = (l_sup - l_inf).clip(0)
    
            # When a lresam bin is not fully (> 99%) covered by lorig pixels, then discard it (leave it with zeros).
            # This will only happen at the edges of lorig.
            if (0 < dl.sum() < 0.99* dlresam):
                dl = 0 * lorig
    
            ResampMat[i_r,:] = dl
    
        if norm:
            ResampMat = ResampMat / dlresam

    # Fix extremes: extrapolate if needed
    if (extrap):
        extrap_ResampMat(ResampMat, lorig_low, lorig_upp, lresam_low, lresam_upp)

    return ResampMat


def ReSamplingMatrixNonUniform(lorig, lresam, extrap = False):
    '''
    Compute resampling matrix R_o2r, useful to convert a spectrum sampled at 
    wavelengths lorig to a new grid lresamp. Here, there is no necessity to have constant gris as on :py:func:`ReSamplingMatrix`.
    Input arrays lorig and lresamp are the bin centres of the original and final lambda-grids.
    ResampMat is a Nlresamp x Nlorig matrix, which applied to a vector F_o (with Nlorig entries) returns
    a Nlresamp elements long vector F_r (the resampled spectrum):

        [[ResampMat]] [F_o] = [F_r]

    Warning! lorig and lresam MUST be on ascending order!
    
    
    Parameters
    ----------
    lorig : array_like
            Original spectrum lambda array.
    
    lresam : array_like
             Spectrum lambda array in which the spectrum should be sampled.        

    extrap : boolean, optional
           Extrapolate values, i.e., values for lresam < lorig[0]  are set to match lorig[0] and
                                     values for lresam > lorig[-1] are set to match lorig[-1].
           

    Returns
    -------
    ResampMat : array_like
                Resample matrix. 
    
    Examples
    --------
    >>> lorig = np.linspace(3400, 8900, 9000) * 1.001
    >>> lresam = np.linspace(3400, 8900, 5000)
    >>> forig = np.random.normal(size=len(lorig))**2
    >>> matrix = slut.ReSamplingMatrixNonUniform(lorig, lresam)
    >>> fresam = np.dot(matrix, forig)
    >>> print np.trapz(forig, lorig), np.trapz(fresam, lresam)
    '''
    
    # Init ResampMatrix
    matrix = np.zeros((len(lresam), len(lorig)))
    
    # Define lambda ranges (low, upp) for original and resampled.
    lo_low = np.zeros(len(lorig))
    lo_low[1:] = (lorig[1:] + lorig[:-1])/2
    lo_low[0] = lorig[0] - (lorig[1] - lorig[0])/2 

    lo_upp = np.zeros(len(lorig))
    lo_upp[:-1] = lo_low[1:]
    lo_upp[-1] = lorig[-1] + (lorig[-1] - lorig[-2])/2

    lr_low = np.zeros(len(lresam))
    lr_low[1:] = (lresam[1:] + lresam[:-1])/2
    lr_low[0] = lresam[0] - (lresam[1] - lresam[0])/2
    
    lr_upp = np.zeros(len(lresam))
    lr_upp[:-1] = lr_low[1:]
    lr_upp[-1] = lresam[-1] + (lresam[-1] - lresam[-2])/2

    
    # Find out if the original array is simply a subset of the resampled.
    # If it is, this routine already returns the correct resampling matrix
    subset = lresam_subset_lorig(lresam, lorig, matrix)

    if (not subset):

        # Iterate over resampled lresam vector
        for i_r in range(len(lresam)): 
            
            # Find in which bins lresam bin within lorig bin
            bins_resam = np.where( (lr_low[i_r] < lo_upp) & (lr_upp[i_r] > lo_low) )[0]
    
            # On these bins, eval fraction of resamled bin is within original bin.
            for i_o in bins_resam:
                
                aux = 0
                
                d_lr = lr_upp[i_r] - lr_low[i_r]
                d_lo = lo_upp[i_o] - lo_low[i_o]
                d_ir = lo_upp[i_o] - lr_low[i_r]  # common section on the right
                d_il = lr_upp[i_r] - lo_low[i_o]  # common section on the left
                
                # Case 1: resampling window is smaller than or equal to the original window.
                # This is where the bug was: if an original bin is all inside the resampled bin, then
                # all flux should go into it, not then d_lr/d_lo fraction. --Natalia@IoA - 21/12/2012
                if (lr_low[i_r] >= lo_low[i_o]) & (lr_upp[i_r] <= lo_upp[i_o]):
                    aux += 1.
                    
                # Case 2: resampling window is larger than the original window.
                if (lr_low[i_r] < lo_low[i_o]) & (lr_upp[i_r] > lo_upp[i_o]):
                    aux += d_lo / d_lr
    
                # Case 3: resampling window is on the right of the original window.
                if (lr_low[i_r] >= lo_low[i_o]) & (lr_upp[i_r] > lo_upp[i_o]):
                    aux += d_ir / d_lr
    
                # Case 4: resampling window is on the left of the original window.
                if (lr_low[i_r] < lo_low[i_o]) & (lr_upp[i_r] <= lo_upp[i_o]):
                    aux += d_il / d_lr
    
                matrix[i_r, i_o] += aux


    # Fix matrix to be exactly = 1 ==> TO THINK
    #print np.sum(matrix), np.sum(lo_upp - lo_low), (lr_upp - lr_low).shape

    
    # Fix extremes: extrapolate if needed
    if (extrap):
        extrap_ResampMat(matrix, lo_low, lo_upp, lr_low, lr_upp)
        
    return matrix


def ReSamplingMatrixNonUniform_alt(lorig , lresam, norm='True', extrap = False):
    '''
    This is the pythonic equivalent to ReSamplingMatrixNonUniform.
    However, because the resampling matrix is so sparse, this version
    is *not* as fast as ReSamplingMatrixNonUniform when tested
    against arrays of sizes ~3000. It might be faster in some cases,
    so it remains here. It *is* faster than ReSamplingMatrix.

    ----
    
    Compute resampling matrix R_o2r, useful to convert a spectrum sampled at 
    wavelengths lorig to a new grid lresamp. Both are assumed to be uniform grids (ie, constant step).
    Input arrays lorig and lresamp are the bin centres of the original and final lambda-grids.
    ResampMat is a Nlresamp x Nlorig matrix, which applied to a vector F_o (with Nlorig entries) returns
    a Nlresamp elements long vector F_r (the resampled spectrum):

        [[ResampMat]] [F_o] = [F_r]

    Based on (but less general than) STARLIGHT's routine RebinSpec. Conserves flux, except for 
    possible loss at the blue & red edges of lorig (1st and last pixels).

    ElCid@Sanchica - 09/Feb/2012
    
    
    Parameters
    ----------
    lorig : array_like
            Original spectrum lambda array.
    
    lresam : array_like
             Spectrum lambda array in which the spectrum should be sampled.        
        
    Returns
    -------
    ResampMat : array_like
                Resample matrix. 
    
    Examples
    --------
    >>> lorig = np.linspace(3400, 8900, 4623)
    >>> lresam = np.linspace(3400, 8900, 9000)
    >>> forig = np.random.normal(size=len(lorig))**2
    >>> matrix = slut.ReSamplingMatrix(lorig, lresam)
    >>> fresam = np.dot(matrix, forig)
    >>> np.trapz(fresam, lresam)
    5588.7178984840939
    >>> np.trapz(forig, lorig)
    5588.7178984824877
    '''

    # Init ResampMatrix
    ResampMat__ro = np.zeros((len(lresam), len(lorig)))
    
    # Define lambda ranges (low, upp) for original and resampled.
    lo_low = np.zeros(len(lorig))
    lo_low[1:] = (lorig[1:] + lorig[:-1])/2
    lo_low[0] = lorig[0] - (lorig[1] - lorig[0])/2 

    lo_upp = np.zeros(len(lorig))
    lo_upp[:-1] = lo_low[1:]
    lo_upp[-1] = lorig[-1] + (lorig[-1] - lorig[-2])/2

    lr_low = np.zeros(len(lresam))
    lr_low[1:] = (lresam[1:] + lresam[:-1])/2
    lr_low[0] = lresam[0] - (lresam[1] - lresam[0])/2
    
    lr_upp = np.zeros(len(lresam))
    lr_upp[:-1] = lr_low[1:]
    lr_upp[-1] = lresam[-1] + (lresam[-1] - lresam[-2])/2

    
    # Find out if the original array is simply a subset of the resampled.
    # If it is, this routine already returns the correct resampling matrix
    subset = lresam_subset_lorig(lresam, lorig, ResampMat__ro)

    if (not subset):
        
        # Create comparison matrixes for lower and upper bin limits
        na = np.newaxis
        lo_low__ro = lo_low[na, ...]
        lo_upp__ro = lo_upp[na, ...]
        lr_low__ro = lr_low[..., na]
        lr_upp__ro = lr_upp[..., na]
    
        # Find in which bins lresam bin within lorig bin
        ff = (lr_low__ro < lo_upp__ro) & (lr_upp__ro > lo_low__ro)
    
        # Eval fraction of resamled bin is within original bin
        d_lr = lr_upp__ro - lr_low__ro
        d_lo = lo_upp__ro - lo_low__ro
        d_ir = lo_upp__ro - lr_low__ro  # common section on the right
        d_il = lr_upp__ro - lo_low__ro  # common section on the left
    
        # Case 1: resampling window is smaller than or equal to the original window.
        f1 = (ff) & (lr_low__ro >= lo_low__ro) & (lr_upp__ro <= lo_upp__ro)
        if (f1.sum() > 0):
            ResampMat__ro[f1] += 1.
    
        # Case 2: resampling window is larger than the original window.
        f2 = (ff) & (lr_low__ro < lo_low__ro) & (lr_upp__ro > lo_upp__ro)
        if (f2.sum() > 0):
            ResampMat__ro[f2] += (d_lo / d_lr)[f2]
    
        # Case 3: resampling window is on the right of the original window.
        f3 = (ff) & (lr_low__ro > lo_low__ro) & (lr_upp__ro > lo_upp__ro)
        if (f3.sum() > 0):
            ResampMat__ro[f3] += (d_ir / d_lr)[f3]
    
        # Case 4: resampling window is on the left of the original window.
        f4 = (ff) & (lr_low__ro < lo_low__ro) & (lr_upp__ro < lo_upp__ro)
        if (f4.sum() > 0):
            ResampMat__ro[f4] += (d_il / d_lr)[f4]

    # Fix extremes: extrapolate if needed
    if (extrap):
        extrap_ResampMat(ResampMat__ro, lo_low, lo_upp, lr_low, lr_upp)

    return ResampMat__ro


def lresam_subset_lorig(lresam, lorig, ResampMat__ro):
    '''
    Finds out if lorig is a subset of lresam. If so,
    create a `diagonal' resampling matrix (in place).
    '''

    subset = False
    tol = 1.e-5
    
    ff = (lresam >= lorig[0]) & (lresam <= lorig[-1])
    lresam_sub = lresam[ff]

    N_lo = len(lorig)

    if (len(lresam_sub) == N_lo):
        Nok = (abs(lorig - lresam_sub) < tol).sum()

        if (Nok == N_lo):
            subset = True
            ilow = np.where(ff)[0][0]
            iupp = np.where(ff)[0][-1]
    
            io_diag = np.arange(len(lorig))
            ir_diag = np.arange(ilow, iupp+1)
            ResampMat__ro[ir_diag, io_diag] = 1.

    return subset


def extrap_ResampMat(ResampMat__ro, lo_low, lo_upp, lr_low, lr_upp):
    '''
    Extrapolate resampling matrix on the borders, i.e.,
    by making it copy the first and the last bins.

    This modifies ResampMat__ro in place.
    '''
    
    bins_extrapl = np.where( (lr_low < lo_low[0])  )[0]
    bins_extrapr = np.where( (lr_upp > lo_upp[-1]) )[0]

    if (len(bins_extrapl) > 0) & (len(bins_extrapr) > 0):
        io_extrapl = np.where( (lo_low >= lr_low[bins_extrapl[0]]) )[0][0]
        io_extrapr = np.where( (lo_upp <= lr_upp[bins_extrapr[0]]) )[0][-1]

        ResampMat__ro[bins_extrapl, io_extrapl] = 1.
        ResampMat__ro[bins_extrapr, io_extrapr] = 1.

    
###############################################################################
## Resample an histogram.
###############################################################################

def hist_resample(bins_o, bins_r, v, density=False):
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
    density : boolean, optional
        ``v`` is a density, not a histogram.
        Default: ``False``.
        
    Returns
    -------
    v_r : array
        The heights of the resampled histogram
        (or resampled densities if ``density=True``),
        with length ``len(bins_r) - 1``.
        If ``bins_r`` completely overlaps ``bins_o``,
        then the sums ``v.sum()`` and ``v_r.sum()``
        will be equal (within numeric error).
    '''
    n_v_r = len(bins_r) - 1
    v_r = np.zeros(n_v_r, dtype=v.dtype)
    i_0 = 0
    j = 0
    lbo = bins_o[i_0]
    rbo = bins_o[i_0+1]
    lbr = bins_r[j]
    rbr = bins_r[j+1]
    dbr = rbr - lbr
    
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
            v_r[j] += frac * v[i_0] * dbo
        else:
            v_r[j] += frac * v[i_0]
        i_0 += 1
        last_edge = rbo
        v_r[j] /= dbr
    else:
        # Skip leading resampled bins that do not overlap the original bins.
        while rbr < lbo:
            j += 1
            lbr = bins_r[j]
            rbr = bins_r[j+1]
        last_edge = rbr
        
    for i in range(i_0, len(v)):
        rbo = bins_o[i+1]
        lbo = bins_o[i]
        dbo = rbo - lbo
        if rbo < rbr:
            # This original bin is entirely contained in the resampled bin.
            if density:
                v_r[j] += v[i] * dbo
            else:
                v_r[j] += v[i]
            last_edge = rbo
            continue
        #print 'dbo', dbo
        while rbr < rbo:
            # Compute the slices of the original bin going to the resampled bins.
            frac = (rbr - last_edge) / dbo
            if density:
                v_r[j] += frac * v[i] * dbo
                v_r[j] /= dbr
            else:
                v_r[j] += frac * v[i]
            last_edge = rbr
            j += 1
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
            v_r[j] += frac * v[i] * dbo
        else:
            v_r[j] += frac * v[i]
        last_edge = rbo
    return v_r


###############################################################################
## Resample a spectrum
###############################################################################

def spec_resample(l_orig, l_resam, flux):
    '''
    Resample a spectrum into another set of wavelengths.
    
    Parameters
    ----------
    bins_o : array
        Original wavelengths.
    
    bins_r : array
        Wavelengths to resample.
        
    flux : array
        The flux. Must be of length ``len(l_orig) - 1``.
        
    Returns
    -------
    flux_r : array
        Fluxes resampled to ``l_resam``, with length
        ``len(bins_r) - 1``. If ``bins_r`` completely
        overlaps ``bins_o``, then the sums ``v.sum()``
        and ``v_r.sum()`` will be equal (within
        numeric error).
    '''
    bins_orig = bin_edges(l_orig)
    bins_resam = bin_edges(l_resam)
    
    return hist_resample(bins_orig, bins_resam, flux, density=True) 
    
###############################################################################

def log_age_bins(ages):
    '''
    Age bin edges, logarithmically spaced.
    
    Computes the bin edges as the bissection
    of the bins in log, expanding the borders
    accordingly.
    
    Parameters
    ----------
    ages : array
        Bin centers.
        
    Returns
    -------
    ages_bins : array
        Bin edges, logarithmically spaced,
        with length ``len(ages) + 1``.
    '''
    return 10**bin_edges(np.log10(ages))

###############################################################################

def bin_edges(bin_center):
    '''
    Computes the bin edges as the bissection
    of the bins, expanding the borders accordingly.
    
    Parameters
    ----------
    bin_center : array
        Bin centers.
        
    Returns
    -------
    bin_edges : array
        Bin edges, with length ``len(bin_center) + 1``.
    '''
    bin_edges = np.empty(len(bin_center) + 1)
    bin_edges[1:-1] = (bin_center[:-1] + bin_center[1:]) / 2.0
    bin_edges[0] = bin_center[0] - (bin_center[1] - bin_center[0]) / 2.0
    bin_edges[-1] = bin_center[-1] + (bin_center[-1] - bin_center[-2]) / 2.0
    return bin_edges

###############################################################################

def ageSmoothingKernel(logAgeBase, logTc, logtc_FWHM=0.1):
    '''
    Given an array of logAgeBase = log10(base ages), and an array logTc of "continuous" log ages 
    (presumably uniformly spaced), computes the smoothing kernel s_bc, which resamples and smooths
    any function X(logAgeBase) to Y(logTc). 
    If X = X_b, where b is the logAgeBase index, and Y = Y_c, with c as the index in logTc, then 

    Y_c = sum-over-all-b's of X_b s_bc

    gives the smoothed & resampled version of X. 
    The smoothing is performed in log-time, with gaussian function of FWHM = logtc_FWHM. 
    Conservation of X is ensured (ie, X_b = sum-over-all-c's of Y_c s_bc).

    Notice that logTc and logtc_FWHM are given default values, in case of lazy function callers...
    However, I do NOT know how to call the function using the default logTc but setting a FWHM different from default!!!
    ???Andre????

    Input:  logAgeBase = array of log base ages [in log yr]
            logTc = array of log "continous" ages [in log yr]
            logtc_FWHM = width of smoothing filter [in dex]
    Output: s__bc = (len(logAgeBase) , len(logTc)) matrix [adimensional]

    ElCid@Sanchica - 18/Mar/2012
    '''
    s__bc = np.zeros( (len(logAgeBase) , len(logTc)) )
    logtc_sig =  logtc_FWHM / (np.sqrt(8 * np.log(2)))
    for i_b , a_b in enumerate(logAgeBase):
        aux1 = np.exp(-0.5 * ((logTc - a_b) / logtc_sig)**2)
        s__bc[i_b,:] = aux1 / aux1.sum()
    return s__bc


#############################################################################
## MStars utility
#############################################################################

class MStars(object):
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
    


#############################################################################

def interpAge(prop, logAgeBase, logAgeInterp):
    '''
    Interpolate linearly Mstars or fbase_norm in log time.
    
    Parameters
    ----------
    prop : array
        Array containing ``Mstars`` or ``fbase_norm``.
        
    logAgeBase : array
        The age base, the same legth as the first
        dimension of ``prop``.

    logAgeInterp : array
        The age to which interpolate ``prop``. The
        returned value will have the same length in
        the first dimension.
        
    Returns
    -------
    propInterp : array
        The same ``prop``, interpolated to ``logAgeInterp``.
    '''
    nAgeInterp = len(logAgeInterp)
    nMet = prop.shape[1]
    if prop.ndim == 3:
        nAFe = prop.shape[2]
        new_shape = (nAgeInterp, nMet, nAFe)
    else:
        nAFe = 1
        new_shape = (nAgeInterp, nMet)
    propInterp = np.empty(new_shape, dtype='>f8')
    for z in range(nMet):
        if nAFe == 1:
            propInterp[:,z] = np.interp(logAgeInterp, logAgeBase, prop[:,z])
        else:
            for a in range(nAFe):
                propInterp[:,z, a] = np.interp(logAgeInterp, logAgeBase, prop[:,z, a])
    return propInterp


#############################################################################

def light2MassIni(popx, fbase_norm, Lobs_norm, q_norm, A_V):

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
    
    Mini = Lobs / arrayAsRowMatrix(fbase_norm, Lobs.ndim)
    return Mini


#############################################################################

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
    smoothKernel = ageSmoothingKernel(logtb, logtc, logtc_FWHM)
    popx_sm = np.tensordot(smoothKernel, popx, (0,0))
    
    fbase_norm_interp = interpAge(fbase_norm, logtb, logtc)
    Mini_sm = light2MassIni(popx_sm, fbase_norm_interp, Lobs_norm, q_norm, A_V)
    return Mini_sm

#############################################################################

def calcSFR(popx, fbase_norm, Lobs_norm, q_norm, A_V, logtb, logtc, logtc_FWHM):
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
    tc = 10.0**logtc
    
    # FIXME: how calculate/verify the time step properly?
    logtc_step = logtc[1] - logtc[0]
    
    smoothKernel = ageSmoothingKernel(logtb, logtc, logtc_FWHM)
    popx_sm = np.tensordot(smoothKernel, popx, (0,0))
    
    fbase_norm_interp = interpAge(fbase_norm, logtb, logtc)
    Mini_sm = light2MassIni(popx_sm, fbase_norm_interp, Lobs_norm, q_norm, A_V)
    
    if Mini_sm.ndim == 4:
        # We have a/Fe in the base.
        Mini_sm = Mini_sm.sum(axis=2)
    Mini_sm = Mini_sm.sum(axis=1)
    tmp = np.log10(np.exp(1.0)) / (logtc_step * tc)
    SFR_sm = Mini_sm * arrayAsRowMatrix(tmp, len(Mini_sm.shape))
    
    return SFR_sm


#############################################################################

def calcSFR_alt(popx, fbase_norm, Lobs_norm, q_norm, A_V,
               ageBase, logtc_step=0.05, logtc_FWHM=0.5, dt=0.5e9):
    logtb = np.log10(ageBase)
    logtc = np.arange(logtb.min(), logtb.max() + logtc_step, logtc_step)
    Mini = smooth_Mini(popx, fbase_norm, Lobs_norm,
                       q_norm, A_V, logtb, logtc, logtc_FWHM)
    
    if Mini.ndim == 4:
        # We have a/Fe in the base.
        Mini = Mini.sum(axis=2)
    Mini = Mini.sum(axis=2).sum(axis=1)
    logtc_bins = bin_edges(logtc)
    tc_bins = 10**logtc_bins
    tl = np.arange(tc_bins.min(), tc_bins.max()+dt, dt)
    tl_bins = bin_edges(tl)
    Mini_r = hist_resample(tc_bins, tl_bins, Mini)
    SFR = Mini_r / dt
    # Add bondary points so that np.trapz(SFR, tl) == Mini.sum().
    SFR = np.hstack((0, SFR, 0))
    tl = np.hstack((tl[0] - dt, tl, tl[-1] + dt))
    return SFR, tl

#############################################################################

def calcPopmu(popx, fbase_norm, Lobs_norm, Mstars, q_norm, A_V):
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
    Mini = light2MassIni(popx, fbase_norm, Lobs_norm, q_norm, A_V)
    ndim = len(Mini.shape)
    Mcor = Mini * arrayAsRowMatrix(Mstars, ndim)
    
    popmu_ini = Mini / Mini.sum(axis=1).sum(axis=0) * 100.0
    popmu_cor = Mcor / Mcor.sum(axis=1).sum(axis=0) * 100.0
    return popmu_ini, popmu_cor 


#############################################################################

def arrayAsRowMatrix(a, ndim):
    '''
    Reshape ``a`` to a "row matrix" of bigger rank, ``ndim``.
    
    Parameters
    ----------
    a : array
        Array to reshape.
        
    ndim : integer
        Number of dimensions to reshape.
        
    Returns
    -------
    a_reshaped : array
        The same array as ``a``, reshaped to ``ndim`` dimensions.
    '''
    if a.ndim == ndim:
        return a
    if a.ndim > ndim:
        raise ValueError('number of dimensions of input must be smaller than %d.' % ndim)
    new_shape = np.ones(ndim, dtype=int)
    new_shape[:a.ndim] = a.shape
    return a.reshape(new_shape)


################################################################################

def gaussVelocitySmooth(lo, fo, v0, sig, ls=None, n_u=31, n_sig=6):
    '''
    c # Smooth an input spectrum fo(lo) with a Gaussian centered at velocity
    c # v = v0 and with sig as velocity dispersion. The output is fs(ls).
    c # The convolution is done with a BRUTE FORCE integration of n_u points
    c # from u = -n_sig to u = +n_sig, where u = (v - v0) / sig.
    c # OBS: Assumes original spectrum fo(lo) is EQUALLY SPACED in lambda!!
    C #      Units of v0 & sig = km/s
    c # Cid@Lynx - 05/Dec/2002
    
    c # Added n_u as a parameter!
    c # Cid@Lynx - 16/Mar/2003
    
    *ETC* !AKI! gfortran does not like do u=u_low,u_upp,du! But works! Fix
    *     after other tests are done! Cid@Lagoa - 11/Jan/2011
    
    *     This version of GaussSmooth replaces the u-loop by one in an
    *     integer index i_u. The differences wrt are minute (< 1e-5) and due
    *     to precision...
    *
    *      ElCid@Sanchica - 16/Ago/2011
    
    *ATT* Promoted to new oficial GaussSmooth without much testing!
    *      Previous version removed to skip compilation warnings...
    *    ElCid@Sanchica - 08/Oct/2011
    
    *ATT* Found an array bounds-bug when compiling with gfortran ... -fbounds-check
    *      Ex: Fortran runtime error: Index '3402' of dimension 1 of array 'fo' above upper bound of 3401
    *      Happens only when sig is very large. The old routine is kept below, and documents the bug briefly.
    *    ElCid@Sanchica - 05/Nov/2011
    '''
    c  = 2.997925e5

    if n_u < 5:
        raise ValueError('n_u=%d too small to integrate properly.' % n_u)

    if ls is None:
        ls = lo
        
# Parameters for the brute-force integration in velocity
    u = np.linspace(-n_sig, n_sig, n_u)
    du = u[1] - u[0]
    d_lo = lo[1] - lo[0]

    Ns = len(ls)
    No = len(lo)
    fs = np.empty(Ns, dtype=fo.dtype)

# loop over ls, convolving fo(lo) with a gaussian
    for i_s in range(Ns):

# reset integral of {fo[ll = ls/(1+v/c)] * Gaussian} & start integration
        sum_fg = 0.
        for _u in u:

# define velocity & lambda corresponding to u
            v  = v0 + sig * _u
            ll = ls[i_s] / (1.0 + v/c)

# find fo flux for lambda = ll
            ind = int((ll - lo[0]) / d_lo)
            if ind < 0: 
                ff = fo[0]
            elif ind >= No-1:
                ff = fo[No-1]
            else:
                a  = (fo[ind+1] - fo[ind]) / (lo[ind+1] - lo[ind])
                b  = fo[ind] - a * lo[ind]
                ff = a * ll + b

# smoothed spectrum
            sum_fg = sum_fg + ff * np.exp(-(_u**2/2.)) 
        fs[i_s] = sum_fg * du / np.sqrt(2. * np.pi)
    return fs


################################################################################\


