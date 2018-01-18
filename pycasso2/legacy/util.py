'''
Created on Oct 25, 2012

@author: andre
'''

from ..geometry import convexhull, Polygon
import numpy as np
from numpy import ma



def gen_rebin(a, e, bin_e, mean=True):
    '''
    Rebinning function. Given the value array `a`, with generic
    positions `e` such that `e.shape == a.shape`, return the sum
    or the mean values of `a` inside the bins defined by `bin_e`.
    
    Parameters
    ----------
    a : array like
        The array values to be rebinned.
    
    e : array like
        Generic positions of the values in `a`.
    
    bin_e : array like
        Bins of `e` to be used in the rebinning.
    
    mean : boolean
        Divide the sum by the number of points inside the bin.
        Defaults to `True`.
        
    Returns
    -------
    a_e : array
        An array of length `len(bin_e)-1`, containing the sum or
        mean values of `a` inside each bin.
        
    Examples
    --------
    
    TODO: Add examples for gen_rebin.
    '''
    a_e = np.histogram(e.ravel(), bin_e, weights=a.ravel())[0]
    if mean:
        N = np.histogram(e.ravel(), bin_e)[0]
        mask = N > 0
        a_e[mask] /= N[mask]
    return a_e


def getGenHalfRadius(X, r, r_max=None):
    '''
    Evaluate radius where the cumulative value of `X` reaches half of its value.

    Parameters
    ----------
    X : array like
        The property whose half radius will be evaluated.
    
    r : array like
        Radius associated to each value of `X`. Must be the
        same shape as `X`.

    r_max : int
        Integrate up to `r_max`. Defaults to `np.max(r)`. 
    
    Returns
    -------
    HXR : float
        The "half X radius."

    Examples
    --------
    
    Find the radius containing half of the volume of a gaussian.
    
    >>> import numpy as np
    >>> xx, yy = np.indices((100, 100))
    >>> x0, y0, A, a = 50.0, 50.0, 1.0, 20.0
    >>> z = A * np.exp(-((xx-x0)**2 + (yy-y0)**2)/a**2)
    >>> r = np.sqrt((xx - 50)**2 + (yy-50)**2)
    >>> getGenHalfRadius(z, r)
    16.786338066912215

    '''
    if r_max is None:
        r_max = np.max(r)
    bin_r = np.arange(0, r_max, 1)
    cumsum_X = gen_rebin(X, r, bin_r, mean=False).cumsum()

    from scipy.interpolate import interp1d
    invX_func = interp1d(cumsum_X, bin_r[1:])
    halfRadiusPix = invX_func(cumsum_X.max() / 2.0)
    return float(halfRadiusPix)


def getAverageFilledImage(X, to_fill, r__yx, X__r, bin_r):
    '''
    TODO: getAverageFilledImage documentation.
    '''
    _X = X.copy()
    _X[to_fill] = np.interp(r__yx[to_fill], bin_r, X__r)
    return _X


def getApertureMask(shape, x0, y0, pa, ba, apertures, method='exact', ring=True):
    '''
    TODO: Documentation for apertures.
    '''
    raise NotImplementedError('Elliptical apertures not implemented.')
#     from .photutils import EllipticalAnnulus
# 
#     N_y, N_x = shape
#     apertures = np.asarray(apertures)
#     apertures_in = apertures[:-1]
#     apertures_out = apertures[1:]
#     x_min = -x0
#     x_max = N_x - x0
#     y_min = -y0
#     y_max = N_y - y0
#     N_r = len(apertures_in)
#     mask = np.zeros((N_r, N_y, N_x))
#     area = np.zeros((N_r))
#     for i in range(apertures_in.size):
#         a_in = apertures_in[i] if ring else apertures_in[0]
#         a_out = apertures_out[i]
#         b_out = a_out * ba
#         an = EllipticalAnnulus(a_in, a_out, b_out, pa)
#         mask[i] = an.encloses(x_min, x_max, y_min, y_max, N_x, N_y, method=method)
#         area[i] = an.area()
#     return mask, area


def convexHullMask(mask):
    '''
    Compute the convex hull of a boolean image mask.
    
    Parameters
    ----------
    mask : array
        2-D image where the ``True`` pixels mark the data.
        
    Returns
    -------
    convex_mask : array
        2-D image of same shape and type as ``mask``,
        where the convex hull of ``mask`` is marked as
        ``True`` values.
    '''
    mask_points = np.array(np.where(mask)).T
    is_hull = convexhull(mask_points)
    hull_poly = Polygon(mask_points[is_hull], conv=False)

    ny, nx = mask.shape
    xx, yy = np.meshgrid(np.arange(nx),np.arange(ny))
    image_points = np.vstack((yy.ravel(), xx.ravel())).T
    
    inside_hull_points = [(y,x) for y,x in image_points if hull_poly.collidepoint((y,x))]
    
    convex_mask = np.zeros_like(mask)
    yy_inside, xx_inside = zip(*inside_hull_points)
    convex_mask[yy_inside, xx_inside] = True
    return convex_mask


def getEllipseParams(image, x0, y0, mask=None):
    '''
    Estimate ellipticity and orientation of the galaxy using the
    "Stokes parameters", as described in:
    http://adsabs.harvard.edu/abs/2002AJ....123..485S
    The image used is ``qSignal``.
    
    Parameters
    ----------
    image : array
        Image to use when calculating the ellipse parameters.

    x0 : float
        X coordinate of the origin.
    
    y0 : float
        Y coordinate of the origin.
    
    mask : array, optional
        Mask containing the pixels to take into account.
    
    Returns
    -------
    pa : float
        Position angle in radians, counter-clockwise relative
        to the positive X axis.
    
    ba : float
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`).
    '''
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    yy, xx = np.indices(image.shape)
    y = yy - y0
    x = xx - x0
    x2 = x**2
    y2 = y**2
    xy = x * y
    r2 = x2 + y2
    
    mask &= (r2 > 0.0)
    norm = image[mask].sum()
    Mxx = ((x2[mask] / r2[mask]) * image[mask]).sum() / norm
    Myy = ((y2[mask] / r2[mask]) * image[mask]).sum() / norm
    Mxy = ((xy[mask] / r2[mask]) * image[mask]).sum() / norm
    
    Q = Mxx - Myy
    U = Mxy
    
    pa = np.arctan2(U, Q) / 2.0
    
    # b/a ratio
    ba = (np.sin(2 * pa) - U) / (np.sin(2 * pa) + U)
    # Should be the same as ba
    #ba_ = (np.cos(2*pa) - Q) / (np.cos(2*pa) + Q)
    
    return pa, ba


def getDistance(x, y, x0, y0, pa=0.0, ba=1.0):
    '''
    Return an image (:class:`numpy.ndarray`)
    of the distance from the center ``(x0, y0)`` in pixels,
    assuming a projected disk.
    
    Parameters
    ----------
    x : array
        X coordinates to get the pixel distances.
    
    y : array
        y coordinates to get the pixel distances.
    
    x0 : float
        X coordinate of the origin.
    
    y0 : float
        Y coordinate of the origin.
    
    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis.
    
    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`).

    Returns
    -------
    pixelDistance : array
        Pixel distances.

    See also
    --------
    getImageDistance

    '''
    y = np.asarray(y) - y0
    x = np.asarray(x) - x0
    x2 = x**2
    y2 = y**2
    xy = x * y

    a_b = 1.0/ba
    cos_th = np.cos(pa)
    sin_th = np.sin(pa)

    A1 = cos_th ** 2 + a_b ** 2 * sin_th ** 2
    A2 = -2.0 * cos_th * sin_th * (a_b ** 2 - 1.0)
    A3 = sin_th ** 2 + a_b ** 2 * cos_th ** 2

    return np.sqrt(A1 * x2 + A2 * xy + A3 * y2)


def getImageDistance(shape, x0, y0, pa=0.0, ba=1.0):
    '''
    Return an image (:class:`numpy.ndarray`)
    of the distance from the center ``(x0, y0)`` in pixels,
    assuming a projected disk.
    
    Parameters
    ----------
    shape : (float, float)
        Shape of the image to get the pixel distances.
    
    x0 : float
        X coordinate of the origin.
    
    y0 : float
        Y coordinate of the origin.
    
    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis. Defaults to ``0.0``.
    
    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`). Defaults to ``1.0``.

    Returns
    -------
    pixelDistance : 2-D array
        Image containing the distances.

    See also
    --------
    getDistance, getEllipseParams

    '''
    y, x = np.indices(shape)
    return getDistance(x, y, x0, y0, pa, ba)


def getAngle(x, y, x0, y0, pa=0.0, ba=1.0):
    '''
    Return an image (:class:`numpy.ndarray` of same shape as :attr`qSignal`)
    of the angle in radians of each pixel, relative from the axis of the
    position angle ``pa``. The projection is fixed assuming the galaxy is
    a disk, throught the ellipticity parameter ``ba``.
    
    The angle is obtained "de-rotating" the pixel positions, stretching the
    y-coordinates to account for the perspective, and then calculating
    the arc tangent of the resulting y/x.
    
    Parameters
    ----------
    x : array
        X coordinates to calculate the angle.
    
    y : array
        Y coordinates to calculate the angle.
    
    x0 : float
        X coordinate of the origin.
    
    y0 : float
        Y coordinate of the origin.
    
    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis. Defaults to ``0``.
    
    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`). This controls
        the corretion of the projection of the galaxy. Set to ``1.0``
        (default) to disable the correction.

    Returns
    -------
    pixelAngle : 2-D array
        Image containing the angles in radians.
        
    See also
    --------
    getPixelDistance
    '''
    x = np.asarray(x) - x0
    y = np.asarray(y) - y0
    
    cos_th = np.cos(pa)
    sin_th = np.sin(pa)
    x_prime = x * cos_th + y * sin_th
    y_prime = - x * sin_th + y * cos_th
    return np.arctan2(y_prime / ba, x_prime)

        
def getImageAngle(shape, x0, y0, pa=0.0, ba=1.0):
    '''
    Return an image (:class:`numpy.ndarray` of same shape as :attr`qSignal`)
    of the angle in radians of each pixel, relative from the axis of the
    position angle ``pa``. The projection is fixed assuming the galaxy is
    a disk, throught the ellipticity parameter ``ba``.
    
    The angle is obtained "de-rotating" the pixel positions, stretching the
    y-coordinates to account for the perspective, and then calculating
    the arc tangent of the resulting y/x.
    
    Parameters
    ----------
    shape : (float, float)
        Shape of the image to get the angles.
    
    x0 : float
        X coordinate of the origin.
    
    y0 : float
        Y coordinate of the origin.
    
    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis. Defaults to ``0``.
    
    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`). This controls
        the corretion of the projection of the galaxy. Set to ``1.0``
        (default) to disable the correction.

    Returns
    -------
    pixelAngle : 2-D array
        Image containing the angles in radians.
        
    See also
    --------
    getPixelDistance
    '''
    y, x = np.indices(shape)
    return getAngle(x, y, x0, y0, pa, ba)


def radialProfile(prop, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0,
                  mask=None, mode='mean', return_npts=False):
    '''
    Calculate the radial profile of an N-D image.
    
    Parameters
    ----------
    prop : array
        Image of property to calculate the radial profile.
        
    bin_r : array
        Semimajor axis bin boundaries in units of ``rad_scale``.
        
    x0 : float
        X coordinate of the origin.
    
    y0 : float
        Y coordinate of the origin.
    
    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis.
    
    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`).

    rad_scale : float, optional
        Scale of the bins, in pixels. Defaults to 1.0.
        
    mask : array, optional
        Mask containing the pixels to use in the radial profile.
        Must be bidimensional and have the same shape as the last
        two dimensions of ``prop``. Default: no mask.
        
    mode : string, optional
        One of:
            * ``'mean'``: Compute the mean inside the radial bins (default).
            * ``'median'``: Compute the median inside the radial bins.
            * ``'sum'``: Compute the sum inside the radial bins.
            * ``'var'``: Compute the variance inside the radial bins.
            * ``'std'``: Compute the standard deviation inside the radial bins.
            
    return_npts : bool, optional
        If set to ``True``, also return the number of points inside
        each bin. Defaults to ``False``.
        
        
    Returns
    -------
    radProf : [masked] array
        Array containing the radial profile as the last dimension.
        Note that ``radProf.shape[-1] == (len(bin_r) - 1)``
        If ``prop`` is a masked aray, this and ``npts`` will be
        a masked array as well.
        
    npts : [masked] array, optional
        The number of points inside each bin, only if ``return_npts``
        is set to ``True``.
        
           
    See also
    --------
    radialProfileExact, getPixelDistance, getEllipseParams
    '''
    def red(func, x, fill_value):
        if x.size == 0: return fill_value, fill_value
        if x.ndim == 1: return func(x), len(x)
        return func(x, axis=-1), x.shape[-1]

    imshape = prop.shape[-2:]
    nbins = len(bin_r) - 1
    new_shape = prop.shape[:-2] + (nbins,)
    r__yx = getImageDistance(imshape, x0, y0, pa, ba) / rad_scale
    if mask is None:
        mask = np.ones(imshape, dtype=bool)
    if mode == 'mean':
        reduce_func = np.mean
    elif mode == 'median':
        reduce_func = np.median
    elif mode == 'sum':
        reduce_func = np.sum
    elif mode == 'var':
        reduce_func = np.var
    elif mode == 'std':
        reduce_func = np.std
    else:
        raise ValueError('Invalid mode: %s' % mode)
    
    if isinstance(prop, ma.MaskedArray):
        n_bad = prop.mask.astype('int')
        max_bad = 1.0
        while n_bad.ndim > 2:
            max_bad *= n_bad.shape[0]
            n_bad = n_bad.sum(axis=0)
        mask = mask & (n_bad / max_bad < 0.5)            
        prop_profile = ma.masked_all(new_shape)
        npts = ma.masked_all((nbins,))
        prop_profile.fill_value = prop.fill_value
        reduce_fill_value = ma.masked
    else:
        prop_profile = np.empty(new_shape)
        npts = np.empty((nbins,))
        reduce_fill_value = np.nan
    if mask.any():
        dist_flat = r__yx[mask]
        dist_idx = np.digitize(dist_flat, bin_r)
        prop_flat = prop[...,mask]
        for i in range(0, nbins):
            prop_profile[..., i], npts[i] = red(reduce_func, prop_flat[..., dist_idx == i+1], reduce_fill_value)

    if return_npts:
        return prop_profile, npts
    return prop_profile


def radialProfileExact(prop, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0,
                       mask=None, return_npts=False):
    '''
    Calculate the radial profile of a property, using exact elliptic apertures.
    This is suited for a one-shot radial profile. See :meth:`getYXToRadialBinsTensorExact`
    for a more efficient approach.
    
    Parameters
    ----------
    prop : array
        Image of property to calculate the radial profile.
        
    bin_r : array
        Semimajor axis bin boundaries in units of ``rad_scale``.

    x0 : float
        X coordinate of the origin.
    
    y0 : float
        Y coordinate of the origin.
    
    pa : float, optional
        Position angle in radians, counter-clockwise relative
        to the positive X axis.
    
    ba : float, optional
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`).

    rad_scale : float, optional
        Scale of the bins, in pixels. Defaults to ``1.0``.
        
    mask : array, optional
        Mask containing the pixels to use in the radial profile.
        Must be bidimensional and have the same shape as the last
        two dimensions of ``prop``. Default: no mask.
        
    return_npts : bool, optional
        If set to ``True``, also return the number of points inside
        each bin. Defaults to ``False``.
        
        
    Returns
    -------
    radProf : masked array
        Array containing the radial profile as the last dimension.
        Note that ``radProf.shape[-1] == (len(bin_r) - 1)``
        
    npts : masked array, optional
        The number of points inside each bin, only if ``return_npts``
        is set to ``True``.
        
    Examples
    --------
    TODO: examples of radialProfileExact
    
    See also
    --------
    radialProfile, getPixelDistance, getEllipseParams
    '''
    shape = prop.shape[-2:]
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    else:
        mask = mask.copy()
        
    if isinstance(prop, ma.MaskedArray):
        prop = prop.filled(0.0)
    
    ryx, _ = getApertureMask(shape, x0, y0, pa, ba, np.asarray(bin_r) * rad_scale)
    ryx[:, ~mask] = 0.0
    area_pix = ma.masked_less(ryx.sum(axis=2).sum(axis=1), 0.5)
    area_pix.fill_value = 0.0
    ryx /= area_pix[:, np.newaxis, np.newaxis]
    ryx[area_pix.mask] = 0.0
    prof = np.tensordot(prop, ryx, [(-2, -1), (-2, -1)])
    prop_profile = np.ma.array(prof)
    prop_profile[...,area_pix.mask] = np.ma.masked
    prop_profile.fill_value = 0.0
    if return_npts:
        return prop_profile, area_pix
    else:
        return prop_profile


def fillImage(prop, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0):
    '''
    Fill a 2-D the masked pixels of an image of a property
    using the values of a radial profile. The pixels to fill
    are chosen by ``mode``.
    
    Parameters
    ----------
    prop : array
        Property (2-D) to fill the hollow pixels.
        
    prop__r : the radial profile to use to fill the data.
        If not set, calculate from prop.

    r : array
        The radial distances for ``prop__r`` values.
        If not set (or ``prop__r`` is not set), use
        a 1 pixel step.
        
    r__yx : array
        A 2-D image containing the geometry of the image.
        If not set, use :attr:`pixelDistance__yx`.
        
    mode : {'convex', 'hollow'}, string
        If mode is ``'convex'``, fill entire convex hull.
        If mode is ``'hollow'``, fill only hollow pixels.
        Default is ``'convex'``.
            
    Returns
    -------
    prop_fill : array
        A 2-D image of ``prop``, with the missing pixels filled.
        
    mask : array
        The effective mask for the filled image.
        
    '''
    r__yx = getImageDistance(prop.shape, x0, y0, pa, ba)
    r__yx = np.ma.array(r__yx, mask=prop.mask)
    bin_r = np.arange(0.0, r__yx.max(), 1.0)
    r = bin_r[:-1] + 0.5
    prop__r = radialProfile(prop, bin_r, x0, y0, pa, ba, mode='mean')
    to_fill = convexHullMask(~prop.mask) & prop.mask
    
    _prop = prop.copy()
    _prop[to_fill] = np.interp(r__yx[to_fill], r[~prop__r.mask], prop__r.compressed())
    return _prop


def get_polygon_mask(xy_poly, shape):
    '''
    Calculates a boolean mask of points inside a polygon.
    
    Parameters
    ----------
    xy_poly : list of tuples
        List containing the (x,y) positions of the polygon.
        
    shape : tuple
        Shape (y, x) of the mask to be built.
    '''
    p = Polygon(xy_poly, conv=False)
    mask = np.zeros(shape, dtype='bool')
    for y in range(shape[0]):
        for x in range(shape[1]):
            mask[y, x] = p.collidepoint((x, y))
    return mask


def getMstarsForTime(Mstars, ageBase, metBase, Tc=None):
    '''
    Given an array of time values (in years), return the Mstars (:math:`M^{\star}`)
    tensor which transforms the initial mass for each population (``popmu_ini``)
    into mass values for evolutionary time, correcting for the mass loss.
    
    Parameters
    ----------
    Mstars : 2-d array
        Fraction of the initial stellar mass for a given
        population that is still trapped inside stars.
    
    ageBase : array
        The age base, the same legth as the first
        dimension of ``Mstars``.
    
    
    metBase : array
        The metallicity base, the same legth as the second
        dimension of ``Mstars``.
    
    
    Tc : float or array-like of floats
        An array of ages (in years) in crescent order.

    Returns
    -------
    mstars : 3-D array
        The transform tensor, shape (len(Tc), len(ageBase), len(metBase))
        
    '''
    from ..starlight.analysis import MStarsEvolution as InterpMstars
    interpMstars = InterpMstars(ageBase, metBase, Mstars)
    return interpMstars.forTime(ageBase, Tc)

