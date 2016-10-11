'''
Created on 28 de set de 2016

@author: andre
'''

import numpy as np

__all__ = ['get_ellipse_params', 'get_image_distance', 'get_image_angle', 'radial_profile']


def get_ellipse_params(image, x0, y0):
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
    
    Returns
    -------
    pa : float
        Position angle in radians, counter-clockwise relative
        to the positive X axis.
    
    ba : float
        Ellipticity, defined as the ratio between the semiminor
        axis and the semimajor axis (:math:`b/a`).
    '''
    if not isinstance(image, np.ma.MaskedArray):
        image = np.ma.array(image)
    
    yy, xx = np.indices(image.shape)
    y = yy - y0
    x = xx - x0
    x2 = x**2
    y2 = y**2
    xy = x * y
    r2 = x2 + y2
    
    mask = image.mask.copy()
    mask[r2 == 0.0] = True
    
    x2 = np.ma.array(x2, mask=mask).compressed()
    y2 = np.ma.array(y2, mask=mask).compressed()
    xy = np.ma.array(xy, mask=mask).compressed()
    r2 = np.ma.array(r2, mask=mask).compressed()
    image = np.ma.array(image, mask=mask).compressed()
    norm = image.sum()
    
    Mxx = ((x2 / r2) * image).sum() / norm
    Myy = ((y2 / r2) * image).sum() / norm
    Mxy = ((xy / r2) * image).sum() / norm
    
    Q = Mxx - Myy
    U = Mxy
    
    pa = np.arctan2(U, Q) / 2.0
    
    # b/a ratio
    ba = (np.sin(2 * pa) - U) / (np.sin(2 * pa) + U)
    # Should be the same as ba
    #ba_ = (np.cos(2*pa) - Q) / (np.cos(2*pa) + Q)
    
    return pa, ba


def get_distance(x, y, x0, y0, pa=0.0, ba=1.0):
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
    distance : array
        Distances in pixel units.

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


def get_image_distance(shape, x0, y0, pa=0.0, ba=1.0):
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
    distance : 2-D array
        Image containing the distances in pixel units.

    '''
    y, x = np.indices(shape)
    return get_distance(x, y, x0, y0, pa, ba)


def get_angle(x, y, x0, y0, pa=0.0, ba=1.0):
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
    angle : array
        Angles in radians.
    '''
    x = np.asarray(x) - x0
    y = np.asarray(y) - y0
    
    cos_th = np.cos(pa)
    sin_th = np.sin(pa)
    x_prime = x * cos_th + y * sin_th
    y_prime = - x * sin_th + y * cos_th
    return np.arctan2(y_prime / ba, x_prime)

        
def get_image_angle(shape, x0, y0, pa=0.0, ba=1.0):
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
    angle : 2-D array
        Image containing the angles in radians.
        
    '''
    y, x = np.indices(shape)
    return get_angle(x, y, x0, y0, pa, ba)


def radial_profile(prop, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0,
                   mode='mean', return_npts=False):
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
    radProf : masked array
        Array containing the radial profile as the last dimension.
        Note that ``radProf.shape[-1] == (len(bin_r) - 1)``
        
    npts : masked array, optional
        The number of points inside each bin, only if ``return_npts``
        is set to ``True``.
        
    '''
    def red(func, x, fill_value):
        if x.size == 0: return fill_value, fill_value
        if x.ndim == 1: return func(x), len(x)
        return func(x, axis=-1), x.shape[-1]

    redfunc_mode = {'mean': np.mean,
                    'median': np.median,
                    'sum': np.sum,
                    'var': np.var,
                    'std': np.std,
                    }
    if not mode in redfunc_mode.keys():
        raise ValueError('Invalid mode: %s' % mode)
    reduce_func = redfunc_mode[mode]
    
    if not isinstance(prop, np.ma.MaskedArray):
        prop = np.ma.array(prop)
    imshape = prop.shape[-2:]
    nbins = len(bin_r) - 1
    new_shape = prop.shape[:-2] + (nbins,)
    r__yx = get_image_distance(imshape, x0, y0, pa, ba) / rad_scale
    
    n_bad = prop.mask.astype('int')
    max_bad = 1.0
    while n_bad.ndim > 2:
        max_bad *= n_bad.shape[0]
        n_bad = n_bad.sum(axis=0)
    good = n_bad / max_bad < 0.5            
    prop_profile = np.ma.masked_all(new_shape)
    npts = np.ma.masked_all((nbins,))
    prop_profile.fill_value = prop.fill_value
    reduce_fill_value = np.ma.masked
    if good.any():
        dist_flat = r__yx[good]
        dist_idx = np.digitize(dist_flat, bin_r)
        prop_flat = prop[..., good]
        for i in range(0, nbins):
            prop_profile[..., i], npts[i] = red(reduce_func, prop_flat[..., dist_idx == i+1], reduce_fill_value)

    if return_npts:
        return prop_profile, npts
    return prop_profile

