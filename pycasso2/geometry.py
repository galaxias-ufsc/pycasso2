'''
Created on 28 de set de 2016

@author: andre
'''

from .resampling import gen_rebin
from .external.pylygon import convexhull, Polygon

from astropy import log
import numpy as np

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

__all__ = ['get_ellipse_params', 'get_image_distance',
           'get_image_angle', 'radial_profile']


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
    yy, xx = np.indices(image.shape)
    y = yy - y0
    x = xx - x0
    x2 = x**2
    y2 = y**2
    xy = x * y
    r2 = x2 + y2

    image = np.ma.array(image)
    sel = ~np.ma.getmaskarray(image)
    sel[r2 == 0.0] = False

    x2 = x2[sel]
    y2 = y2[sel]
    xy = xy[sel]
    r2 = r2[sel]
    image = image[sel]
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

    a_b = 1.0 / ba
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


def weighted_mean_radprof(prop, w, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0, exact=False, return_npts=False):
    '''
    Calculate the mean radial profile of ``prop``, weighted by ``w``.

    Parameters
    ----------
    prop : array
        Image of property to calculate the radial profile.

    w : array
        Image of weights.

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

    exact : bool, optional
        If set to ``True``, use an apertures algorithm to calculate the profile.
        Otherwise, use discrete binning.
        Defaults to ``True``.

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
    w_prop = w * prop
    w = np.ma.masked_where(w_prop.mask, w)
    sum_wm, npts = radial_profile(w_prop, bin_r, x0, y0, pa, ba, rad_scale=rad_scale,
                                  mode='sum', exact=exact, return_npts=True)
    sum_w = radial_profile(w, bin_r, x0, y0, pa, ba, rad_scale=rad_scale, mode='sum', exact=exact)
    radprof = sum_wm / sum_w
    if return_npts:
        return radprof, npts
    else:
        return radprof


def radial_profile(prop, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0,
                   mode='mean', exact=True, return_npts=False):
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
                            Not available for ``exact=True``.
            * ``'sum'``: Compute the sum inside the radial bins.
            * ``'var'``: Compute the variance inside the radial bins.
            * ``'std'``: Compute the standard deviation inside the radial bins.

    exact : bool, optional
        If set to ``True``, use an apertures algorithm to calculate the profile.
        Otherwise, use discrete binning.
        Defaults to ``True``.

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
    if exact:
        radprof = radial_profile_exact
    else:
        radprof = radial_profile_discrete
    prop_profile, npts = radprof(prop, bin_r, x0, y0, pa, ba, rad_scale, mode)
    if return_npts:
        return prop_profile, npts
    else:
        return prop_profile


def radial_profile_discrete(prop, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0, mode='mean'):
    '''
    Calculate the radial profile of an N-D image, where each pixel goes to a single radial bin.

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

    Returns
    -------
    radProf : masked array
        Array containing the radial profile as the last dimension.
        Note that ``radProf.shape[-1] == (len(bin_r) - 1)``

    npts : masked array
        The number of points inside each bin.

    '''
    def red(func, x, fill_value):
        if x.size == 0:
            return fill_value, fill_value
        if x.ndim == 1:
            return func(x), len(x)
        return func(x, axis=-1), x.shape[-1]

    redfunc_mode = {'mean': np.mean,
                    'median': np.median,
                    'sum': np.sum,
                    'var': np.var,
                    'std': np.std,
                    }
    if not mode in list(redfunc_mode.keys()):
        raise ValueError('Invalid mode: %s' % mode)
    reduce_func = redfunc_mode[mode]

    if not isinstance(prop, np.ma.MaskedArray):
        prop = np.ma.array(prop)
    imshape = prop.shape[-2:]
    nbins = len(bin_r) - 1
    new_shape = prop.shape[:-2] + (nbins,)
    r__yx = get_image_distance(imshape, x0, y0, pa, ba) / rad_scale

    n_bad = np.ma.getmaskarray(prop).astype('int')
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
            prop_profile[..., i], npts[i] = red(
                reduce_func, prop_flat[..., dist_idx == i + 1], reduce_fill_value)

    return prop_profile, npts


def radial_profile_exact(prop, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0, mode='mean'):
    '''
    Calculate the radial profile of a N-d image, using exact elliptic apertures.
    
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
        
    mode : string, optional
        One of:
            * ``'mean'``: Compute the mean inside the radial bins (default).
            * ``'sum'``: Compute the sum inside the radial bins.
                
        
    Returns
    -------
    radProf : masked array
        Array containing the radial profile as the last dimension.
        Note that ``radProf.shape[-1] == (len(bin_r) - 1)``
        
    npts : masked array
        The number of points inside each bin.
        
    '''
    if mode == 'median':
        raise Exception('Median mode not supported for exact profile.')
    shape = prop.shape[-2:]
    if isinstance(prop, np.ma.MaskedArray):
        masked = np.ma.getmaskarray(prop)
        prop = prop.filled(0.0)
    else:
        masked = np.zeros_like(prop, dtype='bool')

    ryx, _ = get_aperture_mask(shape, x0, y0, pa, ba, tuple(bin_r), rad_scale)
    ryx[:, masked] = 0.0
    area_pix = np.ma.masked_less(ryx.sum(axis=2).sum(axis=1), 0.05)
    prop_sum = np.tensordot(prop, ryx, [(-2, -1), (-2, -1)])
    prop_sum = np.ma.array(prop_sum, mask=area_pix.mask)
    if mode =='sum':
        return prop_sum, area_pix
    prop_mean = prop_sum / area_pix
    if mode == 'mean':
        return prop_mean, area_pix
    prop_sum2 = np.tensordot(prop**2, ryx, [(-2, -1), (-2, -1)])
    prop_sum2 = np.ma.array(prop_sum2, mask=area_pix.mask)
    prop_var = prop_sum2 / area_pix - prop_mean**2
    if mode == 'var':
        return prop_var, area_pix
    if mode =='std':
        return np.sqrt(prop_var), area_pix
    raise Exception('Unknown mode %s.' % mode)


def get_half_radius(X, r, r_max=None):
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
    >>> get_half_radius(z, r)
    16.786338066912215

    '''
    if r_max is None:
        r_max = np.max(r)
    bin_r = np.arange(0, r_max, 1)
    cumsum_X = gen_rebin(X, r, bin_r, mean=False).cumsum()

    from scipy.interpolate import interp1d
    invX_func = interp1d(cumsum_X, bin_r[1:])
    half_radius_pix = invX_func(cumsum_X.max() / 2.0)
    return float(half_radius_pix)


def convex_hull_mask(mask):
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
    if len(is_hull) == 0:
        raise Exception('No data points found.')
    hull_poly = Polygon(mask_points[is_hull], conv=False)

    ny, nx = mask.shape
    xx, yy = np.meshgrid(np.arange(nx),np.arange(ny))
    image_points = np.vstack((yy.ravel(), xx.ravel())).T
    
    inside_hull_points = [(y,x) for y,x in image_points if hull_poly.collidepoint((y,x))]
    
    convex_mask = np.zeros_like(mask)
    yy_inside, xx_inside = zip(*inside_hull_points)
    convex_mask[yy_inside, xx_inside] = True
    return convex_mask


def find_image_max(image):
    y0, x0 = np.unravel_index(np.argmax(image), image.shape)
    return x0, y0


def find_peak(image, x0=None, y0=None, delta=10, max_iter=20):
    from scipy.ndimage.measurements import center_of_mass
    rtol = 0.01
    if x0 is None or y0 is None:
        x0, y0 = find_image_max(image)
    delta = int(delta)
    for i in range(max_iter):
        _x0 = int(np.rint(x0))
        _y0 = int(np.rint(y0))
        xx = slice(_x0 - delta, _x0 + delta)
        yy = slice(_y0 - delta, _y0 + delta)
        y0_c, x0_c = center_of_mass(image[yy, xx])
        x0_c += xx.start
        y0_c += yy.start
        if np.isclose(x0, x0_c, rtol=rtol) and  np.isclose(y0, y0_c, rtol=rtol):
            break
        x0 = x0_c
        y0 = y0_c
    if i >= (max_iter - 1):
        log.warn('find_peak reached maximum interations: %d' % max_iter)
    return x0_c, y0_c


@lru_cache(maxsize=10)
def get_aperture_mask(shape, x0, y0, pa, ba, apertures, rad_scale=1.0, method='exact', ring=True):
    '''
    TODO: Documentation for apertures.
    '''
    from .external.photutils import EllipticalAnnulus

    N_y, N_x = shape
    apertures = np.asarray(apertures) * rad_scale
    apertures_in = apertures[:-1]
    apertures_out = apertures[1:]
    x_min = -x0 - 0.5
    x_max = N_x - x0 - 0.5
    y_min = -y0 - 0.5
    y_max = N_y - y0 - 0.5
    N_r = len(apertures_in)
    mask = np.zeros((N_r, N_y, N_x))
    area = np.zeros((N_r))
    for i in range(apertures_in.size):
        a_in = apertures_in[i] if ring else apertures_in[0]
        a_out = apertures_out[i]
        b_out = a_out * ba
        an = EllipticalAnnulus(a_in, a_out, b_out, pa)
        mask[i] = an.encloses(x_min, x_max, y_min, y_max, N_x, N_y, method=method)
        area[i] = an.area()
    return mask, area


def fill_image(image, x0, y0, pa=0.0, ba=1.0):
    '''
    Fill a 2-D the masked pixels of an image of a property
    using the values of a radial profile. The pixels to fill
    are chosen by ``mode``.
    
    Parameters
    ----------
    image : array
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
    r__yx = get_image_distance(image.shape, x0, y0, pa, ba)
    r__yx = np.ma.array(r__yx, mask=image.mask)
    bin_r = np.arange(0.0, r__yx.max(), 1.0)
    r = bin_r[:-1] + 0.5
    image_r = radial_profile(image, bin_r, x0, y0, pa, ba, mode='mean')
    to_fill = convex_hull_mask(~image.mask) & image.mask
    
    _image = image.copy()
    _image[to_fill] = np.interp(r__yx[to_fill], r[~image_r.mask], image_r.compressed())
    return _image


