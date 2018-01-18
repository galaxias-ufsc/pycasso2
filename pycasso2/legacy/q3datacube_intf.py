'''
Created on Mar 16, 2012

@author: Andre Luiz de Amorim
'''

import numpy as np
from numpy import ma
from .StarlightUtils import calcSFR, smooth_Mini, bin_edges, hist_resample
from .util import getGenHalfRadius, getDistance, getImageDistance, getImageAngle, getAngle, getEllipseParams,\
    radialProfileExact, radialProfile

class IQ3DataCube(object):
    '''
    Abstract class for Q3 datacubes manipulation.
    
    This class defines the high level operations on the data, such as
    conversion from zones to spatial coordinates and radial profiles.
    
    Do not use this class directly, use one of the implementations
    instead.
    
    See also
    --------
    h5Q3DataCube, fitsQ3DataCube
    '''
    
    def __init__(self):
        '''
        IQ3DataCube constructor, not much work is done here.
        
        See also
        --------
        fitsQ3DataCube, h5Q3DataCube
        
        '''
        pass

        
    def _preprocess(self, smooth=True, set_geometry=True): 
        self.zoneArea_pix = self._getZoneArea_pix()
        self.zoneArea_pc2 = self.zoneArea_pix * self.parsecPerPixel**2
        self.setSmoothDezonification(smooth)
        if set_geometry:
            self.setGeometry(pa=0.0, ba=1.0)
        
        
        
    def loadGalaxy(self):
        '''
        Abstract method used to load galaxy data if allowed by
        the underlying infrastructure.
        '''
        raise NotImplementedError('This method is not allowed by the backend.')
    
    
    def _getZoneArea_pix(self):
        '''
        Calculate the area of the voronoi zones by summing its pixels.
        
        Returns
        -------
        zoneArea : array
            An array of length ``N_zone``.
        '''
        bins = np.arange(self.N_zone+1)
        qZones_flat = self.qZones.ravel()
        zoneArea,_ = np.histogram(qZones_flat, bins=bins)
        return zoneArea
    

    def setSmoothDezonification(self, smooth=True, prop=None):
        '''
        Enable or disable smooth dezonification. If smooth is True, 
        use :attr:`qSignal` image to weight the pixels in each zone. Otherwise
        use the zone area.
        
        Parameters
        ----------
        smooth : boolean, optional
            Enable or disable smooth dezonification. Defaults to ``True``.
        
        prop : array, optional
            Image to use as dezonification weights if ``smooth`` is ``True``.
            If set to ``None``, use :attr:`qSignal`.
        '''
        self._dezonificationWeight = self.getDezonificationWeight(smooth, prop)
        
        
    def getDezonificationWeight(self, smooth, prop=None):
        '''
        Create the weight image for dezonification. If smooth is True, 
        use ``prop`` image to weight the pixels in each zone. Otherwise
        use the zone area. If ``prop`` is not set, use :attr:`qSignal`.
        
        Here we use a scheme similar to :meth:`zoneToYX`, when using smooth
        dezonification, except that we use :func:`numpy.histogram` to calculate
        the weight of the pixels.
        
        Parameters
        ----------
        smooth : boolean
            Enable or disable smooth dezonification.

        prop : array, optional
            Image to use as dezonification weights if ``smooth`` is ``True``.
            If set to ``None``, use :attr:`qSignal`.
        '''
        bins = np.arange(self.N_zone+1)
        qZones_flat = self.qZones.ravel()
        imgShape = self.qZones.shape
        if smooth:
            if prop is None:
                prop = self.qSignal
            elif isinstance(prop, ma.MaskedArray):
                prop = prop.filled(0.0)
            prop_flat = prop.ravel()
            zoneSum,_ = np.histogram(qZones_flat, weights=prop_flat, bins=bins)
            weight = prop_flat / zoneSum[qZones_flat]
        else:
            zoneArea = self.zoneArea_pix
            weight = (1.0/zoneArea)[qZones_flat]
        
        return weight.reshape(imgShape)


    def zoneToYX(self, prop, extensive=True, surface_density=True, fill_value=None):
        '''
        Convert a zone array to an image.

        This scheme takes advantage of the qZones image, which has, for
        every pixel (x, y) the index of the corresponding zone. Using
        this array as a "smart index" for ``prop``, we get to reconstruct
        the image.
        
        Parameters
        ----------
        prop : array
            Property to be converted to an image.
            The zone dimension must be the rightmost dimension.
        
        extensive : boolean, optional
            If ``True``, ``prop`` is extensive, use dezonification weights.
            Defaults to ``True``.
        
        surface_density : boolean, optional
            If ``True``, and ``extensive`` is ``True``, divide the
            return value by the area in parsec^2 of the pixel.
            Defaults to ``True``.
        
        fill_value : float, optional
            Fill value for masked pixels. Defaults to ``numpy.nan``.
            
        Returns
        -------
        prop__yx : masked array
            The ``prop`` array converted to image. All dimensions are
            kept the same, except for the rightmost one, which is
            replaced by y and x.
            
        Examples
        --------
        Load the dataset:
        
        >>> from pycasso.h5datacube import h5Q3DataCube
        >>> K = h5Q3DataCube('qalifa-synthesis.002.h5', 'run001', 'K0277')
            
        The :attr:`A_V` attribute contains the extincion for each zone.
        
        >>> K.A_V.shape
        (1638,)
        
        Convert :attr:`A_V` to spatial coordinates. Note that the extinction
        is not extensive.
        
        >>> A_V__yx = K.zoneToYX(K.A_V, extensive=False)
        >>> A_V__yx.shape
        (73, 77)

        Plot the image.
        
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(A_V__yx)
        
        See also
        --------
        fillImage, getDezonificationWeight, setSmoothDezonification
        '''
        if prop.shape[-1] != self.N_zone:
            raise ValueError('zoneToYX(): last dimension of prop must be N_zone (%d).' % self.N_zone)
            
        if fill_value is None:
            fill_value = self.fill_value
        new_shape = prop.shape[:-1] + (self.N_y, self.N_x)
        prop__yx = ma.masked_all(new_shape, dtype=prop.dtype)
        if extensive:
            prop__yx[..., self.qMask] = prop[..., self.qZones[self.qMask]] * self._dezonificationWeight[self.qMask]
            if surface_density:
                prop__yx[..., self.qMask] /= self.parsecPerPixel ** 2
        else:
            prop__yx[..., self.qMask] = prop[..., self.qZones[self.qMask]]
        prop__yx.fill_value = fill_value
        return prop__yx

    
    
    def setGeometry(self, pa, ba, HLR_pix=None, center=None):
        '''
        Change the geometry of the rings used when calculating radial profiles.
        
        Parameters
        ----------
        pa : float
            Position angle in radians, counter-clockwise relative
            to the positive X axis.
        
        ba : float
            Ellipticity, defined as the ratio between the semiminor
            axis and the semimajor axis (:math:`b/a`).

        HLR_pix : float, optional
            Effective radius

        center : (float, float), optional
            A tuple containing the x and y coodinates of the center
            of the galaxy, in pixels.

        Examples
        --------
        Load the dataset:
        
        >>> from pycasso.h5datacube import h5Q3DataCube
        >>> K = h5Q3DataCube('qalifa-synthesis.002.h5', 'run001', 'K0277')
            
        Find the ellipse parameters.
        
        >>> pa, ba = K.getEllipseParams()

        Set the geometry, using a predefined value for HLR_pix.
        
        >>> K.setGeometry(pa, ba, HLR_pix=10.5)
        
        Get the distance for each pixel from the galaxy center.
        
        >>> dist__yx = K.getPixelDistance()

        Plot the distance image. Note that its shape should resemble
        the ellipticity of the galaxy (if it is well-behaved).
        
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(dist__yx)
        
        See also
        --------
        getEllipseParams, getPixelDistance
        '''
        self.pa = pa
        self.ba = ba
        if center is not None:
            self.x0 = center[0]
            self.y0 = center[1]
        self.pixelDistance__yx = self.getPixelDistance(use_HLR_units=False, pixel_scale=1.0)
        self.pixelAngle__yx = self.getPixelAngle()
        if HLR_pix is None:
            self.HLR_pix = self.getHLR_pix()
        else:
            self.HLR_pix = HLR_pix
        self.HLR_pc = self.HLR_pix * self.parsecPerPixel
        # TODO: remove old stuff like HLR_pix and HLR_pc
        self.pixelsPerHLR = self.HLR_pix
        self.parsecsPerHLR = self.HLR_pc
            

    def getYXToRadialBinsTensorExact(self, bin_r, rad_scale=None, mask=None):
        '''
        Generate an operator for calculating the radial profile
        using exact elliptic apertures. See the examples below
        for the usage.
        
        Parameters
        ----------
        bin_r : array
            Semimajor axis bin boundaries in units of ``rad_scale``.
            
        rad_scale : float, optional
            Scale of the bins, in pixels. Defaults to :attr:`HLR_pix`.
            
        mask : array, optional
            Mask containing the pixels to use in the radial profile.
            Must have the shape ``(N_y, N_x)``. Defaults to :attr:`qMask`.
            
        Returns
        -------
        ryx : masked array
            Operator for calculating the radial profile.
            
        area_pix : masked array
            The number of points inside each bin.
            
        Examples
        --------
        Load the dataset:
        
        >>> from pycasso.h5datacube import h5Q3DataCube
        >>> K = h5Q3DataCube('qalifa-synthesis.002.h5', 'run001', 'K0277')
            
        Create the bins from 0.0 to 3.0 in 0.1 steps,
        using ``rad_scale`` units.
        
        >>> import numpy as np
        >>> bin_r = np.arange(0.0, 3.0 + 0.1, 0.1)

        Create the radial profile operator.
        
        >>> ryx, area = K.getYXToRadialBinsTensorExact(bin_r)
        
        Calculate the radial profile for some properties. Note
        the :func:`np.tensordot` index convention. In general,
        one has to match the XY indices in both the ``ryx`` array
        (1, 2) and the property (the last two indices). The
        order of the arguments is important.
        
        >>> 
        
            
        '''
        raise NotImplementedError('Exact apertures not yet implemented.')
#         from pycasso.util import getApertureMask
#         if rad_scale is None:
#             rad_scale = self.HLR_pix
#         shape = self.N_y, self.N_x
#         ryx, _ = getApertureMask(shape, self.x0, self.y0, self.pa, self.ba, np.asarray(bin_r) * rad_scale)
#         ryx[:, ~self.qMask] = 0.0
#         area_pix = ma.masked_less(ryx.sum(axis=2).sum(axis=1), 1.0, copy=False)
#         area_pix.fill_value = self.fill_value
#         ryx /= area_pix[:, np.newaxis, np.newaxis]
#         return ryx, area_pix

    
    def radialProfile(self, prop, bin_r, rad_scale=None, mask=None, r__yx=None, mode='mean', return_npts=False):
        '''
        Calculate the radial profile of a property. The last two dimensions of
        ``prop`` must be of length ``N_y`` and ``N_x``.
        
        Parameters
        ----------
        prop : array
            Image of property to calculate the radial profile.
            
        bin_r : array
            Semimajor axis bin boundaries in units of ``rad_scale``.
            
        rad_scale : float, optional
            Scale of the bins, in pixels. Defaults to :attr:`HLR_pix`.
            
        mask : array, optional
            Mask containing the pixels to use in the radial profile.
            Must have the shape ``(N_y, N_x)``. Defaults to :attr:`qMask`.
            
        r__yx : array, optional
            Distance of each pixel to the galaxy center, in pixels. Must have
            the shape ``(N_y, N_x)``. Defaults to :attr:`pixelDistance__yx`.
            Not used when ``mode='mean_exact'``.

        mode : {'mean', mean_exact', 'median', 'sum'}, optional
            The operation to perform inside the bin. Default is ``'mean'``.
            The mode 'mean_exact' computes the intersection of an ellipse
            and the pixels, this avoids pixelated bins but is very slow.
                
        return_npts : bool, optional
            If set to ``True``, also return the number of points inside
            each bin. Defaults to ``False``.
            
            
        Returns
        -------
        
        radProf : array
            Array containing the radial profile as the last dimension.
            Note that ``radProf.shape[-1] == (len(bin_r) - 1)``
            
        npts : array, optional
            The number of points inside each bin, only if ``return_npts``
            is set to ``True``.
            
        Examples
        --------
        
        Load the dataset:
        
        >>> from pycasso.h5datacube import h5Q3DataCube
        >>> K = h5Q3DataCube('qalifa-synthesis.002.h5', 'run001', 'K0277')
            
        Create the bins from 0.0 to 3.0 in 0.1 steps,
        using ``rad_scale`` units.
        
        >>> import numpy as np
        >>> bin_r = np.arange(0.0, 3.0 + 0.1, 0.1)

        Calculate the radial profile of the time resolved mass surface
        density, using a radial scale of 10.5 pixels.
        
        >>> McorSD__tyx = K.McorSD__tZyx.sum(axis=1)
        >>> McorSD__tZr = K.radialProfile(McorSD__tyx, bin_r, rad_scale=10.5)
        
        Plot the Radial profile.
        
        >>> import matplotlib.pyplot as plt
        >>> plt.pcolormesh(np.log10(K.ageBase), bin_r, McorSD__r)
        
        See also
        --------
        azimuthalProfileNd, radialProfile, zoneToRad, zoneToYX
        '''
        if rad_scale is None:
            rad_scale = self.HLR_pix
        if mask is None:
            mask = self.qMask
        if mode == 'mean_exact':
            return radialProfileExact(prop, bin_r, self.x0, self.y0, self.pa, self.ba, rad_scale, mask, return_npts)
        else:
            return radialProfile(prop, bin_r, self.x0, self.y0, self.pa, self.ba, rad_scale, mask, mode, return_npts)
    

    def zoneToRad(self, prop, bin_r, rad_scale=None, mask=None, r__yx=None, mode='mean',
                  extensive=True, surface_density=True):
        '''
        Calculates the radial profile from a zone array. This method
        is a wrapper for :meth:`zoneToYX` and :meth:`radialProfile`.
        
        Parameters
        ----------
        prop : array
            Image of property to calculate the radial profile.
            
        bin_r : array
            Semimajor axis bin boundaries in units of ``rad_scale``.
            
        rad_scale : float, optional
            Scale of the bins, in pixels. Defaults to :attr:`HLR_pix`.
            
        mask : array, optional
            Mask containing the pixels to use in the radial profile.
            Must have the same dimensions as ``prop``. Defaults to :attr:`qMask`.
            
        r__yx : array, optional
            Distance of each pixel to the galaxy center, in pixels. Must have
            the same dimensions as ``prop``. Defaults to :attr:`pixelDistance__yx`.
            
        mode : {'mean', mean_exact', 'median', 'sum'}, optional
            The operation to perform inside the bin. Default is ``'mean'``.
            The mode 'mean_exact' computes the intersection of an ellipse
            and the pixels, this avoids pixelated bins but is very slow.
                
        extensive : boolean, optional
            If ``True``, ``prop`` is extensive, use dezonification weights.
            Defaults to ``True``.

        surface_density : boolean, optional
            If ``True``, and ``extensive`` is ``True``, divide the
            return value by the area in parsec^2 of the pixel.
            Defaults to ``True``.       
        
        Returns
        -------
        radProf : array
            Array containing the radial profile.
            Note that ``len(radProf) == (len(bin_r) - 1)``
            
        Examples
        --------
        Load the dataset:
        
        >>> from pycasso.h5datacube import h5Q3DataCube
        >>> K = h5Q3DataCube('qalifa-synthesis.002.h5', 'run001', 'K0277')
            
        Create the bins from 0.0 to 3.0 in 0.1 steps,
        using ``rad_scale`` units.
        
        >>> import numpy as np
        >>> bin_r = np.arange(0.0, 3.0 + 0.1, 0.1)

        Calculate the radial profile of mass surface density, using
        a radial scale of 10.5 pixels. The ``extensive`` option is the
        key for "surface density".
        
        >>> McorSD__r = K.zoneToRad(K.Mcor__z, bin_r, rad_scale=10.5, extensive=True)
        
        Note that ``bin_r`` is the bin boundaries, it is not fit for
        plotting along with ``McorSD__r``. We need the bin centers.
          
        >>> bin_center = (bin_r[:-1] + bin_r[1:]) / 2.0

        Plot the Radial profile.
        
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(bin_center, McorSD__r)
       
        See also
        --------
        radialProfile, radialProfileNd, zoneToYX
        '''
        prop__yx = self.zoneToYX(prop, extensive, surface_density)
        return self.radialProfile(prop__yx, bin_r, rad_scale, mask, r__yx, mode)


    def azimuthalProfileNd(self, prop, bin_a, bin_r, rad_scale=None, mask=None,
                           a__yx=None, r__yx=None, mode='mean', return_npts=False):
        '''
        Calculate the azimuthal profile of a property. The last two dimensions of
        ``prop`` must be of legth ``N_y`` and ``N_x``.
        
        Parameters
        ----------
        prop : array
            Image of property to calculate the radial profile.
            
        bin_a : array
            Angular bin boundaries in radians.
            
        bin_r : array
            Semimajor axis bin boundaries in units of ``rad_scale``.
            
        rad_scale : float, optional
            Scale of the bins, in pixels. Defaults to :attr:`HLR_pix`.
            
        mask : array, optional
            Mask containing the pixels to use in the radial profile.
            Must have the shape ``(N_y, N_x)``. Defaults to :attr:`qMask`.
            
        r__yx : array, optional
            Distance of each pixel to the galaxy center, in pixels. Must have the
            shape ``(N_y, N_x)``. Defaults to :attr:`pixelDistance__yx`.

        a__yx : array, optional
            Angle associated with each pixel. Must have the shape ``(N_y, N_x)``.
            Defaults to :attr:`pixelAngle__yx`.

        mode : {'mean', 'median', 'sum'}, optional
            The operation to perform inside the bin. Default is ``'mean'``.
                
        return_npts : bool, optional
            If set to ``True``, also return the number of points inside
            each bin. Defaults to ``False``.
            
            
        Returns
        -------
        azProf : array
            Array containing the radial and azimuthal profiles as the
            last dimensions.
            Note that ``radProf.shape[-2] == (len(bin_a) - 1)`` and
            ``radProf.shape[-1] == (len(bin_r) - 1)``.
            
        npts : array, optional
            The number of points inside each bin, only if ``return_npts``
            is set to ``True``.
            
        Examples
        --------
        Load the dataset:
        
        >>> from pycasso.h5datacube import h5Q3DataCube
        >>> K = h5Q3DataCube('qalifa-synthesis.002.h5', 'run001', 'K0277')
            
        Create the radial bins from 0.0 to 3.0 in 0.5 steps,
        using ``rad_scale`` units, and the angular bins as
        21 boundaries between -180 and 180 degrees, converted to radians.
        
        >>> import numpy as np
        >>> bin_r = np.arange(0.0, 3.0 + 0.5, 0.5)
        >>> bin_a = np.linspace(-180.0, 180.0, 21) / 180.0 * np.pi

        Calculate the azimuthal profile of the time resolved mass surface
        density, using a radial scale of 10.5 pixels.
        
        >>> McorSD__tyx = K.McorSD__tZyx.sum(axis=1)
        >>> McorSD__taR = K.azimuthalProfileNd(McorSD__tyx, bin_a, bin_r, rad_scale=10.5)
        
        Note that ``bin_a`` contains the bin boundaries, it is not fit for
        plotting along with ``McorSD__tZaR``. We will use the bin centers.
        
        >>> bin_a_center = (bin_a[:-1] + bin_a[1:]) / 2.0

        Plot the azimuthal profile for the first and last radial bins,
        summing in all ages. Note that McorSD__taR.shape is
        ``(N_ages, len(bin_a) - 1, len(bin_r) - 1)``.
        
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(bin_a_center, McorSD__taR[...,0].sum(axis=0)
        >>> plt.plot(bin_a_center, McorSD__taR[...,-1].sum(axis=0)
        
        See also
        --------
        radialProfileNd, radialProfile, zoneToRad, zoneToYX
        '''
        if rad_scale is None:
            rad_scale = self.HLR_pix
        if mask is None:
            mask = self.qMask
        if a__yx is None:
            a__yx = self.pixelAngle__yx
        if r__yx is None:
            r__yx = self.pixelDistance__yx / rad_scale
        if mode == 'mean':
            reduce_func = np.mean
        elif mode == 'median':
            reduce_func = np.median
        elif mode == 'sum':
            reduce_func = np.sum
        else:
            raise ValueError('Invalid mode: %s' % mode)
        
        if prop.ndim == 2:
            prop_flat = prop[mask]

        else:
            prop_flat = prop[...,mask]
            
        r_flat = r__yx[mask]
        r_idx = np.digitize(r_flat, bin_r)
        
        a_flat = a__yx[mask]
        a_idx = np.digitize(a_flat, bin_a)
        
        # composing the angle-radius grid.
        len_a = len(bin_a)
        len_r = len(bin_r)
        rr, aa = np.meshgrid(np.arange(1, len_r), np.arange(1, len_a))
        ra_grid = np.array(zip(rr.ravel(), aa.ravel()))
        
        if prop.ndim == 2:
            prop_profile = np.array([reduce_func(prop_flat[(a_idx == a) & (r_idx == r)]) for r, a in ra_grid])
        else:
            prop_profile = np.array([reduce_func(prop_flat[..., (a_idx == a) & (r_idx == r)], axis=-1) for r, a in ra_grid])

        # Putting the arrays back into shape.
        shape = (len_a - 1, len_r - 1) + prop_flat.shape[0:-1] 
        prop_profile = prop_profile.reshape(shape)
        prop_profile = np.rollaxis(prop_profile, 0, prop.ndim)
        prop_profile = np.rollaxis(prop_profile, 0, prop.ndim).copy()
        
        if return_npts:
            npts = np.array([np.sum((r_idx == r) & (a_idx == r)) for r, a in ra_grid])
            return prop_profile, npts
        return prop_profile
    

    def fillImage(self, prop, prop__r=None, r=None, r__yx=None, mode='convex'):
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
        if r__yx is None:
            r__yx = self.pixelDistance__yx
        if not isinstance(prop, np.ma.MaskedArray):
            input_masked = False
            prop = np.ma.masked_where(self.qMask != 1, prop)
        else:
            input_masked = True
        if prop__r is None or r is None:
            max_r = r__yx.max()
            bin_r = np.arange(0, max_r+1, 1)
            r = (bin_r[:-1] + bin_r[1:]) / 2.0
            prop__r = self.radialProfile(prop, bin_r, rad_scale=1.0, mode='mean')
            
        if mode == 'convex':
            to_fill = self.qConvexHull & prop.mask
        elif mode == 'hollow':
            to_fill = self.qHollowPixels & prop.mask
        else:
            raise ValueError('%s is not a valid mode.' % mode)
        
        _prop = prop.copy()
        _prop[to_fill] = np.interp(r__yx[to_fill], r[~prop__r.mask], prop__r.compressed())
        
        if input_masked:
            return _prop
        else:
            return _prop.data, ~_prop.mask


    def getEllipseParams(self, prop=None, mask=None):
        '''
        Estimate ellipticity and orientation of the galaxy using the
        "Stokes parameters", as described in:
        http://adsabs.harvard.edu/abs/2002AJ....123..485S
        The image used is ``qSignal``.
        
        Parameters
        ----------
        prop : array, optional
            Image to use when calculating the ellipse parameters.
            If not set, :attr:`qSignal` will be used.

        mask : array, optional
            Mask containing the pixels to take into account.
            If not set, :attr:`qMask` will be used.
        
        Returns
        -------
        pa : float
            Position angle in radians, counter-clockwise relative
            to the positive X axis.
        
        ba : float
            Ellipticity, defined as the ratio between the semiminor
            axis and the semimajor axis (:math:`b/a`).
        '''
        if prop is None:
            prop = self.qSignal
        if mask is None:
            mask = self.qMask.copy()
        return getEllipseParams(prop, self.x0, self.y0, mask)


    def getPixelDistance(self, use_HLR_units=True, pixel_scale=None, x=None, y=None, pa=None, ba=None):
        '''
        Return an image (:class:`numpy.ndarray` of same shape as :attr`qSignal`)
        of the distance from the center of the galaxy ``(x0, y0)`` in HLR units
        (default), assuming a projected disk.
        
        Parameters
        ----------
        use_HLR_units : boolean, optional
            Whether to use units of half light radius or pixels.
            
        pixel_scale : float, optional
            Pixel distance scale, used if ``use_HLR_units`` is ``False``.
            If not set, do not scale the distance. 
            
        x : array, optional
            X coordinates to calculate the distance. If not set, the
            coordinates of the core images will be used.
        
        y : array, optional
            Y coordinates to calculate the distance. Must have the same
            length as ``x``. If not set, the coordinates of the core
            images will be used.
        
        pa : float, optional
            Position angle in radians, counter-clockwise relative
            to the positive X axis.
        
        ba : float, optional
            Ellipticity, defined as the ratio between the semiminor
            axis and the semimajor axis (:math:`b/a`).
            
        Returns
        -------
        pixelDistance : array
            Array (or image) containing the pixel distances.
            
        See also
        --------
        getPixelAngle

        '''
        if pa is None or ba is None:
            pa = self.pa
            ba = self.ba
        if x is not None or y is not None:
            pixelDistance = getDistance(x, y, self.x0, self.y0, pa, ba)
        else:
            pixelDistance = getImageDistance(self.qSignal.shape, self.x0, self.y0, pa, ba)
        if use_HLR_units:
            return pixelDistance / self.HLR_pix
        if pixel_scale is not None:
            return pixelDistance / pixel_scale
        return pixelDistance


    def getPixelAngle(self, units='radians', x=None, y=None, pa=None, ba=None):
        '''
        Return an image (:class:`numpy.ndarray` of same shape as :attr`qSignal`)
        of the angle in radians (default) of each pixel, relative from the axis of the
        position angle ``pa``. The projection is fixed assuming the galaxy is
        a disk, throught the ellipticity parameter ``ba``.
        
        Parameters
        ----------
        units : {'radians', 'degrees'}, optional
            If ``'radians'``, angles are in radians, from :math:`-\pi` to :math:`+\pi` (default).
            If ``'degrees'``, angles are in degrees, from :math:`-180.0` to :math:`+180.0`.
            
        x : array, optional
            X coordinates to calculate the distance. If not set, the
            coordinates of the core images will be used.
        
        y : array, optional
            Y coordinates to calculate the distance. Must have the same
            length as ``x``. If not set, the coordinates of the core
            images will be used.
        
        pa : float, optional
            Position angle in radians, counter-clockwise relative
            to the positive X axis.
        
        ba : float, optional
            Ellipticity, defined as the ratio between the semiminor
            axis and the semimajor axis (:math:`b/a`).
            
        Returns
        -------
        pixelDistance : 2-D array
            Image containing the pixel distances.

        '''
        if pa is None or ba is None:
            pa = self.pa
            ba = self.ba
        if x is not None or y is not None:
            pixelAngle = getAngle(x, y, self.x0, self.y0, pa, ba)
        else:
            pixelAngle = getImageAngle(self.qSignal.shape, self.x0, self.y0, pa, ba)
        if units == 'radians':
            return pixelAngle
        elif units == 'degrees':
            return pixelAngle * (180.0 / np.pi)
        else:
            raise ValueError('Wrong unit: %s' )


    def getHalfRadius(self, prop, fill=False, mask=None):
        '''
        Find the half radius of the desired property. Using radial
        bins of 1 pixel, calculate the cumulative sum of ``prop``. The "half
        prop radius" is the radius where the cumulative sum reaches 50% of
        its peak value.
        
        Parameters
        ----------
        prop : array
            Image to get the half radius.

        fill : boolean, optional
            Whether to fill the hollow areas before the calculations.
            The filling is done using the average value in the respective
            radial distance of the missing pixels. Default is ``False``.
            
        mask : array(bool), optional
            Boolean array with the valid data. Defaults to ``qMask``.
            
        Returns
        -------
        HXR : float
            The half ``prop`` radius, in pixels.
            
        Notes
        -----
        This value should be close to $\dfrac{HLR_{circular}}{\sqrt{b/a}}$
        if the isophotes of the galaxy are all ellipses with parameters
        p.a. and b/a.
        
        Examples
        --------
        Load the dataset:
        
        >>> from pycasso.h5datacube import h5Q3DataCube
        >>> K = h5Q3DataCube('qalifa-synthesis.002.h5', 'run001', 'K0277')

        Calculate the half mass radius, in pixels.
        
        >>> HMR_pix = K.getHalfRadius(K.McorSD)
        
        Calculate the radial profile of extinction by dust, using
        ``HMR_pix`` as radial scale, in bins from 0.0 to 3.0,
        in 0.1 steps.
        
        >>> import numpy as np
        >>> bin_r = np.arange(0.0, 3.0 + 0.1, 0.1)
        >>> A_V__r = K.radialProfile(K.A_V__yx, bin_r, rad_scale=HMR_pix)
        
        Note that ``bin_r`` is the bin boundaries, it is not fit for
        plotting along with ``A_V__r``. We need the bin centers.
          
        >>> bin_center = bin_center = (bin_r[:-1] + bin_r[1:]) / 2.0

        Plot the Radial profile.
        
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(bin_center, A_V__r)

        See also
        --------
        setGeometry, radialProfile
        '''
        
        if fill:
            prop, mask = self.fillImage(prop)
        else:
            if mask is None:
                mask = self.qMask
        return getGenHalfRadius(prop[mask], self.pixelDistance__yx[mask])


    def getHLR_pix(self, pa=None, ba=None, fill=False):
        '''
        Find the half light radius using the image at the normalization window
        (:attr:`qSignal`). Using radial bins of 1 pixel, calculate the cumulative
        sum of luminosity. The HLR is the radius where the cum. sum reaches 50% 
        of its peak value.
        
        Parameters
        ----------
        fill : boolean
            Whether to fill the hollow areas before calculating the HLR.
            
        Returns
        -------
        HLR : float
            The half light radius, in pixels.
            
        Notes
        -----
        This value should be close to $\dfrac{HLR_{circular}}{\sqrt{b/a}}$
        if the isophotes of the galaxy are all ellipses with parameters
        p.a. and b/a.
        
        See also
        --------
        getHalfRadius, setGeometry, fillImage
        '''
        return self.getHalfRadius(self.qSignal, fill)


    def growthInAge(self, prop, mask=None, relative=False):
        '''
        TODO: move to utils?
        
        Calculate the cumulative growth in age space.
        In age = 0, we have the sum of ``prop`` in age, and as
        age increases the sum of ``prop`` decreases. 
        
        Parameters
        ----------
        prop : array
            A property having the age as first dimension.
            
        mask : array
            Boolean mask for all dimensions except the first (age).
            If set, the values will be summed, that is, the return array
            will only have the age dimension.
            
        relative : bool
            If True, normalize by the maximum value of the cumulative
            sum of ``prop``.
            
        Returns
        -------
        cs : array
            The growth in age space of ``prop``.

        Examples
        --------
        Load the dataset:
        
        >>> from pycasso.h5datacube import h5Q3DataCube
        >>> K = h5Q3DataCube('qalifa-synthesis.002.h5', 'run001', 'K0277')
            
        Get the spatially resolved mass surface density for each age.

        >>> MiniSD__tyx = K.MiniSD__tZyx.sum(axis=1)
        
        Calculate the mass build up for the whole galaxy (notice
        the "all data" mask, ``K.qMask``). The mask could be any
        selection of data pixels.
        
        >>> MtotGrow__t = K.growthInAge(MiniSD__tyx, mask=K.qMask)
        
        ``MtotGrow__t`` is a surface mass density, in :math:`M_\odot / pc^2`.
        Converting to absolute mass.
        
        >>> MtotGrow__t *= K.parsecPerPixel ** 2
        
        Plot the mass buildup as a function of log. time.

        >>> ages = np.log10(K.ageBase)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(ages, MtotGrow__t)

        '''
        if mask is not None:
            prop = prop[...,mask].sum(axis=-1)

        cs = prop[::-1].cumsum(axis=0)[::-1]
        if relative:
            cs /= cs[0]            
        return cs


    def getSFR(self, logtc_ini=None, logtc_fin=None, logtc_step=0.05, logtc_FWHM=0.5):
        '''
        Calculate the star formation rate using a smooth-resampled age base,
        as prescribed by Asari et al. (2007) <http://adsabs.harvard.edu/abs/2007MNRAS.381..263A>.
        
        Parameters
        ----------
        logtc_ini : float
            Logarithm (base 10) of initial age. Defaults to
            ``logtb.min()``.
    
        logtc_fin : float
            Logarithm (base 10) of final age. Defaults to
            ``logtb.max()``.
    
        logtc_step : float
            Logarithm (base 10) of age step. Defaults to ``0.05``.
    
        logtc_FWHM : float
            Width of the age smoothing kernel. Defaults to ``0.5``.

        Returns
        -------
        SFR : array
            Star formation rate.
            
        logtc : array
            Logarithm (base 10) of smooth-resampled age base.
            
        See also
        --------
        pystarlight.util.StarlightUtils.calcSFR
        '''
        logtb = np.log10(self.ageBase)
        if logtc_ini is None:
            logtc_ini = logtb.min()
        if logtc_fin is None:
            logtc_fin = logtb.max()
        logtc = np.arange(logtc_ini, logtc_fin + logtc_step, logtc_step)
        SFR = calcSFR(self.popx, self.fbase_norm, self.Lobs_norm,
                      self.q_norm, self.A_V,
                      logtb, logtc, logtc_FWHM)
        return SFR, logtc

    
    def getSFR_alt(self,  logtc_ini=None, logtc_fin=None, logtc_step=0.05, logtc_FWHM=0.5, dt=0.5e9):
        logtb = np.log10(self.ageBase)
        if logtc_ini is None:
            logtc_ini = logtb.min()
        if logtc_fin is None:
            logtc_fin = logtb.max()
        logtc = np.arange(logtb.min(), logtb.max() + logtc_step, logtc_step)
        Mini = smooth_Mini(self.popx, self.fbase_norm, self.Lobs_norm,
                           self.q_norm, self.A_V,
                           logtb, logtc, logtc_FWHM)
        if self.hasAlphaEnhancement:
            Mini = Mini.sum(axis=2)
        Mini = Mini.sum(axis=1)
        logtc_bins = bin_edges(logtc)
        tc_bins = 10**logtc_bins
        tl = np.arange(tc_bins.min(), tc_bins.max()+dt, dt)
        tl_bins = bin_edges(tl)
        Mini_r = np.zeros((len(tl) + 2, self.N_zone))
        for z in range(self.N_zone):
            Mini_r[1:-1, z] = hist_resample(tc_bins, tl_bins, Mini[:, z])
        SFR = Mini_r / dt
        # Add bondary points so that np.trapz(SFR, tl) == Mini.sum().
        tl = np.hstack((tl[0] - dt, tl, tl[-1] + dt))
        return SFR, tl
    
    
    def getSFR_nosmooth(self, dt=0.5e9):        
        logtb = np.log10(self.ageBase)
        fbase_norm = self.fbase_norm.copy()
        fbase_norm[fbase_norm == 0] = 1.0
        logtb_bins = bin_edges(logtb)
        tb_bins = 10**logtb_bins
        tl = np.arange(tb_bins.min(), tb_bins.max()+dt, dt)
        tl_bins = bin_edges(tl)
        if self.hasAlphaEnhancement:
            Mini = self.Mini__tZaz.sum(axis=2).sum(axis=1)
        else:
            Mini = self.Mini__tZz.sum(axis=1)
        Mini_r = np.zeros((len(tl) + 2, self.N_zone))
        for z in range(self.N_zone):
            Mini_r[1:-1, z] = hist_resample(tb_bins, tl_bins, Mini[:, z])
        SFR = Mini_r / dt
        # Add bondary points so that np.trapz(SFR, tl) == Mini.sum().
        tl = np.hstack((tl[0] - dt, tl, tl[-1] + dt))
        return SFR, tl
    
    
    def filterResidual(self, fnorm_max=10.0, percentage=60.0, w1=None, w2=None, method='percentage'):
        '''
        Return a boolean array selecting good spectra based on residual criteria.
        This is intended for indexing zone arrays.
        
        Parameters
        ----------
        fnorm_max : float, optional
            Maximum deviation of the residual spectrum in normalized units
            (fobs_norm). This value represents the maximum percentage of
            allowed deviation (being 0% a perfect fit). A value of 10 mask
            as bad points all values larger than 10% (either positive or
            negative). Default: ``10.0``
            
        percentage : float, optional
            Parameter of the "percentage" method. Is the maximum percentage
            of wavelength units (steps) in the selected wavelength range
            (see w1, w2) of points with values larger that fnorm_max. Above
            this percentage, the spectrum is consider a bad fit.
            Default: ``60.0``
            
        w1 : float, optional
            Lower wavelength, in Angstroms. Default: The lowest wavelength in ``l_obs``.

        w2 : float, optional
            Upper wavelength, in Angstroms. Default: The highest wavelength in ``l_obs``.
            
        method : string, optional
            Available methods:  ``'percentage'`` or ``'median'``. 
            ``'percentage'``: This methods counts the total number of wavelength
            steps in the selected wavelength range (does not take into account
            masked values with f_wei = 0) and the number of pixels with values
            larger that "fnorm_max". If the percentage of the bad pixels (later)
            is larger that the "percentage" parameter, the spectrum is consider
            a bad fit.
            ``'median'``: This methods estimates the median value of the residual
            spectrum (does not take into account masked values with f_wei = 0).
            If the output is larger than "fnorm_max", the spectrum is masked.
            Default: ``'percentage'``
        
        Returns
        -------
        mask : array
            An array containing the good spectra marked as ``True``.
            Shape: ``(N_zone,)``
        '''
        residuals = np.abs(100.0 * (self.f_obs - self.f_syn) / self.fobs_norm)
        if w1 is None:
            w1 = self.l_obs[0]
        if w2 is None:
            w2 = self.l_obs[-1]
        wave_mask = (self.l_obs >= w1) & (self.l_obs <= w2)
        global_mask = wave_mask[..., np.newaxis] & (self.f_wei != 0)
        if method == 'percentage':
            total_mask = global_mask & (residuals > fnorm_max)
            mask_number_pixels = np.zeros_like(residuals, dtype=np.bool)
            mask2D = np.zeros_like(residuals, dtype=np.bool)
            mask2D[total_mask] = True
            mask_number_pixels[global_mask] = True
            zonesMask = mask2D.sum(axis=0) <= mask_number_pixels.sum(axis=0) * percentage/ 100.0
        elif method == 'median':
            masked_residuals = np.ma.array(residuals, mask=~global_mask)
            zonesMask =  np.ma.median(masked_residuals, axis=0) <= fnorm_max
        else:
            raise Exception('Method "s" NOT available (percentage | median)')
        return zonesMask
    
    
    def filterResidual__yx(self, fnorm_max=10.0, percentage=60.0, w1=None, w2=None, method='percentage'):
        '''
        Return a boolean array selecting the spectra that meet some residual criteria.
        This is intended for indexing spatially resolved arrays.
        
        Parameters
        ----------
        fnorm_max : float, optional
            Default: ``10.0``
            
        percentage : float, optional
            Default: ``60.0``
            
        w1 : float, optional
            Default: ``None``

        w2 : float, optional
            Default: ``None``
            
        method : string, optional
            Default: ``'percentage'``
        
        Returns
        -------
        mask : array
            An array containing the good spectra marked as ``True``.
            Shape: ``(N_y, N_x)``
        '''
        fm = self.filterResidual(fnorm_max, percentage, w1, w2, method)
        return self.zoneToYX(fm, extensive=False, surface_density=False).filled(False)

    
    @property
    def area_qMask(self):
        '''
        Area, in pixels, of the data regions.
        This is the area to use when dealing with :attr:`qMask`.
        '''
        return np.array(self.qMask, dtype=float).sum()
    
    
    @property
    def area_qHollowPixels(self):
        '''
        Aarea, in pixels, of the masked regions.
        This is the area to use when dealing with :attr:`qHollowPixels`.
        '''
        return np.array(self.qHollowPixels, dtype=float).sum()
    
    
    @property
    def area_qConvexHull(self):
        '''
        Area, in pixels, of the convex hull of :attr:`qMask`.
        This is the area to use when dealing with :attr:`qConvexHull`.
        '''
        return np.array(self.qConvexHull, dtype=float).sum()
    
    
    @property
    def qHollowPixels(self):
        '''
        Masked (bad) pixels inside the data mask.
        
            * Units: bool
            * Shape: ``(N-y, N_x)``
        '''
        hollow_bit = 0b10
        return self.qFilledMask & hollow_bit > 0
    
    
    @property
    def qConvexHull(self):
        '''
        Convex hull of data mask.
        
            * Units: bool
            * Shape: ``(N-y, N_x)``
        '''
        if self._qConvexHull is not None:
            return self._qConvexHull
        hollow_bit = 0b100
        return self.qFilledMask & hollow_bit > 0
    
    
    @property
    def popx__tZayx(self):
        '''
        Spatially resolved light fractions for each population.
        
            * Units: :math:`[\%]`
            * Shape: ``(N_age, N_met, N_aFe, N_y, N_x)``
        '''
        self.checkAlphaEnhancement()
        return self.zoneToYX(self.popx / 100.0, extensive=False)

    
    @property
    def popx__tZyx(self):
        '''
        Spatially resolved light fractions for each population.
        
            * Units: :math:`[\%]`
            * Shape: ``(N_age, N_met, N_y, N_x)``
        '''
        if self.hasAlphaEnhancement:
            return self.popx__tZayx.sum(axis=2)
        else:
            return self.zoneToYX(self.popx / 100.0, extensive=False)

    
    @property
    def popmu_cor__tZayx(self):
        '''
        Spatially resolved corrected mass fractions for each population.

            * Units: :math:`[\%]`
            * Shape: ``(N_age, N_met, N_aFe, N_y, N_x)``
        '''
        self.checkAlphaEnhancement()
        return self.zoneToYX(self.popmu_cor / 100.0, extensive=False)

    
    @property
    def popmu_cor__tZyx(self):
        '''
        Spatially resolved corrected mass fractions for each population.

            * Units: :math:`[\%]`
            * Shape: ``(N_age, N_met, N_y, N_x)``
        '''
        if self.hasAlphaEnhancement:
            return self.popmu_cor__tZayx.sum(axis=2)
        else:
            return self.zoneToYX(self.popmu_cor / 100.0, extensive=False)

    
    @property
    def popmu_ini__tZayx(self):
        '''
        Spatially resolved initial mass fractions for each population.

            * Units: :math:`[\%]`
            * Shape: ``(N_age, N_met, N_aFe, N_y, N_x)``
        '''
        self.checkAlphaEnhancement()
        return self.zoneToYX(self.popmu_ini / 100.0, extensive=False)

    
    @property
    def popmu_ini__tZyx(self):
        '''
        Spatially resolved initial mass fractions for each population.
        
            * Units: :math:`[\%]`
            * Shape: ``(N_age, N_met, N_y, N_x)``
        '''
        if self.hasAlphaEnhancement:
            return self.popmu_ini__tZayx.sum(axis=2)
        else:
            return self.zoneToYX(self.popmu_ini / 100.0, extensive=False)

    
    @property
    def popAV_tot__tZayx(self):
        '''
        Spatially resolved extinction by dust for each population.
        
            * Units: :math:`[mag]`
            * Shape: ``(N_age, N_met, N_aFe, N_y, N_x)``
        '''
        self.checkAlphaEnhancement()
        return self.zoneToYX(self.popAV_tot, extensive=False)

    
    @property
    def popAV_tot__tZyx(self):
        '''
        Spatially resolved extinction by dust for each population.
        
            * Units: :math:`[mag]`
            * Shape: ``(N_age, N_met, N_y, N_x)``
        '''
        if self.hasAlphaEnhancement:
            return self.popAV_tot__tZayx.sum(axis=2)
        else:
            return self.zoneToYX(self.popAV_tot, extensive=False)

    
    @property
    def A_V__yx(self):
        '''
        Spatially resolved extinction by dust.

            * Units: :math:`[mag]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.A_V, extensive=False)

    
    @property
    def tau_V__z(self):
        '''
        Dust optical depth in the V band.

            * Units: dimensionless
            * Shape: ``(N_z)``
        '''
        return self.A_V / (2.5 * np.log10(np.exp(1.)))

    
    @property
    def tau_V__yx(self):
        '''
        Spatially resolved dust optical depth in the V band.

            * Units: dimensionless
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.tau_V__z, extensive=False)
    
    
    @property
    def integrated_tau_V(self):
        '''
        Dust optical depth in the V band.

            * Units: dimensionless
            * Type: float
        '''
        return self.integrated_keywords['A_V'] / (2.5 * np.log10(np.exp(1.)))

    
    @property
    def v_0__yx(self):
        '''
        Spatially resolved velocity displacement.

            * Units: :math:`[km/s]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.v_0, extensive=False)

    
    @property
    def v_d__yx(self):
        '''
        Spatially resolved velocity dispersion.

            * Units: :math:`[km/s]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.v_d, extensive=False)


    @property
    def integrated_Lobn__tZa(self):
        '''
        Luminosity of each population in normalization window 
        of the integrated spectrum.
 
            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met, n_aFe)``
        '''
        self.checkAlphaEnhancement()
        tmp = self.integrated_popx / 100.0
        tmp *= self.integrated_keywords['LOBS_NORM']
        return tmp


    @property
    def integrated_Lobn__tZ(self):
        '''
        Luminosity of each population in normalization window 
        of the integrated spectrum.
 
            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met)``
        '''
        if self.hasAlphaEnhancement:
            return self.integrated_Lobn__tZa.sum(axis=2)
        
        tmp = self.integrated_popx / 100.0
        tmp *= self.integrated_keywords['LOBS_NORM']
        return tmp


    @property
    def integrated_Lobn(self):
        '''
        Luminosity in normalization window of the integrated spectrum.

            * Units: :math:`[L_\odot]`
            *  Type: float
        '''
        return self.integrated_Lobn__tZ.sum(axis=1).sum(axis=0)


    @property
    def Lobn__tZaz(self):
        '''
        Luminosity of each population in normalization window,
        in voronoi zones.
 
            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met, N_aFe, N_zone)``
        '''
        self.checkAlphaEnhancement()
        tmp = self.popx / 100.0
        tmp *= self.Lobs_norm
        return tmp


    @property
    def Lobn__tZz(self):
        '''
        Luminosity of each population in normalization window,
        in voronoi zones.
 
            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met, N_zone)``
        '''
        if self.hasAlphaEnhancement:
            return self.Lobn__tZaz.sum(axis=2)
        
        tmp = self.popx / 100.0
        tmp *= self.Lobs_norm
        return tmp


    @property
    def Lobn__z(self):
        '''
        Luminosity in normalization window, in voronoi zones.

            * Units: :math:`[L_\odot]`
            * Shape: ``(N_zone)``
        '''
        return self.Lobn__tZz.sum(axis=1).sum(axis=0)

    
    @property
    def LobnSD__tZayx(self):
        '''
        Spatially resolved luminosity surface density of
        each population in normalization window.

            * Units: :math:`[L_\odot / pc^2]`
            * Shape: ``(N_age, N_met, N_aFe, N_y, N_x)``
        '''
        return self.zoneToYX(self.Lobn__tZaz, extensive=True)


    @property
    def LobnSD__tZyx(self):
        '''
        Spatially resolved luminosity surface density of
        each population in normalization window.

            * Units: :math:`[L_\odot / pc^2]`
            * Shape: ``(N_age, N_met, N_y, N_x)``
        '''
        return self.zoneToYX(self.Lobn__tZz, extensive=True)


    @property
    def LobnSD__yx(self):
        '''
        Luminosity surface density of each population
        in normalization window.

            * Units: :math:`[L_\odot / pc^2]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.Lobn__z, extensive=True)
 

    @property
    def integrated_DeRed_Lobn__tZa(self):
        '''
        "Dereddened" luminosity of each population in normalization 
	    window of the integrated spectrum.

            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met, N_aFe)``
        '''
        return self.integrated_Lobn__tZa * np.power(10.0, 0.4 * self.q_norm * self.integrated_keywords['A_V'])


    @property
    def integrated_DeRed_Lobn__tZ(self):
        '''
        "Dereddened" luminosity of each population in normalization 
	    window of the integrated spectrum.

            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met)``
        '''
        return self.integrated_Lobn__tZ * np.power(10.0, 0.4 * self.q_norm * self.integrated_keywords['A_V'])


    @property
    def integrated_DeRed_Lobn(self):
        '''
        "Dereddened" luminosity in normalization window 
        of the integrated spectrum.

            * Units: :math:`[L_\odot]`
            *  Type: float
        '''
        return self.integrated_DeRed_Lobn__tZ.sum(axis=1).sum(axis=0)


    @property
    def DeRed_Lobn__tZaz(self):
        '''
        "Dereddened" luminosity of each population
        in normalization window, in voronoi zones.

            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met, N_aFe, N_zone)``
        '''
        return self.Lobn__tZaz * 10.0**(0.4 * self.q_norm * self.A_V)

    
    @property
    def DeRed_Lobn__tZz(self):
        '''
        "Dereddened" luminosity of each population
        in normalization window, in voronoi zones.

            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met, N_zone)``
        '''
        return self.Lobn__tZz * 10.0**(0.4 * self.q_norm * self.A_V)

    
    @property
    def DeRed_Lobn__z(self):
        '''
        "Dereddened" luminosity in normalization window,
        in voronoi zones.

            * Units: :math:`[L_\odot]`
            * Shape: ``(N_zone)``
        '''
        return self.DeRed_Lobn__tZz.sum(axis=1).sum(axis=0)

    
    @property
    def DeRed_LobnSD__tZayx(self):
        '''
        Spatially resolved "dereddened" luminosity surface density of
        each population in normalization window.

            * Units: :math:`[L_\odot / pc^2]`
            * Shape: ``(N_age, N_met, N_aFe, N_y, N_x)``
        '''
        return self.zoneToYX(self.DeRed_Lobn__tZaz, extensive=True)


    @property
    def DeRed_LobnSD__tZyx(self):
        '''
        Spatially resolved "dereddened" luminosity surface density of
        each population in normalization window.

            * Units: :math:`[L_\odot / pc^2]`
            * Shape: ``(N_age, N_met, N_y, N_x)``
        '''
        return self.zoneToYX(self.DeRed_Lobn__tZz, extensive=True)


    @property
    def DeRed_LobnSD__yx(self):
        '''
        "Dereddened" luminosity surface density of each population
        in normalization window.

            * Units: :math:`[L_\odot / pc^2]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.DeRed_Lobn__z, extensive=True)


    @property
    def integrated_Mcor__tZa(self):
        '''
        Current mass of each population of the integrated spectrum.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_aFe)``
        '''
        self.checkAlphaEnhancement()
        tmp = self.integrated_popmu_cor / 100.0
        tmp *= self.integrated_keywords['MCOR_TOT']
        return tmp
      

    @property
    def integrated_Mcor__tZ(self):
        '''
        Current mass of each population of the integrated spectrum.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met)``
        '''
        if self.hasAlphaEnhancement:
            return self.integrated_Mcor__tZa.sum(axis=2)
        
        tmp = self.integrated_popmu_cor / 100.0
        tmp *= self.integrated_keywords['MCOR_TOT']
        return tmp
      

    @property
    def integrated_Mcor(self):
        '''
        Current mass of the integrated spectrum.

            * Units: :math:`[M_\odot]`
            *  Type: float
        '''
        return self.integrated_Mcor__tZ.sum(axis=1).sum(axis=0)

      
    @property
    def Mcor__tZaz(self):
        '''
        Current mass of each population, in voronoi zones.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_aFe, N_zone)``
        '''
        self.checkAlphaEnhancement()
        tmp = self.popmu_cor / 100.0
        tmp *= self.Mcor_tot
        return tmp

    
    @property
    def Mcor__tZz(self):
        '''
        Current mass of each population, in voronoi zones.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_zone)``
        '''
        if self.hasAlphaEnhancement:
            return self.Mcor__tZaz.sum(axis=2)
        tmp = self.popmu_cor / 100.0
        tmp *= self.Mcor_tot
        return tmp

    
    @property
    def Mcor__z(self):
        '''
        Current mass, in voronoi zones.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_zone)``
        '''
        return self.Mcor__tZz.sum(axis=1).sum(axis=0)

    
    @property
    def McorSD__tZayx(self):
        '''
        Spatially resolved current mass surface density of
        each population.

            * Units: :math:`[M_\odot / pc^2]`
            * Shape: ``(N_age, N_met, N_aFe, N_y, N_x)``
        '''
        return self.zoneToYX(self.Mcor__tZaz, extensive=True)



    @property
    def McorSD__tZyx(self):
        '''
        Spatially resolved current mass surface density of
        each population.

            * Units: :math:`[M_\odot / pc^2]`
            * Shape: ``(N_age, N_met, N_y, N_x)``
        '''
        return self.zoneToYX(self.Mcor__tZz, extensive=True)



    @property
    def McorSD__yx(self):
        '''
        Spatially resolved current mass surface density.

            * Units: :math:`[M_\odot / pc^2]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.Mcor__z, extensive=True)


    @property
    def integrated_Mini__tZa(self):
        '''
        Initial mass of each population of the integrated spectrum.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_aFe)``
        '''
        self.checkAlphaEnhancement()
        tmp = self.integrated_popmu_ini / 100.0
        tmp *= self.integrated_keywords['MINI_TOT']
        return tmp
      

    @property
    def integrated_Mini__tZ(self):
        '''
        Initial mass of each population of the integrated spectrum.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met)``
        '''
        if self.hasAlphaEnhancement:
            return self.integrated_Mini__tZa.sum(axis=2)
        
        tmp = self.integrated_popmu_ini / 100.0
        tmp *= self.integrated_keywords['MINI_TOT']
        return tmp
      

    @property
    def integrated_Mini(self):
        '''
        Initial mass of the integrated spectrum.

            * Units: :math:`[M_\odot]`
            *  Type: float
        '''
        return self.integrated_Mini__tZ.sum(axis=1).sum(axis=0)

      
    @property
    def Mini__tZaz(self):
        '''
        Initial mass of each population, in voronoi zones.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_aFe, N_zone)``
        '''
        self.checkAlphaEnhancement()
        tmp = self.popmu_ini / 100.0
        tmp *= self.Mini_tot
        return tmp

    
    @property
    def Mini__tZz(self):
        '''
        Initial mass of each population, in voronoi zones.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_zone)``
        '''
        if self.hasAlphaEnhancement:
            return self.Mini__tZaz.sum(axis=2)
        
        tmp = self.popmu_ini / 100.0
        tmp *= self.Mini_tot
        return tmp

    
    @property
    def Mini__z(self):
        '''
        Initial mass, in voronoi zones.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_zone)``
        '''
        return self.Mini__tZz.sum(axis=1).sum(axis=0)

    
    @property
    def MiniSD__tZayx(self):
        '''
        Spatially resolved initial mass surface density of
        each population.

            * Units: :math:`[M_\odot / pc^2]`
            * Shape: ``(N_age, N_met, N_aFe, N_y, N_x)``
        '''
        return self.zoneToYX(self.Mini__tZaz, extensive=True)


    @property
    def MiniSD__tZyx(self):
        '''
        Spatially resolved initial mass surface density of
        each population.

            * Units: :math:`[M_\odot / pc^2]`
            * Shape: ``(N_age, N_met, N_y, N_x)``
        '''
        return self.zoneToYX(self.Mini__tZz, extensive=True)


    @property
    def MiniSD__yx(self):
        '''
        Spatially resolved initial mass surface density.

            * Units: :math:`[M_\odot / pc^2]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.Mini__z, extensive=True)


    @property
    def integrated_M2L(self):
        '''
        Mass to light ratio of the integrated spectrum.

            * Units: :math:`[M_\odot / L_\odot]`
            *  Type: float
        '''
        return self.integrated_Mcor / self.integrated_Lobn


    @property
    def M2L__yx(self):
        '''
        Spatially resolved mass to light ratio.

            * Units: :math:`[M_\odot / L_\odot]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.Mcor__z / self.Lobn__z, extensive=False)


    @property
    def integrated_DeRed_M2L(self):
        '''
        "Dereddened" mass to light ratio of the integrated spectrum.

            * Units: :math:`[M_\odot / L_\odot]`
            *  Type: float
        '''
        return self.integrated_Mcor / self.integrated_DeRed_Lobn

      
    @property
    def DeRed_M2L__yx(self):
        '''
        Spatially resolved "dereddened" mass to light ratio.

            * Units: :math:`[M_\odot / L_\odot]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.Mcor__z / self.DeRed_Lobn__z, extensive=False)


    @property
    def integrated_at_flux(self):
        '''
        Flux-weighted average log. age of the integrated spectrum.

            * Units: :math:`[\log Gyr]`
            *  Type: float
        '''
        if self.hasAlphaEnhancement:
            popx_sumZ = self.integrated_popx.sum(axis=2).sum(axis=1)
        else:  
            popx_sumZ = self.integrated_popx.sum(axis=1)
        popx_sum = popx_sumZ.sum(axis=0)
        popx_sumZ /= popx_sum
        return np.tensordot(popx_sumZ, np.log10(self.ageBase), (0, 0))

      
    @property
    def at_flux__z(self):
        '''
        Flux-weighted average log. age, in voronoi zones.

            * Units: :math:`[\log Gyr]`
            * Shape: ``(N_zone)``
        '''
        if self.hasAlphaEnhancement:
            popx_sumZ = self.popx.sum(axis=2).sum(axis=1)
        else:  
            popx_sumZ = self.popx.sum(axis=1)
        popx_sum = popx_sumZ.sum(axis=0)
        popx_sumZ /= popx_sum
        return np.tensordot(popx_sumZ, np.log10(self.ageBase), (0, 0))

      
    @property
    def at_flux__yx(self):
        '''
        Spatially resolved, flux-weighted average log. age.

            * Units: :math:`[\log Gyr]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.at_flux__z, extensive=False)


    @property
    def integrated_at_mass(self):
        '''
        Mass-weighted average log. age of the integrated spectrum.

            * Units: :math:`[\log Gyr]`
            *  Type: float
        '''
        if self.hasAlphaEnhancement:
            popmu_cor_sumZ = self.integrated_popmu_cor.sum(axis=2).sum(axis=1)
        else:  
            popmu_cor_sumZ = self.integrated_popmu_cor.sum(axis=1)
        popmu_cor_sum = popmu_cor_sumZ.sum(axis=0)
        popmu_cor_sumZ /= popmu_cor_sum
        return np.tensordot(popmu_cor_sumZ, np.log10(self.ageBase), (0, 0))

      
    @property
    def at_mass__z(self):
        '''
        Mass-weighted average log. age, in voronoi zones.

            * Units: :math:`[\log Gyr]`
            * Shape: ``(N_zone)``
        '''
        if self.hasAlphaEnhancement:
            popmu_cor_sumZ = self.popmu_cor.sum(axis=2).sum(axis=1)
        else:  
            popmu_cor_sumZ = self.popmu_cor.sum(axis=1)
        popmu_cor_sum = popmu_cor_sumZ.sum(axis=0)
        popmu_cor_sumZ /= popmu_cor_sum
        return np.tensordot(popmu_cor_sumZ, np.log10(self.ageBase), (0, 0))

      
    @property
    def at_mass__yx(self):
        '''
        Spatially resolved, mass-weighted average log. age.

            * Units: :math:`[\log Gyr]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.at_mass__z, extensive=False)


    @property
    def integrated_aZ_flux(self):
        '''
        Flux-weighted average metallicity of the integrated spectrum.

            * Units: dimensionless
            *  Type: float
        '''
        if self.hasAlphaEnhancement:
            popx_sumt = self.integrated_popx.sum(axis=2).sum(axis=0)
        else:  
            popx_sumt = self.integrated_popx.sum(axis=0)
        popx_sum = popx_sumt.sum(axis=0)
        popx_sumt /= popx_sum
        return np.tensordot(popx_sumt, self.metBase, (0, 0))

      
    @property
    def aZ_flux__z(self):
        '''
        Flux-weighted average metallicity, in voronoi zones.

            * Units: dimensionless
            * Shape: ``(N_zone)``
        '''
        if self.hasAlphaEnhancement:
            popx_sumt = self.popx.sum(axis=2).sum(axis=0)
        else:  
            popx_sumt = self.popx.sum(axis=0)
        popx_sum = popx_sumt.sum(axis=0)
        popx_sumt /= popx_sum
        return np.tensordot(popx_sumt, self.metBase, (0, 0))

      
    @property
    def aZ_flux__yx(self):
        '''
        Spatially resolved, flux-weighted average metallicity.

            * Units: dimensionless
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.aZ_flux__z, extensive=False)


    @property
    def integrated_aZ_mass(self):
        '''
        Mass-weighted average metallicity of the integrated spectrum.

            * Units: dimensionless
            *  Type: float
        '''
        if self.hasAlphaEnhancement:
            popmu_cor_sumt = self.integrated_popmu_cor.sum(axis=2).sum(axis=0)
        else:  
            popmu_cor_sumt = self.integrated_popmu_cor.sum(axis=0)
        popmu_cor_sum = popmu_cor_sumt.sum(axis=0)
        popmu_cor_sumt /= popmu_cor_sum
        return np.tensordot(popmu_cor_sumt, self.metBase, (0, 0))

      
    @property
    def aZ_mass__z(self):
        '''
        Mass-weighted average metallicity, in voronoi zones.

            * Units: dimensionless
            * Shape: ``(N_zone)``
        '''
        if self.hasAlphaEnhancement:
            popmu_cor_sumt = self.popmu_cor.sum(axis=2).sum(axis=0)
        else:  
            popmu_cor_sumt = self.popmu_cor.sum(axis=0)
        popmu_cor_sum = popmu_cor_sumt.sum(axis=0)
        popmu_cor_sumt /= popmu_cor_sum
        return np.tensordot(popmu_cor_sumt, self.metBase, (0, 0))

      
    @property
    def aZ_mass__yx(self):
        '''
        Spatially resolved, mass-weighted average metallicity.

            * Units: dimensionless
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.aZ_mass__z, extensive=False)

      
    @property
    def integrated_alogZ_flux(self):
        '''
        Flux-weighted average log of metallicity of the integrated spectrum.

            * Units: dimensionless
            *  Type: float
        '''
        if self.hasAlphaEnhancement:
            popx_sumt = self.integrated_popx.sum(axis=2).sum(axis=0)
        else:  
            popx_sumt = self.integrated_popx.sum(axis=0)
        popx_sum = popx_sumt.sum(axis=0)
        popx_sumt /= popx_sum
        return np.tensordot(popx_sumt, np.log10(self.metBase / 0.019), (0, 0))

      
    @property
    def alogZ_flux__z(self):
        '''
        Flux-weighted average log of metallicity, in voronoi zones.

            * Units: dimensionless
            * Shape: ``(N_zone)``
        '''
        if self.hasAlphaEnhancement:
            popx_sumt = self.popx.sum(axis=2).sum(axis=0)
        else:  
            popx_sumt = self.popx.sum(axis=0)
        popx_sum = popx_sumt.sum(axis=0)
        popx_sumt /= popx_sum
        return np.tensordot(popx_sumt, np.log10(self.metBase / 0.019), (0, 0))

      
    @property
    def alogZ_flux__yx(self):
        '''
        Spatially resolved, flux-weighted average log of metallicity.

            * Units: dimensionless
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.alogZ_flux__z, extensive=False)


    @property
    def integrated_alogZ_mass(self):
        '''
        Mass-weighted average log of metallicity of the integrated spectrum.

            * Units: dimensionless
            *  Type: float
        '''
        if self.hasAlphaEnhancement:
            popmu_cor_sumt = self.integrated_popmu_cor.sum(axis=2).sum(axis=0)
        else:  
            popmu_cor_sumt = self.integrated_popmu_cor.sum(axis=0)
        popmu_cor_sum = popmu_cor_sumt.sum(axis=0)
        popmu_cor_sumt /= popmu_cor_sum
        return np.tensordot(popmu_cor_sumt, np.log10(self.metBase / 0.019), (0, 0))

      
    @property
    def alogZ_mass__z(self):
        '''
        Mass-weighted average log of metallicity, in voronoi zones.

            * Units: dimensionless
            * Shape: ``(N_zone)``
        '''
        if self.hasAlphaEnhancement:
            popmu_cor_sumt = self.popmu_cor.sum(axis=2).sum(axis=0)
        else:  
            popmu_cor_sumt = self.popmu_cor.sum(axis=0)
        popmu_cor_sum = popmu_cor_sumt.sum(axis=0)
        popmu_cor_sumt /= popmu_cor_sum
        return np.tensordot(popmu_cor_sumt, np.log10(self.metBase / 0.019), (0, 0))

      
    @property
    def alogZ_mass__yx(self):
        '''
        Spatially resolved, mass-weighted average log of metallicity.

            * Units: dimensionless
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.alogZ_mass__z, extensive=False)

      
    @property
    def adev__yx(self):
        '''
        Mean absolute relative deviation, in percent, only
        for the ``Nl_eff`` points actually used in the synthesis.

            * Units: :math:`[\%]`
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.adev, extensive=False)


    @property
    def chi2__yx(self):
        '''
        :math:`\chi^2 / Nl_{eff}` of the fit.

            * Units: dimensionless
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.chi2, extensive=False)


    @property
    def f_obs__lyx(self):
        '''
        Spatially resolved observed flux
        (input spectra for the synthesis).

            * Units: :math:`[erg / s / cm^2 / \overset{\circ}{A}]`
            * Shape: ``(Nl_obs, N_y, N_x)``
        '''
        return self.zoneToYX(self.f_obs, extensive=True, surface_density=False)

      
    @property
    def f_flag__lyx(self):
        '''
        Spatially resolved flagged spaxels.
        
        FIXME: describe flags.

            * Units: dimensionless
            * Shape: ``(Nl_obs, N_y, N_x)``
        '''
        return self.zoneToYX(self.f_flag, extensive=False)

      
    @property
    def f_syn__lyx(self):
        '''
        Spatially resolved synthetic spectra.

            * Units: :math:`[erg / s / cm^2 / \overset{\circ}{A}]`
            * Shape: ``(Nl_obs, N_y, N_x)``
        '''
        return self.zoneToYX(self.f_syn, extensive=True, surface_density=False)

      
    @property
    def f_wei__lyx(self):
        '''
        Spatially resolved weight of the spaxels in
        the input spectra. This is the weight actually used
        by the synthesis, after clipping, etc.

        FIXME: describe flags and weights.

            * Units: dimensionless
            * Shape: ``(Nl_obs, N_y, N_x)``
        '''
        return self.zoneToYX(self.f_wei, extensive=False)

      
    @property
    def f_err__lyx(self):
        '''
        Spatially resolved error in observed spetra.

            * Units: :math:`[erg / s / cm^2 / \overset{\circ}{A}]`
            * Shape: ``(Nl_obs, N_y, N_x)``
        '''
        return np.sqrt(self.zoneToYX(self.f_err**2, extensive=True, surface_density=False))

    
    @property
    def logAgeBaseBins(self):
        '''
        Age bin edges, in log. Computes the bin edges as the
        bissection of the bins, expanding the borders accordingly.

            * Units: :math:`[\log age / yr]`
            * Shape: ``(N_age + 1)``
        '''
        logAge = np.log10(self.ageBase)
        bin_logAge = np.empty(logAge.size + 1)
        bin_logAge[1:-1] = (logAge[:-1] + logAge[1:]) / 2.0
        bin_logAge[0] = logAge[0] - (logAge[1] - logAge[0]) / 2.0
        bin_logAge[-1] = logAge[-1] + (logAge[-1] - logAge[-2]) / 2.0
        return bin_logAge
        
        
    @property
    def adevS(self):
        '''
        From Cid @ 26/05/2012:
        Here's my request for pycasso reader: That it defines a new figure of
        merit analogous to ``adev``, but which uses the synthetic flux in the
        denominator instead of the observed one. This is the ``adevS__z`` thing
        defined below in awful python. Why? Well, despite all our care there
        are still some non-flagged pixels with very low fluxes, which screws
        up adev, and this alternative definition fixes it.
        
        Original code:
        
        >>> adevS__z = np.zeros((self.nZones))
        >>> for i_z in np.arange(self.nZones):
        >>>    _a = np.abs(self.f_obs[:,i_z] - self.f_syn[:,i_z]) / self.f_syn[:,i_z]
        >>>    _b = _a[self.f_wei[:,i_z] > 0]
        >>>    adevS__z[i_z] = 100.0 * _b.mean()
        >>> return adevS__z
        
        Returns
        -------
        aDevS : array of length (nZones)

        '''
        _a__lz = ma.array(np.abs(self.f_obs - self.f_syn) / self.f_syn, mask=(self.f_wei <= 0))
        return 100.0 * _a__lz.mean(axis=0).data


    @property
    def integrated_adevS(self):
        _a__l = np.abs(self.integrated_f_obs - self.integrated_f_syn) / self.integrated_f_syn
        _a__l = ma.array(_a__l, mask=(self.integrated_f_wei <= 0))
        return 100.0 * _a__l.mean()

        
    def integrated_fluxRatio(self, lrl, lru, lbl, lbu):
        '''
        FIXME:

            * Units: dimensionless
            *  Type: float
        '''
        red = (self.l_obs > lrl) & (self.l_obs < lru)
        blue = (self.l_obs > lbl) & (self.l_obs < lbu)
        f_obs_red__l = np.ma.masked_where(self.integrated_f_flag[red] > 0.0, self.integrated_f_obs[red], copy=True)
        f_obs_blue__l = np.ma.masked_where(self.integrated_f_flag[blue] > 0.0, self.integrated_f_obs[blue], copy=True)
        
        f_obs_red = np.trapz(f_obs_red__l, self.l_obs[red])
        f_obs_blue = np.trapz(f_obs_blue__l, self.l_obs[blue])
        
        return f_obs_red / f_obs_blue

      
    def fluxRatio__z(self, lrl, lru, lbl, lbu):
        '''
        FIXME:

            * Units: dimensionless
            * Shape: ``(N_zone)``
        '''
        red = (self.l_obs > lrl) & (self.l_obs < lru)
        blue = (self.l_obs > lbl) & (self.l_obs < lbu)
        f_obs_red__lz = np.ma.masked_where(self.f_flag[red] > 0.0, self.f_obs[red], copy=True)
        f_obs_blue__lz = np.ma.masked_where(self.f_flag[blue] > 0.0, self.f_obs[blue], copy=True)
        
        f_obs_red__z = np.trapz(f_obs_red__lz, self.l_obs[red], axis=0)
        f_obs_blue__z = np.trapz(f_obs_blue__lz, self.l_obs[blue], axis=0)
        
        return f_obs_red__z / f_obs_blue__z

      
    def fluxRatio__yx(self, lrl, lru, lbl, lbu):
        '''
        FIXME:

            * Units: dimensionless
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.fluxRatio__z(lrl, lru, lbl, lbu), extensive=False)


    @property
    def integrated_Dn4000__z(self):
        '''
        FIXME:

            * Units: dimensionless
            * Type: float
        '''
        return self.integrated_fluxRatio__z(4000, 4100, 3850, 3950)


    @property
    def Dn4000__z(self):
        '''
        FIXME:

            * Units: dimensionless
            * Shape: ``(N_zone)``
        '''
        return self.fluxRatio__z(4000, 4100, 3850, 3950)
    

    @property
    def Dn4000__yx(self):
        '''
        FIXME:

            * Units: dimensionless
            * Shape: ``(N_y, N_x)``
        '''
        return self.zoneToYX(self.Dn4000__z, extensive=False)

    
    @property
    def zoneDistance_pix(self):
        '''
        FIXME:
        
            * Units: pixels
            * Shape: ``(N_zone)``
        '''
        return getDistance(self.zonePos['x'], self.zonePos['y'], self.x0, self.y0, pa=self.pa, ba=self.ba)

      
    @property
    def zoneDistance_HLR(self):
        '''
        FIXME:
        
            * Units: HLR
            * Shape: ``(N_zone)``
        '''
        return self.zoneDistance_pix / self.pixelsPerHLR


    @property
    def zoneDistance_pc(self):
        '''
        FIXME:
        
            * Units: parsecs
            * Shape: ``(N_zone)``
        '''
        return self.zoneDistance_pix * self.parsecPerPixel
