'''
Created on 22/06/2015

@author: andre
'''

from .resampling import resample_spectra, reshape_spectra
from .wcs import get_axis_coordinates, set_axis_WCS, copy_WCS, get_reference_pixel, get_shape, d3d_fix_crpix

from astropy.io import fits
import numpy as np

__all__ = ['D3DFitsCube']


def safe_getheader(f, ext=0):
    with fits.open(f) as hl:
        hdu = hl[ext]
        hdu.verify('fix')
        return hdu.header
        

class D3DFitsCube(object):
    _ext_f_obs = 'PRIMARY'
    _ext_f_syn = 'F_SYN'
    _ext_f_wei = 'F_WEI'
    _ext_f_flag = 'F_FLAG'
    _masterlist_prefix = 'MASTERLIST'
    
    def __init__(self, cubefile=None):
        self.masterlist = {}
        if cubefile is None:
            return
        self._load(cubefile)
        
        
    @classmethod
    def from_reduced(cls, redcube, obscube, **kwargs):
        '''
        FIXME: doc me! 
        '''
        # FIXME: sanitize kwargs
        l_ini = kwargs['l_ini']
        l_fin = kwargs['l_fin']
        dl = kwargs['dl']
        Nx = kwargs['width']
        Ny = kwargs['height']
        ml = kwargs['ml']
        flux_unit = kwargs['flux_unit']

        # FIXME: sanitize file I/O
        header = safe_getheader(redcube)
        d3d_fix_crpix(header, 1)
        d3d_fix_crpix(header, 2)
        obs_header = safe_getheader(obscube)
        for k in obs_header.keys():
            if k in header or k == 'COMMENT' or k == '': continue
            header[k] = obs_header[k]
        header['HIERARCH PIPE FLUX_UNIT'] = flux_unit
        
        f_obs_orig = fits.getdata(redcube)
        
        # TODO: how to handle redshift?
        l_obs_orig = get_axis_coordinates(header, 3, dtype='float64')
        l_obs = np.arange(l_ini, l_fin + dl, dl)
        f_obs, f_flag = resample_spectra(f_obs_orig, l_obs_orig, l_obs)
        # FIXME: read gap and other flags
        
        new_shape = (len(l_obs), Ny, Nx)
        center = get_reference_pixel(header)
        f_obs, f_flag, new_center = reshape_spectra(f_obs, f_flag, center, new_shape)

        # Update WCS
        set_axis_WCS(header, ax=1, crpix=new_center[2], naxis=new_shape[2])
        set_axis_WCS(header, ax=2, crpix=new_center[1], naxis=new_shape[1])
        set_axis_WCS(header, ax=3, crpix=0, crval=l_obs[0], cdelt=dl, naxis=new_shape[0])

        d3dcube = cls()
        d3dcube._initFits(f_obs, header)
        d3dcube._addExtension('F_FLAG', data=f_flag, kind='cube')
        
        d3dcube._saveMasterList(ml)
        d3dcube._populateMasterList()
        return d3dcube
    
    
    def _initFits(self, data, header):
        phdu = fits.PrimaryHDU(data, header)
        self._HDUList = fits.HDUList([phdu])
        self._header = phdu.header

    
    def _saveMasterList(self, ml):
        header_ignored = ['cube', 'cube_obs']
        for key in ml.dtype.names:
            if key in header_ignored: continue
            hkey = 'HIERARCH %s %s' % (self._masterlist_prefix, key.upper())
            self._header[hkey] = ml[key]
    
    
    def _populateMasterList(self):
        for hkey in self._header.keys():
            if not hkey.startswith(self._masterlist_prefix): continue
            key = hkey.replace(self._masterlist_prefix, '')
            key = key.strip()
            self.masterlist[key] = self._header[hkey]
    
    
    def _load(self, cubefile):
        self._HDUList = fits.open(cubefile, memmap=True)
        self._header = self._HDUList[self._ext_f_obs].header
        self._populateMasterList()
        
        
    def write(self, filename, overwrite=False):
        self._HDUList.writeto(filename, clobber=overwrite, output_verify='fix')
    
    
    def createSynthesisCubes(self):
        if not self._hasExtension(self._ext_f_syn):
            self._addExtension(self._ext_f_syn)
        if not self._hasExtension(self._ext_f_wei):
            self._addExtension(self._ext_f_wei)
        
        
    def _hasExtension(self, name):
        return name in self._HDUList
    
     
    def _addExtension(self, name, data=None, dtype=None, kind='cube', overwrite=False):
        if self._hasExtension(name) and not overwrite:
            raise Exception('Extension %s already exists, you may use overwrite=True.' % name)
        if data is None:
            if dtype is None:
                dtype='float64'
            data = np.zeros(get_shape(self._header), dtype=dtype)
        imhdu = fits.ImageHDU(data, name=name)
        if kind == 'cube':
            copy_WCS(self._header, imhdu.header, axes=[1, 2, 3])
        else:
            raise Exception('Unknown extension kind "%s".' % kind)
        self._HDUList.append(imhdu)
    
    
    def _getExtensionData(self, name):
        return self._HDUList[name].data
    
    
    @property
    def f_obs(self):
        return self._getExtensionData(self._ext_f_obs)
    

    @property
    def f_syn(self):
        return self._getExtensionData(self._ext_f_syn)
    

    @property
    def f_wei(self):
        return self._getExtensionData(self._ext_f_wei)
    

    @property
    def f_flag(self):
        return self._getExtensionData(self._ext_f_flag)

    
    @property
    def l_obs(self):
        return get_axis_coordinates(self._header, 3)


    @property
    def y_coords(self):
        return get_axis_coordinates(self._header, 2)


    @property
    def x_coords(self):
        return get_axis_coordinates(self._header, 1)


    @property
    def flux_unit(self):
        return self._header['PIPE FLUX_UNIT']
    
    
    @property
    def id(self):
        return self.masterlist['ID']
    
    
    @property
    def object_name(self):
        return self.masterlist['NAME']
    
    
    def getSpatialMask(self, threshold=0.5):
        '''
        Return a spatial mask containing spaxels that have less than
        a given fraction of masked spectral pixels.
        
        Parameters
        ----------
        threshold : float, optional
            Fraction of spectral pixels that must be flagged
            in the spaxel for it to be masked.
            Default: ``0.5``
            
        Returns
        -------
        mask : array
            A 2-d boolean image with the same x and y dimensions
            as the cube, where ``True`` means the pixel is masked.
        '''
        flagged = (self.f_flag > 0).astype(int).sum(axis=0)
        return flagged > threshold * len(self.l_obs)
