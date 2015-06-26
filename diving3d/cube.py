'''
Created on 22/06/2015

@author: andre
'''

from .resampling import resample_spectra, reshape_spectra
from .wcs import get_axis_coordinates, set_axis_WCS, copy_WCS, get_reference_pixel

from astropy.io import fits
import numpy as np

__all__ = ['D3DFitsCube']


def safe_getheader(f, ext=0):
    with fits.open(f) as hl:
        hdu = hl[ext]
        hdu.verify('fix')
        return hdu.header
        

class D3DFitsCube(object):
    
    def __init__(self, cubefile=None):
        self.keywords = {}
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

        # FIXME: sanitize file I/O
        header = safe_getheader(redcube)
        obs_header = safe_getheader(obscube)
        for k in obs_header.keys():
            if k in header or k == 'COMMENT' or k == '': continue
            header[k] = obs_header[k]
        
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
        d3dcube._addExtension(f_flag, name='F_FLAG', kind='cube')
        
        d3dcube.updateMasterList(ml)
        return d3dcube
    
    
    def _initFits(self, data, header):
        phdu = fits.PrimaryHDU(data, header)
        self._HDUList = fits.HDUList([phdu])
        self._header = phdu.header

    
    def updateMasterList(self, ml):
        ignored = ['cube', 'cube_obs']
        for key in ml.dtype.names:
            if key in ignored: continue
            self._header['HIERARCH MASTERLIST ' + key.upper()] = ml[key]
    
    
    def _load(self, cubefile):
        self._HDUList = fits.open(cubefile, memmap=True)
        self._header = self._HDUList['PRIMARY'].header
        
        
    def write(self, filename, overwrite=False):
        self._HDUList.writeto(filename, clobber=overwrite, output_verify='fix')

    
    def _addExtension(self, data, name, kind='cube', overwrite=False):
        if name in self._HDUList and not overwrite:
            raise Exception('Extension %s already exists, you may use overwrite=True.')
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
        return self._getExtensionData('PRIMARY')
    

    @property
    def f_flag(self):
        return self._getExtensionData('F_FLAG')

    
    @property
    def l_obs(self):
        return get_axis_coordinates(self._header, 3)


    @property
    def y_coords(self):
        return get_axis_coordinates(self._header, 2)


    @property
    def x_coords(self):
        return get_axis_coordinates(self._header, 1)



