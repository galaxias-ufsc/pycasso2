'''
Created on 22/06/2015

@author: andre
'''

from astropy.io import fits
import numpy as np
from diving3d.resampling import resample_spectra, reshape_spectra

__all__ = ['D3DFitsCube']

def get_axis_coordinates(header, ax, dtype='float64'):
    N = header['NAXIS']
    if ax < 0 or ax > N:
        raise Exception('Axis %d not in range (1, %d)' % (ax, N))
    crpix = float(header['CRPIX%d' % ax]) - 1
    crval = header['CRVAL%d' % ax]
    cdelt = header['CDELT%d' % ax]
    naxis = header['NAXIS%d' % ax]
    c_ini = crval - crpix * cdelt
    c_fin = c_ini + cdelt * (naxis - 1)
    return np.linspace(c_ini, c_fin, naxis, dtype=dtype)


def set_axis_coordinates(header, ax, crpix=None, crval=None, cdelt=None, N=None):
    naxes = header['NAXIS']
    if ax < 0 or ax > naxes:
        raise Exception('Axis %d not in range (1, %d)' % (ax, N))
    if crpix is not None:
        header['CRPIX%d' % ax] = crpix + 1
    if crval is not None:
        header['CRVAL%d' % ax] = crval
    if cdelt is not None:
        header['CDELT%d' % ax] = cdelt
    if N is not None:
        header['NAXIS%d' % ax] = N


def get_reference_pixel(header):
    crval_x = float(header['CRPIX1']) - 1
    crval_y = float(header['CRPIX2']) - 1
    crval_l = float(header['CRPIX3']) - 1
    return (crval_l, crval_y, crval_x)


class D3DFitsCube(object):
    
    def __init__(self, cubefile=None):
        if cubefile is None:
            return
        self._load(cubefile)
        
        
    @classmethod
    def from_reduced(cls, redcube, **kwargs):
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
        header = fits.getheader(redcube)
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
        set_axis_coordinates(header, ax=1, crpix=new_center[2], N=new_shape[2])
        set_axis_coordinates(header, ax=2, crpix=new_center[1], N=new_shape[1])
        set_axis_coordinates(header, ax=3, crpix=0, crval=l_obs[0], cdelt=dl, N=new_shape[0])

        d3dcube = cls()
        d3dcube._initFits(f_obs, header)
        d3dcube._addExtension(f_flag, name='F_FLAG')
        
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

    
    def _addExtension(self, data, name, overwrite=False):
        if name in self._HDUList and not overwrite:
            raise Exception('Extension %s already exists, you may use overwrite=True.')
        self._HDUList.append(fits.ImageHDU(data, name=name))
    
    
    @property
    def f_obs(self):
        return self._HDUList['PRIMARY'].data
    

    @property
    def f_flag(self):
        return self._HDUList['F_FLAG'].data

    
    @property
    def l_obs(self):
        return get_axis_coordinates(self._header, 3)


    @property
    def y_coords(self):
        return get_axis_coordinates(self._header, 2)


    @property
    def x_coords(self):
        return get_axis_coordinates(self._header, 1)



