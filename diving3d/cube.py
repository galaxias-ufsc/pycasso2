'''
Created on 22/06/2015

@author: andre
'''

from .resampling import resample_spectra, reshape_spectra
from .wcs import get_axis_coordinates, set_axis_WCS, copy_WCS, get_reference_pixel, get_shape, d3d_fix_crpix, \
get_pixel_area_srad, get_pixel_length_rad
from . import flags

from astropy.io import fits
from astropy import log
import numpy as np

__all__ = ['D3DFitsCube']


def safe_getheader(f, ext=0):
    with fits.open(f) as hl:
        hdu = hl[ext]
        hdu.verify('fix')
        return hdu.header
        

class D3DFitsCube(object):
    _masterlist_prefix = 'MASTERLIST'

    _ext_f_obs = 'PRIMARY'
    _ext_f_err = 'F_ERR'
    _ext_f_syn = 'F_SYN'
    _ext_f_wei = 'F_WEI'
    _ext_f_flag = 'F_FLAG'

    _ext_popZ_base = 'POPZ_BASE'
    _ext_popage_base = 'POPAGE_BASE'
    _ext_popx = 'POPX'
    _ext_popmu_ini = 'POPMU_INI'
    _ext_popmu_cor = 'POPMU_COR'
    
    _pop_len = None
    
    _ext_keyword_list = ['Lobs_norm', 'Mini_tot', 'Mcor_tot', 'fobs_norm',
                         'A_V', 'v_0', 'v_d', 'adev', 'Ntot_clipped',
                         'Nglobal_steps', 'chi2']
    
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
        d3dcube._addExtension(cls._ext_f_flag, data=f_flag, kind='spectra')
        d3dcube._addExtension(cls._ext_f_err, data=np.zeros_like(f_obs), overwrite=True)
        
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
    

    def _initMasks(self):
        self.synthesisMask = self.getSpatialMask(flags.starlight_failed_run)
        self.spectraMask = (self.f_flag & flags.no_obs) > 0
    
    
    def _load(self, cubefile):
        self._HDUList = fits.open(cubefile, memmap=True)
        self._header = self._HDUList[self._ext_f_obs].header
        self._populateMasterList()
        self._initMasks()
        
        
    def write(self, filename, overwrite=False):
        self._HDUList.writeto(filename, clobber=overwrite, output_verify='fix')
    
    
    def createSynthesisCubes(self, pop_len):
        self._pop_len = pop_len
        self._addExtension(self._ext_f_syn, overwrite=True)
        self._addExtension(self._ext_f_wei, overwrite=True)
        self._addExtension(self._ext_popage_base, kind='population', overwrite=True)
        self._addExtension(self._ext_popZ_base, kind='population', overwrite=True)
        self._addExtension(self._ext_popx, kind='population', overwrite=True)
        self._addExtension(self._ext_popmu_ini, kind='population', overwrite=True)
        self._addExtension(self._ext_popmu_cor, kind='population', overwrite=True)

        for ext in self._ext_keyword_list:
            self._addExtension(ext, kind='image', overwrite=True)
        
        
    def _hasExtension(self, name):
        return name in self._HDUList
    
     
    def _addExtension(self, name, data=None, dtype=None, kind='spectra', overwrite=False):
        name = name.upper()
        shape = self._getExtensionShape(kind)
        if data is None:
            if dtype is None:
                dtype='float64'
            data = np.zeros(shape, dtype=dtype)
        if self._hasExtension(name):
            if not overwrite:
                raise Exception('Tried to create extension %s but it already exists.' % name)
            else:
                log.warn('Deleting existing extension %s.' % name)
                self._delExtension(name)
        imhdu = fits.ImageHDU(data, name=name)
        self._setExtensionWCS(imhdu, kind)
        self._HDUList.append(imhdu)
    
    
    def _setExtensionWCS(self, hdu, kind):
        if kind == 'spectra':
            copy_WCS(self._header, hdu.header, axes=[1, 2, 3])
        elif kind == 'image':
            copy_WCS(self._header, hdu.header, axes=[1, 2])
        elif kind == 'population':
            copy_WCS(self._header, hdu.header, axes=[1, 2], dest_axes=[2, 3])
        else:
            raise Exception('Unknown extension kind "%s".' % kind)
    
    
    def _getExtensionShape(self, kind):
        spectra_shape = get_shape(self._header)
        if kind == 'spectra':
            return spectra_shape
        elif kind == 'image':
            return spectra_shape[1:]
        elif kind == 'population':
            if self._pop_len is None:
                raise Exception('Undefined population vector length.')
            return spectra_shape[1:] + (self._pop_len,)
        else:
            raise Exception('Unknown extension kind "%s".' % kind)
    
    
    def _getExtensionData(self, name):
        return self._HDUList[name].data
    
    
    def _delExtension(self, name):
        if not self._hasExtension(name):
            raise Exception('Extension %s not found.' % name)
        del self._HDUList[name]
    
    
    def _getSynthExtension(self, name):
        data = self._getExtensionData(name)
        if data.ndim == 2:
            data = np.ma.array(data, mask=self.synthesisMask)
        if data.ndim == 3:
            data = np.ma.array(data)
            data[self.synthesisMask] = np.ma.masked
        return data
    
    
    @property
    def f_obs(self):
        data = self._getExtensionData(self._ext_f_obs)
        return np.ma.array(data, mask=self.spectraMask)
    

    @property
    def f_err(self):
        data = self._getExtensionData(self._ext_f_err)
        return np.ma.array(data, mask=self.spectraMask)
    

    @property
    def f_syn(self):
        data = self._getExtensionData(self._ext_f_syn)
        data = np.ma.array(data)
        data[:, self.synthesisMask] = np.ma.masked
        return data
    

    @property
    def f_wei(self):
        data = self._getExtensionData(self._ext_f_wei)
        data = np.ma.array(data)
        data[:, self.synthesisMask] = np.ma.masked
    

    @property
    def f_flag(self):
        return self._getExtensionData(self._ext_f_flag)

    
    @property
    def popage_base(self):
        return self._getSynthExtension(self._ext_popage_base)

    
    @property
    def popZ_base(self):
        return self._getSynthExtension(self._ext_popZ_base)

    
    @property
    def popx(self):
        return self._getSynthExtension(self._ext_popx)

    
    @property
    def popmu_ini(self):
        return self._getSynthExtension(self._ext_popmu_ini)

    
    @property
    def popmu_cor(self):
        return self._getSynthExtension(self._ext_popmu_cor)
    
    
    @property
    def pixelArea_pc2(self):
        lum_dist_pc = self.masterlist['DL'] * 1e6
        solid_angle = get_pixel_area_srad(self._header)
        return solid_angle * lum_dist_pc * lum_dist_pc
    
    
    @property
    def pixelLength_pc(self):
        lum_dist_pc = self.masterlist['DL'] * 1e6
        angle = get_pixel_length_rad(self._header)
        return angle * lum_dist_pc
    
    
    @property
    def Mcor_tot(self):
        return self._getSynthExtension('MCOR_TOT')

    
    @property
    def Mini_tot(self):
        return self._getSynthExtension('MINI_TOT')

    
    @property
    def Lobs_norm(self):
        return self._getSynthExtension('LOBS_NORM')

    
    @property
    def McorSD(self):
        popmu_cor = self.popmu_cor.copy()
        popmu_cor /= popmu_cor.sum(axis=2)[..., np.newaxis]
        return popmu_cor * (self.Mcor_tot[..., np.newaxis] / self.pixelArea_pc2) 

    
    @property
    def MiniSD(self):
        popmu_ini = self.popmu_ini.copy()
        popmu_ini /= popmu_ini.sum()
        return popmu_ini * self.Mini_tot / self.pixelArea_pc2 

    
    @property
    def LobnSD(self):
        popx = self.popx.copy()
        popx /= popx.sum()
        return popx * self.Lobs_norm / self.pixelArea_pc2 

    
    @property
    def at_flux(self):
        popx = self.popx
        return (popx * np.log10(self.popage_base)).sum(axis=2) / popx.sum(axis=2)

    
    @property
    def A_V(self):
        return self._getSynthExtension('A_V')

    
    @property
    def v_0(self):
        return self._getSynthExtension('V_0')
    
    
    @property
    def v_d(self):
        return self._getSynthExtension('V_D')
    
    
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
    def center(self):
        return get_reference_pixel(self._header)


    @property
    def flux_unit(self):
        return self._header['PIPE FLUX_UNIT']
    
    
    @property
    def id(self):
        return self.masterlist['ID']
    
    
    @property
    def object_name(self):
        return self.masterlist['NAME']
    
    
    def getSpatialMask(self, flags, threshold=0.5):
        '''
        Return a spatial mask containing spaxels that have less than
        a given fraction of masked spectral pixels.
        
        Parameters
        ----------
        flags : int
            Flags to take into account when creating the mask
            
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
        flagged = ((self.f_flag & flags) > 0).astype(int).sum(axis=0)
        return flagged > threshold * len(self.l_obs)
