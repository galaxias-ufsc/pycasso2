'''
Created on 22/06/2015

@author: andre
'''

from .wcs import get_wavelength_coordinates, get_celestial_coordinates, copy_WCS, get_reference_pixel, get_shape
from .wcs import get_pixel_area_srad, get_pixel_scale_rad, get_wavelength_sampling, get_Nx, get_Ny, get_Nwave
from .starlight.synthesis import get_base_grid
from .starlight.analysis import smooth_Mini, SFR
from .lick import get_Lick_index
from .geometry import radial_profile, get_ellipse_params, get_image_distance, get_half_radius
from .resampling import find_nearest_index
from . import flags

from astropy.io import fits
from astropy.utils.decorators import lazyproperty
from astropy import log, wcs
import numpy as np

__all__ = ['FitsCube']


def safe_getheader(f, ext=0):
    with fits.open(f) as hl:
        hdu = hl[ext]
        hdu.verify('fix')
        return hdu.header


def get_median_flux(wave, flux, wave1, wave2):
    l1, l2 = find_nearest_index(wave, [wave1, wave2])
    return np.median(flux[l1:l2], axis=0)
        

class FitsCube(object):
    _ext_f_obs = 'F_OBS'
    _ext_f_err = 'F_ERR'
    _ext_f_syn = 'F_SYN'
    _ext_f_wei = 'F_WEI'
    _ext_f_flag = 'F_FLAG'

    _ext_popZ_base = 'POPZ_BASE'
    _ext_popage_base = 'POPAGE_BASE'
    _ext_popaFe_base = 'POPAFE_BASE'
    _ext_popx = 'POPX'
    _ext_popmu_ini = 'POPMU_INI'
    _ext_popmu_cor = 'POPMU_COR'
    _ext_mstars = 'MSTARS'
    _ext_fbase_norm = 'FBASE_NORM'
    
    _h_lum_dist_Mpc = 'PIPE LUM_DIST_MPC'
    _h_redshift = 'PIPE REDSHIFT'
    _h_flux_unit = 'PIPE FLUX_UNIT'
    _h_name = 'PIPE CUBE_NAME'
    
    # FIXME: remove legacy code
    _ext_old_f_obs = 'PRIMARY'
    _h_old_name = 'PIPE OBJECT_NAME'
    
    _Z_sun = 0.019
    
    _ext_keyword_list = ['Lobs_norm', 'Mini_tot', 'Mcor_tot', 'fobs_norm',
                         'A_V', 'q_norm', 'v_0', 'v_d', 'adev', 'Ntot_clipped',
                         'Nglobal_steps', 'chi2', 'SN_normwin']
    
    def __init__(self, cubefile=None):
        self._pop_len = None
        if cubefile is None:
            return
        self._load(cubefile)
        
        
    def _fix_name(self):
        # FIXME: remove legacy code
        if self._h_name in self._header:
            return
        if self._h_old_name in self._header:
            name = self._header[self._h_old_name] 
            log.warn('Using old cube name header. This check will be removed in the future.')
            self.name = name
        
        
    def _fix_f_obs_ext(self):
        # FIXME: remove legacy code
        if self._ext_f_obs in self._HDUList:
            return
        if self._ext_old_f_obs in self._HDUList:
            self._HDUList[self._ext_old_f_obs].update_ext_name(self._ext_f_obs)
            log.warn('Fixed primary extension name. This check will be removed in the future.')
        
        
    def _initFits(self, f_obs, f_err, f_flag, header):
        phdu = fits.PrimaryHDU(f_obs, header)
        phdu.update_ext_name(FitsCube._ext_f_obs)
        self._HDUList = fits.HDUList([phdu])
        self._header = phdu.header
        self._wcs = wcs.WCS(self._header)
        self._addExtension(FitsCube._ext_f_err, data=f_err, kind='spectra')
        self._addExtension(FitsCube._ext_f_flag, data=f_flag, kind='spectra')
        self._initMasks()
        self._calcEllipseParams()

    
    def _initMasks(self):
        self.synthImageMask = self.getSpatialMask(flags.no_obs | flags.no_starlight)
        self.synthSpectraMask = (self.f_flag & flags.no_starlight) > 0
        self.spectraMask = (self.f_flag & flags.no_obs) > 0
    
     
    def _calcEllipseParams(self, image=None):
        if image is None:
            image = self.flux_norm_window
        self.pa, self.ba = get_ellipse_params(image, self.x0, self.y0)
    
    
    def _load(self, cubefile):
        self._HDUList = fits.open(cubefile, memmap=True)
        self._fix_f_obs_ext()
        self._header = self._HDUList[self._ext_f_obs].header
        self._fix_name()
        self._wcs = wcs.WCS(self._header)
        self._initMasks()
        self._calcEllipseParams()
        
        
    def write(self, filename, overwrite=False):
        self._HDUList.writeto(filename, clobber=overwrite, output_verify='fix')
    
    
    def createSynthesisCubes(self, pop_len):
        self._pop_len = pop_len
        self._addExtension(self._ext_f_syn, kind='spectra', overwrite=True)
        self._addExtension(self._ext_f_wei, kind='spectra', overwrite=True)
        self._addExtension(self._ext_popx, kind='population', overwrite=True)
        self._addExtension(self._ext_popmu_ini, kind='population', overwrite=True)
        self._addExtension(self._ext_popmu_cor, kind='population', overwrite=True)
        self._addExtension(self._ext_popage_base, kind='base', overwrite=True)
        self._addExtension(self._ext_popZ_base, kind='base', overwrite=True)
        self._addExtension(self._ext_popaFe_base, kind='base', overwrite=True)
        self._addExtension(self._ext_mstars, kind='base', overwrite=True)
        self._addExtension(self._ext_fbase_norm, kind='base', overwrite=True)

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
            copy_WCS(self._header, hdu.header, axes=[1, 2])
        elif kind == 'base':
            pass
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
            return (self._pop_len,) + spectra_shape[1:]
        elif kind == 'base':
            if self._pop_len is None:
                raise Exception('Undefined population vector length.')
            return (self._pop_len,)
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
            data = np.ma.array(data, mask=self.synthImageMask, copy=False)
        if data.ndim == 3:
            data = np.ma.array(data, copy=False)
            data[:, self.synthImageMask] = np.ma.masked
        return data
    
    
    def toRectBase(self, a, fill_value=0.0):
        shape = (self._baseMask.shape) + a.shape[1:]
        a__Zt = np.ma.masked_all(shape, dtype=a.dtype)
        a__Zt.fill_value = fill_value
        a__Zt[self._baseMask, ...] = a
        if a__Zt.ndim == 2:
            return a__Zt.T
        elif a__Zt.ndim == 4:
            return a__Zt.transpose(1, 0, 2, 3)
        else:
            raise Exception('Unsupported number of dimensions.')


    def radialProfile(self, prop, bin_r, x0=None, y0=None, pa=None, ba=None,
                      rad_scale=1.0, mode='mean', return_npts=False):
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
        if pa is None:
            pa = self.pa
        if ba is None:
            ba = self.ba
        return radial_profile(prop, bin_r, x0, y0, pa, ba, rad_scale, mode, return_npts)
    
    
    @property
    def x0(self):
        return self.center[2]


    @property
    def y0(self):
        return self.center[1]


    @lazyproperty
    def Nx(self):
        return get_Nx(self._header)
    
    
    @lazyproperty
    def Ny(self):
        return get_Ny(self._header)
    
    
    @lazyproperty
    def Nwave(self):
        return get_Nwave(self._header)
    
    
    @lazyproperty
    def f_obs(self):
        data = self._getExtensionData(self._ext_f_obs)
        return np.ma.array(data, mask=self.spectraMask, copy=False)
    

    @lazyproperty
    def f_err(self):
        data = self._getExtensionData(self._ext_f_err)
        return np.ma.array(data, mask=self.spectraMask, copy=False)
    

    @lazyproperty
    def f_syn(self):
        data = self._getExtensionData(self._ext_f_syn)
        data = np.ma.array(data, copy=False)
        data[self.synthSpectraMask] = np.ma.masked
        return data
    

    @lazyproperty
    def f_wei(self):
        data = self._getExtensionData(self._ext_f_wei)
        data = np.ma.array(data, copy=False)
        data[self.synthSpectraMask] = np.ma.masked
        return data
    

    @lazyproperty
    def f_flag(self):
        return self._getExtensionData(self._ext_f_flag)

    
    @lazyproperty
    def popage_base(self):
        return self._getExtensionData(self._ext_popage_base)

    
    @lazyproperty
    def age_base(self):
        return np.unique(self.popage_base)

    
    @lazyproperty
    def popZ_base(self):
        return self._getExtensionData(self._ext_popZ_base)

    
    @lazyproperty
    def Z_base(self):
        return np.unique(self.popZ_base)

    
    @lazyproperty
    def popaFe_base(self):
        return self._getExtensionData(self._ext_popaFe_base)

    
    @lazyproperty
    def aFe_base(self):
        return np.unique(self.popaFe_base)

    
    @lazyproperty
    def _baseMask(self):
        base_mask, _, _ = get_base_grid(self.popage_base, self.popZ_base)
        return base_mask


    @lazyproperty
    def Mstars(self):
        return self._getExtensionData(self._ext_mstars)

    
    @lazyproperty
    def fbase_norm(self):
        return self._getExtensionData(self._ext_fbase_norm)

    
    @lazyproperty
    def popx(self):
        return self._getSynthExtension(self._ext_popx)

    
    @lazyproperty
    def popmu_ini(self):
        return self._getSynthExtension(self._ext_popmu_ini)

    
    @lazyproperty
    def popmu_cor(self):
        return self._getSynthExtension(self._ext_popmu_cor)
    
    
    @lazyproperty
    def pixelArea_pc2(self):
        lum_dist_pc = self.lumDistMpc * 1e6
        solid_angle = get_pixel_area_srad(self._wcs)
        return solid_angle * lum_dist_pc * lum_dist_pc
    
    
    @lazyproperty
    def pixelScale_pc(self):
        lum_dist_pc = self.lumDistMpc * 1e6
        angle = get_pixel_scale_rad(self._wcs)
        return angle * lum_dist_pc
    
    
    @lazyproperty
    def HLR(self):
        r = get_image_distance((self.Ny, self.Nx), self.x0, self.y0, self.pa, self.ba)
        return get_half_radius(self.flux_norm_window, r)
     
    
    @lazyproperty
    def Mcor_tot(self):
        return self._getSynthExtension('MCOR_TOT')

    
    @lazyproperty
    def Mini_tot(self):
        return self._getSynthExtension('MINI_TOT')

    
    @lazyproperty
    def Lobs_norm(self):
        return self._getSynthExtension('LOBS_NORM')

    
    @lazyproperty
    def flux_norm_window(self):
        norm_lambda = 5635.0
        return get_median_flux(self.l_obs, self.f_obs, norm_lambda - 50.0, norm_lambda + 50.0)
    
    
    @property
    def McorSD(self):
        popmu_cor = self.popmu_cor.copy()
        popmu_cor /= popmu_cor.sum(axis=0)[np.newaxis, ...]
        return popmu_cor * (self.Mcor_tot[np.newaxis, ...] / self.pixelArea_pc2) 

    
    @property
    def MiniSD(self):
        popmu_ini = self.popmu_ini.copy()
        popmu_ini /= popmu_ini.sum(axis=0)[np.newaxis, ...]
        return popmu_ini * (self.Mini_tot[np.newaxis, ...] / self.pixelArea_pc2) 

    
    @property
    def LobnSD(self):
        popx = self.popx.copy()
        popx /= popx.sum(axis=0)[np.newaxis, ...]
        return popx * (self.Lobs_norm[np.newaxis, ...] / self.pixelArea_pc2) 

    
    @property
    def at_flux(self):
        popx = self.popx
        popage_base = self.popage_base[:, np.newaxis, np.newaxis]
        return (popx * np.log10(popage_base)).sum(axis=0) / popx.sum(axis=0)

    
    @property
    def at_mass(self):
        mu = self.popmu_cor
        popage_base = self.popage_base[:, np.newaxis, np.newaxis]
        return (mu * np.log10(popage_base)).sum(axis=0) / mu.sum(axis=0)

    
    @property
    def alogZ_flux(self):
        popx = self.popx
        popZ_base = self.popZ_base[:, np.newaxis, np.newaxis]
        return (popx * np.log10(popZ_base / self._Z_sun)).sum(axis=0) / popx.sum(axis=0)

    
    @property
    def alogZ_mass(self):
        mu = self.popmu_cor
        popZ_base = self.popZ_base[:, np.newaxis, np.newaxis]
        return (mu * np.log10(popZ_base / self._Z_sun)).sum(axis=0) / mu.sum(axis=0)

    
    @property
    def aaFe_flux(self):
        popx = self.popx
        popaFe_base = self.popaFe_base[:, np.newaxis, np.newaxis]
        return (popx * popaFe_base).sum(axis=0) / popx.sum(axis=0)

    
    @property
    def aaFe_mass(self):
        popmu_cor = self.popmu_cor
        popaFe_base = self.popaFe_base[:, np.newaxis, np.newaxis]
        return (popmu_cor * popaFe_base).sum(axis=0) / popmu_cor.sum(axis=0)

    
    def SFRSD(self, dt=0.5e9):
        Mini = self.toRectBase(self.MiniSD).sum(axis=1)
        return SFR(Mini, self.age_base, dt)
    
    
    def SFRSD_smooth(self,  logtc_step=0.05, logtc_FWHM=0.5, dt=0.5e9):
        logtb = np.log10(self.age_base)
        logtc = np.arange(logtb.min(), logtb.max() + logtc_step, logtc_step)
        popx = self.toRectBase(self.popx)
        fbase_norm = self.toRectBase(self.fbase_norm)
        Mini = smooth_Mini(popx, fbase_norm, self.Lobs_norm,
                           self.q_norm, self.A_V,
                           logtb, logtc, logtc_FWHM)
        Mini = Mini.sum(axis=1) / self.pixelArea_pc2
        tc = 10.0**logtc
        return SFR(Mini, tc, dt)


    @lazyproperty
    def A_V(self):
        return self._getSynthExtension('A_V')

    
    @lazyproperty
    def q_norm(self):
        return self._getSynthExtension('q_norm')

    
    @property
    def tau_V(self):
        return self.A_V / (2.5 * np.log10(np.exp(1.)))

    
    @lazyproperty
    def v_0(self):
        return self._getSynthExtension('V_0')
    
    
    @lazyproperty
    def v_d(self):
        return self._getSynthExtension('V_D')
    
    
    @lazyproperty
    def chi2(self):
        return self._getSynthExtension('CHI2')


    @lazyproperty
    def adev(self):
        return self._getSynthExtension('ADEV')
    
    
    @lazyproperty
    def SN_normwin(self):
        return self._getSynthExtension('SN_NORMWIN')
    
    
    @property
    def Nclipped(self):
        return (self.f_wei == -1.0).astype('float').sum(axis=0)
    
    
    @lazyproperty
    def l_obs(self):
        return get_wavelength_coordinates(self._wcs, self.Nwave)


    @lazyproperty
    def dl(self):
        return get_wavelength_sampling(self._wcs)


    @lazyproperty
    def celestial_coords(self):
        return get_celestial_coordinates(self._wcs, self.Nx, self.Ny, relative=True)


    @lazyproperty
    def center(self):
        return get_reference_pixel(self._wcs)


    @property
    def flux_unit(self):
        key = self._h_flux_unit
        if not key in self._header:
            raise Exception('Flux unit not set. Header key: %s' % key)
        return self._header[key]

    @flux_unit.setter
    def flux_unit(self, value):
        key = 'HIERARCH %s' % self._h_flux_unit
        self._header[key] = value
    
    
    @property
    def lumDistMpc(self):
        key = self._h_lum_dist_Mpc
        if not key in self._header:
            raise Exception('Luminosity distance not set. Header key: %s' % key)
        return self._header[key]


    @lumDistMpc.setter
    def lumDistMpc(self, value):
        key = 'HIERARCH %s' % self._h_lum_dist_Mpc
        self._header[key] = value
    
    
    @property
    def redshift(self):
        key = self._h_redshift
        if not key in self._header:
            raise Exception('Redshift not set. Header key: %s' % key)
        return self._header[key]


    @redshift.setter
    def redshift(self, value):
        key = 'HIERARCH %s' % self._h_redshift
        self._header[key] = value
    
    
    @property
    def name(self):
        key = self._h_name
        if not key in self._header:
            raise Exception('Object name not set. Header key: %s' % key)
        return self._header[key]
    
    
    @name.setter
    def name(self, value):
        key = 'HIERARCH %s' % self._h_name
        self._header[key] = value
    
    
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
    
    
    def LickIndex(self, index_id, calc_error=False):
        if calc_error:
            return get_Lick_index(index_id, self.l_obs, self.f_obs, self.f_err)
        else:
            idx, _ = get_Lick_index(index_id, self.l_obs, self.f_obs, error=None)
            return idx
    
    
