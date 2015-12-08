'''
Created on 22/06/2015

@author: andre
'''

from .resampling import bin_edges, hist_resample
from .wcs import get_axis_coordinates, copy_WCS, get_reference_pixel, get_shape, get_pixel_area_srad, get_pixel_length_rad
from .starlight.synthesis import get_base_grid
from .starlight.analysis import smooth_Mini
from . import flags

from astropy.io import fits
from astropy import log
import numpy as np

__all__ = ['FitsCube']


def safe_getheader(f, ext=0):
    with fits.open(f) as hl:
        hdu = hl[ext]
        hdu.verify('fix')
        return hdu.header
        

class FitsCube(object):
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
    _ext_mstars = 'MSTARS'
    _ext_fbase_norm = 'FBASE_NORM'
    
    _h_lum_dist_Mpc = 'PIPE LUM_DIST_MPC'
    _h_flux_unit = 'PIPE FLUX_UNIT'
    _h_object_name = 'PIPE OBJECT_NAME'
    
    _pop_len = None
    
    _Z_sun = 0.019
    
    _ext_keyword_list = ['Lobs_norm', 'Mini_tot', 'Mcor_tot', 'fobs_norm',
                         'A_V', 'q_norm', 'v_0', 'v_d', 'adev', 'Ntot_clipped',
                         'Nglobal_steps', 'chi2']
    
    def __init__(self, cubefile=None):
        if cubefile is None:
            return
        self._load(cubefile)
        
        
    def _initFits(self, f_obs, f_err, f_flag, header):
        phdu = fits.PrimaryHDU(f_obs, header)
        self._HDUList = fits.HDUList([phdu])
        self._header = phdu.header
        self._addExtension(FitsCube._ext_f_err, data=f_err, kind='spectra')
        self._addExtension(FitsCube._ext_f_flag, data=f_flag, kind='spectra')

    
    def _initMasks(self):
        self.synthesisMask = self.getSpatialMask(flags.no_obs | flags.starlight_failed_run | flags.starlight_masked_pix)
        self.spectraMask = (self.f_flag & flags.no_obs) > 0
    
    
    def _load(self, cubefile):
        self._HDUList = fits.open(cubefile, memmap=True)
        self._header = self._HDUList[self._ext_f_obs].header
        self._initMasks()
        
        
    def write(self, filename, overwrite=False):
        self._HDUList.writeto(filename, clobber=overwrite, output_verify='fix')
    
    
    def createSynthesisCubes(self, pop_len):
        self._pop_len = pop_len
        self._addExtension(self._ext_f_syn, kind='spectra', overwrite=True)
        self._addExtension(self._ext_f_wei, kind='spectra', overwrite=True)
        self._addExtension(self._ext_popx, kind='population', overwrite=True)
        self._addExtension(self._ext_popmu_ini, kind='population', overwrite=True)
        self._addExtension(self._ext_popmu_cor, kind='population', overwrite=True)
        self._addExtension(self._ext_popmu_cor, kind='population', overwrite=True)
        self._addExtension(self._ext_popage_base, kind='base', overwrite=True)
        self._addExtension(self._ext_popZ_base, kind='base', overwrite=True)
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
            data = np.ma.array(data, mask=self.synthesisMask, copy=False)
        if data.ndim == 3:
            data = np.ma.array(data, copy=False)
            data[:, self.synthesisMask] = np.ma.masked
        return data
    
    
    def _reshapeBase(self, a, fill_value=0.0):
        mask, _, _ = get_base_grid(self.popage_base, self.popZ_base)
        shape = (mask.shape) + a.shape[1:]
        a__Zt = np.ma.masked_all(shape, dtype=a.dtype)
        a__Zt.fill_value = fill_value
        a__Zt[mask, ...] = a
        if a__Zt.ndim == 2:
            return a__Zt.T
        elif a__Zt.ndim == 4:
            return a__Zt.transpose(1, 0, 2, 3)
        else:
            raise Exception('Unsupported number of dimensions.')


    @property
    def f_obs(self):
        data = self._getExtensionData(self._ext_f_obs)
        return np.ma.array(data, mask=self.spectraMask, copy=False)
    

    @property
    def f_err(self):
        data = self._getExtensionData(self._ext_f_err)
        return np.ma.array(data, mask=self.spectraMask, copy=False)
    

    @property
    def f_syn(self):
        data = self._getExtensionData(self._ext_f_syn)
        data = np.ma.array(data, copy=False)
        data[:, self.synthesisMask] = np.ma.masked
        return data
    

    @property
    def f_wei(self):
        data = self._getExtensionData(self._ext_f_wei)
        data = np.ma.array(data, copy=False)
        data[:, self.synthesisMask] = np.ma.masked
        return data
    

    @property
    def f_flag(self):
        return self._getExtensionData(self._ext_f_flag)

    
    @property
    def popage_base(self):
        return self._getExtensionData(self._ext_popage_base)

    
    @property
    def age_base(self):
        return np.unique(self.popage_base)

    
    @property
    def popZ_base(self):
        return self._getExtensionData(self._ext_popZ_base)

    
    @property
    def Z_base(self):
        return np.unique(self.popZ_base)

    
    @property
    def Mstars(self):
        return self._getExtensionData(self._ext_mstars)

    
    @property
    def fbase_norm(self):
        return self._getExtensionData(self._ext_fbase_norm)

    
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
        lum_dist_pc = self.lumDistMpc * 1e6
        solid_angle = get_pixel_area_srad(self._header)
        return solid_angle * lum_dist_pc * lum_dist_pc
    
    
    @property
    def pixelLength_pc(self):
        lum_dist_pc = self.lumDistMpc * 1e6
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

    
    def SFRSD(self, dt=0.5e9):
        logtb = np.log10(self.popage_base)
        logtb_bins = bin_edges(np.unique(logtb))
        tb_bins = 10**logtb_bins
        tl = np.arange(tb_bins.min(), tb_bins.max()+dt, dt)
        tl_bins = bin_edges(tl)
        Mini = self._reshapeBase(self.MiniSD).sum(axis=1)
        sfr_shape = (len(tl) + 2,) + Mini.shape[1:]
        sfr = np.zeros(sfr_shape)
        for j in xrange(sfr_shape[1]):
            for i in xrange(sfr_shape[2]):
                if Mini[:, j, i].mask.all():
                    continue
                Mini_resam = hist_resample(tb_bins, tl_bins, Mini[:, j, i])
                sfr[1:-1, j, i] = Mini_resam / dt
        
        tl = np.hstack((tl[0] - dt, tl, tl[-1] + dt))
        return sfr, tl
    
    
    def SFRSD_smooth(self,  logtc_ini=None, logtc_fin=None, logtc_step=0.05, logtc_FWHM=0.5, dt=0.5e9):
        logtb = np.unique(np.log10(self.popage_base))
        if logtc_ini is None:
            logtc_ini = logtb.min()
        if logtc_fin is None:
            logtc_fin = logtb.max()
        logtc = np.arange(logtc_ini, logtc_fin + logtc_step, logtc_step)
        popx = self._reshapeBase(self.popx)
        fbase_norm = self._reshapeBase(self.fbase_norm)
        Mini = smooth_Mini(popx, fbase_norm, self.Lobs_norm,
                           self.q_norm, self.A_V,
                           logtb, logtc, logtc_FWHM)
        Mini /= self.pixelArea_pc2
        logtc_bins = bin_edges(logtc)
        tc_bins = 10**logtc_bins
        tl = np.arange(tc_bins.min(), tc_bins.max()+dt, dt)
        tl_bins = bin_edges(tl)
        Mini = Mini.sum(axis=1)
        sfr_shape = (len(tl) + 2,) + Mini.shape[1:]
        sfr = np.zeros(sfr_shape)
        for j in xrange(sfr_shape[1]):
            for i in xrange(sfr_shape[2]):
                Mini_resam = hist_resample(tc_bins, tl_bins, Mini[:, j, i])
                sfr[1:-1, j, i] = Mini_resam / dt
        
        tl = np.hstack((tl[0] - dt, tl, tl[-1] + dt))
        return sfr, tl

    @property
    def A_V(self):
        return self._getSynthExtension('A_V')

    
    @property
    def q_norm(self):
        return self._getSynthExtension('q_norm')

    
    @property
    def tau_V(self):
        return self.A_V / (2.5 * np.log10(np.exp(1.)))

    
    @property
    def v_0(self):
        return self._getSynthExtension('V_0')
    
    
    @property
    def v_d(self):
        return self._getSynthExtension('V_D')
    
    
    @property
    def chi2(self):
        return self._getSynthExtension('CHI2')


    @property
    def adev(self):
        return self._getSynthExtension('ADEV')
    
    
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
    def objectName(self):
        key = self._h_object_name
        if not key in self._header:
            raise Exception('Object name not set. Header key: %s' % key)
        return self._header[key]
    
    
    @objectName.setter
    def objectName(self, value):
        key = 'HIERARCH %s' % self._h_object_name
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
