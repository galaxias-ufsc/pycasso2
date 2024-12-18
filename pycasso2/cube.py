'''
Created on 22/06/2015

@author: andre
'''

from .wcs import get_wavelength_coordinates, get_celestial_coordinates, write_WCS, get_reference_pixel
from .wcs import get_pixel_scale, get_pixel_scale_rad, get_wavelength_sampling, get_Naxis
from .cosmology import get_angular_distance
from .starlight.synthesis import get_base_grid
from .starlight.analysis import smooth_Mini, SFR
from .starlight.io import pop_table_dtype, spec_table_dtype
from .starlight.plots import plot_spec
from .dobby.utils import el_lc_dtype, el_table_dtype
from .dobby.models.resampled_gaussian import ResampledGaussian
from .dobby.models.gaussian import Gaussian
from .lick import get_Lick_index
from .geometry import radial_profile, get_ellipse_params, get_image_distance, get_half_radius
from .resampling import find_nearest_index
from .segmentation import spatialize
from .importer import read_type, preprocess_obs
from .config import default_config_path, get_config
from . import flags
from . import modeling


from astropy.io import fits
from astropy.table import Table
from astropy.utils.decorators import lazyproperty
from astropy.wcs import WCS
from astropy import log
import numpy as np

__all__ = ['FitsCube']

class FitsCube(object):
    _ext_f_obs = 'F_OBS'
    _ext_f_err = 'F_ERR'
    _ext_f_syn = 'F_SYN'
    _ext_f_wei = 'F_WEI'
    _ext_f_flag = 'F_FLAG'
    _ext_f_disp = 'F_DISP'
    _ext_segmask = 'SEGMASK'
    _ext_seg_good_frac = 'SEG_GOOD_FRAC'
    _ext_fobs_norm = 'FOBS_NORM'
    
    _ext_popZ_base = 'POPZ_BASE'
    _ext_popage_base = 'POPAGE_BASE'
    _ext_popage_base_t2 = 'POPAGE_BASE_T2'
    _ext_popaFe_base = 'POPAFE_BASE'
    _ext_popx = 'POPX'
    _ext_popmu_ini = 'POPMU_INI'
    _ext_popmu_cor = 'POPMU_COR'
    _ext_mstars = 'MSTARS'
    _ext_fbase_norm = 'FBASE_NORM'
    
    _ext_integ_spectra = 'integ_spectra'
    _ext_integ_pop = 'integ_population'
    
    _ext_el_info = 'EL_INFO'
    _ext_el_flux = 'EL_F'
    _ext_el_flag = 'EL_FLAG'   
    _ext_el_ew = 'EL_EW'   
    _ext_el_v_0 = 'EL_V0'   
    _ext_el_v_d = 'EL_VD'   
    _ext_el_v_d_inst = 'EL_VDINS'   
    _ext_el_lc = 'EL_LC'   
    _ext_el_lc_rms = 'EL_LCRMS'   
    _ext_el_integ = 'EL_INTEG'   
    _ext_el_integ_lc = 'EL_INTEG_LC'   
    _ext_el_flux_integ = 'EL_F_INTEG'
    _ext_el_flux_imed = 'EL_F_IMED'
    
    _h_lum_dist_Mpc = 'PYCASSO LUM_DIST_MPC'
    _h_redshift = 'PYCASSO REDSHIFT'
    _h_flux_unit = 'PYCASSO FLUX_UNIT'
    _h_name = 'PYCASSO CUBE_NAME'
    _h_has_segmap = 'PYCASSO HAS_SEGMAP'

    _Z_sun = 0.019

    _ext_keyword_list = ['Lobs_norm', 'Mini_tot', 'Mcor_tot', 'fobs_norm',
                         'Av', 'exAv', 'v0', 'vd', 'adev', 'Ntot_clipped',
                         'Nglobal_steps', 'chi2', 'SN_normwin']

    def __init__(self, cubefile=None, name=None, cube_type='pycasso', mask_file=None, import_cfg=None,
                 memmap=True, mode='readonly'):
        self._pop_len = None
        if cubefile is None:
            # FIXME: needed by segmentation code, which should moved here.
            return
        if cube_type == 'pycasso':
            self._load(cubefile, memmap, mode)
            if name is not None:
                self.name = name
        else:
            if cube_type not in read_type.keys():
                raise Exception('Unknown cube type: %s.' % cube_type)
            read = read_type[cube_type]
            if import_cfg is None:
                import_cfg = get_config(default_config_path)
            log.info('Importing %s, type=%s' % (cubefile, cube_type))
            if type(cubefile) is not list:
                cubefile = [cubefile]
            obs = read(cubefile, name, import_cfg)
            self.mask_file = mask_file
            self._fromObs(obs, import_cfg)
        self._l_norm = None
        self._dl_norm = None

    def close(self):
        self._HDUList.close()
        
    def flush(self):
        self._HDUList.flush()

    def _initFits(self, f_obs, f_err, f_flag, header, wcs, segmask=None, good_frac=None, f_disp=None):
        phdu = fits.PrimaryHDU(header=header)
        phdu.name = 'PRIMARY'
        self._HDUList = fits.HDUList([phdu])
        self._header = phdu.header
        self._wcs = wcs
        if f_flag is None:
            f_flag = np.ones_like(f_obs, dtype='int32')
        if f_disp is None:
            f_disp = np.zeros_like(f_obs)
        if good_frac is not None:
            self._addExtension(FitsCube._ext_seg_good_frac, data=good_frac, wcstype='spectra')
        if segmask is not None:
            self.hasSegmentationMask = True
            f_obs = f_obs.T
            f_err = f_err.T
            f_flag = f_flag.T
            f_disp = f_disp.T
            good_frac = good_frac.T
            f_flag |= np.where(good_frac == 0, flags.no_data, 0)
            self._addExtension(FitsCube._ext_segmask, data=segmask, wcstype='segmask')
            self._addExtension(FitsCube._ext_seg_good_frac, data=good_frac, wcstype='spectra', overwrite=True)
        self._addExtension(FitsCube._ext_f_obs, data=f_obs, wcstype='spectra')
        self._addExtension(FitsCube._ext_f_err, data=f_err, wcstype='spectra')
        self._addExtension(FitsCube._ext_f_flag, data=f_flag, wcstype='spectra')
        self._addExtension(FitsCube._ext_f_disp, data=f_disp, wcstype='spectra')
        self._initMasks()
        self._readKeywords()

    def _initMasks(self):
        self.synthImageMask = self.getSpatialMask(
            flags.no_obs | flags.no_starlight)
        self.synthSpectraMask = (self.f_flag & flags.no_starlight) > 0
        self.spectraMask = (self.f_flag & flags.no_obs) > 0

    def _calcEllipseParams(self):
        image = self.flux_norm_window
        if self.hasSegmentationMask:
            try:
                image = self.spatialize(image, extensive=True)
            except:
                log.debug('Could not calculate ellipse parameters due to segmentation.')
                self.pa = None
                self.ba = None
                return
        self.pa, self.ba = get_ellipse_params(image, self.x0, self.y0)

    def _load(self, cubefile, memmap=True, mode='readonly'):
        self._HDUList = fits.open(cubefile, memmap=memmap, mode=mode)
        self._header = self._HDUList[0].header
        if self.hasSegmentationMask:
            log.debug('Cube has segmentation data.')
            w = WCS(naxis=3)
            w_xy = WCS(self._HDUList[self._ext_segmask].header).celestial
            w_l = WCS(self._HDUList[self._ext_f_obs].header).sub([1])
            w.wcs.pc[:2, :2] = w_xy.wcs.pc
            w.wcs.pc[2, 2] = w_l.wcs.pc
            w.wcs.crpix[:2] = w_xy.wcs.crpix
            w.wcs.crpix[2] = w_l.wcs.crpix
            w.wcs.crval[:2] = w_xy.wcs.crval
            w.wcs.crval[2] = w_l.wcs.crval
            w.wcs.ctype[0] = w_xy.wcs.ctype[0]
            w.wcs.ctype[1] = w_xy.wcs.ctype[1]
            w.wcs.ctype[2] = w_l.wcs.ctype[0]
            self._wcs = w
        else:
            self._wcs = WCS(self._HDUList[self._ext_f_obs].header)
        log.debug('Initializing masks.')
        self._initMasks()
        log.debug('Reading keywords.')
        self._readKeywords()

    def _fromObs(self, obs, cfg):
        preprocess_obs(obs, cfg, self.mask_file)
        self._initFits(obs.f_obs, obs.f_err, obs.f_flag, obs.header, obs.wcs, segmask=None, f_disp=obs.f_disp, good_frac=obs.good_frac)
        self.flux_unit = obs.flux_unit
        self.lumDistMpc = obs.lumDist_Mpc
        self.redshift = obs.redshift
        self.name = obs.name


    def _readKeywords(self):
        self.x0 = self.center[2]
        self.y0 = self.center[1]
        self.ba = 1.0
        self.pa = 0.0

        self.keywords = {k.split()[1]: v for k, v in self._header.items() if 'PYCASSO' in k}
        self.synthKeywords = {k.split()[1]: v for k, v in self._header.items() if 'STARLIGHT' in k}
        if self.hasIntegratedData:
            integ_h = self._HDUList[self._ext_integ_pop].header
            self.synthIntegKeywords = {k.split()[1]: v for k, v in integ_h.items() if 'STARLIGHT' in k}

    def write(self, filename, overwrite=False):
        self._HDUList.writeto(filename, overwrite=overwrite, output_verify='fix')

    def createSynthesisCubes(self, pop_len):
        self._pop_len = pop_len
        if self.hasSegmentationMask:
            spec_shape = (self.Nzone, self.Nwave)
            pop_shape = (self.Nzone, pop_len)
            kw_shape = (self.Nzone,)
        else:
            spec_shape = (self.Nwave, self.Ny, self.Nx)
            pop_shape = (pop_len, self.Ny, self.Nx)
            kw_shape = (self.Ny, self.Nx)
        base_shape = (pop_len,)
        self._addExtension(self._ext_f_syn, wcstype='spectra', shape=spec_shape, overwrite=True)
        self._addExtension(self._ext_f_wei, wcstype='spectra', shape=spec_shape, overwrite=True)
        self._addExtension(self._ext_popx, wcstype='image', shape=pop_shape, overwrite=True)
        self._addExtension(self._ext_popmu_ini, wcstype='image', shape=pop_shape, overwrite=True)
        self._addExtension(self._ext_popmu_cor, wcstype='image', shape=pop_shape, overwrite=True)
        self._addExtension(self._ext_popage_base, wcstype='image', shape=base_shape, overwrite=True)
        self._addExtension(self._ext_popage_base_t2, wcstype='image', shape=base_shape, overwrite=True)
        self._addExtension(self._ext_popZ_base, wcstype='image', shape=base_shape, overwrite=True)
        self._addExtension(self._ext_popaFe_base, wcstype='image', shape=base_shape, overwrite=True)
        self._addExtension(self._ext_mstars, wcstype='image', shape=base_shape, overwrite=True)
        self._addExtension(self._ext_fbase_norm, wcstype='image', shape=base_shape, overwrite=True)

        for ext in self._ext_keyword_list:
            self._addExtension(ext, wcstype='image', shape=kw_shape, overwrite=True)

        if self.isSegmentationOverlapping:
            log.debug('Segmentation mask has overlapping regions, disabled integrated spectra.')
            return

        specdata = np.zeros((self.Nwave), dtype=spec_table_dtype)
        specdata['l_obs'] = self.l_obs
        self._addTableExtension(self._ext_integ_spectra, Table(specdata), overwrite=True)
        popdata = np.zeros((pop_len), dtype=pop_table_dtype)
        self._addTableExtension(self._ext_integ_pop, Table(popdata), overwrite=True)

    def createELinesCubes(self, el_info):
        self._addTableExtension(self._ext_el_info, data=el_info, overwrite=True)
        N_lines = len(el_info['lambda'])
        if self.hasSegmentationMask:
            spec_shape = (self.Nzone, self.Nwave)
            eline_shape = (self.Nzone, N_lines)
        else:
            spec_shape = (self.Nwave, self.Ny, self.Nx)
            eline_shape = (N_lines, self.Ny, self.Nx)        

        self._addExtension(self._ext_el_flux, wcstype='image', shape=eline_shape, overwrite=True)
        self._addExtension(self._ext_el_v_0, wcstype='image', shape=eline_shape, overwrite=True)
        self._addExtension(self._ext_el_v_d, wcstype='image', shape=eline_shape, overwrite=True)
        self._addExtension(self._ext_el_flag, wcstype='image', shape=eline_shape, dtype='int32', overwrite=True)
        self._addExtension(self._ext_el_ew, wcstype='image', shape=eline_shape, overwrite=True)
        self._addExtension(self._ext_el_lc_rms, wcstype='image', shape=eline_shape, overwrite=True)
        self._addExtension(self._ext_el_v_d_inst, wcstype='image', shape=eline_shape, overwrite=True)
        self._addExtension(self._ext_el_lc, wcstype='image', shape=spec_shape, overwrite=True)
        self._addExtension(self._ext_el_flux_integ, wcstype='image', shape=eline_shape, overwrite=True)
        self._addExtension(self._ext_el_flux_imed, wcstype='image', shape=eline_shape, overwrite=True)

        if self.isSegmentationOverlapping:
            log.debug('Segmentation mask has overlapping regions, disabled integrated spectra.')
            return

        specdata = np.zeros((self.Nwave), dtype=el_lc_dtype)
        specdata['l_obs'] = self.l_obs
        self._addTableExtension(self._ext_el_integ_lc, Table(specdata), overwrite=True)
        el_data = np.zeros((N_lines), dtype=el_table_dtype)
        self._addTableExtension(self._ext_el_integ, Table(el_data), overwrite=True)

    def deleteELinesCubes(self):
        self._delExtension(self._ext_el_info)
        self._delExtension(self._ext_el_flux)
        self._delExtension(self._ext_el_v_0)
        self._delExtension(self._ext_el_v_d)
        self._delExtension(self._ext_el_flag)
        self._delExtension(self._ext_el_ew)
        self._delExtension(self._ext_el_lc_rms)
        self._delExtension(self._ext_el_v_d_inst)
        self._delExtension(self._ext_el_lc)
        self._delExtension(self._ext_el_integ_lc)
        self._delExtension(self._ext_el_integ)
        self._delExtension(self._ext_el_flux_integ)
        self._delExtension(self._ext_el_flux_imed)

    @lazyproperty
    def isSegmentationOverlapping(self):
        if not self.hasSegmentationMask:
            return False
        else:
            sum_segmask = self.segmentationMask.sum(axis=0)
            return (sum_segmask > 1).any()

    def _hasExtension(self, name):
        return name in self._HDUList

    def _addExtension(self, name, data=None, dtype=None, shape=None, wcstype=None, overwrite=False):
        name = name.upper()
        if data is None:
            if shape is None:
                raise Exception('shape must me specified if data is not set.')
            if dtype is None:
                dtype = 'float64'
            data = np.zeros(shape, dtype=dtype)
        if self._hasExtension(name):
            if not overwrite:
                raise Exception(
                    'Tried to create extension %s but it already exists.' % name)
            else:
                log.warn('Deleting existing extension %s.' % name)
                self._delExtension(name)
        imhdu = fits.ImageHDU(data, name=name)
        self._setExtensionWCS(imhdu, wcstype)
        self._HDUList.append(imhdu)

    def _setExtensionWCS(self, hdu, wcstype):
        if wcstype is None:
            w = None
        elif wcstype == 'spectra':
            if self.hasSegmentationMask:
                w = self._wcs.sub([3])
            else:
                w = self._wcs
        elif wcstype == 'image':
            if self.hasSegmentationMask:
                w = None
            else:
                w = self._wcs.celestial
        elif wcstype == 'segmask':
            w = self._wcs.celestial
        else:
            raise Exception ('Unknown WCS type %s.' % wcstype)
        write_WCS(hdu.header, w)
        
    def _addTableExtension(self, name, data=None, overwrite=False):
        name = name.upper()
        if self._hasExtension(name):
            if not overwrite:
                raise Exception(
                    'Tried to create extension %s but it already exists.' % name)
            else:
                log.warn('Deleting existing extension %s.' % name)
                self._delExtension(name)
        imhdu = fits.BinTableHDU(data.as_array(), name=name)
        self._HDUList.append(imhdu)

    def _getExtensionData(self, name):
        data = self._HDUList[name].data
        if (data.ndim > 1) and self.hasSegmentationMask and (name != self._ext_segmask):
            data = np.moveaxis(data, 0, -1)
        return data

    def _getTableExtensionData(self, name):
        data = self._HDUList[name].data
        return data

    def _delExtension(self, name):
        if not self._hasExtension(name):
            raise Exception('Extension %s not found.' % name)
        del self._HDUList[name]

    def _getSynthExtension(self, name):
        data = self._getExtensionData(name)
        if self.hasSegmentationMask:
            spatial_dims = 1
        else:
            spatial_dims = 2
        if data.ndim == spatial_dims:
            data = np.ma.array(data, mask=self.synthImageMask, copy=False)
        elif data.ndim == (spatial_dims + 1):
            data = np.ma.array(data, copy=False)
            data[:, self.synthImageMask] = np.ma.masked
        return data

    def toRectBase(self, a, fill_value=0.0):
        shape = (self._baseMask.shape) + a.shape[1:]
        a__Zt = np.ma.masked_all(shape, dtype=a.dtype)
        a__Zt.fill_value = fill_value
        a__Zt[self._baseMask, ...] = a
        return np.swapaxes(a__Zt, 0, 1)

    def spatialize(self, a, extensive=False):
        if not self.hasSegmentationMask:
            log.warn('Tried to spatialize a non-segmented cube.')
            return a
        return spatialize(a, self.segmentationMask, extensive)

    def radialProfile(self, prop, bin_r, x0=None, y0=None, pa=None, ba=None,
                      rad_scale=1.0, mode='mean', exact=True, return_npts=False):
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
        if pa is None:
            pa = self.pa
        if ba is None:
            ba = self.ba
        return radial_profile(prop, bin_r, x0, y0, pa, ba, rad_scale, mode, exact, return_npts)

    @property
    def hasSynthesis(self):
        return self._ext_f_syn in self._HDUList

    @property
    def hasSegmentationMask(self):
        key = self._h_has_segmap
        if not key in self._header:
            return False
        return bool(self._header[key])

    @hasSegmentationMask.setter
    def hasSegmentationMask(self, value):
        key = 'HIERARCH %s' % self._h_has_segmap
        self._header[key] = value

    @lazyproperty
    def isSpatializable(self):
        if not self.hasSegmentationMask:
            return True
        sum_segmask = self.segmentationMask.sum(axis=0)
        return (sum_segmask <= 1).all()

    @property
    def hasIntegratedData(self):
        return self._ext_integ_spectra in self._HDUList

    @property
    def hasELines(self):
        return self._ext_el_flux in self._HDUList

    @lazyproperty
    def Nx(self):
        if self.hasSegmentationMask:
            hdu = self._ext_segmask
        else:
            hdu = self._ext_f_obs
        return get_Naxis(self._HDUList[hdu].header, 1)

    @lazyproperty
    def Ny(self):
        if self.hasSegmentationMask:
            hdu = self._ext_segmask
        else:
            hdu = self._ext_f_obs
        return get_Naxis(self._HDUList[hdu].header, 2)

    @lazyproperty
    def Nwave(self):
        if self.hasSegmentationMask:
            axis = 1
        else:
            axis = 3
        return get_Naxis(self._HDUList[self._ext_f_obs].header, axis)

    @lazyproperty
    def Nzone(self):
        if not self.hasSegmentationMask:
            raise Exception('Cube does not have segmentantion mask.')
        return self.segmentationMask.shape[0]

    @lazyproperty
    def seg_good_frac(self):
        if self._ext_seg_good_frac not in self._HDUList:
            return None
        return self._getExtensionData(self._ext_seg_good_frac)
    
    @lazyproperty
    def segmentationMask(self):
        return self._getExtensionData(self._ext_segmask)
    
    @lazyproperty
    def zoneArea_pix(self):
        if self.hasSegmentationMask:
            return self.segmentationMask.sum(axis=(1, 2))
        else:
            return 1

    @lazyproperty
    def zoneArea_pc2(self):
        return self.zoneArea_pix * self.pixelArea_pc2

    @lazyproperty
    def fobs_norm(self):
        return self._getSynthExtension(self._ext_fobs_norm)

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
    def f_disp(self):
        return self._getExtensionData(self._ext_f_disp)
    
    @lazyproperty
    def popage_base(self):
        return self._getExtensionData(self._ext_popage_base)

    @lazyproperty
    def popage_base_t2(self):
        return self._getExtensionData(self._ext_popage_base_t2)

    @lazyproperty
    def age_base(self):
        return np.unique(self.popage_base)

    @lazyproperty
    def age_base_t2(self):
        return np.unique(self.popage_base_t2)

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
        return self.pixelScale_pc**2

    @lazyproperty
    def pixelScale_pc(self):
        angle = get_pixel_scale_rad(self._wcs)
        return get_angular_distance(self.redshift, angle)

    @lazyproperty
    def pixelScale_arcsec(self):
        return get_pixel_scale(self._wcs) * 3600.0

    @lazyproperty
    def HLR(self):
        image = self.flux_norm_window
        if self.hasSegmentationMask:
            image = self.spatialize(image, extensive=True)
        r = get_image_distance(
            (self.Ny, self.Nx), self.x0, self.y0, self.pa, self.ba)
        return get_half_radius(image, r)

    @lazyproperty
    def Mcor_tot(self):
        return self._getSynthExtension('MCOR_TOT')

    @lazyproperty
    def Mini_tot(self):
        return self._getSynthExtension('MINI_TOT')

    @lazyproperty
    def Lobs_norm(self):
        return self._getSynthExtension('LOBS_NORM')
    
    @property
    def l_norm(self):
        if self._l_norm is not None:
            return self._l_norm
        if not self.hasSynthesis:
            raise Exception('l_norm unset, and cube does not have synthesis data.')
        return self.synthKeywords['l_norm']

    @l_norm.setter
    def l_norm(self, l):
        self._l_norm = l
        
    @property
    def dl_norm(self):
        if self._dl_norm is not None:
            return self._dl_norm
        if not self.hasSynthesis:
            raise Exception('dl_norm not set, and cube does not have synthesis data.')
        return self.synthKeywords['lupp_norm'] - self.synthKeywords['llow_norm']

    @dl_norm.setter
    def dl_norm(self, dl):
        self._dl_norm = dl
        
    def statFluxWindow(self, flux, ll, dl):
        l1, l2 = find_nearest_index(self.l_obs, [ll - dl, ll + dl])
        y = modeling.cube_continuum(self.l_obs[l1:l2], flux[l1:l2], degr=1, niterate=0)
        return np.mean(y, axis=0), np.std(flux[l1:l2] - y, axis=0)

    @lazyproperty
    def flux_norm_window(self):
        flux, _ = self.statFluxWindow(self.f_obs, self.l_norm, self.dl_norm)
        return flux

    @lazyproperty
    def noise_norm_window(self):
        _, noise = self.statFluxWindow(self.f_obs, self.l_norm, self.dl_norm)
        return noise

    @property
    def McorSD(self):
        popmu_cor = self.popmu_cor.copy()
        popmu_cor /= 100.0
        return popmu_cor * (self.Mcor_tot[np.newaxis, ...] / self.zoneArea_pc2)

    @property
    def MiniSD(self):
        popmu_ini = self.popmu_ini.copy()
        popmu_ini /= 100.0
        return popmu_ini * (self.Mini_tot[np.newaxis, ...] / self.zoneArea_pc2)

    @property
    def LobnSD(self):
        popx = self.popx.copy()
        popx /= 100.0
        return popx * (self.Lobs_norm[np.newaxis, ...] / self.zoneArea_pc2)

    @property
    def at_flux(self):
        popx = np.moveaxis(self.popx, 0, -1)
        log_t1 = np.log10(self.popage_base)
        log_t2 = np.log10(self.popage_base_t2)
        log_t = (log_t1 + log_t2) / 2.0
        return (popx * log_t).sum(axis=-1) / popx.sum(axis=-1)

    @property
    def at_mass(self):
        mu = np.moveaxis(self.popmu_cor, 0, -1)
        log_t1 = np.log10(self.popage_base)
        log_t2 = np.log10(self.popage_base_t2)
        log_t = (log_t1 + log_t2) / 2.0
        return (mu * log_t).sum(axis=-1) / mu.sum(axis=-1)

    @property
    def alogZ_flux(self):
        popx = np.moveaxis(self.popx, 0, -1)
        return (popx * np.log10(self.popZ_base / self._Z_sun)).sum(axis=-1) / popx.sum(axis=-1)

    @property
    def alogZ_mass(self):
        mu = np.moveaxis(self.popmu_cor, 0, -1)
        return (mu * np.log10(self.popZ_base / self._Z_sun)).sum(axis=-1) / mu.sum(axis=-1)

    @property
    def aaFe_flux(self):
        popx = np.moveaxis(self.popx, 0, -1)
        return (popx * self.popaFe_base).sum(axis=-1) / popx.sum(axis=-1)

    @property
    def aaFe_mass(self):
        mu = np.moveaxis(self.popmu_cor, 0, -1)
        return (mu * self.popaFe_base).sum(axis=-1) / mu.sum(axis=-1)

    def recentSFRSD(self, t_SF=32e6, xY_min=0.03):
        young_pop = self.popage_base_t2 < t_SF
        sfr = self.MiniSD[young_pop].sum(axis=0) / t_SF
        xY = self.popx[young_pop].sum(axis=0)
        sfr[xY < xY_min] = np.ma.masked
        return sfr

    def SFRSD(self, dt=0.5e9):
        Mini = self.toRectBase(self.MiniSD).sum(axis=1)
        return SFR(Mini, tb1=self.age_base, tb2=self.age_base_t2, dt=dt)

    def SFRSD_smooth(self,  logtc_step=0.05, logtc_FWHM=0.5, dt=0.5e9):
        if not np.allclose(self.popage_base, self.popage_base_t2):
            log.warn('Smooth SFR not implemented properly for CSP. '
                     'Converting to SSP (deltas) and hoping for the best!')
        logtb = (np.log10(self.age_base) + np.log10(self.age_base_t2)) / 2
        logtc = np.arange(logtb.min(), logtb.max() + logtc_step, logtc_step)
        popx = self.toRectBase(self.popx)
        fbase_norm = self.toRectBase(self.fbase_norm)
        Mini = smooth_Mini(popx, fbase_norm, self.Lobs_norm,
                           self.q_norm, self.A_V,
                           logtb, logtc, logtc_FWHM)
        Mini = Mini.sum(axis=1) / self.pixelArea_pc2
        tc = 10.0**logtc
        return SFR(Mini, tc, dt=dt)

    @lazyproperty
    def A_V(self):
        return self._getSynthExtension('Av')

    @lazyproperty
    def exA_V(self):
        return self._getSynthExtension('exAv')
    
    @lazyproperty
    def q_norm(self):
        return self.synthKeywords['q_norm']

    @property
    def tau_V(self):
        return self.A_V / (2.5 * np.log10(np.exp(1.)))

    @lazyproperty
    def v_0(self):
        return self._getSynthExtension('v0')

    @lazyproperty
    def v_d(self):
        return self._getSynthExtension('vd')

    @lazyproperty
    def chi2(self):
        return self._getSynthExtension('CHI2')

    @lazyproperty
    def adev(self):
        return self._getSynthExtension('ADEV')

    @lazyproperty
    def Nglobal_steps(self):
        return self._getSynthExtension('Nglobal_steps')

    @lazyproperty
    def Ntot_clipped(self):
        return self._getSynthExtension('Ntot_clipped')

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
        return get_reference_pixel(self._wcs, as_int=True)

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
            raise Exception(
                'Luminosity distance not set. Header key: %s' % key)
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

    def getSpatialMask(self, flags, threshold=0.8):
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
            Default: ``0.8``

        Returns
        -------
        mask : array
            A 2-d boolean image with the same x and y dimensions
            as the cube, where ``True`` means the pixel is masked.
        '''
        flagged = ((self.f_flag & flags) > 0).astype(int).sum(axis=0)
        return flagged > (threshold * len(self.l_obs))

    def LickIndex(self, index_id, calc_error=False, use_obs=False):
        if use_obs:
            flux = self.f_obs
        elif self.hasSynthesis:
            flux = self.f_syn
        else:
            log.warn('Synthetic spectra not found, calculating Lick index on observed spectra.')
            flux = self.f_obs
        if calc_error:
            return get_Lick_index(index_id, self.l_obs, flux, self.f_err)
        else:
            idx, _ = get_Lick_index(
                index_id, self.l_obs, flux, error=None)
            return idx

    def integLickIndex(self, index_id, calc_error=False):
        if self.hasSynthesis:
            flux = self.integ_f_syn
        else:
            log.warn('Synthetic spectra not found, calculating Lick index on observed spectra.')
            flux = self.integ_f_obs
        if calc_error:
            return get_Lick_index(index_id, self.l_obs, flux, self.f_err)
        else:
            idx, _ = get_Lick_index(
                index_id, self.l_obs, flux, error=None)
            return idx

    @lazyproperty
    def integ_f_obs(self):
        t = self._getTableExtensionData(self._ext_integ_spectra)
        flagged = (self.integ_f_flag & flags.no_obs) > 0
        return np.ma.masked_where(flagged, t['f_obs'])
    
    @lazyproperty
    def integ_f_err(self):
        t = self._getTableExtensionData(self._ext_integ_spectra)
        flagged = (self.integ_f_flag & flags.no_obs) > 0
        return np.ma.masked_where(flagged, t['f_err'])
    
    @lazyproperty
    def integ_f_flag(self):
        t = self._getTableExtensionData(self._ext_integ_spectra)
        return t['f_flag']
    
    @lazyproperty
    def integ_f_disp(self):
        t = self._getTableExtensionData(self._ext_integ_spectra)
        flagged = (self.integ_f_flag & flags.no_obs) > 0
        return np.ma.masked_where(flagged, t['f_disp'])
    
    @lazyproperty
    def integ_f_syn(self):
        t = self._getTableExtensionData(self._ext_integ_spectra)
        flagged = (self.integ_f_flag & flags.no_starlight) > 0
        return np.ma.masked_where(flagged, t['f_syn'])
    
    @lazyproperty
    def integ_f_wei(self):
        t = self._getTableExtensionData(self._ext_integ_spectra)
        flagged = (self.integ_f_flag & flags.no_starlight) > 0
        return np.ma.masked_where(flagged, t['f_wei'])
    
    @lazyproperty
    def integ_popx(self):
        t = self._getTableExtensionData(self._ext_integ_pop)
        return t['popx']
    
    @lazyproperty
    def integ_popmu_ini(self):
        t = self._getTableExtensionData(self._ext_integ_pop)
        return t['popmu_ini']
    
    @lazyproperty
    def integ_popmu_cor(self):
        t = self._getTableExtensionData(self._ext_integ_pop)
        return t['popmu_cor']
    
    @property
    def integ_tau_V(self):
        '''
        Dust optical depth in the V band.

            * Units: dimensionless
            * Type: float
        '''
        return self.synthIntegKeywords['Av'] / (2.5 * np.log10(np.exp(1.)))

    @property
    def integ_Lobn(self):
        '''
        Luminosity of each population in normalization window 
        of the integrated spectrum.
 
            * Units: :math:`[L_\odot]`
            * Shape: ``(N_age, N_met)``
        '''
        tmp = self.integ_popx / 100.0
        tmp *= self.synthIntegKeywords['Lobs_norm']
        return tmp

    @property
    def integ_Mcor(self):
        '''
        Current mass of each population of the integrated spectrum.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_aFe)``
        '''
        tmp = self.integ_popmu_cor / 100.0
        tmp *= self.synthIntegKeywords['Mcor_tot']
        return tmp
      
    @property
    def integ_Mini(self):
        '''
        Initial mass of each population of the integrated spectrum.

            * Units: :math:`[M_\odot]`
            * Shape: ``(N_age, N_met, N_aFe)``
        '''
        tmp = self.integ_popmu_ini / 100.0
        tmp *= self.synthIntegKeywords['Mini_tot']
        return tmp

    @property
    def integ_at_flux(self):
        '''
        Flux-weighted average log. age of the integrated spectrum.

            * Units: :math:`[\log Gyr]`
            *  Type: float
        '''
        logt = np.log10((self.popage_base + self.popage_base_t2) / 2)
        return (self.integ_popx * logt).sum() / self.integ_popx.sum()

    @property
    def integ_at_mass(self):
        '''
        Mass-weighted average log. age of the integrated spectrum.

            * Units: :math:`[\log Gyr]`
            *  Type: float
        '''
        logt = np.log10((self.popage_base + self.popage_base_t2) / 2)
        return (self.integ_popmu_cor * logt).sum() / self.integ_popmu_cor.sum()

    @property
    def integ_alogZ_flux(self):
        '''
        Flux-weighted average log of metallicity of the integrated spectrum.

            * Units: dimensionless
            *  Type: float
        '''
        logZ = np.log10(self.popZ_base / 0.019)
        return (self.integ_popx * logZ).sum() / self.integ_popx.sum()

    @property
    def integ_alogZ_mass(self):
        '''
        Mass-weighted average log of metallicity of the integrated spectrum.

            * Units: dimensionless
            *  Type: float
        '''
        logZ = np.log10(self.popZ_base / 0.019)
        return (self.integ_popmu_cor * logZ).sum() / self.integ_popmu_cor.sum()

    @lazyproperty
    def EL_info(self):
        return self._getTableExtensionData(self._ext_el_info)
    
    @lazyproperty
    def EL_names(self):
        return list(self.EL_info['name'])
        
    @lazyproperty
    def EL_lambda(self):
        return list(self.EL_info['lambda'])
        
    @lazyproperty
    def _EL_name_map(self):
        return {k: i for i, k in enumerate(self.EL_names)}
    
    @lazyproperty
    def _EL_lambda_map(self):
        return {k: i for i, k in enumerate(self.EL_lambda)}
    
    @lazyproperty
    def _EL_id_map(self):
        return {**self._EL_lambda_map, **self._EL_name_map}
    
    @lazyproperty
    def _EL_flag(self):
        return self._getSynthExtension(self._ext_el_flag)

    @lazyproperty
    def _EL_badfit(self):
        return self._EL_flag > 0

    def _getELExtension(self, name):
        data = self._getSynthExtension(name)
        data[self._EL_badfit] = np.ma.masked
        return data

    @lazyproperty
    def _EL_flux(self):
        return self._getELExtension(self._ext_el_flux)

    @lazyproperty
    def _EL_EW(self):
        return self._getELExtension(self._ext_el_ew)

    @lazyproperty
    def _EL_v_0(self):
        return self._getELExtension(self._ext_el_v_0)

    @lazyproperty
    def _EL_v_d(self):
        return self._getELExtension(self._ext_el_v_d)

    @lazyproperty
    def _EL_v_d_inst(self):
        return self._getELExtension(self._ext_el_v_d_inst)

    @lazyproperty
    def _EL_continuum_RMS(self):
        return self._getELExtension(self._ext_el_lc_rms)

    @lazyproperty
    def EL_continuum(self):
        return self._getSynthExtension(self._ext_el_lc)

    @lazyproperty
    def _El_F_integ(self):
        return self._getELExtension(self._ext_el_flux_integ)

    @lazyproperty
    def _El_F_imed(self):
        return self._getELExtension(self._ext_el_flux_imed)
    
    def _getELProperty(self, line, prop):
        try:
            i = self._EL_id_map[line]
        except KeyError:
            raise Exception('Emission line not found: %s' % line)
        return prop[i]
        
    def EL_flag(self, line):
        return self._getELProperty(line, self._EL_flag)
    
    def EL_flux(self, line):
        return self._getELProperty(line, self._EL_flux)
    
    def EL_EW(self, line):
        return self._getELProperty(line, self._EL_EW)
    
    def EL_v_0(self, line):
        return self._getELProperty(line, self._EL_v_0)
    
    def EL_v_d(self, line):
        return self._getELProperty(line, self._EL_v_d)
    
    def EL_v_d_inst(self, line):
        return self._getELProperty(line, self._EL_v_d_inst)
    
    def EL_continuum_RMS(self, line):
        return self._getELProperty(line, self._EL_continuum_RMS)
    
    @lazyproperty
    def _EL_integ(self):
        t = self._getTableExtensionData(self._ext_el_integ)
        data = np.ma.array(t)
#        data[data['El_flag'] > 0] = np.ma.masked
        return data

    def EL_integ(self, line):
        return self._getELProperty(line, self._EL_integ)

    @lazyproperty
    def EL_integ_continuum(self):
        t = self._getTableExtensionData(self._ext_el_integ_lc)
        return np.array(t['total_lc'])

    def EL_model(self, line, iy, ix):
        if not self.hasELines:
            log.warning('This cube does not have emission line measurements.')
            return None
            
        _, _, l0, model, _, _ = self._getELProperty(line, self.EL_info)
        
        if model == 'gaussian':
            Model = Gaussian
        elif model == 'resampled_gaussian':
            Model = ResampledGaussian
        
        mod = Model(l0, self.EL_flux(line)[iy, ix],
                    self.EL_v_0(line)[iy, ix],
                    self.EL_v_d(line)[iy, ix],
                    self.EL_v_d_inst(line)[iy, ix])
        return mod
        
    def EL_total_flux(self, iy, ix):
        if not self.hasELines:
            log.warning('This cube does not have emission line measurements.')
            return None

        flux = np.zeros_like(self.l_obs)
        for line in self.EL_lambda:
            if self.EL_flag(line)[iy, ix] > 0:
                continue
            mod = self.EL_model(line, iy, ix)
            l0 = mod.l0.value
            l1 = l0 - 100.0
            l2 = l0 + 100.0
            m = (self.l_obs >= l1) & (self.l_obs < l2)
            flux[m] += mod(self.l_obs[m])
        return flux

    def plotPixel(self, iy, ix, fig=None):
        f_obs = self.f_obs[:, iy, ix]
        f_err = self.f_err[:, iy, ix]
        f_flag = self.f_flag[:, iy, ix]
        f_syn = self.f_syn[:, iy, ix]
        if self.hasELines:
            f_emline = self.EL_continuum[:, iy, ix] + self.EL_total_flux(iy, ix)
            vlines = self.EL_info['l0']
        else:
            f_emline = None
            vlines = None
        fig = plot_spec(self.l_obs, f_obs, f_err, f_syn, f_flag, f_emline, vlines, fig)
        return fig
        
