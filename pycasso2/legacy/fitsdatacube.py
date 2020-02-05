'''
Created on Mar 8, 2012

@author: Andre Luiz de Amorim

'''

from .q3datacube_intf import IQ3DataCube
from ..geometry import convex_hull_mask as convexHullMask
from ..wcs import get_pixel_scale_rad, get_Naxis, get_reference_pixel, get_wavelength_coordinates
from .. import flags, modeling
from ..resampling import find_nearest_index
from ..starlight.synthesis import get_base_grid

from astropy import log as logger
from astropy.io import fits
from astropy.utils.decorators import lazyproperty
from astropy.wcs import WCS

import numpy as np
import ast
from distutils.version import LooseVersion


#############################################################################
## Zone cube reader
#############################################################################

class ZoneCube(object):
    '''
    Load data from the 11-layer zone info cube in a fits file.
    
    ima_5650: Image at 5590.0-5680.0 A (ima_5650.data)
    ima_sn: S/N at 5590.0-5680.0 A (ima_sn.data)
    qima_5650: Image at 5590.0-5680.0 A (quick, not masked)
    mask: Mask used for data (mask.data)
    zones_ima_sn20: Voronoi Zones (voronoi_bins)
    zones_sn: S/N Voronoi Zones (resampled) at 5590.0-5680.0(zones_sn)
    zones_rms: RMS Voronoi Zones (resampled) at 5590.0-5680.0(zones_rms)
    bzones_sn: S/N Voronoi Zones (UNresampled, beta) at 5590.0-5680.0(bzones_sn)
    bzones_rms: RMS Voronoi Zones (UNresampled, beta) at 5590.0-5680.0(bzones_rms)
    vzones_sn: Voronoi program output S/N at 5590.0-5680.0(vzones_sn, binSN)
    ima_noise_ferr: Noise image derived from formal errors (UNresampled) at 5590.0-5680.0 A
    zima_noise_ferr: Noise image derived from formal errors in Voronoi Zone (UNresampled) at 5590.0-5680.0 A
    
    Since version 0.3.1
    
    Signal: Signal @ 5590.0-5680.0 A
    Noise: Noise @ 5590.0-5680.0 A
    Sn: S/N @ 5590.0-5680.0 A
    SingalUnmasked: Signal (NON-masked) @ 5590.0-5680.0 A
    PipeNoise: Noise derived from given (Pipeline) errors @ 5590.0-5680.0 A
    Zones: Voronoi/Segmented Zones
    SegmentationSn: Segmentation program output S/N @ 5590.0-5680.0 A (binSN)
    ZonesSn: S/N in Segmented Zones (resampled) @ 5590.0-5680.0 A
    ZonesNoise: Noise in Segmented Zones (resampled) @ 5590.0-5680.0 A 
    ZonesSnOrig: S/N in Segmented Zones (original, NON-resampled) @ 5590.0-5680.0 A
    ZonesNoiseOrig: Noise in Segmented Zones (original, NON-resampled) @ 5590.0-5680.0 A
    PipeZonesNoiseOrig: Noise in Segmented Zones derived from given 
    (Pipeline) errors in Segmentated Zones (original, NON-resampled) @ 5590.0-5680.0 A
    PipeZonesMedianNoiseOrig: Noise in Segmented Zones derived as the 
    Median of the given (Pipeline) errors in Segmentated Zones (original, NON-resampled)
    FlagRatio: Ratio (100%) of flags in window at  3650.0-4150.0 A
    Mask: Mask used for data

    '''

    def __init__(self, hdu):
        self._hdu = hdu
        self.header = hdu.header
        self.qVersion = self.header['QVERSION'].strip()
        
        self.galaxyName = self.header.get('NED_NAME', '').strip()
        self.califaID = self.header.get('CALIFA_ID', '').strip()
        self.N_x = self.header['naxis1']
        self.N_y = self.header['naxis2']
        self.N_zone = self.header['NZONES']
        
        zoneXpos = ast.literal_eval(self.header['XPOSzone'])
        zoneYpos = ast.literal_eval(self.header['YPOSzone'])
        data = list(zip(zoneXpos, zoneYpos))
        self.zonePos = np.array(data, dtype=[('x', np.float64), ('y', np.float64)])

        self.HLR_pix = self.header['HLR']
        self.HLR_pc = self.header['HLR_PC']
        self.distance_Mpc = self.header['D_MPC']
        self.parsecPerPixel = self.header['PPP']
        self.x0, self.y0 = self.zonePos[0]
        
        self._planeSetup()
        
        if LooseVersion(self.qVersion) < LooseVersion('0.3.3'):
            logger.debug('qVersion < 0.3.3')
            self.masterListData = {}
            self.qSignal = self.getPlane('ima_5650')
            self.qNoise = self.getPlane(None)
            self.qSn = self.getPlane('ima_sn')
            self.qSignalUnmasked = self.getPlane('qima_5650')
            self.qNoiseUnmasked = self.getPlane(None)
            self.qPipeNoise = self.getPlane(None)
            # NOTE: qMask may be wrong, qZones image is more reliable.
            #self.qMask = self.getPlane('mask') > 0
            self.qZones = np.array(self.getPlane('zones_ima_sn20'), dtype=np.int) - 1
            self.qZonesSn = self.getPlane('zones_sn')
            self.qZonesNoise = self.getPlane('zones_rms')
            self.qFlagRatio = self.getPlane('zflag_ratio')
            self.qZonesSnOrig = self.getPlane('bzones_sn')
            self.qZonesNoiseOrig = self.getPlane('bzones_rms')
            self.qSegmentationSn = self.getPlane('vzones_sn')
            self.qPipeNoiseOrig = self.getPlane('ima_noise_ferr')
            self.qPipeZonesNoiseOrig = self.getPlane('zima_noise_ferr')
            self.qSpatialMask = self.getPlane(None)
            self.qSnMask = self.getPlane(None)
            
            self.qMask = self.qZones >= 0
            self.qConvexHull = convexHullMask(self.qMask)
            self.qFilledMask = np.array(self.qMask, dtype=np.int)
        else:
            logger.debug('qVersion > 0.3.6')
            self.masterListData = ast.literal_eval(self.header['DICTPROP'])
            self.qSignal = self.getPlane('Signal')
            self.qNoise = self.getPlane('Noise')
            self.qSn = self.getPlane('Sn')
            self.qSignalUnmasked = self.getPlane('SingalUnmasked')
            self.qNoiseUnmasked = self.getPlane('NoiseUnmasked')
            self.qPipeNoise = self.getPlane('PipeNoise')
            # NOTE: qMask may be wrong, qFilledZones image is more reliable.
            #self.qMask = self.getPlane('Mask') > 0
            self.qZones = np.array(self.getPlane('Zones'), dtype=np.int) - 1
            self.qZonesSn = self.getPlane('ZonesSn')
            self.qZonesNoise = self.getPlane('ZonesNoise')
            self.qFlagRatio = self.getPlane('FlagRatio')
            self.qZonesSnOrig = self.getPlane('ZonesSnOrig')
            self.qZonesNoiseOrig = self.getPlane('ZonesNoiseOrig')
            self.qSegmentationSn = self.getPlane('SegmentationSn')
            self.qPipeNoiseOrig = self.getPlane('PipeZonesMedianNoiseOrig')
            self.qPipeZonesNoiseOrig = self.getPlane('PipeZonesNoiseOrig')
            self.qSpatialMask = self.getPlane('SpatialMask')
            self.qSnMask = self.getPlane('SnMask')
            
            self.qFilledMask = np.array(self.getPlane('FilledMask'), dtype=np.int)
            self.qConvexHull = None
            self.qMask = (self.qFilledMask & 1) > 0
            
        if LooseVersion(self.qVersion) == LooseVersion('0.3.3'):
            logger.warn('Fixing qMask for unsupported qVersion 0.3.3')
            self.qMask = self.getPlane('Mask') > 0
        
        if LooseVersion(self.qVersion) == LooseVersion('0.4.3'):
            sn_vor = self.header['SN_VOR']
            if type(sn_vor) is str and 'Pixel-wise' in sn_vor:
                logger.warn('Fixing swapped x0, y0 for pixel-wise q043.')
                self.x0, self.y0 = self.y0, self.x0
            

    def _planeSetup(self):    
        '''
        Create a dictionary of plane names to index in image.
        '''   
        nplanes = self.header['NAXIS3']
        self._planeIndex = {}
        for pId in range(nplanes):
            key = 'PLANE_%d' % pId
            pname = self.header[key].split(':')[0]
            self._planeIndex[pname] = pId
 
 
    def getPlane(self, pname):
        '''
        Get the qbick plane by plane name.
        If not found, return an array of NaN.
        '''
        if pname not in self._planeIndex:
            if pname is not None:
                logger.warn('plane %s not found. Filling with NaN.' % pname) 
            nan = np.empty((self.N_y, self.N_x), dtype=np.float)
            nan.fill(np.nan)
            return nan
        planeId = self._planeIndex[pname]
        return self._hdu.data[planeId].copy()
    
    
#############################################################################
## Synthesis cube reader
#############################################################################

class fitsQ3DataCube(IQ3DataCube):
    '''
    Implementation of ``IQ3DataCube``.
    Load the cubes from the synthesis output fits file. This class builds
    upon ZoneCube, it uses the extended HDUs for synthesis data. The header
    from the primary HDU is also available, it contains keywords from
    earlier steps.

    The synthesis data is stored as cubes, 1-D images and keywords:
    
        cubes: 3-D images with shape (ageBase, metallicityBase, zone)
        
        1-D images: scalar values for each zone, with shape (zone)
        
        keywords: a dictionary with keywords from the synthesis.
    
    Note: Mstars is the same for all zones, therefore it is a 2-D image,
        with shape (ageBase, metallicityBase).
        
    Usage:
        c = SynthesisZoneCube('K0001_synthesis_suffix.fits')
        
        # Plot population vector for the integrated synthesis.
        imshow(c.integrated_popx)
        
        # Calculate Lominosity for each population in base.
        Lobn = c.popx / 100.0 * c.Lobs_norm
    '''
    
    _ext_f_obs = 'F_OBS'
    _ext_f_err = 'F_ERR'
    _ext_f_syn = 'F_SYN'
    _ext_f_wei = 'F_WEI'
    _ext_f_flag = 'F_FLAG'
    _ext_segmask = 'SEGMASK'
    _ext_seg_good_frac = 'SEG_GOOD_FRAC'
    
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
    
    _ext_qplanes = 'qPlanes'

    _h_lum_dist_Mpc = 'PYCASSO LUM_DIST_MPC'
    _h_redshift = 'PYCASSO REDSHIFT'
    _h_flux_unit = 'PYCASSO FLUX_UNIT'
    _h_name = 'PYCASSO CUBE_NAME'
    _h_has_segmap = 'PYCASSO HAS_SEGMAP'

    _Z_sun = 0.019

    _ext_keyword_list = ['Lobs_norm', 'Mini_tot', 'Mcor_tot', 'fobs_norm',
                         'Av', 'v0', 'vd', 'adev', 'Ntot_clipped',
                         'Nglobal_steps', 'chi2', 'SN_normwin']

    def __init__(self, synthesisFile, smooth=True):
        '''
        Parameters
        ----------
         synthesisFile : string
             Path to a FITS file containing the synthesis.
             
       smooth : bool
            Enable smooth dezonification. Defaults to ``True``.
            See the method ``setSmoothDezonification``.
        
        See also
        --------
        h5Q3DataCube, IQ3DataCube
        
        '''
        self.baseId = ''
        self.fill_value = np.nan
        
        self._hdulist = fits.open(synthesisFile)
        self.header = self._hdulist[0].header
        self._wcs = WCS(self.header)
        if self._hasZoneCube:
            self._zoneCube = ZoneCube(self._hdulist[self._ext_qplanes])
        IQ3DataCube.__init__(self)
        self._loadMetadata()
        self._loadQData()
        self._preprocess(smooth)
        self.EL = None


    def close(self):
        self._hdulist.close()


    def getExt(self, name):
        if name not in self._hdulist:
            raise Exception('Extension %s not found.' % name)
        data = self._hdulist[name].data
        if (data.ndim > 1) and self._hasSegmask and (name != self._ext_segmask):
            data = np.moveaxis(data, 0, -1)
        return data.copy()
        

    def getTableExt(self, name):
        data = self._hdulist[name].data
        return data

    def _loadQData(self):
        if self._hasZoneCube:
            q = self._zoneCube
            self.masterListData = q.masterListData
            self.qSignal = q.qSignal
            self.qNoise = q.qNoise
            self.qSn = q.qSn
            self.qSignalUnmasked = q.qSignalUnmasked
            self.qNoiseUnmasked = q.qNoiseUnmasked
            self.qPipeNoise = q.qPipeNoise
            self.qZones = q.qZones
            self.qZonesSn = q.qZonesSn
            self.qZonesNoise = q.qZonesNoise
            self.qFlagRatio = q.qFlagRatio
            self.qZonesSnOrig = q.qZonesSnOrig
            self.qZonesNoiseOrig = q.qZonesNoiseOrig
            self.qSegmentationSn = q.qSegmentationSn
            self.qPipeNoiseOrig = q.qPipeNoiseOrig
            self.qPipeZonesNoiseOrig = q.qPipeZonesNoiseOrig
            self.qSpatialMask = q.qSpatialMask
            self.qSnMask = q.qSnMask
            
            self.qFilledMask = q.qFilledMask
            self._qConvexHull = None
            self._qMask = (self.qFilledMask & 1) > 0
        else:
            logger.warn('QBICK data not implemented correctly for this cube.')
            self.masterListData = {}
            if self._hasSegmask:
                self.qZones = self._qZonesFromSegmask()
                self._qMask = self.qZones >= 0
            else:
                # FIXME:
                self._qMask = self._qMaskFromFlags()
                self.qZones = np.ones((self.N_y, self.N_x), dtype=np.int)
                self.qZones.fill(-1)
                self.qZones[self.qMask] = np.arange(self.qMask.sum())
            # HACK: setup dezonification even though we dont support it
            # yet, to allow bootstrap dezonification of qSignal.
            self._preprocess(smooth=False, set_geometry=False)
            self.qSignal =self.zoneToYX(self._qSignal, extensive=True, surface_density=False).filled(0.0)
            self.qNoise =self.zoneToYX(self._qNoise, extensive=True, surface_density=False).filled(0.0)
            self.qSn = self.qSignal / self.qNoise
            self.qFlagRatio = self._qFlagRatio
            self.qSignalUnmasked = self.qSignal
            self.qNoiseUnmasked = self.qNoise
            self.qSegmentationSn = self.qSn
            self.qZonesSn = self.qSn
            self.qPipeNoise = None
            self.qZonesNoise = None
            self.qZonesSnOrig = None
            self.qZonesNoiseOrig = None
            self.qPipeNoiseOrig = None
            self.qPipeZonesNoiseOrig = None
            self.qSpatialMask = None
            self.qSnMask = None
            self.zonePos = self._zonePos
            
            self._qConvexHull = convexHullMask(self.qMask)
            self.qFilledMask = np.array(self.qMask, dtype=np.int)
        

    def _loadMetadata(self):
        self._readKeywords()
        self.integrated_keywords['A_V'] = self.integrated_keywords['AV']
        self.integrated_keywords['V_0'] = self.integrated_keywords['V0']
        self.integrated_keywords['V_D'] = self.integrated_keywords['VD']

        # If the version is not set, assume it the old version, 0.8.x
        self.pycassoVersion = self.keywords.get('VERSION', '0.8')
        logger.debug('PyCASSO version %s'% self.pycassoVersion)

        if self._hasZoneCube:
            self.qVersion = self._zoneCube.qVersion
            self.galaxyName = self._zoneCube.galaxyName
            self.califaID = self._zoneCube.califaID
            self.zonePos = self._zoneCube.zonePos
            self.x0, self.y0 = self.zonePos[0]
    
            self.HLR_pix = self._zoneCube.HLR_pix
            self.HLR_pc = self.HLR_pix * self.parsecPerPixel
        else:
            self.qVersion = self.pycassoVersion
            self.galaxyName = self.keywords['CUBE_NAME'].strip()
            self.header['NED_NAME'] = self.galaxyName
            self.califaID = self.keywords['CUBE_NAME'].strip()
            _, self.y0, self.x0 = get_reference_pixel(self._wcs, as_int=True)
    
            # FIXME:
            self.zonePos = None
            self.HLR_pix = None
            self.HLR_pc = None

        # Useful definitions
        self.N_age = len(self.ageBase) 
        self.N_met = len(self.metBase)
        self.N_aFe = len(self.aFeBase)
        self.N_base = self.N_age * self.N_met * self.N_aFe
        self.q_norm = self.keywords['Q_NORM']
        self.dl = self.keywords['DL']
        self.l_ini = self.keywords['L_INI']
        self.l_fin = self.keywords['L_FIN']
        self.flux_unit = self.keywords['FLUX_UNIT']
        self.Nl_obs = self.keywords['NL_OBS']

    @property
    def hasSegmentationMask(self):
        key = self._h_has_segmap
        if not key in self.header:
            return False
        return bool(self.header[key])


    @lazyproperty
    def Nwave(self):
        if self.hasSegmentationMask:
            axis = 1
        else:
            axis = 3
        return get_Naxis(self._hdulist[self._ext_f_obs].header, axis)


    @lazyproperty
    def l_obs(self):
        return get_wavelength_coordinates(self._wcs, self.Nwave)

    def _toRectBase(self, a, fill_value=0.0):
        shape = (self._baseMask.shape) + a.shape[1:]
        a__Zt = np.ma.masked_all(shape, dtype=a.dtype)
        a__Zt.fill_value = fill_value
        a__Zt[self._baseMask, ...] = a
        return np.swapaxes(a__Zt, 0, 1)

    def _toZone(self, x):
        if self._hasSegmask:
            return x
        else:
            if x.ndim > 2:
                return x[..., self.qMask]
            else:
                return x[self.qMask]
       
        
    def _readKeywords(self):
        '''
        Read synthesis keywords from the fits header.
        The base (age and metallicity) is also read here.
        '''
        self.keywords = {}
        self.integrated_keywords = {}

        self.keywords = {k.split()[1].upper(): v for k, v in self.header.items() if 'PYCASSO' in k}
        synth_keywords = {k.split()[1].upper(): v for k, v in self.header.items() if 'STARLIGHT' in k}
        self.keywords.update(synth_keywords)
        
        integ_h = self._hdulist[self._ext_integ_pop].header
        self.integrated_keywords = {k.split()[1].upper(): v for k, v in integ_h.items() if 'STARLIGHT' in k}

                
    def _qZonesFromSegmask(self):
        return np.sum(self._segmask * np.arange(self.N_zone + 1))
        

    def _qMaskFromFlags(self):
        if self._hasSegmask:
            raise Exception('_qMaskFromFlags only works for spatially resolved data. Use segmask.')
        f = flags.no_obs | flags.no_starlight
        threshold=0.8
        f_flag = self.getExt(self._ext_f_flag)
        flagged = ((f_flag & f) > 0).astype(int).sum(axis=0)
        return (flagged < (threshold * len(self.l_obs)))


    @lazyproperty
    def _qSignal(self):
        norm_lambda = 5635.0
        delta_l = 45.0
        l1, l2 = find_nearest_index(
            self.l_obs, [norm_lambda - delta_l, norm_lambda + delta_l])
        l_obs = self.l_obs[l1:l2]
        f_obs = np.ma.masked_where(self.f_flag[l1:l2] > 0.0, self.f_obs[l1:l2], copy=True)
        y = modeling.cube_continuum(l_obs, f_obs, degr=1, niterate=0)
        return np.mean(y, axis=0)

    @lazyproperty
    def _qFlagRatio(self):
        norm_lambda = 5635.0
        delta_l = 45.0
        l1, l2 = find_nearest_index(
            self.l_obs, [norm_lambda - delta_l, norm_lambda + delta_l])
        f_flag = self._hdulist[self._ext_f_flag].data[l1:l2]
        return ((f_flag & (flags.no_obs | flags.no_starlight)) > 0).sum(axis=0) / (l2 - l1)

    @lazyproperty
    def _qNoise(self):
        norm_lambda = 5635.0
        delta_l = 45.0
        l1, l2 = find_nearest_index(
            self.l_obs, [norm_lambda - delta_l, norm_lambda + delta_l])
        l_obs = self.l_obs[l1:l2]
        f_obs = np.ma.masked_where(self.f_flag[l1:l2] > 0.0, self.f_obs[l1:l2], copy=True)
        y = modeling.cube_continuum(l_obs, f_obs, degr=1, niterate=0)
        return np.std(f_obs - y, axis=0)


    @lazyproperty
    def _zonePos(self):
        y, x = np.indices((self.N_y, self.N_x))
        data = list(zip(x[self.qMask], y[self.qMask]))
        return np.array(data, dtype=[('x', np.float64), ('y', np.float64)])


    @property
    def qMask(self):
        return self._qMask
    

    @qMask.setter
    def qMask(self, m):
        '''
        FIXME: deprecated setter for qMask.
        '''
        logger.warn('qMask is being changed, and this is depracated.')
        # Convert the mask to boolean
        if m.dtype is not bool:
            m = m > 0.0
        self._qMask = m

            
    def loadEmLinesDataCube(self, emLinesFile):
        raise NotImplementedError('Emission lines not supported.')  
 
 
    def checkAlphaEnhancement(self):
        if not self.hasAlphaEnhancement:
            raise Exception('This cube was fitted using a base with constant aFe.')


    @property
    def hasAlphaEnhancement(self):
        return self.N_aFe > 1


    @lazyproperty
    def ageBase(self):
        popage = self.getExt(self._ext_popage_base)
        popage_t2 = self.getExt(self._ext_popage_base_t2)
        return (np.unique(popage) + np.unique(popage_t2)) / 2
    
        
    @lazyproperty
    def metBase(self):
        popZ = self.getExt(self._ext_popZ_base)
        return np.unique(popZ)
        

    @lazyproperty
    def aFeBase(self):
        popaFe = self.getExt(self._ext_popaFe_base)
        return np.unique(popaFe)


    @lazyproperty
    def _baseMask(self):
        popage = self.getExt(self._ext_popage_base)
        popZ = self.getExt(self._ext_popZ_base)
        base_mask, _, _ = get_base_grid(popage, popZ)
        return base_mask


    @property
    def _hasZoneCube(self):
        return self._ext_qplanes in self._hdulist
    
    
    @property
    def _hasSegmask(self):
        return self._ext_segmask in self._hdulist
    
    
    @property
    def PIXSIZE(self):
        if self._hasZoneCube:
            return self._zoneCube.header['PIXSIZE']
        else:
            raise NotImplementedError('PIXSIZE only defined for QBICK cubes.')

    
    @lazyproperty
    def parsecPerPixel(self):
        lum_dist_pc = self.distance_Mpc * 1e6
        angle = get_pixel_scale_rad(self._wcs)
        return angle * lum_dist_pc


    @property
    def distance_Mpc(self):
        key = self._h_lum_dist_Mpc
        if not key in self.header:
            raise Exception(
                'Luminosity distance not set. Header key: %s' % key)
        return self.header[key]


    @distance_Mpc.setter
    def distance_Mpc(self, value):
        key = 'HIERARCH %s' % self._h_lum_dist_Mpc
        self.header[key] = value


    @lazyproperty
    def N_zone(self):
        if self._hasSegmask:
            return get_Naxis(self._hdulist[self._ext_segmask].header, 3)
        else:
            return self.qZones.max() + 1


    @lazyproperty
    def N_x(self):
        if self._hasSegmask:
            hdu = self._ext_segmask
        else:
            hdu = self._ext_f_obs
        return get_Naxis(self._hdulist[hdu].header, 1)


    @lazyproperty
    def N_y(self):
        if self._hasSegmask:
            hdu = self._ext_segmask
        else:
            hdu = self._ext_f_obs
        return get_Naxis(self._hdulist[hdu].header, 2)


    @property
    def zone_invalid(self):
        logger.warn('Zone mask is deprecated, nothing will be masked.')
        return np.zeros((self.N_zone), dtype='bool')
    
    
    @property
    def redshift(self):
        return self.keywords['REDSHIFT']
    
    
    @property
    def _segmask(self):
        return self.getExt(self._ext_segmask)


    @lazyproperty
    def _integ_population(self):
        return self.getTableExt(self._ext_integ_pop)
    
    
    @lazyproperty
    def _integ_spectra(self):
        return self.getTableExt(self._ext_integ_spectra)
    
    
    @property
    def popx(self):
        return self._toRectBase(self._toZone(self.getExt('POPX')))


    @property
    def popmu_cor(self):
        return self._toRectBase(self._toZone(self.getExt('POPMU_COR')))
    

    @property
    def popmu_ini(self):
        return self._toRectBase(self._toZone(self.getExt('POPMU_INI')))
    

    @property
    def popAV_tot(self):
        raise NotImplementedError('popAV_tot not available.')
    

    @property
    def popexAV_flag(self):
        raise NotImplementedError('popexAV_flag not available.')


    @property
    def SSP_chi2r(self):
        raise NotImplementedError('SSP_chi2r not available.')


    @property
    def SSP_adev(self):
        raise NotImplementedError('SSP_adev not available.')


    @property
    def SSP_AV(self):
        raise NotImplementedError('SSP_AV not available.')


    @property
    def SSP_x(self):
        raise NotImplementedError('SSP_x not available.')


    @property
    def integrated_popx(self):
        return self._toRectBase(self._integ_population['popx'])


    @property
    def integrated_popmu_cor(self):
        return self._toRectBase(self._integ_population['popmu_cor'])


    @property
    def integrated_popmu_ini(self):
        return self._toRectBase(self._integ_population['popmu_ini'])


    @property
    def integrated_popAV_tot(self):
        raise NotImplementedError('integrated_popAV_tot not available.')


    @property
    def integrated_popexAV_flag(self):
        raise NotImplementedError('integrated_popexAV_flag not available.')
        

    @property
    def integrated_SSP_chi2r(self):
        raise NotImplementedError('integrated_SSP_chi2r not available.')


    @property
    def integrated_SSP_adev(self):
        raise NotImplementedError('integrated_SSP_adev not available.')


    @property
    def integrated_SSP_AV(self):
        raise NotImplementedError('integrated_SSP_AV not available.')


    @property
    def integrated_SSP_x(self):
        raise NotImplementedError('integrated_SSP_x not available.')


    @property
    def Mstars(self):
        return self._toRectBase(self.getExt('MSTARS'))


    @property
    def fbase_norm(self):
        fbase_norm = self.getExt('FBASE_NORM')
        if LooseVersion(self.qVersion) == LooseVersion('0.4.3'):
            logger.debug('Fixing bug in fbase_norm.')
            fbase_norm[fbase_norm == 0] = 1.0
        return self._toRectBase(fbase_norm)
        

    @property
    def Lobs_norm(self):
        return self._toZone(self.getExt('LOBS_NORM'))


    @property
    def fobs_norm(self):
        return self._toZone(self.getExt('FOBS_NORM'))


    @property
    def Mini_tot(self):
        return self._toZone(self.getExt('MINI_TOT'))


    @property
    def Mcor_tot(self):
        return self._toZone(self.getExt('MCOR_TOT'))


    @property
    def A_V(self):
        return self._toZone(self.getExt('AV'))


    @property
    def v_0(self):
        return self._toZone(self.getExt('V0'))


    @property
    def v_d(self):
        return self._toZone(self.getExt('VD'))


    @property
    def adev(self):
        return self._toZone(self.getExt('ADEV'))


    @property
    def index_Best_SSP(self):
        raise NotImplementedError('index_Best_SSP not available.')


    @property
    def NOl_eff(self):
        raise NotImplementedError('NOl_eff not available.')


    @property
    def Nl_eff(self):
        # FIXME:
        raise NotImplementedError('Nl_eff not available.')


    @property
    def Ntot_clipped(self):
        return self._toZone(self.getExt('NTOT_CLIPPED'))


    @property
    def Nglobal_steps(self):
        return self._toZone(self.getExt('NGLOBAL_STEPS'))


    @property
    def chi2(self):
        return self._toZone(self.getExt('CHI2'))


    @property
    def chi2_TOT(self):
        raise NotImplementedError('chi2_TOT not available.')


    @property
    def chi2_Opt(self):
        raise NotImplementedError('chi2_Opt not available.')


    @property
    def chi2_FIR(self):
        raise NotImplementedError('chi2_FIR not available.')


    @property
    def chi2_QHR(self):
        raise NotImplementedError('chi2_QHR not available.')


    @property
    def chi2_PHO(self):
        raise NotImplementedError('chi2_PHO not available.')


    @property
    def f_obs(self):
        return self._toZone(self.getExt('F_OBS'))


    @property
    def f_syn(self):
        return self._toZone(self.getExt('F_SYN'))


    @property
    def f_wei(self):
        return self._toZone(self.getExt('F_WEI'))


    @property
    def f_err(self):
        return self._toZone(self.getExt('F_ERR'))


    @property
    def f_flag(self):
        f_flag = self._toZone(self.getExt('F_FLAG'))
        bad = (f_flag & flags.before_starlight) > 0
        return np.where(bad, 2.0, 0.0)



    @property
    def integrated_f_obs(self):
        return self._integ_spectra['f_obs']


    @property
    def integrated_f_syn(self):
        return self._integ_spectra['f_syn']


    @property
    def integrated_f_wei(self):
        return self._integ_spectra['f_wei']


    @property
    def integrated_f_err(self):
        return self._integ_spectra['f_err']


    @property
    def integrated_f_flag(self):
        f_flag = self._integ_spectra['f_flag']
        bad = (f_flag & flags.before_starlight) > 0
        return np.where(bad, 2.0, 0.0)



    @property
    def chains_best_par(self):
        return self._toZone(self.getExt('CHAINS_BEST_PAR'))


    @property
    def chains_ave_par(self):
        return self._toZone(self.getExt('CHAINS_AVE_PAR'))


    @property
    def chains_par(self):
        return self._toZone(self.getExt('CHAINS_PAR'))            


    @property
    def chains_best_LAx(self):
        return self._toZone(self.getExt('CHAINS_BEST_LAX'))


    @property
    def chains_ave_LAx(self):
        return self._toZone(self.getExt('CHAINS_AVE_LAX'))


    @property
    def chains_LAx(self):
        return self._toZone(self.getExt('CHAINS_LAX'))
            


    @property
    def chains_best_mu_cor(self):
        return self._toZone(self.getExt('CHAINS_BEST_MU_COR'))


    @property
    def chains_ave_mu_cor(self):
        return self._toZone(self.getExt('CHAINS_AVE_MU_COR'))


    @property
    def chains_mu_cor(self):
        return self._toZone(self.getExt('CHAINS_MU_COR'))


    @property
    def best_chi2(self):
        return self._toZone(self.getExt('BEST_CHI2'))


    @property
    def ave_chi2(self):
        return self._toZone(self.getExt('AVE_CHI2'))


    @property
    def cha_chi2(self):
        return self._toZone(self.getExt('CHA_CHI2'))


    @property
    def best_Mcor(self):
        return self._toZone(self.getExt('BEST_MCOR'))


    @property
    def ave_Mcor(self):
        return self._toZone(self.getExt('AVE_MCOR'))


    @property
    def cha_Mcor(self):
        return self._toZone(self.getExt('CHA_MCOR'))


    @property
    def integrated_chains_best_par(self):
        return self.getExt('INTEGRATED_CHAINS_BEST_PAR')


    @property
    def integrated_chains_ave_par(self):
        return self.getExt('INTEGRATED_CHAINS_AVE_PAR')


    @property
    def integrated_chains_par(self):
        return self.getExt('INTEGRATED_CHAINS_PAR')


    @property
    def integrated_chains_best_LAx(self):
        return self.getExt('INTEGRATED_CHAINS_BEST_LAX')


    @property
    def integrated_chains_ave_LAx(self):
        return self.getExt('INTEGRATED_CHAINS_AVE_LAX')


    @property
    def integrated_chains_LAx(self):
        return self.getExt('INTEGRATED_CHAINS_LAX')          


    @property
    def integrated_chains_best_mu_cor(self):
        return self.getExt('INTEGRATED_CHAINS_BEST_MU_COR')


    @property
    def integrated_chains_ave_mu_cor(self):
        return self.getExt('INTEGRATED_CHAINS_AVE_MU_COR')


    @property
    def integrated_chains_mu_cor(self):
        return self.getExt('INTEGRATED_CHAINS_MU_COR')


    @property
    def PHO_logY_TOT(self):
        return self._toZone(self.getExt('PHO_LOGY_TOT'))


    @property
    def PHO_ErrlogY(self):
        return self._toZone(self.getExt('PHO_ERRLOGY'))


    @property
    def PHO_YFrac2Model(self):
        return self._toZone(self.getExt('PHO_YFRAC2MODEL'))


    @property
    def PHO_Chi2ScaleFactor(self):
        return self._toZone(self.getExt('PHO_CHI2SCALEFACTOR'))


    @property
    def PHO_MeanLamb(self):
        return self._toZone(self.getExt('PHO_MEANLAMB'))


    @property
    def PHO_StdDevLamb(self):
        return self._toZone(self.getExt('PHO_STDDEVLAMB'))


    @property
    def PHO_q_MeanLamb(self):
        return self._toZone(self.getExt('PHO_Q_MEANLAMB'))


    @property
    def PHO_ModlogY(self):
        return self._toZone(self.getExt('PHO_MODLOGY'))


    @property
    def PHO_chi2_Y(self):
        return self._toZone(self.getExt('PHO_CHI2_Y'))


    @property
    def integrated_PHO_logY_TOT(self):
        return self.getExt('INTEGRATED_PHO_LOGY_TOT')


    @property
    def integrated_PHO_ErrlogY(self):
        return self.getExt('INTEGRATED_PHO_ERRLOGY')


    @property
    def integrated_PHO_YFrac2Model(self):
        return self.getExt('INTEGRATED_PHO_YFRAC2MODEL')


    @property
    def integrated_PHO_Chi2ScaleFactor(self):
        return self.getExt('INTEGRATED_PHO_CHI2SCALEFACTOR')


    @property
    def integrated_PHO_MeanLamb(self):
        return self.getExt('INTEGRATED_PHO_MEANLAMB')


    @property
    def integrated_PHO_StdDevLamb(self):
        return self.getExt('INTEGRATED_PHO_STDDEVLAMB')


    @property
    def integrated_PHO_q_MeanLamb(self):
        return self.getExt('INTEGRATED_PHO_Q_MEANLAMB')


    @property
    def integrated_PHO_ModlogY(self):
        return self.getExt('INTEGRATED_PHO_MODLOGY')


    @property
    def integrated_PHO_chi2_Y(self):
        return self.getExt('INTEGRATED_PHO_CHI2_Y')

