'''
Created on 15 de dez de 2017

@author: andre
'''

from . import FitsCube
from .geometry import get_image_distance, get_half_radius, radial_profile
from .resampling import age_smoothing_kernel, interp_age
from .starlight.analysis import calc_popmu, MStarsEvolution
from astropy.utils.decorators import lazyproperty
import numpy as np


def MStars(ageBase, metBase, Mstars):
    return MStarsEvolution(ageBase, metBase, Mstars)


def ageSmoothingKernel(logtb, logtc, logtc_FWHM=0.1):
    return age_smoothing_kernel(logtb, logtc, logtc_FWHM)


def interpAge(prop, log_age_base, log_age_interp):
    return interp_age(prop, log_age_base, log_age_interp)


def calcPopmu(popx, fbase_norm, Lobs_norm, Mstars, q_norm, A_V):
    return calc_popmu(popx, fbase_norm, Lobs_norm, Mstars, q_norm, A_V)


def qplane_map(header):
    '''
    Create a dictionary of plane names to index in image.
    '''   
    nplanes = header['NAXIS3']
    plane_index = {}
    for pid in range(nplanes):
        key = 'PLANE_%d' % pid
        pname = header[key].split(':')[0]
        plane_index[pname] = pid
    return plane_index


def qzones2segmask(qZones):
    Nzone = int(qZones.max()) + 1
    segmask = np.zeros((Nzone,) + qZones.shape, dtype='int32')
    for z in range(Nzone):
        this_zone = (qZones == z)
        segmask[z, this_zone] = 1
    return segmask


def l_obs_from_header(header):
    l_ini = header['SYN L_INI']
    l_fin = header['SYN L_FIN']
    dl = header['SYN DL']
    return np.arange(l_ini, l_fin + dl, dl)

    
class fitsQ3DataCube(FitsCube):

    def __init__(self, filename):
        FitsCube.__init__(self, filename)
        self.keywords.update(self.synthKeywords)
        self._pmap = qplane_map(self._HDUList['qPlanes'].header)
        y0, x0 = np.where(self.segmentationMask[0])
        self._x0 = np.asscalar(x0)
        self._y0 = np.asscalar(y0)
        
    @property
    def header(self):
        return self._header
        
    @property
    def baseId(self):
        return self.synthKeywords['ARQ_BASE']
    
    @property
    def qVersion(self):
        return self.keywords['VERSION']
    
    @property
    def califaID(self):
        return self.name
    
    def setGeometry(self, pa, ba):
        self.pa = pa
        self.ba = ba
    
    def getHalfRadius(self, image):    
        r = get_image_distance(
            (self.Ny, self.Nx), self.x0, self.y0, self.pa, self.ba)
        return get_half_radius(image, r)

    @property
    def LobnSD__tZyx(self):
        return self.spatialize(self.LobnSD, extensive=False)
    
    @property
    def LobnSD__yx(self):
        return self.LobnSD__tZyx.sum(axis=(0, 1))
    
    @property
    def DeRed_LobnSD__tZyx(self):
        der_LobnSD = self.LobnSD * 10.0**(0.4 * self.q_norm * self.A_V)
        return self.spatialize(der_LobnSD, extensive=False)
    
    @property
    def DeRed_LobnSD__yx(self):
        return self.DeRed_LobnSD__tZyx.sum(axis=(0,1))
    
    @property
    def DeRed_M2L__yx(self):
        return self.McorSD__yx / self.DeRed_LobnSD__yx

    @property
    def MiniSD__tZyx(self):
        return self.spatialize(self.MiniSD, extensive=False)
    
    @property
    def McorSD__tZyx(self):
        return self.spatialize(self.McorSD, extensive=False)
    
    @property
    def McorSD__yx(self):
        return self.McorSD__tZyx.sum(axis=(0, 1))
    
    @property
    def parsecPerPixel(self):
        return self.pixelScale_pc
    
    @property
    def ageBase(self):
        return self.age_base
    
    @property
    def metBase(self):
        return self.Z_base
    
    @property
    def HLR_pix(self):
        return self.HLR
    
    @property
    def HLR_pc(self):
        return self.HLR * self.pixelScale_pc
    
    @lazyproperty
    def popx(self):
        popx = self._getSynthExtension(self._ext_popx)
        return self.toRectBase(popx, 0.0)

    @property
    def popx__tZyx(self):
        return self.spatialize(self.popx / 100.0, extensive=False)

    @lazyproperty
    def popmu_ini(self):
        popmu_ini = self._getSynthExtension(self._ext_popmu_ini)
        return self.toRectBase(popmu_ini, 0.0)

    @property
    def popmu_ini__tZyx(self):
        return self.spatialize(self.popmu_ini / 100.0, extensive=False)

    @lazyproperty
    def popmu_cor(self):
        popmu_cor = self._getSynthExtension(self._ext_popmu_cor)
        return self.toRectBase(popmu_cor, 0.0)
    
    @property
    def popmu_cor__tZyx(self):
        return self.spatialize(self.popmu_cor / 100.0, extensive=False)

    @lazyproperty
    def Mstars(self):
        Mstars = self._getExtensionData(self._ext_mstars)
        return self.toRectBase(Mstars, 0.0)

    @lazyproperty
    def fbase_norm(self):
        fbn = self._getExtensionData(self._ext_fbase_norm)
        return self.toRectBase(fbn, 0.0)

    @property
    def at_flux(self):
        popx = super(fitsQ3DataCube, self).popx
        popx = np.moveaxis(popx, 0, -1)
        log_t1 = np.log10(self.popage_base)
        log_t2 = np.log10(self.popage_base_t2)
        log_t = (log_t1 + log_t2) / 2.0
        return (popx * log_t).sum(axis=-1) / popx.sum(axis=-1)

    @property
    def at_flux__yx(self):
        return self.spatialize(self.at_flux, extensive=False)
    
    @property
    def at_mass(self):
        mu = super(fitsQ3DataCube, self).popmu_cor
        mu = np.moveaxis(mu, 0, -1)
        log_t1 = np.log10(self.popage_base)
        log_t2 = np.log10(self.popage_base_t2)
        log_t = (log_t1 + log_t2) / 2.0
        return (mu * log_t).sum(axis=-1) / mu.sum(axis=-1)

    @property
    def at_mass__yx(self):
        return self.spatialize(self.at_mass, extensive=False)
    
    @property
    def alogZ_flux(self):
        popx = super(fitsQ3DataCube, self).popx
        popx = np.moveaxis(popx, 0, -1)
        return (popx * np.log10(self.popZ_base / self._Z_sun)).sum(axis=-1) / popx.sum(axis=-1)

    @property
    def alogZ_flux__yx(self):
        return self.spatialize(self.alogZ_flux, extensive=False)
    
    @property
    def alogZ_mass(self):
        mu = super(fitsQ3DataCube, self).popmu_cor
        mu = np.moveaxis(mu, 0, -1)
        return (mu * np.log10(self.popZ_base / self._Z_sun)).sum(axis=-1) / mu.sum(axis=-1)

    @property
    def alogZ_mass__yx(self):
        return self.spatialize(self.alogZ_mass, extensive=False)
    
    @property
    def aaFe_flux(self):
        popx = super(fitsQ3DataCube, self).popx
        popx = np.moveaxis(popx, 0, -1)
        return (popx * self.popaFe_base).sum(axis=-1) / popx.sum(axis=-1)

    @property
    def aaFe_flux__yx(self):
        return self.spatialize(self.aaFe_flux, extensive=False)
    
    @property
    def aaFe_mass(self):
        mu = super(fitsQ3DataCube, self).popmu_cor
        mu = np.moveaxis(mu, 0, -1)
        return (mu * self.popaFe_base).sum(axis=-1) / mu.sum(axis=-1)

    @property
    def aaFe_mass__yx(self):
        return self.spatialize(self.aaFe_mass, extensive=False)
    
    @property
    def adevS(self):
        return self.adev
    
    @property
    def y0(self):
        return self._y0
    
    @property
    def x0(self):
        return self._x0
    
    @property
    def N_y(self):
        return self.Ny
    
    @property
    def N_x(self):
        return self.Nx
    
    @property
    def N_zone(self):
        return self.Nzone
    
    def zoneToYX(self, a, extensive=False):
        a = self.spatialize(a, extensive)
        if extensive:
            a = a / self.pixelArea_pc2
        return a

    def radialProfile(self, prop, bin_r, x0=None, y0=None, pa=None, ba=None,
                      rad_scale=None, mode='mean', return_npts=False):
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
        if pa is None:
            pa = self.pa
        if ba is None:
            ba = self.ba
        if rad_scale is None:
            rad_scale = self.HLR_pix
        if not isinstance(prop, np.ma.MaskedArray):
            prop = np.ma.array(prop)
            prop[..., ~self.qMask] = np.ma.masked
        radprof, npts = radial_profile(prop, bin_r, x0, y0, pa, ba, rad_scale, mode, return_npts=True)
        radprof[..., npts == 0] = np.ma.masked
        if return_npts:
            return radprof, npts
        else:
            return radprof
            
        return radprof

    def getPixelDistance(self):
        dist = get_image_distance((self.Ny, self.Nx), self.x0, self.y0, self.pa, self.ba)
        return dist / self.HLR_pix
    
    @property
    def qPlanes(self):
        return self._HDUList['qPlanes'].data
    
    @property
    def qMask(self):
        pid = self._pmap['Mask']
        return self.qPlanes[pid] > 0
        #return self.spatialize(~self.synthImageMask, extensive=False).filled(0)
    
    @property
    def qFlagRatio(self):
        pid = self._pmap['FlagRatio']
        return self.qPlanes[pid]
    
    @property
    def qSignal(self):
        pid = self._pmap['Signal']
        return self.qPlanes[pid]
    
    @property
    def qSignalUnmasked(self):
        pid = self._pmap['SingalUnmasked']
        return self.qPlanes[pid]
    
    @property
    def qSn(self):
        pid = self._pmap['Sn']
        return self.qPlanes[pid]
    
    @property
    def qZonesSn(self):
        pid = self._pmap['ZonesSn']
        return self.qPlanes[pid]
    
    @property
    def qSegmentationSn(self):
        pid = self._pmap['SegmentationSn']
        return self.qPlanes[pid]
    
    @property
    def A_V__yx(self):
        return self.spatialize(self.A_V, extensive=False)
    
    @property
    def v_0__yx(self):
        return self.spatialize(self.v_0, extensive=False)
    
    @property
    def v_d__yx(self):
        return self.spatialize(self.v_d, extensive=False)
    
    @property
    def adev__yx(self):
        return self.spatialize(self.adev, extensive=False)
    
    @property
    def chi2__yx(self):
        return self.spatialize(self.chi2, extensive=False)
    
    
    
