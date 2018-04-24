'''
Natalia@UFSC - 29/May/2017

Fitting emission lines for Nebulatom3.

Based on dobby.
'''

import numpy as np

import astropy.constants as const
from astropy.modeling import Fittable1DModel, Parameter


c = const.c.to('km/s').value


class ResampledGaussian(Fittable1DModel):

    l0 = Parameter(fixed=True)
    flux = Parameter()
    v0 = Parameter()
    vd = Parameter()
    vd_inst = Parameter(fixed=True)

    def __getitem__(self, i):
        return self
    
    def bounding_box(self, factor=5.5):
        l0 = self.l0.value * (self.v0 / c)
        dl = factor * self.l0.value * self.vd / c
        return (l0 - dl, l0 + dl)

    @staticmethod
    def evaluate(l, l0, flux, v0, vd, vd_inst):

            
        def gaussian_int(l):
            from scipy.special import erf
            v = c * (l - l0) / l0
            vd_obs2 = vd ** 2 + vd_inst ** 2
            sigma_l = np.sqrt(vd_obs2) * l0 / c
            ampl = flux / np.sqrt(2. * np.pi * vd_obs2)
            x = (v - v0) / np.sqrt(2 * vd_obs2)
            return erf(x) * ampl * np.sqrt(vd_obs2 * np.pi / 2.)
        
        dl = l[1] - l[0]
        lbin_low = l - dl/2.
        lbin_upp = l + dl/2.
        aux1 = gaussian_int(lbin_low)
        aux2 = gaussian_int(lbin_upp)
        fbin = (aux2 - aux1) / dl
        
        return fbin
    
    @staticmethod
    def fit_deriv(l, l0, flux, v0, vd, vd_inst):
        #v = c * (l - l0) / l0
        vd_obs2 = vd ** 2 + vd_inst ** 2
        d_flux = np.exp( -0.5 * (v - v0) ** 2 / vd_obs2 ) / np.sqrt(2. * np.pi * vd_obs2)
        d_v0   = flux * d_flux * (v - v0) / vd_obs2
        d_vd   = flux * d_flux * (v - v0) ** 2 * vd / vd_obs2 ** 2
        return [0., d_flux, d_v0, d_vd]


def MultiResampledGaussian(l0, flux, v0, vd, vd_inst, name, v0_min=None, v0_max=None, vd_min=None, vd_max=None):
    # TODO: Sanity checks.

    flux = [flux, ] * len(l0)
    models = [ResampledGaussian(l, a, v0, vd, vdi, name=n)
              for l, a, vdi, n in zip(l0, flux, vd_inst, name)]

    model = models[0]
    for i in np.arange(1, len(models)):
        model += models[i]

    for n in name:
        model[n].flux.min = 0.0

    if v0_min is not None:
        for n in name:
            model[n].v0.min = v0_min

    if v0_max is not None:
        for n in name:
            model[n].v0.max = v0_max

    if vd_min is not None:
        for n in name:
            model[n].vd.min = vd_min

    if vd_max is not None:
        for n in name:
            model[n].vd.max = vd_max

    return model


