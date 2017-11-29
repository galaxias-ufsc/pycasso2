'''
Natalia@UFSC - 29/May/2017

Fitting emission lines for Nebulatom3.

Based on dobby.
'''

import atpy
import pystarlight.io.starlighttable #io.starlighttable #@UnusedImport
from astropy.modeling import Fittable1DModel, Parameter, fitting
from astropy.table import Table
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

c = const.c.to('km/s').value


# TODO: Wrap around Gaussian1D.
class GaussianELModel(Fittable1DModel):

    l0 = Parameter(fixed=True)
    flux = Parameter()
    v0 = Parameter()
    vd = Parameter()
    vd_inst = Parameter(fixed=True)
    
    def bounding_box(self, factor=5.5):
        l0 = self.l0.value * (self.v0 / c)
        dl = factor * self.l0.value * self.vd / c
        return (l0 - dl, l0 + dl)

    @staticmethod
    def evaluate(l, l0, flux, v0, vd, vd_inst):
        v = c * (l - l0) / l0
        vd_obs2 = vd ** 2 + vd_inst ** 2
        sigma_l = np.sqrt(vd_obs2) * l0 / c
        ampl = flux / (sigma_l * np.sqrt(2. * np.pi))
        return ampl * np.exp( -0.5 * (v - v0)** 2 / vd_obs2 )

    @staticmethod
    def fit_deriv(l, l0, flux, v0, vd, vd_inst):
        vd_obs2 = vd ** 2 + vd_inst ** 2
        d_flux = np.exp( -0.5 * (v - v0) ** 2 / vd_obs2 ) / np.sqrt(2. * np.pi * vd_obs2)
        d_v0   = flux * d_flux * (v - v0) / vd_obs2
        d_vd   = flux * d_flux * (v - v0) ** 2 * vd / vd_obs2 ** 2
        return [d_flux, d_v0, d_vd]


def ElementEmissionLines(l0, flux, v0, vd, vd_inst, name, ratio=None, v0_min=None, v0_max=None, vd_min=None, vd_max=None):
    # TODO: Sanity checks.


    flux = [flux, ] * len(l0)
    models = [GaussianELModel(l, a, v0, vd, vdi, name=n)
              for l, a, vdi, n in zip(l0, flux, vd_inst, name)]
    model = reduce(lambda x, y: x + y, models)

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


def fit_strong_lines_starlight(tc, ts, **kwargs):
    # Reading from fits
    ll = ts.spectra.l_obs
    f_obs = ts.spectra.f_obs
    f_syn = ts.spectra.f_syn
    f_res = (f_obs - f_syn)

    # Reading from cxt file
    # Assumes the binning is the same in both files
    ll_cxt = tc['lobs']
    flag_cxt = (ll_cxt >= ll[0]) & (ll_cxt <= ll[-1])
    f_err = tc['ferr']
    f_wei = f_err**-1

    # Cleaning spectra
    flag = ((f_wei > 0) & (tc['flag'] <= 1))[flag_cxt]

    _ll = ll[flag]
    _f_res = f_res[flag]
    _f_wei = f_wei[flag_cxt][flag]

    return fit_strong_lines(_ll, _f_res, _f_wei, **kwargs)
    
def fit_strong_lines(_ll, _f_res, _f_wei, kinematic_ties_on = True):

	# Fitting Hb, Ha and [NII]
    l0 = [4861.325, 6562.80, 6548.04, 6583.46]
    name = ['4861', '6563', '6548', '6584']
    vd_inst = [0., 0., 0., 0.]
    mod_init_HbHaN2 = ElementEmissionLines(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)
    mod_init_HbHaN2['6584'].flux.tied = lambda m: 3 * m['6548'].flux
    if kinematic_ties_on:
        mod_init_HbHaN2['4861'].v0.tied = lambda m: m['6563'].v0
        mod_init_HbHaN2['4861'].vd.tied = lambda m: m['6563'].vd
        mod_init_HbHaN2['6548'].v0.tied = lambda m: m['6584'].v0
        mod_init_HbHaN2['6548'].vd.tied = lambda m: m['6584'].vd
    fitter = fitting.LevMarLSQFitter()
    mod_fit_HbHaN2 = fitter(mod_init_HbHaN2, _ll, _f_res) #, weights=_f_wei)
    
    # Fitting [OIII]
    l0 = [4958.911, 5006.843]
    name = ['4959', '5007']
    vd_inst = [0., 0.]
    mod_init_O3 = ElementEmissionLines(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)
    #mod_init_O3['5007'].flux.tied = lambda m: 2.97 * m['4959'].flux
    if kinematic_ties_on:
        mod_init_O3['4959'].v0.tied = lambda m: m['5007'].v0
        mod_init_O3['4959'].vd.tied = lambda m: m['5007'].vd
    fitter = fitting.LevMarLSQFitter()
    mod_fit_O3 = fitter(mod_init_O3, _ll, _f_res) #, weights=_f_wei)
    
    
    return mod_fit_HbHaN2, mod_fit_O3


def plot_el(ts, el, save = False):

    mod_fit_HbHaN2, mod_fit_O3 = el
    
    ll = ts.spectra.l_obs

    f_obs = ts.spectra.f_obs
    f_syn = ts.spectra.f_syn
    f_wei = ts.spectra.f_wei
    f_err = f_wei**-1
    f_res = (f_obs - f_syn)

    fig=plt.figure(1, figsize=(12,6))
    gs = gridspec.GridSpec(2, 3)

    # Plot the full spectrum
    ax1 = plt.subplot(gs[0, :])
    plt.plot(ll, f_res, 'k', label = 'Residual')
    plt.plot(ll, (mod_fit_HbHaN2(ll) + mod_fit_O3(ll)), 'r', label = 'Fit')
    plt.legend(loc = 'upper right')

    # Write fluxes in FHa units
    FHa =  mod_fit_HbHaN2['6563'].flux.value
    
    # Plot [OIII]
    ax2 = plt.subplot(gs[1, 0])
    flag = (ll >= 4900) & (ll < 5100)
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_O3(ll[flag]), 'r', label = 'Fit')
    for i, name in enumerate(mod_fit_O3.submodel_names):
        ax2.text(0.99, 0.50 + i*0.45, '$\lambda_0 = %s$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$' 
                 % (name, mod_fit_O3[name].v0.value, mod_fit_O3[name].vd.value, mod_fit_O3[name].flux.value / FHa),
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax2.transAxes)

    # Plot Hbeta 
    ax3 = plt.subplot(gs[1, 1])
    flag = (ll >= 4750) & (ll < 4950)
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_HbHaN2(ll[flag]), 'r', label = 'Fit')
    ax3.text(0.99, 0.95, '$\lambda_0 = 4861$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$' 
             % (mod_fit_HbHaN2['4861'].v0.value, mod_fit_HbHaN2['4861'].vd.value, mod_fit_HbHaN2['4861'].flux.value / FHa),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax3.transAxes)
    
    # Plot [NII, Halpha]
    ax4 = plt.subplot(gs[1, 2])
    flag = (ll >= 6500) & (ll < 6600)
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_HbHaN2(ll[flag]), 'r', label = 'Fit')
    for i, name in enumerate(mod_fit_HbHaN2.submodel_names[1:]):
        xlab, ylab = 0.99, 0.50 + (i-1)*0.45
        if name == '6563':
            xlab, ylab = 0.50, 0.95
        ax4.text(xlab, ylab, '$\lambda_0 = %s$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$' 
                 % (name, mod_fit_HbHaN2[name].v0.value, mod_fit_HbHaN2[name].vd.value, mod_fit_HbHaN2[name].flux.value / FHa),
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax4.transAxes)

    
if __name__ == "__main__":

    tc = Table.read('STARLIGHTv04/0414.51901.393.cxt', format = 'ascii', names = ('lobs', 'fobs', 'ferr', 'flag'))
    ts = atpy.TableSet('STARLIGHTv04/0414.51901.393.cxt.sc4.C11.im.CCM.BN', type = 'starlightv4')
    
    #tc = Table.read('STARLIGHTv04/0404.51812.036.7xt', format = 'ascii', names = ('lobs', 'fobs', 'ferr', 'flag'))
    #ts = atpy.TableSet('STARLIGHTv04/0404.51812.036.sc4.NA3.gm.CCM.BS', type = 'starlightv4')

    el = fit_strong_lines_starlight(tc, ts, kinematic_ties_on = False)
    plot_el(ts, el)
    
    
    # Test flux
    ll = ts.spectra.l_obs
    f_obs = ts.spectra.f_obs
    f_syn = ts.spectra.f_syn
    f_res = (f_obs - f_syn)
    
    flag_Hb = (ll > 4850) & (ll < 4870)
    F_int = f_res[flag_Hb].sum()
    print F_int / el[0]['4861'].flux.value

    flag_Ha = (ll > 6554) & (ll < 6570)
    F_int = f_res[flag_Ha].sum()
    print F_int / el[0]['6563'].flux.value
