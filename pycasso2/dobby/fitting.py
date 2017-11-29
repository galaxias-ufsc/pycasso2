'''
Natalia@UFSC - 29/Nov/2017
'''

from os import path

import numpy as np

from astropy.table import Table
from astropy.modeling import fitting

lines_windows = Table.read(path.join(path.dirname(__file__), 'lines.dat'), format = 'ascii.commented_header')

def calc_cont_EW(_ll, _f_syn, flux_integrated, linename):
    # Select a line from our red/blue continua table
    flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)

    # Get the blue and red continuum median
    blue_median = np.median(_f_syn[(_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line]['blue2'])])
    red_median  = np.median(_f_syn[(_ll >= lines_windows[flag_line][ 'red1']) & (_ll <= lines_windows[flag_line][ 'red2'])])

    # Get the midpoint wavelengths
    blue_lambda = (lines_windows[flag_line]['blue1'] + lines_windows[flag_line]['blue2']) / 2.
    red_lambda  = (lines_windows[flag_line][ 'red1'] + lines_windows[flag_line][ 'red2']) / 2.
    central_lambda = lines_windows[flag_line]['central']

    # Calculate line parameters
    a = (red_median - blue_median) / (red_lambda - blue_lambda)
    b = blue_median - a * blue_lambda

    # Calculate continuum and equivalent width
    C  = a * central_lambda + b
    EW = flux_integrated / C

    return EW




def fit_strong_lines(_ll, _f_res, _f_syn, _f_wei, kinematic_ties_on = True,
                     model = 'resampled_gaussian',
                     saveAll = False, saveHDF5 = False, saveTXT = False,
                     outname = None, outdir = None, **kwargs):

    if model == 'resampled_gaussian':
        from .models.resampled_gaussian import MultiResampledGaussian
        elModel = MultiResampledGaussian

        
    # Normalize spectrum by the median, to avoid problems in the fit
    med = np.median(_f_syn)
    f_res = _f_res / np.abs(med)
    
	# Fitting Hb, Ha and [NII]
    l0 = [4861.325, 6562.80, 6548.04, 6583.46]
    name = ['4861', '6563', '6548', '6584']
    vd_inst = [0., 0., 0., 0.]
    mod_init_HbHaN2 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)
    mod_init_HbHaN2['6584'].flux.tied = lambda m: 3 * m['6548'].flux
    if kinematic_ties_on:
        #mod_init_HbHaN2['4861'].v0.tied = lambda m: m['6563'].v0
        #mod_init_HbHaN2['4861'].vd.tied = lambda m: m['6563'].vd
        mod_init_HbHaN2['6548'].v0.tied = lambda m: m['6584'].v0
        mod_init_HbHaN2['6548'].vd.tied = lambda m: m['6584'].vd
    fitter = fitting.LevMarLSQFitter()
    mod_fit_HbHaN2 = fitter(mod_init_HbHaN2, _ll, f_res) #, weights=_f_wei)
    
    # Fitting [OIII]
    l0 = [4958.911, 5006.843]
    name = ['4959', '5007']
    vd_inst = [0., 0.]
    mod_init_O3 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)
    mod_init_O3['5007'].flux.tied = lambda m: 2.97 * m['4959'].flux
    if kinematic_ties_on:
        mod_init_O3['4959'].v0.tied = lambda m: m['5007'].v0
        mod_init_O3['4959'].vd.tied = lambda m: m['5007'].vd
    fitter = fitting.LevMarLSQFitter()
    mod_fit_O3 = fitter(mod_init_O3, _ll, f_res) #, weights=_f_wei)


    # Rescale by the median
    el = [mod_fit_HbHaN2, mod_fit_O3]
    for model in el:
        for name in model.submodel_names:
            model[name].flux.value *= np.abs(med)

            
    # Integrate fluxes (be careful not to use the one normalized)
    flag = (_ll > 6554) & (_ll < 6573)
    flux_Ha_int =  _f_res[flag].sum()
    #print('Fluxes: ', flux_Ha_int, mod_fit_HbHaN2['6563'].flux)


    # Get EWs
    el_extra = {}
    name     = ['4861', '6563',    '6548',      '6584']
    linename = ['Hbeta', 'Halpha', '[NII]6548', '[NII]6584' ]
    for n, ln in zip(name, linename):
        el_extra[n] = {'EW': calc_cont_EW(_ll, _f_syn, mod_fit_HbHaN2[n].flux, n),
                       'linename' :  ln}
        
    name     = ['4959',       '5007']
    linename = ['[OIII]4959', '[OIII]5007']
    for n, ln in zip(name, linename):
        el_extra[n] = {'EW': calc_cont_EW(_ll, _f_syn, mod_fit_O3[n].flux, n),
                       'linename' :  ln}

    el.append(el_extra)
    
    if saveAll:
        saveHDF5 = True
        saveTXT = True

    if saveHDF5 or saveTXT:
        from .utils import save_summary_to_file
        save_summary_to_file(el, outdir, outname, saveHDF5 = saveHDF5, saveTXT = saveTXT, **kwargs)

    return el

        


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
    print(F_int / el[0]['4861'].flux.value)

    flag_Ha = (ll > 6554) & (ll < 6570)
    F_int = f_res[flag_Ha].sum()
    print(F_int / el[0]['6563'].flux.value)
