'''
Natalia@UFSC - 29/Nov/2017
'''

from os import path

import numpy as np

from astropy.table import Table
from astropy.modeling import fitting

lines_windows = Table.read(path.join(path.dirname(__file__), 'lines.dat'), format = 'ascii.commented_header')


def calc_cont_EW(_ll, _f_syn, flux_integrated, linename):
    # Calculate continuum and equivalent width
    a, b, central_lambda = local_continuum(_ll, _f_syn, linename, return_continuum = False)
    C  = a * central_lambda + b
    EW = flux_integrated / C

    return EW


def local_continuum(_ll, _f_res, linename, return_continuum = True, debug = False):
    # Select a line from our red/blue continua table
    flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)

    # Get the blue and red continuum median
    blue_median = np.median(_f_res[(_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line]['blue2'])])
    red_median  = np.median(_f_res[(_ll >= lines_windows[flag_line][ 'red1']) & (_ll <= lines_windows[flag_line][ 'red2'])])

    # Get the midpoint wavelengths
    blue_lambda = (lines_windows[flag_line]['blue1'] + lines_windows[flag_line]['blue2']) / 2.
    red_lambda  = (lines_windows[flag_line][ 'red1'] + lines_windows[flag_line][ 'red2']) / 2.
    central_lambda = lines_windows[flag_line]['central']

    # Calculate line parameters
    a = (red_median - blue_median) / (red_lambda - blue_lambda)
    b = blue_median - a * blue_lambda

    flag_cont = (_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line][ 'red2'])
    local_cont = a * _ll + b

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(blue_lambda, blue_median, 'xb')
        plt.plot(red_lambda, red_median, 'xr')
        plt.plot(central_lambda, a*central_lambda + b, 'xk')
    
    if return_continuum:
        return local_cont, flag_cont
    else:
        return a, b, central_lambda
    

def fit_strong_lines(_ll, _f_res, _f_syn, _f_err, kinematic_ties_on = True,
                     model = 'resampled_gaussian', vd_inst = None,
                     saveAll = False, saveHDF5 = False, saveTXT = False,
                     outname = None, outdir = None, debug = False, **kwargs):

    if model == 'resampled_gaussian':
        from .models.resampled_gaussian import MultiResampledGaussian
        elModel = MultiResampledGaussian
    elif model == 'gaussian':
        from .models.gaussian import MultiGaussian
        elModel = MultiGaussian
    else:
        raise Exception('@@> No model found. Giving up.')
        
    # Normalize spectrum by the median, to avoid problems in the fit
    med = np.median(_f_syn)
    f_res = _f_res / np.abs(med)
    f_err = _f_err / np.abs(med)

    # Get vd_inst
    def get_vd_inst(vd_inst, name):
        if vd_inst is None:
            return [0. for n in name]
        elif np.isscalar(vd_inst):
            return [vd_inst for n in name]
        elif type(vd_inst) is dict:
            return [vd_inst[n] for n in name]
        else:
            sys.exit('Check vd_inst, must be a scalar a dictionary: %s' % vd_inst)


    # Get local pseudocontinuum
    el_extra = {}

    # Continuum only for Ha
    l, f = local_continuum(_ll, _f_res, '6563')
    lc = np.ma.masked_array(l, mask=~f)
    name     = ['6563',   '6548',      '6584']
    linename = ['Halpha', '[NII]6548', '[NII]6584' ]    
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc }

    # Continuum for Hb
    l, f = local_continuum(_ll, _f_res, '4861')
    lc = np.ma.masked_array(l, mask=~f)
    name     = ['4861'  ]
    linename = ['Hbeta' ]
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc }
        
    # Continuum only for 5007
    l, f = local_continuum(_ll, _f_res, '5007')
    lc = np.ma.masked_array(l, mask=~f)
    name     = ['4959',       '5007']
    linename = ['[OIII]4959', '[OIII]5007']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc }

    
	# ** Fitting Ha and [NII]

    # Parameters
    name = ['6563', '6548', '6584']
    _vd_inst = get_vd_inst(vd_inst, name)
    l0 = [0., 0., 0.]
    for il, linename in enumerate(name):
        flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)
        l0[il] = lines_windows[flag_line]['central'][0]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont']

    # Start model
    mod_init_HaN2 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)

    # Ties
    mod_init_HaN2['6584'].flux.tied = lambda m: 3 * m['6548'].flux
    if kinematic_ties_on:
        mod_init_HaN2['6548'].v0.tied = lambda m: m['6584'].v0
        mod_init_HaN2['6548'].vd.tied = lambda m: m['6584'].vd

    # Fit
    fitter = fitting.LevMarLSQFitter()
    mod_fit_HaN2 = fitter(mod_init_HaN2, _ll, f_res-lc, weights=f_err)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit1')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_HaN2(_ll)+lc)

        
	# ** Fitting Hb

    # Parameters
    name = ['4861','6563',]
    _vd_inst = get_vd_inst(vd_inst, name)
    l0 = [0., 0.]
    for il, linename in enumerate(name):
        flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)
        l0[il] = lines_windows[flag_line]['central'][0]
        
    # Get local continuum
    lc = el_extra[name[0]]['local_cont']

    # Start model
    # Fitting Ha too because the single model has a problem,
    # and many things are based on the compounded model.
    mod_init_Hb = elModel(l0, flux=0.0, v0=mod_fit_HaN2['6563'].v0, vd=mod_fit_HaN2['6563'].vd, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)

    # Ties
    if kinematic_ties_on:
        mod_init_Hb['4861'].v0.fixed = True
        mod_init_Hb['4861'].vd.fixed = True
    
    # Fit
    fitter = fitting.LevMarLSQFitter()
    mod_fit_Hb = fitter(mod_init_Hb, _ll, f_res-lc) #, weights=f_err)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit2')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Hb(_ll)+lc)
        #plt.plot(_ll, f_res-lc)
        #plt.plot(_ll, mod_fit_Hb(_ll))
        #plt.xlim(4750, 4950)
        #plt.ylim(-0.5, 2)

        
    # ** Fitting [OIII]

    # Parameters
    name = ['4959', '5007']
    _vd_inst = get_vd_inst(vd_inst, name)
    l0 = [0., 0.]
    for il, linename in enumerate(name):
        flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)
        l0[il] = lines_windows[flag_line]['central'][0]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont']

    # Start model
    mod_init_O3 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)

    # Ties
    mod_init_O3['5007'].flux.tied = lambda m: 2.97 * m['4959'].flux
    if kinematic_ties_on:
        mod_init_O3['4959'].v0.tied = lambda m: m['5007'].v0
        mod_init_O3['4959'].vd.tied = lambda m: m['5007'].vd

    # Fit
    fitter = fitting.LevMarLSQFitter()
    mod_fit_O3 = fitter(mod_init_O3, _ll, f_res-lc, weights=f_err)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O3(_ll)+lc)

    # Rescale by the median
    el = [mod_fit_HaN2, mod_fit_Hb, mod_fit_O3]
    for model in el:
        for name in model.submodel_names:
            model[name].flux.value *= np.abs(med)

            
    # Integrate fluxes (be careful not to use the one normalized)
    flag = (_ll > 6554) & (_ll < 6573)
    flux_Ha_int =  _f_res[flag].sum()
    #print('Fluxes: ', flux_Ha_int, mod_fit_HbHaN2['6563'].flux)

    
    # Get EWs
    name     = ['6563',   '6548',      '6584']
    linename = ['Halpha', '[NII]6548', '[NII]6584' ]
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_HaN2[n].flux, n)
        
    name     = ['4861'  ]
    linename = ['Hbeta' ]
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_Hb[n].flux, n)
        
    name     = ['4959',       '5007']
    linename = ['[OIII]4959', '[OIII]5007']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_O3[n].flux, n)

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
