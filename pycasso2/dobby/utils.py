'''
Natalia@UFSC - 29/Nov/2017
'''

import h5py
from os import path, remove
import numpy as np
from astropy.table import Table


def summary_elines(el):
    '''
    Save emission line info to an easy-to-use array.
    '''

    mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_HaN2, mod_fit_S2, el_extra = el
    el = mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_HaN2, mod_fit_S2
    
    N_models = len(el)

    # Start empty arrays
    El_line  = []
    El_l     = []
    El_F     = []
    El_l0    = []
    El_v0    = []
    El_vd    = []
    El_vdins = []
    El_lcrms = []
    
    for i_model in range(N_models):

        # Get one astropy model
        model = el[i_model]

        # Get all submodel info in the compound model
        El_l.extend(  [                   submodel for submodel in model.submodel_names ] )
        El_F.extend(  [ model[submodel].flux.value for submodel in model.submodel_names ] )
        El_l0.extend( [ model[submodel].l0.value   for submodel in model.submodel_names ] )
        El_v0.extend( [ model[submodel].v0.value   for submodel in model.submodel_names ] )
        El_vd.extend( [ model[submodel].vd.value   for submodel in model.submodel_names ] )

    # And now get info from extra dictionary (outside astropy model)
    El_flag  = [el_extra[l]['flag']     for l in El_l]
    El_EW    = [el_extra[l]['EW']       for l in El_l]
    El_line  = [el_extra[l]['linename'] for l in El_l]
    El_vdins = [el_extra[l]['vd_inst']  for l in El_l]
    El_lcrms = [el_extra[l]['rms_lc']   for l in El_l]

    # Save lines as integers
    El_l = np.int_(El_l)
    
    # Clean nan's and inf's
    El_l0    = replace_nan_inf_by_minus999( np.array(El_l0)    )
    El_F     = replace_nan_inf_by_minus999( np.array(El_F)     )
    El_v0    = replace_nan_inf_by_minus999( np.array(El_v0)    )
    El_vd    = replace_nan_inf_by_minus999( np.array(El_vd)    )
    El_flag  = replace_nan_inf_by_minus999( np.array(El_flag)  )
    El_EW    = replace_nan_inf_by_minus999( np.array(El_EW)    )
    El_vdins = replace_nan_inf_by_minus999( np.array(El_vdins) )
    El_lcrms = replace_nan_inf_by_minus999( np.array(El_lcrms) )

    # Save table
    elines = Table( { "lambda"   : El_l     ,
                      "line"     : El_line  ,
                      "El_l0"    : El_l0    ,
                      "El_F"     : El_F     ,
                      "El_v0"    : El_v0    ,
                      "El_vd"    : El_vd    ,
                      "El_flag"  : El_flag  ,
                      "El_EW"    : El_EW    ,
                      "El_lcrms" : El_lcrms ,
                      "El_vdins" : El_vdins ,
                    } )

    # Convert unicode
    elines.convert_unicode_to_bytestring()

    spec = Table( { 'total_lc'      : el_extra['total_lc'].data ,
                    'total_lc_mask' : el_extra['total_lc'].mask
                  } )
    
    return elines, spec


def save_summary_to_file(el, outdir, outname, saveHDF5 = False, saveTXT = False, overwrite = False):
    '''
    f_h5py must be an open h5py file.
    '''

    elines, spec = summary_elines(el)

    if saveHDF5:
        outfile = path.join(outdir, '%s.hdf5' % outname)

        if overwrite:
            try:
                remove(outfile)
            except OSError:
                pass
            
        with h5py.File(outfile, 'w') as f:
            ds1 = f.create_dataset('elines', data = elines, compression = 'gzip', compression_opts = 4)
            ds2 = f.create_dataset('spec', data = spec, compression = 'gzip', compression_opts = 4)
            
    if saveTXT:
        outfile = path.join(outdir, '%s.txt' % outname)

        if overwrite:
            try:
                remove(outfile)
            except OSError:
                pass
            
        elines.write(outfile, format = 'ascii.fixed_width_two_line')

    # TO DO!
    #    f.write(footer)
    #    f.write('# Emission lines measured by dobby | 27/Sep/2017\n' % os.path.basename(__file__))

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

def plot_el_starlight(ts, el, save = False):
    
    ll = ts.spectra.l_obs

    f_obs = ts.spectra.f_obs
    f_syn = ts.spectra.f_syn
    f_wei = ts.spectra.f_wei
    f_err = f_wei**-1
    f_res = (f_obs - f_syn)

    plot_el(ll, f_res, el)
    
    
def plot_el(ll, f_res, el, ifig = 1):
    import matplotlib
    #++matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    
    fig = plt.figure(ifig, figsize=(12,6))
    gs = gridspec.GridSpec(2, 3)

    mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_HaN2, mod_fit_S2, el_extra = el

    # Start full spectrum
    m = np.full_like(ll, np.nan)
    
    # Write fluxes in FHa units
    FHa =  mod_fit_HaN2['6563'].flux.value
    

    # Plot [OIII]
    ax2 = plt.subplot(gs[1, 0])
    #flag = (ll >= 4900) & (ll < 5100)
    lc = el_extra[mod_fit_O3.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_O3.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_O3(ll[flag]) + lc[flag], 'r', label = 'Fit')
    m[flag] = mod_fit_O3(ll[flag]) + lc[flag]
    for i, name in enumerate(mod_fit_O3.submodel_names):
        ax2.text(0.99, 0.50 + i*0.45, '$\lambda_0 = %s$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$ \n good fit: %s' 
                 % (name, mod_fit_O3[name].v0.value, mod_fit_O3[name].vd.value, mod_fit_O3[name].flux.value / FHa, good_fit),
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax2.transAxes)

    # Plot Hbeta
    ax3 = plt.subplot(gs[1, 1])
    #flag = (ll >= 4750) & (ll < 4950)
    lc = el_extra[mod_fit_HbHg.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_HbHg.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_HbHg(ll[flag]) + lc[flag], 'r', label = 'Fit')
    ax3.text(0.99, 0.95, '$\lambda_0 = 4861$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$ \n good fit: %s' 
             % (mod_fit_HbHg['4861'].v0.value, mod_fit_HbHg['4861'].vd.value, mod_fit_HbHg['4861'].flux.value / FHa, good_fit),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax3.transAxes)

    # Add Hgamma too
    m[flag] = mod_fit_HbHg(ll[flag]) + lc[flag]
    lc = el_extra[mod_fit_HbHg.submodel_names[1]]['local_cont']
    flag = ~lc.mask
    m[flag] = mod_fit_HbHg(ll[flag]) + lc[flag]
    
    # Plot [NII, Halpha]
    ax4 = plt.subplot(gs[1, 2])
    #flag = (ll >= 6500) & (ll < 6600)
    lc = el_extra[mod_fit_HaN2.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_HaN2.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_HaN2(ll[flag]) + lc[flag], 'r', label = 'Fit')
    m[flag] = mod_fit_HaN2(ll[flag]) + lc[flag]
    for i, name in enumerate(mod_fit_HaN2.submodel_names):
        xlab, ylab = 0.99, 0.55 + (i-1)*0.40
        if name == '6563':
            xlab, ylab = 0.50, 0.95
        ax4.text(xlab, ylab, '$\lambda_0 = %s$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$ \n good fit: %s' 
                 % (name, mod_fit_HaN2[name].v0.value, mod_fit_HaN2[name].vd.value, mod_fit_HaN2[name].flux.value / FHa, good_fit),
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax4.transAxes)

    # Plot [OII]
    lc = el_extra[mod_fit_O2.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_O2.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_O2(ll[flag]) + lc[flag]
        
    # Plot [SII]
    lc = el_extra[mod_fit_S2.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_S2.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_S2(ll[flag]) + lc[flag]
    
    # Plot the full spectrum
    ax1 = plt.subplot(gs[0, :])
    plt.plot(ll, f_res, 'k', label = 'Residual', drawstyle = 'steps-mid')
    plt.plot(ll, m, 'r', label = 'Fit', drawstyle = 'steps-mid')
    plt.legend(loc = 'upper right')

    return fig
    
def replace_by_minus999(x, x_old, x_new = -999):

    y = x.copy()

    if x_old == 'nan':
        flag = np.isnan(x)
    elif x_old == 'inf':
        flag = np.isinf(x)
    else:
        flag = (x == x_old)

    y[flag] = x_new

    return y

def replace_nan_inf_by_minus999(x):
    
    y1 = replace_by_minus999(x,  'nan')
    y2 = replace_by_minus999(y1, 'inf')
    
    return y2

