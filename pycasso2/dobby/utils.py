'''
Natalia@UFSC - 29/Nov/2017
'''

import sys
from os import path, remove
from astropy import log

import h5py
import numpy as np
from astropy.table import Table

el_table_dtype = [('lambda', 'int32'),
                  ('line', 'S20'),
                  ('El_l0', 'float64'),
                  ('El_F', 'float64'),
                  ('El_v0', 'float64'),
                  ('El_vd', 'float64'),
                  ('El_flag', 'int32'),
                  ('El_EW', 'float64'),
                  ('El_lcrms', 'float64'),
                  ('El_vdins', 'float64'),
                  ]

el_lc_dtype = [('l_obs', 'float64'),
               ('total_lc', 'float64'),
               ]


def summary_elines(el):
    '''
    Save emission line info to an easy-to-use array.
    '''

    mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_O3_weak, mod_fit_O1, mod_fit_HaN2,mod_fit_N2He2He1, mod_fit_S2, mod_fit_O2_weak, mod_fit_Ar3, mod_fit_Fe3, mod_fit_Ne3, mod_fit_ArIV, mod_fit_Cl3, mod_fit_S3, el_extra = el
    el = mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_O3_weak, mod_fit_O1, mod_fit_HaN2,mod_fit_N2He2He1, mod_fit_S2, mod_fit_O2_weak, mod_fit_Ar3, mod_fit_Fe3, mod_fit_Ne3, mod_fit_ArIV, mod_fit_Cl3, mod_fit_S3


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
    El_l0    = replace_nan_inf_by_minus999( np.hstack(El_l0)   )
    El_F     = replace_nan_inf_by_minus999( np.hstack(El_F)    )
    El_v0    = replace_nan_inf_by_minus999( np.hstack(El_v0)   )
    El_vd    = replace_nan_inf_by_minus999( np.hstack(El_vd)   )
    El_flag  = replace_nan_inf_by_minus999( np.hstack(El_flag) )
    El_EW    = replace_nan_inf_by_minus999( np.hstack(El_EW)   )
    El_vdins = replace_nan_inf_by_minus999( np.hstack(El_vdins))
    El_lcrms = replace_nan_inf_by_minus999( np.hstack(El_lcrms))

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


def new_summary_elines(el):
    '''
    Save emission line info to an easy-to-use array.
    Modified version using a fixed dtype.
    '''

    mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_O3_weak, mod_fit_O1, mod_fit_HaN2, mod_fit_N2He2He1, mod_fit_S2, mod_fit_O2_weak, mod_fit_Ar3, mod_fit_Fe3,mod_fit_Ne3, mod_fit_ArIV, mod_fit_Cl3, mod_fit_S3, el_extra = el
    el = mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_O3_weak, mod_fit_O1, mod_fit_HaN2, mod_fit_N2He2He1, mod_fit_S2, mod_fit_O2_weak, mod_fit_Ar3, mod_fit_Fe3, mod_fit_Ne3, mod_fit_ArIV, mod_fit_Cl3, mod_fit_S3


    N_models = len(el)

    # Start empty arrays
    El_l     = []
    El_F     = []
    El_l0    = []
    El_v0    = []
    El_vd    = []

    for i_model in range(N_models):

        # Get one astropy model
        model = el[i_model]

        # Get all submodel info in the compound model
        El_l.extend(  [                   submodel       for submodel in model.submodel_names ] )
        El_F.extend(  [ model[submodel].flux.value       for submodel in model.submodel_names ] )
        El_l0.extend( [ model[submodel].l0.value         for submodel in model.submodel_names ] )
        El_v0.extend( [ model[submodel].v0.value         for submodel in model.submodel_names ] )
        El_vd.extend( [ np.abs(model[submodel].vd.value) for submodel in model.submodel_names ] )

    # And now get info from extra dictionary (outside astropy model)
    El_flag  = [el_extra[l]['flag']     for l in El_l]
    El_EW    = [el_extra[l]['EW']       for l in El_l]
    El_line  = [el_extra[l]['linename'] for l in El_l]
    El_vdins = [el_extra[l]['vd_inst']  for l in El_l]
    El_lcrms = [el_extra[l]['rms_lc']   for l in El_l]

    # Save lines as integers
    El_l = np.int_(El_l)
    # Clean nan's and inf's
    El_l0    = replace_nan_inf_by_minus999( np.hstack(El_l0)   )
    El_F     = replace_nan_inf_by_minus999( np.hstack(El_F)    )
    El_v0    = replace_nan_inf_by_minus999( np.hstack(El_v0)   )
    El_vd    = replace_nan_inf_by_minus999( np.hstack(El_vd)   )
    El_flag  = replace_nan_inf_by_minus999( np.hstack(El_flag) )
    El_EW    = replace_nan_inf_by_minus999( np.hstack(El_EW)   )
    El_vdins = replace_nan_inf_by_minus999( np.hstack(El_vdins))
    El_lcrms = replace_nan_inf_by_minus999( np.hstack(El_lcrms))

    # Save table
    data = [El_l, El_line, El_l0, El_F, El_v0, El_vd, El_flag, El_EW, El_lcrms, El_vdins]
    elines = Table(np.core.records.fromarrays(data, dtype=el_table_dtype))

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

def read_summary_from_file(filename, readHDF5 = True):

    if readHDF5:
        elines = Table.read(filename, path='elines')
        spec   = Table.read(filename, path='spec')
    else:
        elines = Table.read(filename, format = 'ascii.fixed_width_two_line')
        spec = None

    return elines, spec

def get_el_info(elines, line, info):
    '''
    Get emission line info
    '''
    f = (elines['lambda'] == line)
    return elines[info][f]

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

def plot_el_starlight(ts, el, save = False, display_plot = True):

    ll = ts.spectra.l_obs

    f_obs = ts.spectra.f_obs
    f_syn = ts.spectra.f_syn
    f_wei = ts.spectra.f_wei
    f_err = f_wei**-1
    f_res = (f_obs - f_syn)

    plot_el(ll, f_res, el, display_plot = display_plot)


def plot_el(ll, f_res, el, ifig = 1, display_plot = False):
    import matplotlib
    if not display_plot:
        matplotlib.use('pdf')

    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(ifig, figsize=(12,6))
    gs = gridspec.GridSpec(3, 3)

    mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_O3_weak, mod_fit_O1, mod_fit_HaN2, mod_fit_N2He2He1, mod_fit_S2, mod_fit_O2_weak, mod_fit_Ar3, mod_fit_Fe3,mod_fit_Ne3, mod_fit_ArIV, mod_fit_Cl3, mod_fit_S3, el_extra = el

    # Start full spectrum
    m  = np.full_like(ll, np.nan)
    l = np.full_like(ll, np.nan)

    # Write fluxes in FHa units
    FHa =  mod_fit_HaN2['6563'].flux.value

    # Plot the full spectrum
    ax2 = plt.subplot(gs[1, :])
      #flag = (ll >= 4900) & (ll < 5100)
    lc = el_extra[mod_fit_O3_weak.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_O3_weak.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_O3_weak(ll[flag]) + lc[flag], 'r', label = 'Fit')
    plt.plot(ll[flag], lc[flag], color='grey', label = 'Local continuum', zorder=-1, alpha=0.5)
    m[flag] = mod_fit_O3_weak(ll[flag]) + lc[flag]
    l[flag] = lc[flag]
    for i, name in enumerate(mod_fit_O3_weak.submodel_names):
        ax2.text( 0.20 + i*0.20,0.95, '$\lambda_0 = %s$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$ \n good fit: %s'
                 % (name, mod_fit_O3_weak[name].v0.value, mod_fit_O3_weak[name].vd.value, mod_fit_O3_weak[name].flux.value / FHa, good_fit),
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax2.transAxes)



    # Plot [OIII]weak
    ax5 = plt.subplot(gs[2, 0])
    #flag = (ll >= 4900) & (ll < 5100)
    lc = el_extra[mod_fit_O3_weak.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_O3_weak.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_O3_weak(ll[flag]) + lc[flag], 'r', label = 'Fit')
    plt.plot(ll[flag], lc[flag], color='grey', label = 'Local continuum', zorder=-1, alpha=0.5)
    m[flag] = mod_fit_O3_weak(ll[flag]) + lc[flag]
    l[flag] = lc[flag]
    ax5.text(0.99, 0.95, '$\lambda_0 = 4363$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$ \n good fit: %s'
             % (mod_fit_O3_weak['4363'].v0.value, mod_fit_O3_weak['4363'].vd.value, mod_fit_O3_weak['4363'].flux.value / FHa, good_fit),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax5.transAxes)


    # Plot [NII]weak
    ax6 = plt.subplot(gs[2, 1])
    #flag = (ll >= 4900) & (ll < 5100)
    lc = el_extra[mod_fit_N2He2He1.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_N2He2He1.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag],mod_fit_N2He2He1(ll[flag]) + lc[flag], 'r', label = 'Fit')
    plt.plot(ll[flag], lc[flag], color='grey', label = 'Local continuum', zorder=-1, alpha=0.5)
    plt.xlabel('[NII]5755')
    m[flag] = mod_fit_N2He2He1(ll[flag]) + lc[flag]
    l[flag] = lc[flag]
    ax6.text(0.99, 0.95, '$\lambda_0 = 5755$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$ \n good fit: %s'
             % (mod_fit_N2He2He1['5755'].v0.value, mod_fit_N2He2He1['5755'].vd.value,mod_fit_N2He2He1['5755'].flux.value / FHa, good_fit),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax6.transAxes)



    # Plot [OII] weak
    ax7 = plt.subplot(gs[2, 2])
    lc = el_extra[mod_fit_O2_weak.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_O2_weak.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    plt.plot(ll[flag], f_res[flag], 'k', label = 'Residual')
    plt.plot(ll[flag], mod_fit_O2_weak(ll[flag]) + lc[flag], 'r', label = 'Fit')
    plt.plot(ll[flag], lc[flag], color='grey', label = 'Local continuum', zorder=-1, alpha=0.5)
    plt.xlabel('[OII]7320')
    m[flag] = mod_fit_O2_weak(ll[flag]) + lc[flag]
    l[flag] = lc[flag]
    ax7.text(0.99, 0.95, '$\lambda_0 = 7320$ \n $v_0 = %.2f$ \n $v_d = %.2f$ \n $F = %.2e$ \n good fit: %s'
             % (mod_fit_O2_weak['7320'].v0.value, mod_fit_O2_weak['7320'].vd.value, mod_fit_O2_weak['7320'].flux.value / FHa, good_fit),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax7.transAxes)




    #Plot [OII]
    lc = el_extra[mod_fit_O2.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_O2.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_O2(ll[flag]) + lc[flag]
    l[flag] = lc[flag]


    # Plot [SII]
    lc = el_extra[mod_fit_S2.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_S2.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_S2(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    #Plot[OIII]4363 (weak line)
    lc = el_extra[mod_fit_O3_weak.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_O3_weak.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_O3_weak(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    #Plot[OIII]4363 (weak line)
    lc = el_extra[mod_fit_O3_weak.submodel_names[1]]['local_cont']
    good_fit = el_extra[mod_fit_O3_weak.submodel_names[1]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_O3_weak(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    #Plot [NII]5755 (weak line)
    lc = el_extra[mod_fit_N2He2He1.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_N2He2He1.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_N2He2He1(ll[flag]) + lc[flag]
    l[flag] = lc[flag]


    #Plot [OII]weak
    lc = el_extra[mod_fit_O2_weak.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_O2_weak.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_O2_weak(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    #Plot [FeIII]
    lc = el_extra[mod_fit_Fe3.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_Fe3.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_Fe3(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    #Plot [ArIII]
    lc = el_extra[mod_fit_Ar3.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_Ar3.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_Ar3(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    #Plot [SIII]
    lc = el_extra[mod_fit_S3.submodel_names[0]]['local_cont']
    good_fit = el_extra[mod_fit_S3.submodel_names[0]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_S3(ll[flag]) + lc[flag]
    l[flag] = lc[flag]



    #Plot [HeII]
    lc = el_extra[mod_fit_N2He2He1.submodel_names[1]]['local_cont']
    good_fit = el_extra[mod_fit_N2He2He1.submodel_names[1]]['flag'] == 0
    flag = ~lc.mask
    m[flag] = mod_fit_N2He2He1(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    # [HeI]
    m[flag] = mod_fit_N2He2He1(ll[flag]) + lc[flag]
    lc = el_extra[mod_fit_N2He2He1.submodel_names[2]]['local_cont']
    flag = ~lc.mask
    m[flag] = mod_fit_N2He2He1(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    # [FeIII]
    m[flag] = mod_fit_Fe3(ll[flag]) + lc[flag]
    lc = el_extra[mod_fit_Fe3.submodel_names[0]]['local_cont']
    flag = ~lc.mask
    m[flag] = mod_fit_Fe3(ll[flag]) + lc[flag]
    l[flag] = lc[flag]


    # [SIII]
    m[flag] = mod_fit_Ne3(ll[flag]) + lc[flag]
    lc = el_extra[mod_fit_Ne3.submodel_names[0]]['local_cont']
    flag = ~lc.mask
    m[flag] = mod_fit_Ne3(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    # [NeIII]
    m[flag] = mod_fit_Ne3(ll[flag]) + lc[flag]
    lc = el_extra[mod_fit_Ne3.submodel_names[1]]['local_cont']
    flag = ~lc.mask
    m[flag] = mod_fit_Ne3(ll[flag]) + lc[flag]
    l[flag] = lc[flag]


    # [ArIV]
    m[flag] = mod_fit_ArIV(ll[flag]) + lc[flag]
    lc = el_extra[mod_fit_ArIV.submodel_names[0]]['local_cont']
    flag = ~lc.mask
    m[flag] = mod_fit_ArIV(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    # [ClIII]
    m[flag] = mod_fit_Cl3(ll[flag]) + lc[flag]
    lc = el_extra[mod_fit_Cl3.submodel_names[0]]['local_cont']
    flag = ~lc.mask
    m[flag] = mod_fit_Cl3(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    # [ClIII]
    m[flag] = mod_fit_Cl3(ll[flag]) + lc[flag]
    lc = el_extra[mod_fit_Cl3.submodel_names[1]]['local_cont']
    flag = ~lc.mask
    m[flag] = mod_fit_Cl3(ll[flag]) + lc[flag]
    l[flag] = lc[flag]

    # Plot the full spectrum
    ax1 = plt.subplot(gs[0, :])
    plt.plot(ll, f_res, 'k', label = 'Residual', drawstyle = 'steps-mid')
    plt.plot(ll, m, 'r', label = 'Fit', drawstyle = 'steps-mid')
    plt.plot(ll, l, 'grey', label = 'Local continuum', drawstyle = 'steps-mid', zorder=-1, alpha=0.5)
    plt.legend(loc = 'upper right')

    if display_plot:
        plt.ioff()
        plt.show()
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

def safe_pow(a, b):
    a = safe_x(a)
    b = safe_x(b)
    apb = np.where( ((a > -999.) & (b > -999.)), a**b + (a <= -999.), -999. )
    apb = np.where( ((a == 0.) & (b < 0.)), -999., apb )
    apb = np.ma.masked_where((apb < -990), apb)
    return apb

def safe_x(x):
    x = np.where(np.isfinite(x), x, -999.)
    x = np.ma.masked_where((x < -990), x)
    return x
