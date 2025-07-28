'''
Natalia@UFSC - 29/Nov/2017
'''

from os import path

import numpy as np

from astropy.table import Table
from astropy.modeling import fitting
import astropy.constants as const
from astropy import log
from pycasso2.dobby import flags

from .utils import safe_pow

c = const.c.to('km/s').value


def calc_cont_EW(_ll, _f_syn, flux_integrated, linename, lines_windows):
    # Calculate continuum and equivalent width
    a, b, central_lambda = local_continuum_linear(_ll, _f_syn, linename, lines_windows, return_continuum = False)
    C  = a * central_lambda + b
    if C > 0:
        return flux_integrated / C
    else:
        return np.nan


def local_continuum_legendre(_ll, _f_res, linename, lines_windows, degree, debug = False):
    # Select a line from our red/blue continua table
    flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)

    assert (lines_windows[flag_line]['blue1'] < lines_windows[flag_line]['blue2']), f'@@> Check blue cont lambda for {linename}: {lines_windows[flag_line]["blue1"][0]} >= {lines_windows[flag_line]["blue2"][0]}'
    assert (lines_windows[flag_line]['red1'] < lines_windows[flag_line]['red2']), f'@@> Check red cont lambda for {linename}: {lines_windows[flag_line]["red1"][0]} >= {lines_windows[flag_line]["red2"][0]}'
 
    # Get the blue and red continua
    flag_blue = (_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line]['blue2'])
    flag_red  = (_ll >= lines_windows[flag_line][ 'red1']) & (_ll <= lines_windows[flag_line][ 'red2'])
    flag_cont = (_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line][ 'red2'])
    flag_windows = (flag_blue) | (flag_red)

    # Select window for Legendre polynomial fit
    flag_lc = (~_f_res.mask) & (flag_cont)

    # Deal with completely masked windows
    if (np.count_nonzero(flag_lc) == 0):
        local_cont = np.zeros_like(_ll)

    else:
        # Fit legendre polynomials
        x = np.linspace(-1, 1, np.sum(flag_lc))
        coeffs = np.polynomial.legendre.legfit(x, _f_res[flag_lc], degree)

        local_cont = np.zeros_like(_ll)
        local_cont[flag_lc] = np.polynomial.legendre.legval(x, coeffs)
        local_cont = np.interp(_ll, _ll[flag_lc], np.polynomial.legendre.legval(x, coeffs))

        if debug:
            import matplotlib.pyplot as plt
            plt.figure('local_cont%s' % linename)
            plt.clf()
            plt.plot(_ll, _f_res.data, 'red')
            #plt.plot(_ll[flag_lc], _f_res[flag_lc], 'k', zorder=10)
            plt.axhline(0, color='grey')
            plt.plot(_ll[flag_lc], np.polynomial.legendre.legval(x, coeffs), '.-', label="Degree=%i"%degree)
            plt.legend()
            plt.xlim(lines_windows[flag_line]['blue1'], lines_windows[flag_line][ 'red2'])
            plt.show()

    return local_cont, flag_cont, flag_windows

def local_continuum_linear(_ll, _f_res, linename, lines_windows, return_continuum = True, debug = False):
    # Select a line from our red/blue continua table
    flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)

    assert (lines_windows[flag_line]['blue1'] < lines_windows[flag_line]['blue2']), f'@@> Check blue cont lambda for {linename}: {lines_windows[flag_line]["blue1"][0]} >= {lines_windows[flag_line]["blue2"][0]}'
    assert (lines_windows[flag_line]['red1'] < lines_windows[flag_line]['red2']), f'@@> Check red cont lambda for {linename}: {lines_windows[flag_line]["red1"][0]} >= {lines_windows[flag_line]["red2"][0]}'
    
    def safe_median(x, l, l1, l2):
        mask = (l >= l1) & (l <= l2)
        if mask.any():
            median = np.ma.median(x[mask])
        else:
            median = 0.0
        return median, mask

    # Get the blue and red continuum median
    blue_median, flag_blue = safe_median(_f_res, _ll, lines_windows[flag_line]['blue1'], lines_windows[flag_line]['blue2'])
    red_median, flag_red   = safe_median(_f_res, _ll, lines_windows[flag_line][ 'red1'], lines_windows[flag_line][ 'red2'])

    # Get the midpoint wavelengths
    blue_lambda = (lines_windows[flag_line]['blue1'] + lines_windows[flag_line]['blue2'])[0] / 2.
    red_lambda  = (lines_windows[flag_line][ 'red1'] + lines_windows[flag_line][ 'red2'])[0] / 2.
    central_lambda = lines_windows[flag_line]['central'][0]

    # Calculate line parameters
    a = (red_median - blue_median) / (red_lambda - blue_lambda)
    b = blue_median - a * blue_lambda

    flag_cont = (_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line][ 'red2'])
    local_cont = a * _ll + b

    flag_windows = (flag_blue) | (flag_red)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot(_ll, _f_res)
        plt.plot(blue_lambda, blue_median, 'xb')
        plt.plot(red_lambda, red_median, 'xr')
        plt.plot(central_lambda, a*central_lambda + b, 'xk')

    if return_continuum:
        return local_cont, flag_cont, flag_windows
    else:
        return a, b, central_lambda


def fit_strong_lines(_ll, _f_res, _f_syn, _f_err,
                     kinematic_ties_on = True, balmer_limit_on = True, noHa = False,
                     model = 'resampled_gaussian', vd_inst = None, vd_kms = True,
                     lines_windows_file = None, degree=16, min_good_fraction=.2,
                     saveAll = False, saveHDF5 = False, saveTXT = False,
                     outname = None, outdir = None, debug = False,
                     stellar_v0=0., stellar_vd_stronglines=100., stellar_vd_weaklines=100.,
                     dv0 = 500., vd_max = 500.,
                     use_running_mean = False, N_running_mean = 50, N_clip = 1e30,
                     hii_ties_on = False,
                     **kwargs):

    #############################################################################################################
    # Check options and files
    if lines_windows_file is None:
        lines_windows = Table.read(path.join(path.dirname(__file__), 'lines.dat'), format = 'ascii.commented_header')
    else:
        lines_windows = Table.read(lines_windows_file, format = 'ascii.commented_header')

    if model == 'resampled_gaussian':
        from .models.resampled_gaussian import MultiResampledGaussian
        elModel = MultiResampledGaussian
    elif model == 'gaussian':
        from .models.gaussian import MultiGaussian
        elModel = MultiGaussian
    else:
        raise Exception('@@> No model found. Giving up.')
    #############################################################################################################

    #############################################################################################################
    # Useful functions
    
    # Get central wavelength for each line
    def get_central_wavelength(name):
        l0 = np.zeros(len(name))
        for il, linename in enumerate(name):
            flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)
            l0[il] = lines_windows[flag_line]['central'][0]
        return l0

    # Get vd_inst
    def get_vd_inst(vd_inst, name, l0, vd_kms, ll):

        if vd_inst is None:
            return [0. for n in name]

        elif np.isscalar(vd_inst):
            if vd_kms:
                return [vd_inst for n in name]
            else:
                return [c * vd_inst / l  for l in l0]

        elif type(vd_inst) is dict:
            if vd_kms:
                return [vd_inst[n] for n in name]
            else:
                return [c * vd_inst[n] / l for l, n in zip(l0, name)]

        elif isinstance(vd_inst, (np.ndarray, np.generic) ):
            vdi = np.interp(l0, ll, vd_inst)
            if vd_kms:
                return vdi
            else:
                return c * vdi / l0

        else:
            raise Exception('Check vd_inst, must be a scalar, a dictionary or a dispersion spectrum: %s' % vd_inst)

    def do_fit(model, ll, lc, flux, err, min_good_fraction, ignore_warning=False, maxiter=500):
        good = ~np.ma.getmaskarray(flux) & ~np.ma.getmaskarray(lc)
        Nl_cont = lc.count()
        N_good = good.sum()
        if Nl_cont > 1 and (N_good / Nl_cont) > min_good_fraction:
            fitter = fitting.LevMarLSQFitter()
            fitted_model = fitter(model, ll[good], (flux - lc)[good], weights=safe_pow(err[good], -1), maxiter=maxiter)
            flag = interpret_fit_result(fitter.fit_info, ignore_warning=ignore_warning)
            return fitted_model, flag
        else:
            log.warn('Too few data points for fitting (%.d / %d), flagged.' % (good.sum(), lc.count()))
            return model.copy(), flags.no_data


    def interpret_fit_result(fit_info, ignore_warning=False):
        log.debug('nfev: %d, ierr: %d (%s)' % (fit_info['nfev'], fit_info['ierr'], fit_info['message']))
        if fit_info['ierr'] not in [1, 2, 3, 4]:
            if not ignore_warning:
                log.warning('Bad fit, flagging.')
            return flags.bad_fit
        else:
            return 0
    #############################################################################################################

    #############################################################################################################
    # Pseudo-continuum tricks
    
    # Normalize spectrum by the median, to avoid problems in the fit
    med = np.median(np.ma.array(_f_syn).compressed())
    if not np.isfinite(med):
        raise Exception('Problems with synthetic spectra, median is non-finite.')

    f_res = np.ma.array(_f_res).filled(0) / np.abs(med)
    f_err = _f_err / np.abs(med)

    # Get local pseudocontinuum
    el_extra = {}
    total_lc = np.ma.zeros(len(_ll))
    total_lc.mask = True

    # Starting dictionaries to save results
    name = np.char.mod('%d', lines_windows['namel'])
    linename = lines_windows['name']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for n, ln in zip(name, linename):
        el_extra[n] = {'linename' : ln,
                       'f_integ' : -999.,
                       'f_imed' : -999.,}
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]
    
    # Remove emission lines detected to calculate the continuum with Legendre polynomials
    lc = np.ma.masked_array(np.zeros_like(_ll))
    flag_lc = np.ones(len(_ll), 'bool')

    log.info(f'Using stellar v0 = {stellar_v0:.2f}, vd = {stellar_vd_stronglines:.2f} (strong lines), vd = {stellar_vd_weaklines:.2f} (weak lines)')
    log.info(f'to mask out emission lines for the pseudocontinuum fit.')
    l_cen = l0 * (1. + stellar_v0 / c)
    sig_l = np.where((lines_windows['strong?'] == 1), l0 * (stellar_vd_stronglines / c), l0 * (stellar_vd_weaklines / c))
    Nsig = 5
    for _l_cen, _sig_l in zip(l_cen, sig_l):
        flag_line = (_ll >= (_l_cen - Nsig * _sig_l)) & (_ll <= (_l_cen + Nsig * _sig_l))
        flag_lc[flag_line] = False
    _f_res_lc = np.ma.masked_array(_f_res.copy(), mask=~flag_lc)

    # Interpolate _f_res_lc where it is masked out
    nans, x = _f_res_lc.mask, lambda z: z.nonzero()[0]
    _f_res_lc[nans]= np.interp(x(nans), x(~nans), _f_res_lc[~nans])

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('cont1')
        plt.plot(_ll, f_res, 'b')
        plt.plot(_ll[flag_lc], f_res[flag_lc], 'k')


    
    # Calc a running mean for clipping
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    from scipy.ndimage.filters import uniform_filter1d
    _aux = uniform_filter1d(_f_res_lc, size=N_running_mean)
    running_mean = np.ma.masked_array(_aux, _f_res_lc.mask)
    running_mean.mask[_ll > 9200] = True
    
    # Get continuum for each emission lines:
    # Legendre for continuum, linear for rms (because Legendre may overfit the noise)
    for n, ln in zip(name, linename):
        l, f, fw = local_continuum_linear(_ll, _f_res, n, lines_windows)
        lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])

        # Clipping where the continuum - running mean is greater than N_clip * rms
        flag_clip = (f) & (np.abs(f_res - running_mean) > (N_clip * lc_rms))
        _f_res_lc.mask[flag_clip] = True
        running_mean.mask[flag_clip] = True

        if use_running_mean:
            l, f, fw = local_continuum_legendre(_ll, running_mean, n, lines_windows, degree=degree)
        else:
            l, f, fw = local_continuum_legendre(_ll, _f_res_lc, n, lines_windows, degree=degree)

        # Save local continuum for each line
        lc = np.ma.masked_array(l, mask=~f)
        el_extra[n]['local_cont'] = lc
        el_extra[n]['rms_lc'] = lc_rms
        
        # Add this local continuum to total pseudocontinuum
        fc = ~lc.mask
        total_lc[fc] = lc[fc]
        total_lc.mask[fc] = False

    # Save the total pseudocontinuum
    total_lc = total_lc / np.abs(med)
    el_extra['total_lc'] = total_lc * np.abs(med)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('cont', figsize=(15,10))
        plt.clf()

        plt.suptitle(f'use_running_mean = {use_running_mean}, N_running_mean = {N_running_mean}, N_clip = {N_clip}')

        plt.subplot(221)
        plt.plot(_ll, f_res, 'tab:blue')
        plt.plot(_ll, total_lc, 'tab:orange')
        plt.ylim(-0.1, 0.4)
        plt.xlim(4500, 5200)

        plt.subplot(222)
        plt.plot(_ll, f_res, 'tab:blue')
        plt.plot(_ll, total_lc, 'tab:orange')
        plt.ylim(-0.1, 0.4)
        plt.xlim(5500, 6500)
        
        plt.subplot(223)
        plt.plot(_ll, f_res, 'tab:blue')
        plt.plot(_ll, total_lc, 'tab:orange')
        plt.ylim(-0.1, 0.4)
        plt.xlim(8800, 9300)

        plt.subplot(224)
        plt.plot(_ll, f_res, 'tab:blue')
        plt.plot(_ll, _f_res_lc, 'tab:red')
        #plt.plot(_ll, running_mean)
        plt.plot(_ll, total_lc, 'tab:orange')
        #plt.plot(_ll[flag_line], _f_res_lc.data[flag_line], 'r')
        
        plt.show()
    #############################################################################################################


    #############################################################################################################
    # Start line fitting
    
    log.debug('Fitting Ha and [NII]...')

    # Parameters
    name = ['6563', '6548', '6584']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = stellar_v0
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = 50.
    mod_init_HaN2 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=-vd_max, vd_max=vd_max)

    # Ties
    # From pyneb: [N II] F6584/F6548 =  2.9421684623736306 n_ii_atom_FFT04.dat
    mod_init_HaN2['6584'].flux.tied = lambda m: 2.94 * m['6548'].flux
    if kinematic_ties_on:
        mod_init_HaN2['6548'].v0.tied = lambda m: m['6584'].v0.value
        mod_init_HaN2['6548'].vd.tied = lambda m: m['6584'].vd.value

    # Fit
    mod_fit_HaN2, _flag = do_fit(mod_init_HaN2, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = np.int(_flag)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_HaN2')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_HaN2(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
    #############################################################################################################

    #############################################################################################################
    log.debug('Fitting [NII]weak [HeII] and [HeI]...')

    # Parameters
    name = ['5755','4686', '5876']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = mod_fit_HaN2['6584'].v0.value
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = mod_fit_HaN2['6584'].vd.value
    mod_init_N2He2He1 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    if kinematic_ties_on:
        mod_init_N2He2He1['5755'].v0.fixed = True
        mod_init_N2He2He1['5755'].vd.fixed = True
        mod_init_N2He2He1['4686'].v0.tied = lambda m: m['5876'].v0.value
        mod_init_N2He2He1['4686'].vd.tied = lambda m: m['5876'].vd.value

    if noHa:
        mod_init_N2He2He1['5755'].v0.fixed = False
        mod_init_N2He2He1['5755'].vd.fixed = False

    # Fit
    mod_fit_N2He2He1, _flag = do_fit(mod_init_N2He2He1, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_N2He2He1')
        plt.clf()
        plt.plot(_ll, f_res, 'k')
        plt.plot(_ll, mod_fit_N2He2He1(_ll)+total_lc, 'r')
        plt.plot(_ll, total_lc, 'grey')
        plt.ylim(-0.05, 0.05)
        plt.xlim(5555, 5955)
        for ll in l0:
            plt.axvline(ll, ls=':')
        plt.show()
    #############################################################################################################

    #############################################################################################################
    log.debug('Fitting Hb and Hg...')

    # Parameters
    name = ['4861', '4340']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    # Fitting Hg too because the single model has a problem,
    # and many things are based on the compounded model.
    v0_ini = mod_fit_HaN2['6563'].v0.value
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = mod_fit_HaN2['6563'].vd.value
    mod_init_HbHg = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Ties
    if kinematic_ties_on:
        mod_init_HbHg['4861'].v0.fixed = True
        mod_init_HbHg['4861'].vd.fixed = True
        mod_init_HbHg['4340'].v0.fixed = True
        mod_init_HbHg['4340'].vd.fixed = True

    if balmer_limit_on:
        mod_init_HbHg['4861'].flux.max = mod_fit_HaN2['6563'].flux / 2.6
        mod_init_HbHg['4340'].flux.max = mod_fit_HaN2['6563'].flux / 5.5

    if noHa:
        mod_init_HbHg['4861'].v0.fixed = False
        mod_init_HbHg['4861'].vd.fixed = False
        mod_init_HbHg['4340'].v0.fixed = False
        mod_init_HbHg['4340'].vd.fixed = False
        mod_init_HbHg['4861'].flux.max = None
        mod_init_HbHg['4340'].flux.max = None

    # Fit
    mod_fit_HbHg, _flag = do_fit(mod_init_HbHg, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_HbHg')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_HbHg(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
    #############################################################################################################
        
    #############################################################################################################
    log.debug('Fitting [OIII]...')

    # Parameters
    name = ['4959', '5007']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = stellar_v0
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = 50.
    mod_init_O3 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Ties
    # From pyneb: [O III] F5007/F4959 =  2.983969006971861 o_iii_atom_FFT04-SZ00.dat
    mod_init_O3['5007'].flux.tied = lambda m: 2.98 * m['4959'].flux
    if kinematic_ties_on:
        mod_init_O3['4959'].v0.tied = lambda m: m['5007'].v0.value
        mod_init_O3['4959'].vd.tied = lambda m: m['5007'].vd.value

    # Fit
    mod_fit_O3, _flag = do_fit(mod_init_O3, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_O3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O3(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
        plt.xlim(4500, 5500)
        plt.ylim(-0.1, 1)
    #############################################################################################################
    
    #############################################################################################################
    log.debug('Fitting [OIII]weak...')

    # Parameters
    name = ['4363', '4288', '4360', '4356']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    v0_ini = np.array([mod_fit_O3['5007'].v0.value, mod_fit_HaN2['6563'].v0.value, mod_fit_HaN2['6563'].v0.value, mod_fit_HaN2['6563'].v0.value])
    v0_min = min(v0_ini - dv0)
    v0_max = max(v0_ini + dv0)
    vd_ini = np.array([mod_fit_O3['5007'].vd.value, mod_fit_HaN2['6563'].vd.value, mod_fit_HaN2['6563'].vd.value, mod_fit_HaN2['6563'].vd.value])
    mod_init_O3_weak = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Ties
    if kinematic_ties_on:
        mod_init_O3_weak['4363'].v0.fixed = True
        mod_init_O3_weak['4363'].vd.fixed = True
        mod_init_O3_weak['4288'].v0.fixed = True
        mod_init_O3_weak['4288'].vd.fixed = True
        mod_init_O3_weak['4360'].v0.fixed = True
        mod_init_O3_weak['4360'].vd.fixed = True
        mod_init_O3_weak['4356'].v0.fixed = True
        mod_init_O3_weak['4356'].vd.fixed = True

    # Fit
    mod_fit_O3_weak, _flag = do_fit(mod_init_O3_weak, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_O3weak')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O3_weak(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
    #############################################################################################################
    
    #############################################################################################################
    log.debug('Fitting [OII]...')

    # Parameters
    name = ['3726', '3729']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = stellar_v0
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = 50.
    mod_init_O2 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Ties
    if kinematic_ties_on:
        mod_init_O2['3726'].v0.tied = lambda m: m['3729'].v0.value
        mod_init_O2['3726'].vd.tied = lambda m: m['3729'].vd.value

    # Fit
    mod_fit_O2, _flag = do_fit(mod_init_O2, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_O2')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O2(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')    
    #############################################################################################################

    #############################################################################################################
    log.debug('Fitting [SII]...')

    # Parameters
    name = ['6716', '6731']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = stellar_v0
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = 50.
    mod_init_S2 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Ties
    if kinematic_ties_on:
        mod_init_S2['6716'].v0.tied = lambda m: m['6731'].v0.value
        mod_init_S2['6716'].vd.tied = lambda m: m['6731'].vd.value

    # Fit
    mod_fit_S2, _flag = do_fit(mod_init_S2, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_S2')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_S2(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')  
    #############################################################################################################
    
    #############################################################################################################
    log.debug('Fitting [OII]weak...')

    # Parameters
    name = ['7320', '7330']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    v0_ini = stellar_v0
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = 50.
    mod_init_O2_weak = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Ties
    if kinematic_ties_on:
        mod_init_O2_weak['7320'].v0.tied = lambda m: m['7330'].v0.value
        mod_init_O2_weak['7320'].vd.tied = lambda m: m['7330'].vd.value

    # From pyneb:
    # [O II] F7330/F7320 den=1e0 =  0.8860478527312622 o_ii_atom_FFT04.dat
    # [O II] F7330/F7320 den=1e2 =  0.859372406330488 o_ii_atom_FFT04.dat
    # [O II] F7330/F7320 den=1e4 =  0.6736464205478534 o_ii_atom_FFT04.dat
    # [O II] F7330/F7320 den=1e6 =  0.639507406844097 o_ii_atom_FFT04.dat
    # 0.86 is a good guess for H II regions / SF galaxies
    hii_ties_on = True
    if hii_ties_on:
        mod_init_O2_weak['7330'].flux.tied = lambda m: 0.86 * m['7320'].flux

    # Fit
    mod_fit_O2_weak, _flag = do_fit(mod_init_O2_weak, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_O2weak')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O2_weak(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
        plt.xlim(7300, 7400)
        #plt.ylim(-0.1, 0.4)
    #############################################################################################################

    #############################################################################################################
    log.debug('Fitting [NeIII]...')

    # Parameters
    name = ['3869', '3968']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = stellar_v0
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = 50.
    mod_init_Ne3 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Fit
    mod_fit_Ne3, _flag = do_fit(mod_init_Ne3, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_Ne3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Ne3(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
    #############################################################################################################

    #############################################################################################################
    log.debug('Fitting [ArIII]')

    # Parameters
    name = ['7135', '7751']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = np.array([mod_fit_O3['5007'].v0.value, mod_fit_O3['5007'].v0.value])
    v0_min = min(v0_ini - dv0)
    v0_max = max(v0_ini + dv0)
    vd_ini = np.array([mod_fit_O3['5007'].vd.value, mod_fit_O3['5007'].vd.value])
    mod_init_Ar3 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Ties
    # From pyneb: [Ar III] F7135/F7751 = 4.1443010868494 ar_iii_atom_MB09.dat
    mod_init_Ar3['7135'].flux.tied = lambda m: 4.14 * m['7751'].flux

    if kinematic_ties_on:
        mod_init_Ar3['7135'].v0.fixed = True
        mod_init_Ar3['7135'].vd.fixed = True
        mod_init_Ar3['7751'].v0.fixed = True
        mod_init_Ar3['7751'].vd.fixed = True

    # Fit
    mod_fit_Ar3, _flag = do_fit(mod_init_Ar3, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_Ar3')
        plt.clf()
        plt.plot(_ll, f_res, drawstyle = 'steps-mid')
        plt.plot(_ll, mod_fit_Ar3(_ll)+total_lc, drawstyle = 'steps-mid')
        #plt.plot(_ll, total_lc.mask, 'r')
        for ll in l0:
            plt.axvline(ll, ls=':')
        plt.xlim(7000, 8000)
        plt.ylim(-0.1, 0.4)
    #############################################################################################################
    
    #############################################################################################################
        log.debug('Fitting [OI] & [SIII]...')

    # Parameters
    name = ['6312', '9068', '6300', '6364']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il] 

    # Start model
    v0_ini = np.array([mod_fit_O3['5007'].v0.value, mod_fit_O3['5007'].v0.value, 0., 0.])
    v0_min = min(v0_ini - dv0)
    v0_max = max(v0_ini + dv0)
    vd_ini = np.array([mod_fit_O3['5007'].vd.value, mod_fit_O3['5007'].vd.value, 50., 50.])
    mod_init_O1S3 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)
    
    if kinematic_ties_on:
        mod_init_O1S3['6312'].v0.fixed = True
        mod_init_O1S3['6312'].vd.fixed = True
        mod_init_O1S3['9068'].v0.fixed = True
        mod_init_O1S3['9068'].vd.fixed = True
        mod_init_O1S3['6364'].v0.tied = lambda m: m['6300'].v0.value
        mod_init_O1S3['6364'].vd.tied = lambda m: m['6300'].vd.value

    # Fit
    mod_fit_O1S3, _flag = do_fit(mod_init_O1S3, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_O1S3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O1S3(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
    #############################################################################################################
    
    #############################################################################################################
    log.debug('Fitting [FeIII]...')

    # Parameters
    name = ['4658', '4988']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = np.array([mod_fit_O3['5007'].v0.value, mod_fit_O3['5007'].v0.value])
    v0_min = min(v0_ini - dv0)
    v0_max = max(v0_ini + dv0)
    vd_ini = np.array([mod_fit_O3['5007'].vd.value, mod_fit_O3['5007'].vd.value])
    mod_init_Fe3 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    if kinematic_ties_on:
        mod_init_Fe3['4658'].v0.fixed = True
        mod_init_Fe3['4658'].vd.fixed = True
        mod_init_Fe3['4988'].v0.fixed = True
        mod_init_Fe3['4988'].vd.fixed = True

    # Fit
    mod_fit_Fe3, _flag = do_fit(mod_init_Fe3, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_Fe3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Fe3(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
    #############################################################################################################
    
    #############################################################################################################
    log.debug('Fitting [ArIV]...')

    # Parameters
    name = ['4740', '6434']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = stellar_v0
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = 50.
    mod_init_Ar4 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    # Ties
    if kinematic_ties_on:
        mod_init_Ar4['4740'].v0.tied = lambda m: m['6434'].v0.value
        mod_init_Ar4['4740'].vd.tied = lambda m: m['6434'].vd.value

    # Fit
    mod_fit_Ar4, _flag = do_fit(mod_init_Ar4, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_Ar4')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Ar4(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
    #############################################################################################################

    #############################################################################################################
    log.debug('Fitting [ClIII]...')

    # Parameters
    name = ['5517','5539']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Start model
    v0_ini = stellar_v0
    v0_min = v0_ini - dv0
    v0_max = v0_ini + dv0
    vd_ini = 50.
    mod_init_Cl3 = elModel(l0, flux=0., v0=v0_ini, vd=vd_ini, vd_inst=_vd_inst, name=name, v0_min=v0_min, v0_max=v0_max, vd_min=0., vd_max=vd_max)

    if kinematic_ties_on:
        mod_init_Cl3['5517'].v0.tied = lambda m: m['5539'].v0.value
        mod_init_Cl3['5517'].vd.tied = lambda m: m['5539'].vd.value
  
    # Fit
    mod_fit_Cl3, _flag = do_fit(mod_init_Cl3, _ll, total_lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit_Cl3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Cl3(_ll)+total_lc)
        for ll in l0:
            plt.axvline(ll, ls=':')
    #############################################################################################################



    #############################################################################################################
    # Integrate fluxes (be careful not to use the one normalized)

    # [N II]5755
    name = '5755'
    l0 = get_central_wavelength([name])[0]

    try:
        _vd_inst = get_vd_inst(vd_inst, [name], l0, vd_kms, _ll)[0]
    except:
        _vd_inst = get_vd_inst(vd_inst, [name], l0, vd_kms, _ll)
        
    vd_6584 = mod_fit_HaN2['6584'].vd.value
    if (vd_6584 < 0): vd_6584 = 0.

    v0 = mod_fit_HaN2['6584'].v0.value
    vd = np.sqrt(vd_6584**2 + _vd_inst**2)

    lo = l0 * v0 / c
    ld = l0 * vd / c
    lc = l0 + lo

    Ns = 4
    flag_centre = (_ll >= (lc - Ns*ld)) & (_ll <= (lc + Ns*ld))

    Nm1 =  5
    Nm2 = 50
    flag_blue = (_ll <= (lc - Nm1*ld)) & (_ll >= (lc - Nm2*ld))
    flag_red  = (_ll >= (lc + Nm1*ld)) & (_ll <= (lc + Nm2*ld))

    flag_cont = (flag_blue) | (flag_red)
    f_med = np.ma.median(f_res[flag_cont])
    f_new = f_res[flag_centre] - f_med
    f_integ = np.trapz(f_new, x=_ll[flag_centre])

    # Rescale by the median and save
    el_extra[name]['f_integ'] = f_integ * np.abs(med)
    el_extra[name]['f_imed' ] = f_med   * np.abs(med)
    el_extra[f'f_integ_{name}'] = flag_centre * 1 + flag_blue * 2 + flag_red * 3
    
    if debug:
        print(f_integ * np.abs(med), mod_fit_N2He2He1['5755'].flux.value * np.abs(med) )
        import matplotlib.pyplot as plt
        plt.figure('fit_N25775_integ')
        plt.clf()
        plt.plot(_ll, f_res, 'k', zorder=-10)
        plt.plot(_ll[flag_centre], f_res[flag_centre], 'g', label='centre')
        plt.plot(_ll[flag_blue]  , f_res[flag_blue]  , 'b', label='blue')
        plt.plot(_ll[flag_red]   , f_res[flag_red]   , 'r', label='red')
        plt.axhline(f_med, c='grey', ls=':', label='continuum')
        plt.plot(_ll, mod_fit_N2He2He1(_ll)+total_lc, 'gray', label='Gaussian fit')
        #plt.legend()
        plt.ylim(-0.05, 0.05)
        plt.xlim(5555, 5955)
        plt.show()

    # TO DO: Integrate for other emission lines
    #############################################################################################################

    #############################################################################################################
    # Rescale by the median
    el = [mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_O3_weak, mod_fit_O1S3, mod_fit_HaN2, mod_fit_N2He2He1, mod_fit_S2, mod_fit_O2_weak, mod_fit_Ar3, mod_fit_Fe3, mod_fit_Ne3, mod_fit_Ar4, mod_fit_Cl3]
    for model in el:
        for name in model.submodel_names:
            model[name].flux.value *= np.abs(med)

    # Recheck flags.no_data, because all models fit more than one emission line
    min_good_fraction = 0.3
    for model in el:
        for l in model.submodel_names:
            lc = el_extra[l]['local_cont']
            good = ~np.ma.getmaskarray(_f_res) & ~np.ma.getmaskarray(lc)
            Nl_cont = lc.count()
            N_good = good.sum()
            if Nl_cont == 0 or (N_good / Nl_cont) <= min_good_fraction:
                el_extra[l]['flag'] = flags.no_data
    #############################################################################################################

    #############################################################################################################
    # Get EWs
    name     = ['6563',   '6548',      '6584']
    linename = ['Halpha', '[NII]6548', '[NII]6584' ]
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_HaN2[n].flux, n, lines_windows)

    name     = ['4861'  , '4340'  ]
    linename = ['Hbeta' , 'Hgamma']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_HbHg[n].flux, n, lines_windows)

    name     = ['4959',       '5007']
    linename = ['[OIII]4959', '[OIII]5007']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_O3[n].flux, n, lines_windows)

    name     = ['4363', '4288', '4360', '4356']
    linename = ['[OIII]4363', '[FeII]4288', '[FeII]4360', '[FeII]4356']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_O3_weak[n].flux, n, lines_windows)

    name     = ['5755', '4686',       '5876']
    linename = ['[NII]5755','[HeII]4686', '[HeI]5876']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_N2He2He1[n].flux, n, lines_windows)

    name     = ['3726',      '3729']
    linename = ['[OII]3726', '[OII]3729']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_O2[n].flux, n, lines_windows)

    name     = ['6300', '6364', '6312', '9068']
    linename = ['[OI]6300', '[OI]6364', '[SIII]6312', '[SIII]9068']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_O1S3[n].flux, n, lines_windows)

    name     = ['6716',      '6731']
    linename = ['[SII]6716', '[SII]6731']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_S2[n].flux, n, lines_windows)

    name     = ['7320',        '7330']
    linename = ['[OII]7320', '[OII]7330']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_O2_weak[n].flux, n, lines_windows)

    name     = ['7135',        '7751']
    linename = ['[ArIII]7135', '[ArIII]7751']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_Ar3[n].flux, n, lines_windows)

    name     = ['4658', '4988']
    linename = ['[FeIII]4658', '[FeIII]4988']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_Fe3[n].flux, n, lines_windows)

    name     = ['3869', '3968']
    linename = ['[NeIII]3869', '[NeIII]3968']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_Ne3[n].flux, n, lines_windows)

    name     = ['6434', '4740']
    linename = ['[ArIV]6434', '[ArIV]4740']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_Ar4[n].flux, n, lines_windows)

    name     = ['5517', '5539']
    linename = ['[ClIII]5517', '[ClIII]5539']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_Cl3[n].flux, n, lines_windows)
    #############################################################################################################

    #############################################################################################################
    # Save
    el.append(el_extra)

    if saveAll:
        saveHDF5 = True
        saveTXT = True

    if saveHDF5 or saveTXT:
        from .utils import save_summary_to_file
        save_summary_to_file(el, outdir, outname, saveHDF5 = saveHDF5, saveTXT = saveTXT, **kwargs)
    #############################################################################################################

    return el

if __name__ == "__main__":

    tc = Table.read('STARLIGHTv04/0414.51901.393.cxt', format = 'ascii', names = ('lobs', 'fobs', 'ferr', 'flag'))
    ts = atpy.TableSet('STARLIGHTv04/0414.51901.393.cxt.sc4.C11.im.CCM.BN', type = 'starlightv4')

    #tc = Table.read('STARLIGHTv04/0404.51812.036.7xt', format = 'ascii', names = ('lobs', 'fobs', 'ferr', 'flag'))
    #ts = atpy.TableSet('STARLIGHTv04/0404.51812.036.sc4.NA3.gm.CCM.BS', type = 'starlightv4')

    el = fit_strong_lines_starlight(tc, ts, kinematic_ties_on = False)
    plot_el(ts, el, display_plot = True)


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
