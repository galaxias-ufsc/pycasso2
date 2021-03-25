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

    # Select window for Legendre polynomial fit
    flag_lc = (~_f_res.mask) & (_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line][ 'red2'])

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

    # Get the blue and red continua
    flag_blue = (_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line]['blue2'])
    flag_red  = (_ll >= lines_windows[flag_line][ 'red1']) & (_ll <= lines_windows[flag_line][ 'red2'])
    flag_cont = (_ll >= lines_windows[flag_line]['blue1']) & (_ll <= lines_windows[flag_line][ 'red2'])
    flag_windows = (flag_blue) | (flag_red)

    return local_cont, flag_cont, flag_windows

def local_continuum_linear(_ll, _f_res, linename, lines_windows, return_continuum = True, debug = False):
    # Select a line from our red/blue continua table
    flag_line = ( np.char.mod('%d', lines_windows['namel']) == linename)

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
                     kinematic_ties_on = True, balmer_limit_on = True,
                     model = 'resampled_gaussian', vd_inst = None, vd_kms = True,
                     lines_windows_file = None, degree=16, min_good_fraction=.2,
                     saveAll = False, saveHDF5 = False, saveTXT = False,
                     outname = None, outdir = None, debug = False,
                     stellar_v0=0., stellar_vd=100.,
                     **kwargs):

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

    # Normalize spectrum by the median, to avoid problems in the fit
    med = np.median(np.ma.array(_f_syn).compressed())
    if not np.isfinite(med):
        raise Exception('Problems with synthetic spectra, median is non-finite.')

    f_res = _f_res / np.abs(med)
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
        el_extra[n] = {'linename' : ln}
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Remove emission lines detected to calculate the continuum with Legendre polynomials
    lc = np.ma.masked_array(np.zeros_like(_ll))
    flag_lc = np.ones(len(_ll), 'bool')

    print('Using stellar v0 = %.2f, vd = %.2f to mask out emission lines for the pseudocontinuum fit.', (stellar_v0, stellar_vd))
    l_cen = l0 * (1. + stellar_v0 / c)
    sig_l = l0 * (stellar_vd / c)
    Nsig = 4
    for _l_cen, _sig_l in zip(l_cen, sig_l):
        flag_line = (_ll >= (_l_cen - Nsig * _sig_l)) & (_ll <= (_l_cen + Nsig * _sig_l))
        flag_lc[flag_line] = False
    _f_res_lc = np.ma.masked_array(_f_res, mask=~flag_lc)
    if debug:
        import matplotlib.pyplot as plt
        plt.figure('cont')
        plt.plot(_ll, f_res, 'b')
        plt.plot(_ll[flag_lc], f_res[flag_lc], 'k')
        plt.plot(_ll, mod_fit_all(_ll), 'r')


    # Continuum only for Ha - Legendre for continuum, linear for rms (because Legendre may overfit the noise)
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '6563', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '6563', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['6563',   '6548',      '6584']
    linename = ['Halpha', '[NII]6548', '[NII]6584' ]
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

########################################################################################################################################################
    # Continuum only for [NII]weak - Legendre for continuum, linear for rms (because Legendre may overfit the noise)
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '5755', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '5755', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['5755'     ]
    linename = ['[NII]5755']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

####################################################################################################################################################


    # Continuum for Hb - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '4861', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '4861', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['4861'  ]
    linename = ['Hbeta' ]
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

    # Continuum for Hg - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '4340', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '4340', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['4340'   ]
    linename = ['Hgamma' ]
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

    # Continuum only for [OIII]5007 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '5007', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '5007', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['4959',       '5007']
    linename = ['[OIII]4959', '[OIII]5007']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False


#########################################################################################################################################################

    # Continuum only for [OIII]weak - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '4363', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '4363', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['4363', '4288', '4360', '4356']
    linename = ['[OIII]4363', '[FeII]4288', '[FeII]4360', '[FeII]4356']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False


    # Continuum only for [HeII]4686 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '4686', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '4686', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['4686'      ]
    linename = ['[HeII]4686']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

    # Continuum only for [HeI]5876 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '5876', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '5876', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['5876'      ]
    linename = ['[HeI]5876' ]
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

    # Continuum only for [SIII]9068 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '9068', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '9068', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['9068'      ]
    linename = ['[SIII]9068' ]
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

##########################################################################################################################################################



    # Continuum only for [OII]3726 - linear - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '3726', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '3726', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['3726'     , '3729'      ]
    linename = ['[OII]3726', '[OII]3729' ]
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

    # Continuum for [OI]6300 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '6300', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '6300', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['6300', '6364']
    linename = ['[OI]6300', '[OI]6364']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

    # Continuum only for [SII]6716 - linear - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '6716', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '6716', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['6716'     , '6731'      ]
    linename = ['[SII]6716', '[SII]6731' ]
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False


###########################################################################################################################################################
    # Continuum only for [NeIII]3869 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '3869', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '3869', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['3869'       , '3968']
    linename = ['[NeIII]3869', '[NeIII]3968']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False


    # Continuum only for [OII]7330 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '7330', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '7330', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['7320'       , '7330']
    linename = ['[OII]7320', '[OII]7330']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False



    # Continuum only for [ArIII]7135 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '7135', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '7135', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['7135'       , '7751']
    linename = ['[ArIII]7135', '[ArIII]7751']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False



    # Continuum only for [FeIII]4658 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '4658', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '4658', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['4658', '4988' ]
    linename = ['[FeIII]4658' , '[FeIII]4988']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

    # Continuum only for [SIII]6312 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '6312', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '6312', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['6312']
    linename = ['[SIII]6312']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False


    # Continuum only for [ClIII]5518 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '5517', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '5517', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['5517', '5539']
    linename = ['[ClIII]5517', '[ClIII]5539']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False



    # Continuum only for [ArIV]6435 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '6434', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '6434', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['6434']
    linename = ['[ArIV]6434']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False

    # Continuum only for [ArIV]4740 - Legendre for continuum, linear for rms
    l, f, fw = local_continuum_legendre(_ll, _f_res_lc, '4740', lines_windows, degree=degree)
    lc = np.ma.masked_array(l, mask=~f)
    l, f, fw = local_continuum_linear(_ll, _f_res, '4740', lines_windows)
    lc_rms = np.ma.std((_f_res - l)[(f)&(fw)])
    name     = ['4740']
    linename = ['[ArIV]4740']
    for n, ln in zip(name, linename):
        el_extra[n] = { 'linename'   : ln ,
                        'local_cont' : lc ,
                        'rms_lc'     : lc_rms  }
    fc = ~lc.mask
    total_lc[fc] = lc[fc]
    total_lc.mask[fc] = False



##########################################################################################################################################################


    el_extra['total_lc'] = total_lc

    log.debug('Fitting Ha and [NII]...')

    # Parameters
    name = ['6563', '6548', '6584']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    mod_init_HaN2 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Ties
    mod_init_HaN2['6584'].flux.tied = lambda m: 3 * m['6548'].flux
    if kinematic_ties_on:
        mod_init_HaN2['6548'].v0.tied = lambda m: m['6584'].v0.value
        mod_init_HaN2['6548'].vd.tied = lambda m: m['6584'].vd.value

    # Fit
    mod_fit_HaN2, _flag = do_fit(mod_init_HaN2, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit1')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_HaN2(_ll)+lc)

#################################################################################################################################################

    log.debug('Fitting [NII]weak [HeII] and [HeI]...')# como plotar a linha de HeI pq no caso de Hg e Hb plota e nesse não?

    # Parameters
    name = ['5755','4686', '5876']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    #lc = el_extra[name[0]]['local_cont'] / np.abs(med)# talvez eu tenha que alterar algo aqui, quem sabe só estou pegando um contínuo
    lc = np.ma.zeros(len(_ll))
    lc.mask = True
    for n in name:
        fc = ~el_extra[n]['local_cont'].mask
        lc[fc] = el_extra[n]['local_cont'][fc]
        lc.mask[fc] = False
    lc /= np.abs(med)

    # Start model
    mod_init_N2He2He1 = elModel(l0, flux=0.0, v0=mod_fit_HaN2['6584'].v0.value, vd=mod_fit_HaN2['6584'].vd.value, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)


    if kinematic_ties_on:
        mod_init_N2He2He1['5755'].v0.fixed = True
        mod_init_N2He2He1['5755'].vd.fixed = True
        mod_init_N2He2He1['4686'].v0.tied = lambda m: m['5876'].v0.value
        mod_init_N2He2He1['4686'].vd.tied = lambda m: m['5876'].vd.value

    # Fit
    mod_fit_N2He2He1, _flag = do_fit(mod_init_N2He2He1, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_N2He2He1(_ll)+lc)

###########################################################################################################################################



    log.debug('Fitting Hb and Hg...')

    # Parameters
    name = ['4861', '4340']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum for Hb ang Hg
    lc = np.ma.zeros(len(_ll))
    lc.mask = True
    for n in name:
        fc = ~el_extra[n]['local_cont'].mask
        lc[fc] = el_extra[n]['local_cont'][fc]
        lc.mask[fc] = False
    lc /= np.abs(med)

    # Start model
    # Fitting Hg too because the single model has a problem,
    # and many things are based on the compounded model.
    mod_init_HbHg = elModel(l0, flux=0.0, v0=mod_fit_HaN2['6563'].v0.value, vd=mod_fit_HaN2['6563'].vd.value, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Ties
    if kinematic_ties_on:
        mod_init_HbHg['4861'].v0.fixed = True
        mod_init_HbHg['4861'].vd.fixed = True
        mod_init_HbHg['4340'].v0.fixed = True
        mod_init_HbHg['4340'].vd.fixed = True

    if balmer_limit_on:
        mod_init_HbHg['4861'].flux.max = mod_fit_HaN2['6563'].flux / 2.6
        mod_init_HbHg['4340'].flux.max = mod_fit_HaN2['6563'].flux / 5.5

    # Fit
    mod_fit_HbHg, _flag = do_fit(mod_init_HbHg, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit2')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_HbHg(_ll)+lc)

    log.debug('Fitting [OIII]...')

    # Parameters
    name = ['4959', '5007']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    mod_init_O3 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Ties
    mod_init_O3['5007'].flux.tied = lambda m: 2.97 * m['4959'].flux
    if kinematic_ties_on:
        mod_init_O3['4959'].v0.tied = lambda m: m['5007'].v0.value
        mod_init_O3['4959'].vd.tied = lambda m: m['5007'].vd.value

    # Fit
    mod_fit_O3, _flag = do_fit(mod_init_O3, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O3(_ll)+lc)

###########################################################################################################################################


    log.debug('Fitting [OIII]weak...')

    #parameters
    name = ['4363', '4288', '4360', '4356'] # tentar com um modelo simples gaussiano?
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum for OIIIweak
    lc = np.ma.zeros(len(_ll))
    lc.mask = True
    for n in name:
        fc = ~el_extra[n]['local_cont'].mask
        lc[fc] = el_extra[n]['local_cont'][fc]
        lc.mask[fc] = False
    lc /= np.abs(med)

    v0 = [mod_fit_O3['5007'].v0.value, mod_fit_HaN2['6563'].v0.value, mod_fit_HaN2['6563'].v0.value, mod_fit_HaN2['6563'].v0.value]
    vd = [mod_fit_O3['5007'].vd.value, mod_fit_HaN2['6563'].vd.value, mod_fit_HaN2['6563'].vd.value, mod_fit_HaN2['6563'].vd.value]
    mod_init_O3_weak = elModel(l0, flux=0.0, v0=v0, vd=vd, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=0.0, vd_max=500.0)

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
    mod_fit_O3_weak, _flag = do_fit(mod_init_O3_weak, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O3_weak(_ll)+lc)



################################################################################################################################################


    log.debug('Fitting [OII]...')

    # Parameters
    name = ['3726', '3729']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    mod_init_O2 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Ties
    if kinematic_ties_on:
        mod_init_O2['3726'].v0.tied = lambda m: m['3729'].v0.value
        mod_init_O2['3726'].vd.tied = lambda m: m['3729'].vd.value

    # Fit
    mod_fit_O2, _flag = do_fit(mod_init_O2, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit4')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O2(_ll)+lc)

    log.debug('Fitting [OI]...')

    # Parameters
    name = ['6300', '6364']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    mod_init_O1 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Fit
    mod_fit_O1, _flag = do_fit(mod_init_O1, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit5')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O1(_ll)+lc)

    log.debug('Fitting [SII]...')

    # Parameters
    name = ['6716', '6731']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    mod_init_S2 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Ties
    if kinematic_ties_on:
        mod_init_S2['6716'].v0.tied = lambda m: m['6731'].v0.value
        mod_init_S2['6716'].vd.tied = lambda m: m['6731'].vd.value

    # Fit
    mod_fit_S2, _flag = do_fit(mod_init_S2, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit6')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_S2(_ll)+lc)


########################################################################################################################################
    log.debug('Fitting [OII]weak...')

    #parameters
    name = ['7320', '7330'] # tentar com um modelo simples gaussiano?
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum for OIIIweak
    lc = np.ma.zeros(len(_ll))
    lc.mask = True
    for n in name:
        fc = ~el_extra[n]['local_cont'].mask
        lc[fc] = el_extra[n]['local_cont'][fc]
        lc.mask[fc] = False
    lc /= np.abs(med)


    mod_init_O2_weak = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Ties
    if kinematic_ties_on:
        mod_init_O2_weak['7320'].v0.tied = lambda m: m['7330'].v0.value
        mod_init_O2_weak['7320'].vd.tied = lambda m: m['7330'].vd.value

    # Fit
    mod_fit_O2_weak, _flag = do_fit(mod_init_O2_weak, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit8')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_O2_weak(_ll)+lc)


    log.debug('Fitting [NeIII]...')

    # Parameters
    name = ['3869', '3968']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    mod_init_Ne3 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Fit
    mod_fit_Ne3, _flag = do_fit(mod_init_Ne3, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit5')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Ne3(_ll)+lc)


    log.debug('Fitting [ArIII]')

    # Parameters
    name = ['7135', '7751']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    v0 = [mod_fit_O3['5007'].v0.value, mod_fit_O3['5007'].v0.value]
    vd = [mod_fit_O3['5007'].vd.value, mod_fit_O3['5007'].vd.value]
    mod_init_Ar3 = elModel(l0, flux=0.0, v0=v0, vd=vd, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Ties
    mod_init_Ar3['7135'].flux.tied = lambda m: np.sqrt((1.40e-3/8.30e-2)) * m['7751'].flux


    if kinematic_ties_on:
        mod_init_Ar3['7135'].v0.fixed = True
        mod_init_Ar3['7135'].vd.fixed = True
        mod_init_Ar3['7751'].v0.fixed = True
        mod_init_Ar3['7751'].vd.fixed = True


    # Fit
    mod_fit_Ar3, _flag = do_fit(mod_init_Ar3, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit1')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Ar3(_ll)+lc)

    log.debug('Fitting [SIII]')

    # Parameters
    name = ['6312', '9068']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    v0 = [mod_fit_O3['5007'].v0.value, mod_fit_O3['5007'].v0.value]
    vd = [mod_fit_O3['5007'].vd.value, mod_fit_O3['5007'].vd.value]
    mod_init_S3 = elModel(l0, flux=0.0, v0=v0, vd=vd, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    if kinematic_ties_on:
        mod_init_S3['6312'].v0.fixed = True
        mod_init_S3['6312'].vd.fixed = True
        mod_init_S3['9068'].v0.fixed = True
        mod_init_S3['9068'].vd.fixed = True


    # Fit
    mod_fit_S3, _flag = do_fit(mod_init_S3, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit1')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_S3(_ll)+lc)


    log.debug('Fitting [FeIII]...')

    # Parameters
    name = ['4658', '4988']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    v0 = [mod_fit_O3['5007'].v0.value, mod_fit_O3['5007'].v0.value]
    vd = [mod_fit_O3['5007'].vd.value, mod_fit_O3['5007'].vd.value]
    mod_init_Fe3 = elModel(l0, flux=0.0, v0=v0, vd=vd, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    if kinematic_ties_on:
        mod_init_Fe3['4658'].v0.fixed = True
        mod_init_Fe3['4658'].vd.fixed = True
        mod_init_Fe3['4988'].v0.fixed = True
        mod_init_Fe3['4988'].vd.fixed = True




    # Fit
    mod_fit_Fe3, _flag = do_fit(mod_init_Fe3, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit1')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Fe3(_ll)+lc)





    log.debug('Fitting [ArIV]...')

    # Parameters
    name = ['4740', '6434']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    lc = el_extra[name[0]]['local_cont'] / np.abs(med)

    # Start model
    mod_init_ArIV = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min=-500.0, vd_max=500.0)

    # Ties
    if kinematic_ties_on:
        mod_init_ArIV['4740'].v0.tied = lambda m: m['6434'].v0.value
        mod_init_ArIV['4740'].vd.tied = lambda m: m['6434'].vd.value

    # Fit
    mod_fit_ArIV, _flag = do_fit(mod_init_ArIV, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit4')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_ArIV(_ll)+lc)



    log.debug('Fitting [ClIII]...')# como plotar a linha de HeI pq no caso de Hg e Hb plota e nesse não?

    # Parameters
    name = ['5517','5539']
    l0 = get_central_wavelength(name)
    _vd_inst = get_vd_inst(vd_inst, name, l0, vd_kms, _ll)
    for il, ln in enumerate(name):
        el_extra[ln]['vd_inst'] = _vd_inst[il]

    # Get local continuum
    #lc = el_extra[name[0]]['local_cont'] / np.abs(med)# talvez eu tenha que alterar algo aqui, quem sabe só estou pegando um contínuo
    lc = np.ma.zeros(len(_ll))
    lc.mask = True
    for n in name:
        fc = ~el_extra[n]['local_cont'].mask
        lc[fc] = el_extra[n]['local_cont'][fc]
        lc.mask[fc] = False
    lc /= np.abs(med)

    # Start model
    mod_init_Cl3 = elModel(l0, flux=0.0, v0=0.0, vd=50.0, vd_inst=_vd_inst, name=name, v0_min=-500.0, v0_max=500.0, vd_min= 0.0, vd_max=500.0)


    if kinematic_ties_on:
        mod_init_Cl3['5517'].v0.tied = lambda m: m['5539'].v0.value
        mod_init_Cl3['5517'].vd.tied = lambda m: m['5539'].vd.value



    # Fit
    mod_fit_Cl3, _flag = do_fit(mod_init_Cl3, _ll, lc, f_res, f_err, min_good_fraction=min_good_fraction)
    for ln in name:
        el_extra[ln]['flag'] = _flag

    if debug:
        import matplotlib.pyplot as plt
        plt.figure('fit3')
        plt.clf()
        plt.plot(_ll, f_res)
        plt.plot(_ll, mod_fit_Cl3(_ll)+lc)


########################################################################################################################################

    # Rescale by the median
    el = [mod_fit_O2, mod_fit_HbHg, mod_fit_O3, mod_fit_O3_weak, mod_fit_O1, mod_fit_HaN2,mod_fit_N2He2He1, mod_fit_S2, mod_fit_O2_weak, mod_fit_Ar3, mod_fit_Fe3, mod_fit_Ne3, mod_fit_ArIV, mod_fit_Cl3, mod_fit_S3]
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


    # TO DO
    # Integrate fluxes (be careful not to use the one normalized)
    flag = (_ll > 6554) & (_ll < 6573)
    flux_Ha_int =  _f_res[flag].sum()
    #print('Fluxes: ', flux_Ha_int, mod_fit_HbHaN2['6563'].flux)


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

    name     = ['6300', '6364']
    linename = ['[OI]6300', '[OI]6364']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_O1[n].flux, n, lines_windows)

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
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_ArIV[n].flux, n, lines_windows)

    name     = ['5517', '5539']
    linename = ['[ClIII]5517', '[ClIII]5539']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_Cl3[n].flux, n, lines_windows)

    name     = ['6312', '9068']
    linename = ['[SIII]6312', '[SIII]9068']
    for n, ln in zip(name, linename):
        el_extra[n]['EW'] = calc_cont_EW(_ll, _f_syn, mod_fit_S3[n].flux, n, lines_windows)

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
