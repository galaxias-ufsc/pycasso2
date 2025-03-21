'''
Created on 15/06/2015

@author: andre
'''

from ..cosmology import wavelength_apply_redshift

import os
import gzip
import bz2
import numpy as np
from astropy.table import Table
from astropy import log

from .. import flags

__all__ = ['read_wavelength_mask', 'write_input', 'read_output_tables']

pop_table_dtype = [('popx', 'float64'),
                   ('popmu_ini', 'float64'),
                   ('popmu_cor', 'float64'),
                   ]

spec_table_dtype = [('l_obs', 'float64'),
                    ('f_obs', 'float64'),
                    ('f_err', 'float64'),
                    ('f_syn', 'float64'),
                    ('f_wei', 'float64'),
                    ('f_flag', 'int32'),
                    ('f_disp', 'float64'),
                    ]


def read_wavelength_mask(maskfile, wl, z=0.0, dest='rest'):
    '''
    Read a STARLIGHT mask file, optionally applying redshift correction,
    returning a boolean array of masked wavelengths.

    Parameters
    ----------
    maskfile : string
        Path to mask file.

    wl : array
        Wavelength array.

    z : float, optional.
        Redshift. Default: ``0.0`` (no redshift).

    dest : string, optional
        Destination frame. Either ``'rest'`` or ``'observed'``.
        Default: ``'rest'``.

    Returns
    -------
    wl_mask : array
        Boolean array with same shape as ``wl`` marking the masked
        wavelengths as ``True``.
    '''
    t = _read_mask(maskfile)
    masked_wl = np.zeros(wl.shape, dtype='bool')
    for i in range(len(t)):
        l_low, l_upp, line_w, _ = t[i]
        if line_w > 0.0:
            continue
        if z > 0.0:
            l_low, l_upp = wavelength_apply_redshift(
                np.array([l_low, l_upp]), z, dest)
        masked_wl |= (wl > l_low) & (wl < l_upp)
    return masked_wl


def _read_mask(maskfile):
    '''
        Reads StarlightChains_v04.for maskfile
        @param maskfile: Starlight input maskfile.
    '''
    if not os.path.exists(maskfile):
        raise Exception('File not found: %s' % maskfile)

    with open(maskfile) as f:
        data = f.read().splitlines()
    N = int(data[0])
    linedata = [line.split() for line in data[1:N + 1]]
    l_low = []
    l_upp = []
    weight = []
    name = []
    for cols in linedata:
        l_low.append(float(cols[0]))
        l_upp.append(float(cols[1]))
        weight.append(float(cols[2]))
        name.append(' '.join(cols[3:]))

    cols = [l_low, l_upp, weight, name]
    names = ['l_low', 'l_upp', 'weight', 'name']
    return Table(cols, names=names)


def write_input(l_obs, f_obs, f_err, f_flag, filename):
    from astropy.io import ascii
    bad_data = ~np.isfinite(f_obs)
    if bad_data.any():
        log.warning('Fixed some bad values in f_obs, check your masks!')
        f_obs[bad_data] = np.ma.masked
    f_obs = f_obs.filled(0.0)
    if f_flag is not None and f_err is not None:
        bad_data = ~np.isfinite(f_err)
        if bad_data.any():
            log.warning('Fixed some bad values in f_err, check your masks!')
            f_err[bad_data] = np.ma.masked
        f_err = f_err.filled(0.0)
        bad = (f_flag & flags.before_starlight) > 0
        f_flag = np.where(bad, 2.0, 0.0)
        cols = [l_obs, f_obs, f_err, f_flag]
    else:
        cols = [l_obs, f_obs]
    formats = {'col0': lambda x: '%.1f' % round(x, 1)}
    # FIXME: trocar Writer= para format= 
    ascii.write(cols, filename, Writer=ascii.NoHeader, formats=formats, overwrite=True)


def read_base(basefile, basedir='', read_spectra=False):
    from astropy.io import ascii
    colnames = ['sspfile', 'age_base', 'Z_base', 'component', 'Mstars', 'YA_V', 'aFe']
    bt = read_base_file(basefile)

    if not read_spectra:
        return bt
    else:
        colnames = ['wl', 'flux']
        f = []
        for ssp_file in bt['sspfile']:

            t = ascii.read(os.path.join(basedir, ssp_file), names=colnames,
                           Reader=ascii.CommentedHeader, guess=False)
            f.append(t['flux'])
        return bt, np.array(t['wl']), np.vstack(f)


def read_base_file(basefile):
    with open(basefile, 'rt') as f:
        data = [l.split() for l in f.read().splitlines()]
    Nbase = int(data[0][0])
    names = ['sspfile', 'age_base', 'Z_base', 'component', 'Mstars', 'YA_V', 'aFe']
    dt = ['|S128', 'float32', 'float32', '|S128', 'float32', 'float32', 'float32']
    t = Table(names=names, dtype=dt)
    for i in range(1, Nbase + 1):
        t.add_row(data[i])
    return t


def read_output_tables(filename, read_chains=False):
    '''
    Reads STARLIGHT output file - version v06r1.

    ElCid@Sanchica - 13/Feb/2012

    --------------------------------------------------------------------
             Some notes on the structure of STARLIGHT output files
    --------------------------------------------------------------------

     Considering the 1st line to be number 1 (ATT: **subtract** 1 in python!!):
         * The pop-vector block starts at line 64 and has N_base entries from n1 to n2:
            n1 = 64
            n2 = n1 + N_base - 1

        * The Average & Chains light fractions block starts at line 64 + N_base + 5 and has N_par entries
            n1 = 64 + N_base + 5
            n2 = n1 + N_par - 1, where N_par = N_base + 2 + N_exAV
        * then comes the chain-LAx-pop-vectors block, whose N_base elements go from
            n1 = 64 + N_base + 5 + N_par + 2
            n2 = n1 + N_base - 1
        * and the chain-mu_cor-pop-vectors block, whose N_base elements go from
             n1 = 64 + N_base + 5 + N_par + 2 + N_base + 2
            n2 = n1 + N_base - 1
        * The chain chi2's and masses are in lines
             64 + N_base + 5 + N_par + 2 + N_base + 2 + N_base + 2, and
             64 + N_base + 5 + N_par + 2 + N_base + 2 + N_base + 3, respectively,
             followed by 2 lines with v_0_before_EX0s and v_d_before_EX0s.

        * The specral block starts with new line containing Nl_obs, index_Best_SSP,
         and i_SaveBestSingleCompFit at
              64 + N_base + 5 + N_par + 2 + N_base + 2 + N_base + 10
        * The l_obs , f_obs , f_syn , f_wei , Best_f_SSP info has Nl_obs entries, running from
             n1 = 64 + N_base + 5 + N_par + 2 + N_base + 2 + N_base + 11
            n2 = n1 + Nl_obs -1
'''
    if not os.path.exists(filename):
        raise Exception('File not found: %s' % filename)


    mode = 'rt'
    if filename.endswith('.gz'):
        open_func = gzip.open
    elif filename.endswith('.bz2'):
        open_func = bz2.open
    else:
        open_func = open
    with open_func(filename, mode) as f:
        data = [l.split() for l in f.read().splitlines()]

    fileVersion = data[1][5]
    keywords = {}
    tables = {}
    tables['keywords'] = keywords

    #Setting IsELROn in case QHRc are off.
    keywords['IsELROn'] = 0

    keywords['file_version'] = fileVersion
    keywords['arq_synt'] = os.path.basename(filename)

    #--------------------------------------------------------------------
    # Read "header": general info plus pop-vectors
    #--------------------------------------------------------------------

    # Some input info
    keywords['arq_spec'] = data[5][0]
    keywords['obs_dir'] = data[5][2]
    keywords['arq_base'] = data[6][0]
    keywords['arq_masks'] = data[7][0]
    keywords['arq_config'] = data[8][0]
    keywords['N_base'] = int(data[9][0])
    keywords['N_exAV_components'] = int(data[10][0])
    keywords['N_exAV'] = int(data[10][1])

    keywords['IsCFlawOn'] = int(data[10][2])
    keywords['a0_CFl'] = float(data[10][3])
    keywords['a1_CFl'] = float(data[10][4])


    keywords['IsFIRcOn'] = int(data[11][0])
    keywords['IsQHRcOn'] = int(data[11][1])
    keywords['IsPHOcOn'] = int(data[11][2])
    keywords['IsOPTimize_fn_OPT'] = int(data[11][3])


    keywords['ETC_ESM'] = data[12][0]
    keywords['ETC_gamma'] = float(data[12][1])
    keywords['Np_PHO'] = int(data[12][2])
    keywords['Np_QHR'] = int(data[12][3])
    keywords['Np_FIR'] = int(data[12][4])

    keywords['red_law_option'] = data[13][0]
    keywords['q_norm'] = float(data[14][0])
    keywords['flux_unit'] = float(data[14][1])

    # (Re)Sampling Parameters
    keywords['l_ini'] = float(data[17][0])
    keywords['l_fin'] = float(data[18][0])
    keywords['dl'] = float(data[19][0])
    keywords['dl_cushion'] = float(data[19][1])

    # Normalization info
    keywords['l_norm'] = float(data[22][0])
    keywords['llow_norm'] = float(data[23][0])
    keywords['lupp_norm'] = float(data[24][0])
    keywords['fobs_norm'] = float(data[25][0])
    keywords['Lobs_norm'] = float(data[25][1])
    keywords['LumDistInMpc'] = float(data[25][2])

    # S/N
    keywords['llow_SN'] = float(data[28][0])
    keywords['lupp_SN'] = float(data[29][0])
    keywords['SN_snwin'] = float(data[30][0])
    keywords['SN_normwin'] = float(data[31][0])
    keywords['SNerr_snwin'] = float(data[32][0])
    keywords['SNerr_normwin'] = float(data[33][0])

    # etc...
    keywords['idum_orig'] = int(data[36][0])
    keywords['NOl_eff'] = int(data[37][0])
    keywords['Nl_eff'] = int(data[38][0])
    keywords['Ntot_clipped'] = int(data[39][0])
    keywords['clip_method']  = data[39][1]
    keywords['SNmax_threshold'] = float(data[39][2])
    keywords['Nglobal_steps'] = int(data[40][0])
    keywords['N_chains'] = int(data[41][0])
    keywords['NEX0s_base'] = int(data[42][0])
    keywords['iCLIPBUG_flag'] = int(data[43][0])
    keywords['i_RC_CRASH_FLAG'] = int(data[43][1])
    keywords['IsBurInOver_BurnIn'] = int(data[43][2])
    keywords['n_censored_weights'] = int(data[43][3])
    keywords['wei_nsig_threshold'] = float(data[43][4])
    keywords['wei_limit'] = float(data[43][5])
    keywords['idt_all'] = int(data[44][0])
    keywords['wdt_TotTime'] = float(data[44][1])
    keywords['wdt_UsrTime'] = float(data[44][2])
    keywords['wdt_SysTime'] = float(data[44][3])

    ## Synthesis Results - Best model ##
    keywords['chi2/N_eff'] = float(data[49][0])
    keywords['chi2'] = float(data[49][1])
    keywords['adev'] = float(data[50][0])

    keywords['sum_x'] = float(data[51][0])
    keywords['Lum_tot'] = float(data[52][0])
    keywords['Mini_tot'] = float(data[53][0])
    keywords['Mcor_tot'] = float(data[54][0])

    keywords['v0']  = float(data[56][0])
    keywords['vd']  = float(data[57][0])
    keywords['Av']  = float(data[58][0])
    keywords['exAv'] = float(data[59][0])
    keywords['x(exAV>0)'] = float(data[59][-1][0:-1])

    keywords['FIR_GlobalChi2ScaleFactor'] = float(data[61][0])
    keywords['QHR_GlobalChi2ScaleFactor'] = float(data[61][1])
    keywords['PHO_GlobalChi2ScaleFactor'] = float(data[61][2])
    keywords['ETC_GlobalChi2ScaleFactor'] = float(data[61][3])
    keywords['k_FIR'] = float(data[62][0])
    keywords['k_QHR'] = float(data[62][1])
    keywords['k_PHO'] = float(data[62][2])
    keywords['k_FIR*chi2_FIR/chi2_OPT'] = float(data[63][0])
    keywords['k_QHR*chi2_QHR/chi2_OPT'] = float(data[63][1])
    keywords['k_PHO*chi2_PHO/chi2_OPT'] = float(data[63][2])
    keywords['chi2_FIR'] = float(data[64][0])
    keywords['chi2_QHR'] = float(data[64][1])
    keywords['chi2_PHO'] = float(data[64][2])
    keywords['chi2_ETC'] = float(data[64][3])

    # Reset populations lists
    popx = []    # column 2
    popmu_ini = []    # column 3
    popmu_cor = []    # column 4
    popage_base = []    # column 5
    popZ_base = []    # column 6
    popfbase_norm = []    # column 7
    popexAV_flag = []    # column 8
    popMstars = []    # column 9
    aFe = []    # column 11
    SSP_chi2r = []    # column 12
    SSP_adev = []    # column 13
    SSP_AV = []    # column 14
    SSP_x = []    # column 15
    popAV_tot = []    # column 16
    popLAx = []    # column 17

    # j     x_j(%)      Mini_j(%)     Mcor_j(%)     age_j(yr)     Z_j      (L/M)_j   exAV?  Mstars   component_j        new/Fe...    |  SSP_chi2r SSP_adev(%)   SSP_AV   SSP_x(%)    |  AV_tot   <LAx>_j(%)
    # Reads all these things (as lists) from lines _n1 to _n2
    _n1 = 67
    _n2 = _n1 + keywords['N_base']
    for i in range(_n1, _n2):
        popx.append(float(data[i][1]))
        popmu_ini.append(float(data[i][2]))
        popmu_cor.append(float(data[i][3]))
        popage_base.append(float(data[i][4]))
        popZ_base.append(float(data[i][5]))
        popfbase_norm.append(float(data[i][6]))
        popexAV_flag.append(float(data[i][7]))
        popMstars.append(float(data[i][8]))
        aFe.append(float(data[i][10]))
        SSP_chi2r.append(float(data[i][11]))
        SSP_adev.append(float(data[i][12]))
        SSP_AV.append(float(data[i][13]))
        SSP_x.append(float(data[i][14]))
        popAV_tot.append(float(data[i][15]))
        popLAx.append(float(data[i][16]))

    #WARNING: Ignoring Power-law fixes.

    tables['population'] = Table()

    tables['population']['popx']          = popx
    tables['population']['popmu_ini']     = popmu_ini
    tables['population']['popmu_cor']     = popmu_cor
    tables['population']['popage_base']   = popage_base
    tables['population']['popZ_base']     = popZ_base
    tables['population']['popfbase_norm'] = popfbase_norm
    tables['population']['popexAV_flag']  = popexAV_flag
    tables['population']['popMstars']     = popMstars
    tables['population']['aFe']           = aFe
    tables['population']['p_chi2r']       = SSP_chi2r
    tables['population']['p_adev']        = SSP_adev
    tables['population']['p_AV']          = SSP_AV
    tables['population']['p_x']           = SSP_x
    tables['population']['popAV_tot']     = popAV_tot

    #Column 10 may be the upper bound of age bin for composite stellar populations, if so. this value is stored:
    try:
        tables['population']['popage_base_upp'] = [float(data[i][9]) for i in range(_n1, _n2)]
    except ValueError:
    # Upper bin boundary is not present, we are using SSPs.
        tables['population']['popage_base_upp'] = tables['population']['popage_base']

    if read_chains == True:
        #--------------------------------------------------------------------
        # Read chain-related info (in arrays!)
        # ATT: The average solution is counted as an extra chain entry - the 1st one (Chain_*[0])!
        #--------------------------------------------------------------------
        ## Synthesis Results - Average & Chains ##

        # j    par_j: min, <> & last-chain-values for 0 ... N_chains chains (ave is 0'th chain!)
        # Reading chain par-vector (light fractions at l_norm, + extinctions and etc)
        # Notice that Chain_Par contains AV (maybe more than 1!) and fn, as well
        # as x!
        N_par = keywords['N_base'] + 1 + keywords['N_exAV'] + 1
        _n1 = 67 + keywords['N_base'] + 6 - 1
        _n2 = _n1 + N_par - 1 + 1
        Best_Par = []
        Ave_Par = []
        Chain_Par = []
        for unused, i in enumerate(range(_n1, _n2)):
            Best_Par.append(np.float(data[i][1]))
            Ave_Par.append(np.float(data[i][2]))
            x_ = [np.float(x) for x in data[i][3:3 + keywords['N_chains']]]
            Chain_Par.append(x_)

        # j Lambda-Averaged pop-vectors <LAx_*>_j: min, <> & last-chain-values for 0 ... N_chains chains (ave is 0'th chain!)
        # Reading chain LAx pop-vectors
        _n1 = 67 + keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2
        _n2 = _n1 + keywords['N_base']
        Best_LAx = []
        Ave_LAx = []
        Chain_LAx = []
        for unused, i in enumerate(range(_n1, _n2)):
            Best_LAx.append(np.float(data[i][1]))
            Ave_LAx.append(np.float(data[i][2]))
            x_ = [np.float(x) for x in data[i][3:3 + keywords['N_chains']]]
            Chain_LAx.append(x_)

        # j   Mcor_j: min, <> & last-chain-values for 0 ... N_chains chains (ave is 0'th chain!)
        # Reading chain mu_cor pop-vectors
        _n1 = 67 + keywords['N_base'] + 6 - 1 + \
            N_par - 1 + 1 + 2 + keywords['N_base'] + 2
        _n2 = _n1 + keywords['N_base']
        Best_mu_cor = []
        Ave_mu_cor = []
        Chain_mu_cor = []
        for unused, i in enumerate(range(_n1, _n2)):
            Best_mu_cor.append(np.float(data[i][1]))
            Ave_mu_cor.append(np.float(data[i][2]))
            x_ = [np.float(x) for x in data[i][3:3 + keywords['N_chains']]]
            Chain_mu_cor.append(x_)

        # chi2/Nl_eff & Mass for min, <> & i_chain = 0 ...  N_chains chains (ave is 0'th chain!)
        # Read Chain chi2/Nl_eff's , as well as kinematics before_EX0s
        i = 67 + keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + \
            2 + keywords['N_base'] + 2 + keywords['N_base'] + 2
        keywords['best_chi2'] = np.float(data[i][1])
        keywords['ave_chi2'] = np.float(data[i][2])
        keywords['cha_chi2'] = [
            np.float(x) for x in data[i][3:3 + keywords['N_chains']]]

        keywords['best_Mcor'] = np.float(data[i + 1][1])
        keywords['ave_Mcor'] = np.float(data[i + 1][2])
        keywords['cha_Mcor'] = [
            np.float(x) for x in data[i + 1][3:3 + keywords['N_chains']]]

        keywords['v_0_before_EX0s'] = float(data[i + 2][0])
        keywords['v_d_before_EX0s'] = float(data[i + 3][0])


        # Store chains in tables.
        cols = [Best_Par, Ave_Par, Chain_Par]
        chains_names = ['best', 'average', 'chains']
        tables['chains_par'] = Table(cols, names=chains_names)

        cols = [Best_LAx, Ave_LAx, Chain_LAx]
        tables['chains_LAx'] = Table(cols, names=chains_names)

        cols = [Best_mu_cor, Ave_mu_cor, Chain_mu_cor]
        tables['chains_mu_cor'] = Table(cols, names=chains_names)
        #--------------------------------------------------------------------

    N_par = keywords['N_base'] + 1 + keywords['N_exAV'] + 1

    # Synthetic spectrum (Best Model) ##l_obs f_obs f_syn wei Best_f_SSP

    i = 67 + keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + \
        2 + keywords['N_base'] + 2 + keywords['N_base'] + 2 + 8
    keywords['Nl_obs'] = int(data[i][0])
    keywords['index_Best_SSP'] = int(data[i][1])
    keywords['i_SaveBestSingleCompFit'] = int(data[i][2])

    # Reset & read spectral arrays (later turned into numpy.arrays)
    l_obs = []
    f_obs = []
    f_syn = []
    f_wei = []
    Best_f_SSP = []

    # Read spectra. Notice that new 5th column (with Best_f_SSP) is only read
    # & returned if it actually exists!
    _n1 = i + 1
    _n2 = _n1 + keywords['Nl_obs']
    for i in range(_n1, _n2):
        l_obs.append(float(data[i][0]))
        f_obs.append(float(data[i][1]))
        f_syn.append(float(data[i][2]))
        f_wei.append(float(data[i][3]))
        if (keywords['i_SaveBestSingleCompFit'] == 1):
            Best_f_SSP.append(float(data[i][4]))

    cols = [l_obs, f_obs, f_syn, f_wei]
    names = ['l_obs', 'f_obs', 'f_syn', 'f_wei']
    if (keywords['i_SaveBestSingleCompFit'] == 1):
        cols.append(Best_f_SSP)
        names.append('Best_f_SSP')

    tables['spectra'] = Table(cols, names=names)

    #--------------------------------------------------------------------
    # Reading FIR-related output
    #--------------------------------------------------------------------
    if (keywords['IsFIRcOn'] != 0):
        N_par = keywords['N_base'] + 1 + keywords['N_exAV'] + 1

        # Skip spectra
        i = 67 + keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + \
            keywords['N_base'] + 2 + keywords['N_base'] + 2 + 8
        keywords['Nl_obs'] = int(data[i][0])
        _n1 = i + 1
        _n2 = _n1 + keywords['Nl_obs']
        _n3 = _n2 + 8


        keywords['FIR_arq_ETCinfo'] = data[_n3][0]
        _n3 += 1
        keywords['FIR_LumDistInMpc'] = float(data[_n3][0])
        _n3 += 1
        keywords['FIR_logLFIR_TOT'] = float(data[_n3][0])
        _n3 += 1
        keywords['FIR_LFIRFrac2Model'] = float(data[_n3][0])
        _n3 += 1
        keywords['FIR_logLFIR_obs'] = float(data[_n3][0])
        _n3 += 1
        keywords['FIR_ErrlogLFIR'] = float(data[_n3][0])
        _n3 += 1
        keywords['FIR_RangelogLFIR'] = float(data[_n3][0])
        _n3 += 1
        keywords['FIRChi2ScaleFactor'] = float(data[_n3][0])
        _n3 += 1
        keywords['FIR_logLFIR_lowInLsun'] = float(data[_n3][0])
        keywords['FIR_logLFIR_uppInLsun'] = float(data[_n3][1])
        _n3 += 1
        keywords['FIRbeta_D'] = float(data[_n3][0])
        keywords['FIRbeta_I'] = float(data[_n3][1])
        _n3 += 1
        keywords['log_LFIR/LOPT_rough'] = float(data[_n3][0])
        _n3 += 3

        keywords['FIR_logLFIR_mod'] = float(data[_n3][0])
        keywords['FIRModObsRatio'] = float(data[_n3][1])
        _n3 += 1
        keywords['FIR_logLBOL_mod'] = float(data[_n3][0])
        keywords['FIR_BOL_Ratio'] = float(data[_n3][1])
        _n3 += 1
        keywords['chi2_FIR'] = float(data[_n3][0])
        keywords['chi2_OPT'] = float(data[_n3][1])
        _n3 += 3

        # Reset & read FIR-related SSP arrays
        x_FIR = []
        x_BOL = []
        BolCor = []
        FracLion = []
        Lbol_M = []
        Rmat = []
        R_opt = []
        R_Lya = []
        R_LCE = []



        # Read FIR-related SSP arrays
        _n1 = _n3
        _n2 = _n1 + keywords['N_base']
        for i in range(_n1, _n2):
            x_FIR.append(float(data[i][6]))
            x_BOL.append(float(data[i][7]))
            BolCor.append(float(data[i][8]))
            FracLion.append(float(data[i][9]))
            Lbol_M.append(float(data[i][10]))
            Rmat.append(float(data[i][11]))
            R_opt.append(float(data[i][12]))
            R_Lya.append(float(data[i][13]))
            R_LCE.append(float(data[i][14]))


        tables['FIR'] = Table()

        tables['FIR']['x_FIR']    = x_FIR
        tables['FIR']['x_BOL']    = x_BOL
        tables['FIR']['BolCor']   = BolCor
        tables['FIR']['FracLion'] = FracLion
        tables['FIR']['Lbol_M']   = Lbol_M
        tables['FIR']['Rmat']     = Rmat
        tables['FIR']['R_opt']    = R_opt
        tables['FIR']['R_Lya']    = R_Lya
        tables['FIR']['R_LCE']    = R_LCE



    #--------------------------------------------------------------------
    # Reading QHR-related output
    #--------------------------------------------------------------------
    if (keywords['IsQHRcOn'] != 0):
        N_par = keywords['N_base'] + 1 + keywords['N_exAV'] + 1

        # Skip spectra
        i = 67 + keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + \
            keywords['N_base'] + 2 + keywords['N_base'] + 2 + 8
        keywords['Nl_obs'] = int(data[i][0])
        _n1 = i + 1
        _n2 = _n1 + keywords['Nl_obs']
        _n3 = _n2 + 8

        # Skip FIR
        if (keywords['IsFIRcOn'] != 0):
            _n3 = _n3 + 26 + keywords['N_base']

        keywords['QHRbeta_I'] = float(data[_n3][0])
        keywords['IsReadQHfromBaseFile'] = int(data[_n3][1])
        _n3 += 1
        keywords['QHR_arq_ETCinfo'] = data[_n3][0]
        _n3 += 1
        keywords['QHR_LumDistInMpc'] = float(data[_n3][0])
        _n3 += 1
        keywords['QHR_GlobalChi2ScaleFactor'] = float(
            data[_n3][0])
        _n3 += 1
        keywords['NQHR_Ys'] = int(data[_n3][0])
        _n3 += 3

        # Reset & read QHR observed
        QHR_lambda = []
        QHR_frecomb = []
        QHR_logY_TOT = []
        QHR_YFrac2Model = []
        QHR_ErrlogY = []
        QHR_RangelogY = []
        QHR_Chi2ScaleFactor = []
        QHR_logY_obs = []
        QHR_logY_low = []
        QHR_logY_upp = []

        # Read QHR observed
        _n1 = _n3
        _n2 = _n1 + keywords['NQHR_Ys']
        for i in range(_n1, _n2):
            QHR_lambda.append(float(data[i][1]))
            QHR_frecomb.append(float(data[i][2]))
            QHR_logY_TOT.append(float(data[i][3]))
            QHR_YFrac2Model.append(float(data[i][4]))
            QHR_ErrlogY.append(float(data[i][5]))
            QHR_RangelogY.append(float(data[i][6]))
            QHR_Chi2ScaleFactor.append(float(data[i][7]))
            QHR_logY_obs.append(float(data[i][8]))
            QHR_logY_low.append(float(data[i][9]))
            QHR_logY_upp.append(float(data[i][10]))

        tables['QHR'] = Table()

        tables['QHR']['lambda']          = QHR_lambda
        tables['QHR']['frecomb']         = QHR_frecomb
        tables['QHR']['logY_tot']        = QHR_logY_TOT
        tables['QHR']['YFrac2Model']     = QHR_YFrac2Model
        tables['QHR']['ErrlogY']         = QHR_ErrlogY
        tables['QHR']['RangelogY']       = QHR_RangelogY
        tables['QHR']['Chi2ScaleFactor'] = QHR_Chi2ScaleFactor
        tables['QHR']['logY_obs']        = QHR_logY_obs
        tables['QHR']['logY_low']        = QHR_logY_low
        tables['QHR']['logY_upp']        = QHR_logY_upp



        # Read Emission Line Ratio-related things
        _n3 = _n2 + 2

        tables['ELR'] = {}

        keywords['IsELROn'] = int(data[_n3][0])
        tables['ELR']['lambda_A'] = float(data[_n3][1])
        tables['ELR']['lambda_B'] = float(data[_n3][2])
        tables['ELR']['ind_A'] = int(data[_n3][3])
        tables['ELR']['ind_B'] = int(data[_n3][4])
        tables['ELR']['logRint'] = float(data[_n3][5])
        keywords['Av_neb'] = float(data[_n3][6])
        tables['ELR']['errAV_neb'] = float(data[_n3][7])
        _n3 += 1


        tables['ELR']['Err_logR'] = float(data[_n3][0])
        tables['ELR']['RangelogR'] = float(data[_n3][1])
        tables['ELR']['logR_low'] = float(data[_n3][2])
        tables['ELR']['logR_upp'] = float(data[_n3][3])
        tables['ELR']['Chi2ScaleFactor'] = float(data[_n3][4])
        _n3 += 1

        tables['ELR']['logR_obs'] = float(data[_n3][0])
        tables['ELR']['logR_mod'] = float(data[_n3][1])
        tables['ELR']['chi2_ELR'] = float(data[_n3][2])

        _n3 += 3

        keywords['log_QH0_PhotPerSec'] = float(data[_n3][0])
        keywords['log_QHeff_PhotPerSec'] = float(data[_n3][1])
        _n3 += 1
        keywords['chi2_QHR'] = float(data[_n3][0])
        keywords['chi2_OPT'] = float(data[_n3][1])
        _n3 += 1

        _n3 += 2

        # Reset & read QHR model
        QHR_q_lambda = []
        QHR_logY_mod = []
        QHR_chi2_Y = []

        # Read QHR model
        _n1 = _n3
        _n2 = _n1 + keywords['NQHR_Ys']
        for i in range(_n1, _n2):
            QHR_q_lambda.append(float(data[i][2]))
            QHR_logY_mod.append(float(data[i][4]))
            QHR_chi2_Y.append(float(data[i][5]))

        tables['QHR']['q_lambda'] = QHR_q_lambda
        tables['QHR']['logY_mod'] = QHR_logY_mod
        tables['QHR']['chi2_Y']   = QHR_chi2_Y

        _n3 = _n2 + 2

        # Reset & read FIR-related SSP arrays
        qH__40 = []
        QH2Lnorm__40 = []
        QH0_Perc = []
        QHeff_Perc = []
        Y_Perc = dict()
        for il in range(0, keywords['NQHR_Ys']):
            Y_Perc[il] = []

        # Read FIR-related SSP arrays
        _n1 = _n3
        _n2 = _n1 + keywords['N_base']
        for i in range(_n1, _n2):
            qH__40.append(float(data[i][6]))
            QH2Lnorm__40.append(float(data[i][7]))
            QH0_Perc.append(float(data[i][8]))
            QHeff_Perc.append(float(data[i][9]))
            for il in range(0, keywords['NQHR_Ys']):
                Y_Perc[il].append(float(data[i][10 + il]))

        cols = [qH__40, QH2Lnorm__40, QH0_Perc, QHeff_Perc]
        names = ['qH__40', 'QH2Lnorm__40', 'QH0_Perc', 'QHeff_Perc']
        for il in range(0, keywords['NQHR_Ys']):
            cols.append(Y_Perc[il])
            names.append('Y_Perc_Line' + str(il))
        tables['popQHR'] = Table(cols, names=names)

    #--------------------------------------------------------------------
    # Reading PHO-related output
    #--------------------------------------------------------------------
    if (keywords['IsPHOcOn'] != 0):

        #Creating PHO table
        tables['PHO'] = Table()

        N_par = keywords['N_base'] + 1 + keywords['N_exAV'] + 1

        # Skip spectra
        i = 67 + keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + \
            keywords['N_base'] + 2 + keywords['N_base'] + 2 + 8
        keywords['Nl_obs'] = int(data[i][0])
        _n1 = i + 1
        _n2 = _n1 + keywords['Nl_obs']
        _n3 = _n2 + 8

        # Skip FIR
        if (keywords['IsFIRcOn'] != 0):
            _n3 = _n3 + 26 + keywords['N_base']

        # Skip QHR
        if (keywords['IsQHRcOn'] != 0):
            _n3 = _n3 + 28 + 2 * \
                keywords['NQHR_Ys'] + keywords['N_base']

        keywords['PHO_arq_ETCinfo'] = data[_n3][0]
        _n3 += 1
        keywords['PHO_LumDistInMpc'] = float(data[_n3][0])
        _n3 += 1
        keywords['PHO_Redshift'] = float(data[_n3][0])
        _n3 += 1
        keywords['PHO_GlobalChi2ScaleFactor'] = float(
            data[_n3][0])
        _n3 += 1
        keywords['NPHO_Ys'] = int(data[_n3][0])
        _n3 += 1

        _n3 += 2

        # Reset & read PHO observed
        PHO_name = []
        PHO_magY_TOT = []
        PHO_YFrac2Model = []
        PHO_magYErr = []
        PHO_magYRange = []
        PHO_Chi2ScaleFactor = []
        PHO_magY_obs = []
        PHO_magY_low = []
        PHO_magY_upp = []

        # Read PHO observed
        _n1 = _n3
        _n2 = _n1 + keywords['NPHO_Ys']
        for i in range(_n1, _n2):
            PHO_name.append(data[i][0])
            PHO_magY_TOT.append(float(data[i][2]))
            PHO_YFrac2Model.append(float(data[i][3]))
            PHO_magYErr.append(float(data[i][4]))
            PHO_magYRange.append(float(data[i][5]))
            PHO_Chi2ScaleFactor.append(float(data[i][6]))
            PHO_magY_obs.append(float(data[i][7]))
            PHO_magY_low.append(float(data[i][8]))
            PHO_magY_upp.append(float(data[i][9]))


        tables['PHO']['filter']           = PHO_name
        tables['PHO']['magY_TOT']         = PHO_magY_TOT
        tables['PHO']['Yfrac2model']      = PHO_YFrac2Model
        tables['PHO']['magYErr']          = PHO_magYErr
        tables['PHO']['magYRange']        = PHO_magYRange
        tables['PHO']['Chi2ScaleFactor']  = PHO_Chi2ScaleFactor
        tables['PHO']['magY_obs']         = PHO_magY_obs
        tables['PHO']['magY_low']         = PHO_magY_low
        tables['PHO']['magY_upp']         = PHO_magY_upp


        _n3 = _n2 + 2

        keywords['chi2_PHO'] = float(data[_n3][0])
        keywords['chi2_OPT'] = float(data[_n3][1])
        _n3 += 1

        _n3 += 2

#  name/code   MeanLamb PivotLamb StdDevLamb    q_MeanLamb  magY_obs     magY_mod     fY_obs        fY_mod       chi2_Y        chi2_Y/chi2_OPT  chi2_Y/chi2_TOT
        # Reset & read PHO model
        PHO_MeanLamb      = []
        PHO_PivotLamb     = []
        PHO_StdDevLamb    = []
        PHO_q_MeanLamb    = []
        PHO_magY_obs      = []
        PHO_magY_mod      = []
        PHO_fY_obs        = []
        PHO_fY_mod        = []
        PHO_chi2_Y        = []

        # Read PHO model
        _n1 = _n3
        _n2 = _n1 + keywords['NPHO_Ys']
        for i in range(_n1, _n2):
            PHO_name.append(data[i][1])
            PHO_MeanLamb.append(float(data[i][2]))
            PHO_PivotLamb.append(float(data[i][3]))
            PHO_StdDevLamb.append(float(data[i][4]))
            PHO_q_MeanLamb.append(float(data[i][5]))
            PHO_magY_obs.append(float(data[i][6]))
            PHO_magY_mod.append(float(data[i][7]))
            PHO_fY_obs.append(float(data[i][8]))
            PHO_fY_mod.append(float(data[i][9]))
            PHO_chi2_Y.append(float(data[i][10]))


        tables['PHO']['MeanLamb']   = PHO_MeanLamb
        tables['PHO']['PivotLamb']  = PHO_PivotLamb
        tables['PHO']['StdDevLamb'] = PHO_StdDevLamb
        tables['PHO']['q_MeanLamb'] = PHO_q_MeanLamb
        tables['PHO']['magY_obs']   = PHO_magY_obs
        tables['PHO']['magY_mod']   = PHO_magY_mod
        tables['PHO']['fY_obs']     = PHO_fY_obs
        tables['PHO']['fY_mod']     = PHO_fY_mod
        tables['PHO']['chi2_Y']     = PHO_chi2_Y


        _n3 = _n2 + 2

        # Reset & read PHO-related SSP arrays
        Y_Perc = []

        # Read PHO-related SSP arrays
        _n1 = _n3
        _n2 = _n1 + keywords['N_base']
        for i in range(_n1, _n2):
            Y_Perc.append([float(x) for x in data[i][6:]])

        Y_Perc = np.array(Y_Perc)

        tables['PHO']['Y_Perc'] = np.zeros((keywords['NPHO_Ys'], keywords['N_base']))
        for i in range(keywords['NPHO_Ys']):
            tables['PHO']['Y_Perc'][i] = Y_Perc[:,i]

    return tables


def read_output_tables_v4(filename, read_header=True, read_spec=True, verbose=False):
    '''
        Returns an array w/ outputfile from StarlightChains_v05 
                    (http://starlight.ufsc.br/)
    '''
    
    mode = 'rt'
    if filename.endswith('.gz'):
        open_func = gzip.open
    elif filename.endswith('.bz2'):
        open_func = bz2.open
    else:
        open_func = open
    with open_func(filename, mode) as f:
        data = [l.split() for l in f.read().splitlines()]

    fileVersion = data[1][5]
    keywords = {}
    tables = {}
    tables['keywords'] = keywords

    #Setting IsELROn in case QHRc are off.
    keywords['IsELROn'] = 0

    keywords['file_version'] = fileVersion
    keywords['arq_synt'] = os.path.basename(filename)
    
    if(read_header):
        if(verbose): print('read_header set to TRUE! Reading the spectra...')
    
        ## Some input info

        keywords['arq_spec']          = data[5][0]
        keywords['arq_base']          = data[6][0]
        keywords['arq_masks']         = data[7][0]
        keywords['arq_config']        = data[8][0]
        keywords['N_base']            = int(data[9][0])
        keywords['N_YAV_components']  = int(data[10][0])
        keywords['iFitPowerLaw']      = int(data[11][0])
        keywords['alpha_PowerLaw']    = np.float32(data[12][0])
        keywords['red_law_option']    = data[13][0]
        keywords['q_norm']            = np.float32(data[14][0])

        ## (Re)Sampling Parameters

        keywords['l_ini']           = np.float32(data[17][0])
        keywords['l_fin']           = np.float32(data[18][0])
        keywords['dl']              = np.float32(data[19][0])

        ## Normalization info

        keywords['l_norm']          = np.float32(data[22][0])
        keywords['llow_norm']       = np.float32(data[23][0])
        keywords['lupp_norm']       = np.float32(data[24][0])
        keywords['fobs_norm']       = np.float32(data[25][0])

        ## S/N

        keywords['llow_SN']         = np.float32(data[28][0])
        keywords['lupp_SN']         = np.float32(data[29][0])
        keywords['SN_snwin']        = np.float32(data[30][0])
        keywords['SN_normwin']      = np.float32(data[31][0])
        keywords['SNerr_snwin']     = np.float32(data[32][0])
        keywords['SNerr_normwin']   = np.float32(data[33][0])
        keywords['fscale_chi2']     = np.float32(data[34][0])

        ## etc...

        keywords['idum_orig']       = int(data[37][0])
        keywords['NOl_eff']         = int(data[38][0])
        keywords['Nl_eff']          = int(data[39][0])
        keywords['Ntot_clipped']    = int(data[40][0])
        keywords['Nglobal_steps']   = int(data[41][0])
        keywords['N_chains']        = int(data[42][0])
        keywords['NEX0s_base']      = int(data[43][0])

        ## Synthesis Results - Best model ##

        keywords['chi2']            = np.float32(data[49][0])
        keywords['adev']            = np.float32(data[50][0])

        keywords['sum_x']           = np.float32(data[52][0])
        keywords['Flux_tot']        = np.float32(data[53][0])
        keywords['Mini_tot']        = np.float32(data[54][0])
        keywords['Mcor_tot']        = np.float32(data[55][0])

        keywords['v0']              = np.float32(data[57][0])
        keywords['vd']              = np.float32(data[58][0])
        keywords['AV']              = np.float32(data[59][0])
        keywords['exAV']            = np.float32(data[60][0])


        # Read/define x, mu_ini, mu_cor, age_base, Z_base & YAV_flag arrays.
        _nlast = 62+keywords['N_base']
        # Reset populations lists
        #popx 2 popmu_ini 3 popmu_cor 4 popage_base 5 popZ_base 6  popYAV_flag 8 popMstars 9
        popx             = []    # column 2
        popmu_ini        = []    # column 3
        popmu_cor        = []    # column 4
        popage_base      = []    # column 5
        popZ_base        = []    # column 6
        popfbase_norm    = []    # column 7
        popYAV_flag      = []    # column 8
        popMstars        = []    # column 9
        
        
        for i in range(63,_nlast+1):
            tmp = data[i]
            popx.append(          float(tmp[1]))
            popmu_ini.append(     float(tmp[2]))
            popmu_cor.append(     float(tmp[3]))
            popage_base.append(   float(tmp[4]))
            popZ_base.append(     float(tmp[5]))
            popfbase_norm.append( float(tmp[6]))
            popYAV_flag.append(   float(tmp[7]))
            popMstars.append(     float(tmp[8]))
        
        t = Table()
        t['popx'] = np.array(popx)
        t['popmu_ini'] = np.array(popmu_ini)
        t['popmu_cor'] = np.array(popmu_cor)
        t['popage_base'] = np.array(popage_base)
        t['popZ_base'] = np.array(popZ_base)
        t['popfbase_norm'] = np.array(popfbase_norm)
        t['popYAV_flag'] = np.array(popYAV_flag)
        t['popMstars'] = np.array(popMstars)
        tables['population'] = t


        # Renormalize x to 100% sum!!
        #FIXME: Why not?
        #pop[0] = 100.*pop[0]/np.sum(pop[0])


        #FIXME: Skip this also...
        # OBS: PL have age = 0 in the Starlight output file:(
        #      Here I change it so that age_PL = 5e5 yr... & Z_PL = solar
        #      This is all obsolete anyway. The built-in PL is not used anymore.
        #if (int(self.keywords['iFitPowerLaw']) > 0):
        #    print '@@> [Warning!] ...Fixing PL age & Z ...????? CHECK THIS ?????'
        #    pop[3][self.keywords['N_base'] - 1] = 5e5 #popage_bae
        #    pop[4][self.keywords['N_base'] - 1]   = 0.02 #popZ_base

        #self.keywords['pop'] = pop
    

    if(read_spec == True):
        if(verbose): print('read_spec set to TRUE! Reading the spectra...')
        if(read_header == False):
                keywords['N_base']      =    int(data[9][0])
                keywords['fobs_norm']   = np.float32(data[25][0])

        # Read spectra (l_obs, f_obs, f_syn & f_wei)
        #l_obs 1 f_obs 2 f_syn 3 f_wei 4 Best_f_SSP 5
        iaux1 = 62 + keywords['N_base'] + 5 + keywords['N_base'] + 2 + keywords['N_base'] + 11
        keywords['Nl_obs'] = int(data[iaux1][0])
        iaux2 = iaux1 + 1
        #iaux3 = iaux1 + StarlightOut['Nl_obs']
    
        try:
            dt = np.dtype([('wl', 'float32'), ('flux_obs', 'float32'), ('flux_syn', 'float32'), ('wei', 'float32'), ('Best_f_SSP', 'float32')])
            out_spec = np.loadtxt(filename, dtype=dt, skiprows=iaux2)
        except:
            if(verbose): print('Warning: Did not read Best_f_SSP!')
            dt = np.dtype([('wl', 'float32'), ('flux_obs', 'float32'), ('flux_syn', 'float32'), ('wei', 'float32')])
            out_spec = np.loadtxt(filename, dtype=dt, skiprows=iaux2)
        
        t = Table()
        t['l_obs'] = out_spec['wl']
        t['f_obs'] = out_spec['flux_obs']
        t['f_syn'] = out_spec['flux_syn']
        t['f_wei'] = out_spec['wei']
        if (len(out_spec.dtype.names) == 5):
            t['Best_f_SSP'] = np.array(out_spec['Best_f_SSP'])     
        tables['spectra'] = t
    
    return tables

