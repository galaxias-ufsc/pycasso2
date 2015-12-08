'''
Created on 15/06/2015

@author: andre
'''

from ..resampling import apply_redshift

import os
import numpy as np
import atpy
from atpy.registry import register_reader, register_set_reader

__all__ = ['read_wavelength_mask', 'write_input', 'read_output_tables']


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
    t = atpy.Table(maskfile, type='starlight_mask')
    masked_wl = np.zeros(wl.shape, dtype='bool')
    for i in xrange(len(t)):
        l_low, l_upp, line_w, _ = t[i]
        if line_w > 0.0: continue
        if z > 0.0:
            l_low, l_upp = apply_redshift(np.array([l_low, l_upp]), z, dest)
        masked_wl |= (wl > l_low) & (wl < l_upp)
    return masked_wl


def _read_mask(self, maskfile):
    '''
        Reads StarlightChains_v04.for maskfile
        @param maskfile: Starlight input maskfile.
    '''
    self.reset()

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
            
        self.table_name = 'maskfile'
        self.add_column('l_low', l_low)
        self.add_column('l_up', l_upp)
        self.add_column('weight', weight)
        self.add_column('name', name)


def write_input(wl, flux, err, flags, filename):
    from astropy.io import ascii
    
    flags = np.where(flags, 1.0, 0.0)
    if flags is not None and err is not None:
        cols = [wl, flux.data, err.data, flags]
    else:
        cols = [wl, flux.data]
    ascii.write(cols, filename, Writer=ascii.NoHeader)


def read_output_tables(filename):
    return atpy.TableSet(filename, type='starlight')


def _read_output(self, filename, read_header=True , read_chains=True , read_spec=True,
                 read_FIR_SSPs=True, read_QHR_SSPs=True, read_PHO_SSPs=True):
    '''
    Read STARLIGHT output file - version PANcMExStarlightChains_v03.for ...

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
    
        * The FIRc/QHRc/PHOc-related ouput is given at the end of the file, and 
        I still have to implement their reading here!
    --------------------------------------------------------------------
'''
    import gzip
    import bz2

    self.reset()

    if not os.path.exists(filename):
        raise Exception('File not found: %s' % filename)

    if filename.endswith('.gz'):
        f = gzip.GzipFile(filename)
    elif filename.endswith('.bz2'):
        f = bz2.BZ2File(filename)
    else:
        f = open(filename)
    data = f.read().splitlines()
    f.close()

    fileVersion = data[1].split()[3]
    # FIXME: How to handle starlight file versions?
#    if fileVersion != _getStarlightVersion():
#        raise Exception('Incorrect starlight version. Expected: %s, found: %' (_getStarlightVersion(), fileVersion))

    self.keywords['file_version'] = fileVersion
    self.keywords['arq_synt'] = os.path.basename(filename)

    #--------------------------------------------------------------------
    # Read "header": general info plus pop-vectors
    #--------------------------------------------------------------------
    if (read_header == True):

        ## Some input info
        self.keywords['arq_spec']            = data[5].split()[0]
        self.keywords['obs_dir']            = data[5].split()[2]
        self.keywords['arq_base']            = data[6].split()[0]
        self.keywords['arq_masks']            = data[7].split()[0]
        self.keywords['arq_config']            = data[8].split()[0]
        self.keywords['N_base']                   = int(data[9].split()[0])
        self.keywords['N_exAV_components']    = int(data[10].split()[0])
        self.keywords['N_exAV']                  = int(data[10].split()[1])
        self.keywords['IsFIRcOn']              = int(data[10].split()[2])
        self.keywords['IsQHRcOn']              = int(data[10].split()[3])
        self.keywords['IsPHOcOn']              = int(data[10].split()[4])
        self.keywords['iFitPowerLaw']       = int(data[11].split()[0])
        self.keywords['alpha_PowerLaw']     = float(data[12].split()[0])
        self.keywords['red_law_option']     = data[13].split()[0]
        self.keywords['q_norm']             = float(data[14].split()[0])
        self.keywords['flux_unit']          = float(data[14].split()[1])

        ## (Re)Sampling Parameters
        self.keywords['l_ini']         = float(data[17].split()[0])
        self.keywords['l_fin']         = float(data[18].split()[0])
        self.keywords['dl']            = float(data[19].split()[0])
        self.keywords['dl_cushion']    = float(data[19].split()[1])

        ## Normalization info
        self.keywords['l_norm']        = float(data[22].split()[0])
        self.keywords['llow_norm']     = float(data[23].split()[0])
        self.keywords['lupp_norm']     = float(data[24].split()[0])
        self.keywords['fobs_norm']     = float(data[25].split()[0])
        self.keywords['Lobs_norm']     = float(data[25].split()[1])
        self.keywords['LumDistInMpc']  = float(data[25].split()[2])

        ## S/N
        self.keywords['llow_SN']           = float(data[28].split()[0])
        self.keywords['lupp_SN']           = float(data[29].split()[0])
        self.keywords['SN_snwin']          = float(data[30].split()[0])
        self.keywords['SN_normwin']        = float(data[31].split()[0])
        self.keywords['SNerr_snwin']       = float(data[32].split()[0])
        self.keywords['SNerr_normwin']     = float(data[33].split()[0])
        self.keywords['fscale_chi2']       = float(data[34].split()[0])
        self.keywords['IsOptimize_fn_OPT'] = float(data[34].split()[1])

        ## etc...
        self.keywords['idum_orig']           = int(data[37].split()[0])
        self.keywords['NOl_eff']             = int(data[38].split()[0])
        self.keywords['Nl_eff']              = int(data[39].split()[0])
        self.keywords['Ntot_clipped']        = int(data[40].split()[0])
        self.keywords['clip_method']         = data[40].split()[1]
        self.keywords['Nglobal_steps']       = int(data[41].split()[0])
        self.keywords['N_chains']            = int(data[42].split()[0])
        self.keywords['NEX0s_base']          = int(data[43].split()[0])
        self.keywords['iCLIPBUG_flag']       = int(data[44].split()[0])
        self.keywords['i_RC_CRASH_FLAG']     = int(data[44].split()[1])
        self.keywords['IsBurInOver_BurnIn']  = int(data[44].split()[2])
        self.keywords['n_censored_weights']  = int(data[44].split()[3])
        self.keywords['wei_nsig_threshold']  = float(data[44].split()[4])
        self.keywords['wei_limit']           = float(data[44].split()[5])
        self.keywords['idt_all']             = int(data[45].split()[0])
        self.keywords['wdt_TotTime']         = float(data[45].split()[1])
        self.keywords['wdt_UsrTime']         = float(data[45].split()[2])
        self.keywords['wdt_SysTime']         = float(data[45].split()[3])

        ## Synthesis Results - Best model ##
        self.keywords['chi2']      = float(data[49].split()[0])
        self.keywords['adev']      = float(data[50].split()[0])
        self.keywords['chi2_TOT']  = float(data[51].split()[0])
        self.keywords['chi2_Opt']  = float(data[51].split()[1])
        self.keywords['chi2_FIR']  = float(data[51].split()[2])
        self.keywords['chi2_QHR']  = float(data[51].split()[3])
        self.keywords['chi2_PHO']  = float(data[51].split()[4])

        self.keywords['sum_x']     = float(data[52].split()[0])
        self.keywords['Flux_tot']  = float(data[53].split()[0])
        self.keywords['Mini_tot']  = float(data[54].split()[0])
        self.keywords['Mcor_tot']  = float(data[55].split()[0])

        self.keywords['v_0']       = float(data[57].split()[0])
        self.keywords['v_d']       = float(data[58].split()[0])
        self.keywords['A_V']       = float(data[59].split()[0])

        # Reset populations lists
        popx                = []    # column 2
        popmu_ini           = []    # column 3
        popmu_cor           = []    # column 4
        popage_base         = []    # column 5
        popZ_base           = []    # column 6
        popfbase_norm       = []    # column 7
        popexAV_flag        = []    # column 8
        popMstars           = []    # column 9
        aFe                 = []    # column 11
        SSP_chi2r           = []    # column 12
        SSP_adev            = []    # column 13
        SSP_AV              = []    # column 14
        SSP_x               = []    # column 15
        popAV_tot           = []    # column 16
        popLAx              = []    # column 17

        # j     x_j(%)      Mini_j(%)     Mcor_j(%)     age_j(yr)     Z_j      (L/M)_j   exAV?  Mstars   component_j        new/Fe...    |  SSP_chi2r SSP_adev(%)   SSP_AV   SSP_x(%)    |  AV_tot   <LAx>_j(%)
        # Reads all these things (as lists) from lines _n1 to _n2
        _n1 = 63
        _n2 = _n1 + self.keywords['N_base']
        for i in range(_n1,_n2):
            popx.append(          float(data[i].split()[1]) )
            popmu_ini.append(     float(data[i].split()[2]) )
            popmu_cor.append(     float(data[i].split()[3]) )
            popage_base.append(   float(data[i].split()[4]) )
            popZ_base.append(     float(data[i].split()[5]) )
            popfbase_norm.append( float(data[i].split()[6]) )
            popexAV_flag.append(  float(data[i].split()[7]) )
            popMstars.append(     float(data[i].split()[8]) )
            aFe.append(           float(data[i].split()[10]) )
            SSP_chi2r.append(     float(data[i].split()[11]) )
            SSP_adev.append(      float(data[i].split()[12]) )
            SSP_AV.append(        float(data[i].split()[13]) )
            SSP_x.append(         float(data[i].split()[14]) )
            popAV_tot.append(     float(data[i].split()[15]) )
            popLAx.append(        float(data[i].split()[16]) )

        # ATT: In my SM macros I fix PL-age & Z, and renormalize light fractions to 100%.
        #      This could be done like this:
        #          aux = np.array(popx)   ; popx   = np.array(100. * aux / aux.sum())
        #          aux = np.array(popLAx) ; popLAx = np.array(100. * aux / aux.sum())
        #      But I'm **NOT** doing this here since all this function does is to replicate the 
        #       STARLIGHT output to new python dictionary! Furthermore, renomrmalization may
        #        be inconvenient to create luminosity x age 3D-CALIFA-images...

        # ATT2: Ignoring Power-law fixes.
            # OBS: PL have age = 0 in the Starlight output file:(
            #      Here I change it so that age_PL = 5e5 yr... & Z_PL = solar
            #      This is all obsolete anyway. The built-in PL is not used anymore.
            #if (int(StarlightOut['iFitPowerLaw']) > 0):
            #    print '@@> [Warning!] ...Fixing PL age & Z ...Why are u using PLs??'
            #    popage_base][StarlightOut['N_base'] - 1] = 5e5
            #    popZ_base[StarlightOut['N_base'] - 1]    = 0.02
        t = atpy.Table()
        # FIXME: explicit data types
        t.table_name = 'population'
        t.add_column('popx', np.array(popx, dtype='>f8'))
        t.add_column('popmu_ini', np.array(popmu_ini, dtype='>f8'))
        t.add_column('popmu_cor', np.array(popmu_cor, dtype='>f8'))
        t.add_column('popage_base', np.array(popage_base, dtype='>f8'))
        t.add_column('popZ_base', np.array(popZ_base, dtype='>f8'))
        t.add_column('popfbase_norm', np.array(popfbase_norm, dtype='>f8'))
        t.add_column('popexAV_flag', np.array(popexAV_flag, dtype='>f8'))
        t.add_column('popMstars', np.array(popMstars, dtype='>f8'))
        t.add_column('aFe', np.array(aFe, dtype='>f8'))
        t.add_column('SSP_chi2r', np.array(SSP_chi2r, dtype='>f8'))
        t.add_column('SSP_adev', np.array(SSP_adev, dtype='>f8'))
        t.add_column('SSP_AV', np.array(SSP_AV, dtype='>f8'))
        t.add_column('SSP_x', np.array(SSP_x, dtype='>f8'))
        t.add_column('popAV_tot', np.array(popAV_tot, dtype='>f8'))
        t.add_column('popLAx', np.array(popLAx, dtype='>f8'))
        self.append(t)

    # ...Another possibility is to store all pop-arrays in new single one ... (new la William)...
    #    # Put all pop-lists inside the np.array pop ... 
    #    # don't understand why dtype=dt doen not work on aux = ... NOT working!!
    #    dt = np.dtype([('popx','f'),('popmu_ini','f'),('popmu_cor','f'), ('popage_base','f'),('popZ_base','f'), \
    #                ('popfbase_norm','f'), ('popexAV_flag','f'),('popMstars','f'), \
    #                ('SSP_chi2r','f'),('SSP_adev','f'),('SSP_AV','f'),('SSP_x','f'), \
    #                ('popAV_tot','f'),('popLAx','f')])
    #    aux = np.array( [popx,popmu_ini,popmu_cor,popage_base,popZ_base, \
    #          popfbase_norm,popexAV_flag,popMstars, \
    #          SSP_chi2r,SSP_adev,SSP_AV,SSP_x, \
    #          popAV_tot,popLAx] )
    #    pop = np.array(aux,dtype=dt)
    #    StarlightOut['pop'] = pop
    #--------------------------------------------------------------------


    #--------------------------------------------------------------------
    # Read chain-related info (in arrays!)
    # ATT: The average solution is counted as an extra chain entry - the 1st one (Chain_*[0])!
    #--------------------------------------------------------------------
    if (read_chains == True):
        # These things are needed if header was not read!
        if (read_header == False):
            self.keywords['N_base'] = int(data[9].split()[0])
            self.keywords['N_exAV'] = int(data[10].split()[1])
            self.keywords['N_chains'] = int(data[42].split()[0])

    ## Synthesis Results - Average & Chains ##

    # j    par_j: min, <> & last-chain-values for 0 ... N_chains chains (ave is 0'th chain!)
    # Reading chain par-vector (light fractions at l_norm, + extinctions and etc)
    # Notice that Chain_Par contains AV (maybe more than 1!) and fn, as well as x!
        N_par = self.keywords['N_base'] + 1 + self.keywords['N_exAV'] + 1
        _n1 = 63 + self.keywords['N_base'] + 6 - 1
        _n2 = _n1 + N_par - 1 + 1
        Best_Par = []
        Ave_Par = []
        Chain_Par = []
        for unused,i in enumerate( range(_n1,_n2) ):
            Best_Par.append(np.float(data[i].split()[1]))
            Ave_Par.append(np.float(data[i].split()[2]))
            x_ = [np.float(x) for x in data[i].split()[3:3+self.keywords['N_chains']]]
            Chain_Par.append( x_ )

    # j Lambda-Averaged pop-vectors <LAx_*>_j: min, <> & last-chain-values for 0 ... N_chains chains (ave is 0'th chain!)
    # Reading chain LAx pop-vectors
        _n1 = 63 + self.keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2
        _n2 = _n1 + self.keywords['N_base']
        Best_LAx = []
        Ave_LAx = []
        Chain_LAx = []
        for unused,i in enumerate( range(_n1,_n2) ):
            Best_LAx.append(np.float(data[i].split()[1]))
            Ave_LAx.append(np.float(data[i].split()[2]))
            x_ = [np.float(x) for x in data[i].split()[3:3+self.keywords['N_chains']]]
            Chain_LAx.append( x_ )

    # j   Mcor_j: min, <> & last-chain-values for 0 ... N_chains chains (ave is 0'th chain!)
    # Reading chain mu_cor pop-vectors
        _n1 = 63 + self.keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + self.keywords['N_base'] + 2
        _n2 = _n1 + self.keywords['N_base']
        Best_mu_cor = []
        Ave_mu_cor = []
        Chain_mu_cor = []
        for unused,i in enumerate( range(_n1,_n2) ):
            Best_mu_cor.append(np.float(data[i].split()[1]))
            Ave_mu_cor.append(np.float(data[i].split()[2]))
            x_ = [np.float(x) for x in data[i].split()[3:3+self.keywords['N_chains']]]
            Chain_mu_cor.append( x_ )

    # chi2/Nl_eff & Mass for min, <> & i_chain = 0 ...  N_chains chains (ave is 0'th chain!)
    # Read Chain chi2/Nl_eff's , as well as kinematics before_EX0s
        i = 63 + self.keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + self.keywords['N_base'] + 2 + self.keywords['N_base'] + 2
        self.add_keyword('best_chi2', np.float(data[i].split()[1]))
        self.add_keyword('ave_chi2', np.float(data[i].split()[2]))
        self.add_keyword('cha_chi2', [np.float(x) for x in data[i].split()[3:3+self.keywords['N_chains']]])

        self.add_keyword('best_Mcor', np.float(data[i+1].split()[1]))
        self.add_keyword('ave_Mcor', np.float(data[i+1].split()[2]))
        self.add_keyword('cha_Mcor', [np.float(x) for x in data[i+1].split()[3:3+self.keywords['N_chains']]])

        self.add_keyword('v_0_before_EX0s', float( data[i+2].split()[0] ))
        self.add_keyword('v_d_before_EX0s', float( data[i+3].split()[0] ))

    # Store chains in tables.
        t = atpy.Table()
        t.table_name = 'chains_par'
        t.add_column('best', np.array(Best_Par, dtype='>f8'))
        t.add_column('average', np.array(Ave_Par, dtype='>f8'))
        t.add_column('chains', np.array(Chain_Par, dtype='>f8'))
        self.append(t)
        
        t = atpy.Table()
        t.table_name = 'chains_LAx'
        t.add_column('best', np.array(Best_LAx, dtype='>f8'))
        t.add_column('average', np.array(Ave_LAx, dtype='>f8'))
        t.add_column('chains', np.array(Chain_LAx, dtype='>f8'))
        self.append(t)
        
        t = atpy.Table()
        t.table_name = 'chains_mu_cor'
        t.add_column('best', np.array(Best_mu_cor, dtype='>f8'))
        t.add_column('average', np.array(Ave_mu_cor, dtype='>f8'))
        t.add_column('chains', np.array(Chain_mu_cor, dtype='>f8'))
        self.append(t)
        
    #--------------------------------------------------------------------


    #--------------------------------------------------------------------
    # Reading spectral info: l_obs , f_obs , f_syn , f_wei , Best_f_SSP 
    #--------------------------------------------------------------------
    if (read_spec == True):

        # These things are needed if header was not read!
        if (read_header == False):
            self.keywords['N_base']            = int(data[9].split()[0])
            self.keywords['N_exAV']         = int(data[10].split()[1])
            self.keywords['flux_unit']    = float(data[14].split()[1])
            self.keywords['l_norm']       = float(data[22].split()[0])
            self.keywords['llow_norm']    = float(data[23].split()[0])
            self.keywords['lupp_norm']    = float(data[24].split()[0])
            self.keywords['fobs_norm']    = float(data[25].split()[0])
            self.keywords['Lobs_norm']    = float(data[25].split()[1])
            self.keywords['LumDistInMpc'] = float(data[25].split()[2])

        N_par = self.keywords['N_base'] + 1 + self.keywords['N_exAV'] + 1


        ## Synthetic spectrum (Best Model) ##l_obs f_obs f_syn wei Best_f_SSP

        i = 63 + self.keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + self.keywords['N_base'] + 2 + self.keywords['N_base'] + 2 + 8
        self.keywords['Nl_obs']                  = int(data[i].split()[0])
        self.keywords['index_Best_SSP']          = int(data[i].split()[1])
        self.keywords['i_SaveBestSingleCompFit'] = int(data[i].split()[2])

        # Reset & read spectral arrays (later turned into numpy.arrays)
        l_obs = []
        f_obs = []
        f_syn = []
        f_wei = []
        Best_f_SSP = []

        # Read spectra. Notice that new 5th column (with Best_f_SSP) is only read & returned if it actually exists!
        _n1 = i+1
        _n2 = _n1 + self.keywords['Nl_obs']
        for i in range(_n1,_n2):
            l_obs.append( float(data[i].split()[0]) )
            f_obs.append( float(data[i].split()[1]) )
            f_syn.append( float(data[i].split()[2]) )
            f_wei.append( float(data[i].split()[3]) )
            if (self.keywords['i_SaveBestSingleCompFit'] == 1):
                Best_f_SSP.append( float(data[i].split()[4]) )
    
        t = atpy.Table()
        t.table_name = 'spectra'
        t.add_column('l_obs', np.array(l_obs, dtype='>f8'))
        t.add_column('f_obs', np.array(f_obs, dtype='>f8'))
        t.add_column('f_syn', np.array(f_syn, dtype='>f8'))
        t.add_column('f_wei', np.array(f_wei, dtype='>f8'))
        if (self.keywords['i_SaveBestSingleCompFit'] == 1):
            t.add_column('Best_f_SSP', np.array(Best_f_SSP, dtype='>i4'))
            
        self.append(t)


    #--------------------------------------------------------------------
    # Reading FIR-related output 
    #--------------------------------------------------------------------
    if (self.keywords['IsFIRcOn'] != 0):

        # ++NAT++ Should we refactor this block??? def quick_read_header; return _n1, _n2, N_base, Nl_obs, N_par
        #         See also the same block in spec and QHR readers
        # These things are needed if header was not read!
        if (read_header == False):
            self.keywords['N_base']            = int(data[9].split()[0])
            self.keywords['N_exAV']         = int(data[10].split()[1])
            self.keywords['flux_unit']    = float(data[14].split()[1])
            self.keywords['l_norm']       = float(data[22].split()[0])
            self.keywords['llow_norm']    = float(data[23].split()[0])
            self.keywords['lupp_norm']    = float(data[24].split()[0])
            self.keywords['fobs_norm']    = float(data[25].split()[0])
            self.keywords['Lobs_norm']    = float(data[25].split()[1])
            self.keywords['LumDistInMpc'] = float(data[25].split()[2])

        N_par = self.keywords['N_base'] + 1 + self.keywords['N_exAV'] + 1

        # Skip spectra
        i = 63 + self.keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + self.keywords['N_base'] + 2 + self.keywords['N_base'] + 2 + 8
        self.keywords['Nl_obs']                  = int(data[i].split()[0])
        _n1 = i+1
        _n2 = _n1 + self.keywords['Nl_obs']
        _n3 = _n2 + 8

        self.keywords['FIR_arq_ETCinfo']       =      (data[_n3].split()[0]); _n3 += 1
        self.keywords['FIR_LumDistInMpc']      = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['FIR_logLFIR_TOTInLsun'] = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['FIR_LFIRFrac2Model']    = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['FIR_logLFIR_obsInLsun'] = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['FIR_ErrlogLFIRInDex']   = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['FIR_RangelogLFIRInDex'] = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['FIRChi2ScaleFactor']    = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['FIR_logLFIR_lowInLsun'] = float(data[_n3].split()[0])
        self.keywords['FIR_logLFIR_uppInLsun'] = float(data[_n3].split()[1]); _n3 += 1
        self.keywords['FIRbeta_D']             = float(data[_n3].split()[0]);
        self.keywords['FIRbeta_I']             = float(data[_n3].split()[1]); _n3 += 1
        self.keywords['log_LFIR/LOpt_rough']   = float(data[_n3].split()[0]); _n3 += 1

        _n3 += 2

        self.keywords['FIRModlogLFIRInLsun']   = float(data[_n3].split()[0])
        self.keywords['FIRModObsRatio']        = float(data[_n3].split()[1]); _n3 += 1
        self.keywords['FIRModlogLBOLInLsun']   = float(data[_n3].split()[0])
        self.keywords['FIR_BOL_Ratio']         = float(data[_n3].split()[1]); _n3 += 1
        self.keywords['chi2_FIR']              = float(data[_n3].split()[0])
        self.keywords['chi2_Opt']              = float(data[_n3].split()[1]); _n3 += 1
        self.keywords['chi2_FIR/TOT_Perc']     = float(data[_n3].split()[0])
        self.keywords['chi2_Opt/TOT_Perc']     = float(data[_n3].split()[1]); _n3 += 1

        _n3 += 2

        if (read_FIR_SSPs == True):

            # Reset & read FIR-related SSP arrays
            x_FIR    = []
            x_BOL    = []
            BolCor   = []
            FracLion = []
            Lbol_M   = []
            Rmat     = []
            R_opt    = []
            R_Lya    = []
            R_LCE    = []

            # Read FIR-related SSP arrays        
            _n1 = _n3
            _n2 = _n1 + self.keywords['N_base']
            for i in range(_n1,_n2):
                x_FIR.append(    float(data[i].split()[ 6]) )
                x_BOL.append(    float(data[i].split()[ 7]) )
                BolCor.append(   float(data[i].split()[ 8]) )
                FracLion.append( float(data[i].split()[ 9]) )
                Lbol_M.append(   float(data[i].split()[10]) )
                Rmat.append(     float(data[i].split()[11]) )
                R_opt.append(    float(data[i].split()[12]) )
                R_Lya.append(    float(data[i].split()[13]) )
                R_LCE.append(    float(data[i].split()[14]) )
    
            t = atpy.Table()
            t.table_name = 'FIR'
            t.add_column('x_FIR',    np.array(x_FIR, dtype='>f8'))
            t.add_column('x_BOL',    np.array(x_BOL, dtype='>f8'))
            t.add_column('BolCor',   np.array(BolCor, dtype='>f8'))
            t.add_column('FracLion', np.array(FracLion, dtype='>f8'))
            t.add_column('Lbol_M',   np.array(Lbol_M, dtype='>f8'))
            t.add_column('Rmat',     np.array(Rmat, dtype='>f8'))
            t.add_column('R_opt',    np.array(R_opt, dtype='>f8'))
            t.add_column('R_Lya',    np.array(R_Lya, dtype='>f8'))
            t.add_column('R_LCE',    np.array(R_LCE, dtype='>f8'))
            
            self.append(t)


    #--------------------------------------------------------------------
    # Reading QHR-related output 
    #--------------------------------------------------------------------
    if (self.keywords['IsQHRcOn'] != 0):

        # These things are needed if header was not read!
        if (read_header == False):
            self.keywords['N_base']            = int(data[9].split()[0])
            self.keywords['N_exAV']         = int(data[10].split()[1])
            self.keywords['flux_unit']    = float(data[14].split()[1])
            self.keywords['l_norm']       = float(data[22].split()[0])
            self.keywords['llow_norm']    = float(data[23].split()[0])
            self.keywords['lupp_norm']    = float(data[24].split()[0])
            self.keywords['fobs_norm']    = float(data[25].split()[0])
            self.keywords['Lobs_norm']    = float(data[25].split()[1])
            self.keywords['LumDistInMpc'] = float(data[25].split()[2])

        N_par = self.keywords['N_base'] + 1 + self.keywords['N_exAV'] + 1

        # Skip spectra
        i = 63 + self.keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + self.keywords['N_base'] + 2 + self.keywords['N_base'] + 2 + 8
        self.keywords['Nl_obs']                  = int(data[i].split()[0])
        _n1 = i+1
        _n2 = _n1 + self.keywords['Nl_obs']
        _n3 = _n2 + 8

        # Skip FIR
        if (self.keywords['IsFIRcOn'] != 0):
            _n3 = _n3 + 27 + self.keywords['N_base']

        self.keywords['QHRbeta_I']                 = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['QHR_arq_ETCinfo']           =      (data[_n3].split()[0]); _n3 += 1
        self.keywords['QHR_LumDistInMpc']          = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['QHR_GlobalChi2ScaleFactor'] = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['NQHR_Ys']                   =   int(data[_n3].split()[0]); _n3 += 1

        _n3 += 2

        # Reset & read QHR observed
        QHR_lambda        = []
        QHR_frecomb        = []
        QHR_logY_TOT        = []
        QHR_YFrac2Model        = []
        QHR_ErrlogY        = []
        QHR_RangelogY        = []
        QHR_Chi2ScaleFactor = []
        QHR_logY_obs        = []
        QHR_logY_low        = []
        QHR_logY_upp        = []

        # Read QHR observed
        _n1 = _n3
        _n2 = _n1 + self.keywords['NQHR_Ys']
        for i in range(_n1,_n2):
            QHR_lambda.append(          float(data[i].split()[1]))
            QHR_frecomb.append(         float(data[i].split()[2]))
            QHR_logY_TOT.append(        float(data[i].split()[3]))
            QHR_YFrac2Model.append(     float(data[i].split()[4]))
            QHR_ErrlogY.append(         float(data[i].split()[5]))
            QHR_RangelogY.append(       float(data[i].split()[6]))
            QHR_Chi2ScaleFactor.append( float(data[i].split()[7]))
            QHR_logY_obs.append(        float(data[i].split()[8]))
            QHR_logY_low.append(        float(data[i].split()[9]))
            QHR_logY_upp.append(        float(data[i].split()[10]))
            
        t = atpy.Table()
        t.table_name = 'QHR_Obs'
        t.add_column('lambda',          np.array(QHR_lambda, dtype='>f8'))
        t.add_column('frecomb',        np.array(QHR_frecomb, dtype='>f8'))
        t.add_column('logY_TOT',    np.array(QHR_logY_TOT, dtype='>f8'))
        t.add_column('YFrac2Model',    np.array(QHR_YFrac2Model, dtype='>f8'))
        t.add_column('ErrlogY',        np.array(QHR_ErrlogY, dtype='>f8'))
        t.add_column('RangelogY',    np.array(QHR_RangelogY, dtype='>f8'))
        t.add_column('Chi2ScaleFactor',    np.array(QHR_Chi2ScaleFactor, dtype='>f8'))
        t.add_column('logY_obs',    np.array(QHR_logY_obs, dtype='>f8'))
        t.add_column('logY_low',    np.array(QHR_logY_low, dtype='>f8'))
        t.add_column('logY_upp',           np.array(QHR_logY_upp, dtype='>f8'))
            
        self.append(t)


        # Read Emission Line Ratio-related things
        _n3 = _n2 + 2

        self.keywords['IsELROn']       =   int(data[_n3].split()[0])
        self.keywords['ELR_lambda_A']  = float(data[_n3].split()[1])
        self.keywords['ELR_lambda_B']  = float(data[_n3].split()[2])
        self.keywords['ELR_ind_A']     =   int(data[_n3].split()[3])
        self.keywords['ELR_ind_B']     =   int(data[_n3].split()[4])
        self.keywords['ELR_logRint']   = float(data[_n3].split()[5])
        self.keywords['ELR_AV_neb']    = float(data[_n3].split()[6])
        self.keywords['ELR_errAV_neb'] = float(data[_n3].split()[7]); _n3 += 1


        if (self.keywords['IsELROn'] == 1):

            self.keywords['ELR_Err_logR']        = float(data[_n3].split()[0])
            self.keywords['ELR_RangelogR']       = float(data[_n3].split()[1])
            self.keywords['ELR_logR_low']        = float(data[_n3].split()[2])
            self.keywords['ELR_logR_upp']        = float(data[_n3].split()[3])
            self.keywords['ELR_Chi2ScaleFactor'] = float(data[_n3].split()[4]); _n3 += 1

            # WARNING!!! Starlight output labels logRobs & ModlogR in the wrong order!
            self.keywords['ELR_ModlogR']           = float(data[_n3].split()[0])
            self.keywords['ELR_logRobs']           = float(data[_n3].split()[1])
            self.keywords['ELR_chi2_ELR']          = float(data[_n3].split()[2])
            self.keywords['ELR_chi2_ELR/chi2_QHR'] = float(data[_n3].split()[3])
            self.keywords['ELR_chi2_ELR/chi2_TOT'] = float(data[_n3].split()[4]); _n3 += 1
        else:
            _n3 += 2


        _n3 += 2

        self.keywords['log_QH0_PhotPerSec']   = float(data[_n3].split()[0])
        self.keywords['log_QHeff_PhotPerSec'] = float(data[_n3].split()[1]); _n3 += 1
        self.keywords['chi2_QHR/TOT_Perc']    = float(data[_n3].split()[0])
        self.keywords['chi2_Opt/TOT_Perc']    = float(data[_n3].split()[1]); _n3 += 1
        self.keywords['chi2_QHR']             = float(data[_n3].split()[0])
        self.keywords['chi2_Opt']             = float(data[_n3].split()[1])
        self.keywords['chi2_TOT']             = float(data[_n3].split()[2]); _n3 += 1

        _n3 += 2

        # Reset & read QHR model
        QHR_lambda        = []
        QHR_q_lambda      = []
        QHR_logY_obs      = []
        QHR_ModlogY       = []
        QHR_chi2_Y        = []
        QHR_chi2Y_chi2Opt = []
        QHR_chi2Y_chi2TOT = []

        # Read QHR model
        _n1 = _n3
        _n2 = _n1 + self.keywords['NQHR_Ys']
        for i in range(_n1,_n2):
            QHR_lambda.append(        float(data[i].split()[1]))
            QHR_q_lambda.append(      float(data[i].split()[2]))
            QHR_logY_obs.append(      float(data[i].split()[3]))
            QHR_ModlogY.append(       float(data[i].split()[4]))
            QHR_chi2_Y.append(        float(data[i].split()[5]))
            QHR_chi2Y_chi2Opt.append( float(data[i].split()[6]))
            QHR_chi2Y_chi2TOT.append( float(data[i].split()[7]))
        
        t = atpy.Table()
        t.table_name = 'QHR_Mod'
        t.add_column('lambda',        np.array(QHR_lambda, dtype='>f8'))
        t.add_column('q_lambda',      np.array(QHR_q_lambda, dtype='>f8'))
        t.add_column('logY_obs',      np.array(QHR_logY_obs, dtype='>f8'))
        t.add_column('ModlogY',       np.array(QHR_ModlogY, dtype='>f8'))
        t.add_column('chi2_Y',        np.array(QHR_chi2_Y, dtype='>f8'))
        t.add_column('chi2Y_chi2Opt', np.array(QHR_chi2Y_chi2Opt, dtype='>f8'))
        t.add_column('chi2Y_chi2TOT', np.array(QHR_chi2Y_chi2TOT, dtype='>f8'))

        self.append(t)


        _n3 = _n2 + 2

        if (read_QHR_SSPs == True):

            # Reset & read FIR-related SSP arrays
            qH__40       = []
            QH2Lnorm__40 = []
            QH0_Perc     = []
            QHeff_Perc   = []
            Y_Perc = dict()
            for il in range(0,self.keywords['NQHR_Ys']):
                Y_Perc[il] = []

            # Read FIR-related SSP arrays        
            _n1 = _n3
            _n2 = _n1 + self.keywords['N_base']
            for i in range(_n1,_n2):
                qH__40.append(       float(data[i].split()[ 6]) )
                QH2Lnorm__40.append( float(data[i].split()[ 7]) )
                QH0_Perc.append(     float(data[i].split()[ 8]) )
                QHeff_Perc.append(   float(data[i].split()[ 9]) )
                for il in range(0,self.keywords['NQHR_Ys']):
                    Y_Perc[il].append(   float(data[i].split()[ 10+il]) )

            t = atpy.Table()
            t.table_name = 'QHR'
            t.add_column('qH__40'      , np.array(qH__40, dtype='>f8'))
            t.add_column('QH2Lnorm__40', np.array(QH2Lnorm__40, dtype='>f8'))
            t.add_column('QH0_Perc'    , np.array(QH0_Perc, dtype='>f8'))
            t.add_column('QHeff_Perc'  , np.array(QHeff_Perc, dtype='>f8'))
            for il in range(0,self.keywords['NQHR_Ys']):
                t.add_column('Y_Perc_Line' + str(il) , np.array(Y_Perc[il], dtype='>f8'))

            self.append(t)
            
    #--------------------------------------------------------------------
    # Reading PHO-related output 
    #--------------------------------------------------------------------
    if (self.keywords['IsPHOcOn'] != 0):

        # These things are needed if header was not read!
        if (read_header == False):
            self.keywords['N_base']            = int(data[9].split()[0])
            self.keywords['N_exAV']         = int(data[10].split()[1])
            self.keywords['flux_unit']    = float(data[14].split()[1])
            self.keywords['l_norm']       = float(data[22].split()[0])
            self.keywords['llow_norm']    = float(data[23].split()[0])
            self.keywords['lupp_norm']    = float(data[24].split()[0])
            self.keywords['fobs_norm']    = float(data[25].split()[0])
            self.keywords['Lobs_norm']    = float(data[25].split()[1])
            self.keywords['LumDistInMpc'] = float(data[25].split()[2])

        N_par = self.keywords['N_base'] + 1 + self.keywords['N_exAV'] + 1

        # Skip spectra
        i = 63 + self.keywords['N_base'] + 6 - 1 + N_par - 1 + 1 + 2 + self.keywords['N_base'] + 2 + self.keywords['N_base'] + 2 + 8
        self.keywords['Nl_obs']                  = int(data[i].split()[0])
        _n1 = i+1
        _n2 = _n1 + self.keywords['Nl_obs']
        _n3 = _n2 + 8

        # Skip FIR
        if (self.keywords['IsFIRcOn'] != 0):
            _n3 = _n3 + 27 + self.keywords['N_base']

        # Skip QHR
        if (self.keywords['IsQHRcOn'] != 0):
            _n3 = _n3 + 29 + 2*self.keywords['NQHR_Ys'] + self.keywords['N_base']
            
        self.keywords['PHO_arq_ETCinfo']           =      (data[_n3].split()[0]); _n3 += 1
        self.keywords['PHO_LumDistInMpc']          = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['PHO_Redshift']              = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['PHO_GlobalChi2ScaleFactor'] = float(data[_n3].split()[0]); _n3 += 1
        self.keywords['NPHO_Ys']                   =   int(data[_n3].split()[0]); _n3 += 1

        _n3 += 2

        # Reset & read PHO observed
        PHO_name        = []
        PHO_logY_TOT        = []
        PHO_YFrac2Model        = []
        PHO_ErrlogY        = []
        PHO_RangelogY        = []
        PHO_Chi2ScaleFactor = []
        PHO_logY_obs        = []
        PHO_logY_low        = []
        PHO_logY_upp        = []

        # Read PHO observed
        _n1 = _n3
        _n2 = _n1 + self.keywords['NPHO_Ys']
        for i in range(_n1,_n2):
            PHO_name.append(                  data[i].split()[0])
            PHO_logY_TOT.append(        float(data[i].split()[2]))
            PHO_YFrac2Model.append(     float(data[i].split()[3]))
            PHO_ErrlogY.append(         float(data[i].split()[4]))
            PHO_RangelogY.append(       float(data[i].split()[5]))
            PHO_Chi2ScaleFactor.append( float(data[i].split()[6]))
            PHO_logY_obs.append(        float(data[i].split()[7]))
            PHO_logY_low.append(        float(data[i].split()[8]))
            PHO_logY_upp.append(        float(data[i].split()[9]))
            
        t = atpy.Table()
        t.table_name = 'PHO_Obs'
        t.add_column('name',          np.array(PHO_name, dtype='S20'))
        t.add_column('logY_TOT',    np.array(PHO_logY_TOT, dtype='>f8'))
        t.add_column('YFrac2Model',    np.array(PHO_YFrac2Model, dtype='>f8'))
        t.add_column('ErrlogY',        np.array(PHO_ErrlogY, dtype='>f8'))
        t.add_column('RangelogY',    np.array(PHO_RangelogY, dtype='>f8'))
        t.add_column('Chi2ScaleFactor',    np.array(PHO_Chi2ScaleFactor, dtype='>f8'))
        t.add_column('logY_obs',    np.array(PHO_logY_obs, dtype='>f8'))
        t.add_column('logY_low',    np.array(PHO_logY_low, dtype='>f8'))
        t.add_column('logY_upp',           np.array(PHO_logY_upp, dtype='>f8'))
            
        self.append(t)

        _n3 = _n2 + 2

        self.keywords['chi2_PHO/TOT_Perc']    = float(data[_n3].split()[0])
        self.keywords['chi2_Opt/TOT_Perc']    = float(data[_n3].split()[1]); _n3 += 1
        self.keywords['chi2_PHO']             = float(data[_n3].split()[0])
        self.keywords['chi2_Opt']             = float(data[_n3].split()[1])
        self.keywords['chi2_TOT']             = float(data[_n3].split()[2]); _n3 += 1

        _n3 += 2

#  name/code  MeanLamb StdDevLamb  q_MeanLamb   logY_obs      ModlogY       chi2_Y   chi2_Y/chi2_Opt  chi2_Y/chi2_TOT
        # Reset & read PHO model
        PHO_name          = []
        PHO_MeanLamb      = []
        PHO_StdDevLamb    = []
        PHO_q_MeanLamb    = []
        PHO_logY_obs      = []
        PHO_ModlogY       = []
        PHO_chi2_Y        = []
        PHO_chi2Y_chi2Opt = []
        PHO_chi2Y_chi2TOT = []

        # Read PHO model
        _n1 = _n3
        _n2 = _n1 + self.keywords['NPHO_Ys']
        for i in range(_n1,_n2):
            PHO_name.append(                data[i].split()[1])
            PHO_MeanLamb.append(      float(data[i].split()[2]))
            PHO_StdDevLamb.append(    float(data[i].split()[3]))
            PHO_q_MeanLamb.append(    float(data[i].split()[4]))
            PHO_logY_obs.append(      float(data[i].split()[5]))
            PHO_ModlogY.append(       float(data[i].split()[6]))
            PHO_chi2_Y.append(        float(data[i].split()[7]))
            PHO_chi2Y_chi2Opt.append( float(data[i].split()[8]))
            PHO_chi2Y_chi2TOT.append( float(data[i].split()[9]))
        
        t = atpy.Table()
        t.table_name = 'PHO_Mod'
        t.add_column('name',          np.array(PHO_name, dtype='S20'))
        t.add_column('MeanLamb',      np.array(PHO_MeanLamb, dtype='>f8'))
        t.add_column('StdDevLamb',    np.array(PHO_StdDevLamb, dtype='>f8'))
        t.add_column('q_MeanLamb',    np.array(PHO_q_MeanLamb, dtype='>f8'))
        t.add_column('logY_obs',      np.array(PHO_logY_obs, dtype='>f8'))
        t.add_column('ModlogY',       np.array(PHO_ModlogY, dtype='>f8'))
        t.add_column('chi2_Y',        np.array(PHO_chi2_Y, dtype='>f8'))
        t.add_column('chi2Y_chi2Opt', np.array(PHO_chi2Y_chi2Opt, dtype='>f8'))
        t.add_column('chi2Y_chi2TOT', np.array(PHO_chi2Y_chi2TOT, dtype='>f8'))

        self.append(t)

        _n3 = _n2 + 2

        if (read_PHO_SSPs == True):

            # Reset & read PHO-related SSP arrays
            Y_Perc = []

            # Read PHO-related SSP arrays        
            _n1 = _n3
            _n2 = _n1 + self.keywords['N_base']
            for i in range(_n1,_n2):
                Y_Perc.append([float(x) for x in data[i].split()[6:]])

            t = atpy.Table()
            t.table_name = 'PHO'
            t.add_column('Y_Perc', np.array(Y_Perc, dtype='>f8'))

            self.append(t)

####################################################################################################

register_reader('starlight_mask', _read_mask)
register_set_reader('starlight', _read_output)
