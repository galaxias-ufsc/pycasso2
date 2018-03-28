'''
Created on 30/05/2014

@author: andre
'''

from ..resampling import resample_cube, gauss_velocity_smooth
from ..reddening import calc_redlaw
from .io import read_base

from os import path
from tables import open_file, IsDescription, StringCol, Float64Col, Int32Col
import numpy as np

###############################################################################

class BaseTable(IsDescription):
    sspfile = StringCol(itemsize=60)
    age_base = Float64Col()
    Z_base = Float64Col()
    component = Float64Col()
    Mstars = Float64Col()
    YA_V = Int32Col()
    aFe = Float64Col()


###############################################################################

class StarlightBase(object):
    '''
    Starlight base interface and utilities.
    
    Parameters
    ----------
    base_file : string
        File describing the base. This is the one specified
        in the starlight grid file.
        
    base_dir : string
        Directory containing the base spectra.
    
    Examples
    --------
    TODO: examples
    '''

    def __init__(self, base_id=None, base_storage=None, is_hdf5=False, l_obs=None, l_norm=5635.0):
        self.name = path.basename(base_id)
        if is_hdf5 or base_storage.endswith('.hdf5') or base_storage.endswith('.h5'):
            self._loadHDF5(base_storage, base_id)
        else:
            self._loadASCII(base_storage, base_id)
        if l_obs is not None:
            self.f_ssp = self._resample_f_ssp(l_obs)
            self.l_obs = l_obs.copy()
        else:
            self.f_ssp = self._f_ssp.copy()
            self.l_obs = self._l_ssp.copy()
        
        self._l_norm = l_norm
        self.fbase_norm = self._calc_Fnorm(l_norm)
        self.f_ssp /= self.fbase_norm[:, np.newaxis]
        

    def _loadASCII(self, base_dir, base_file):
        btable, l_ssp, f_ssp = read_base(base_file, basedir=base_dir, read_spectra=True)
        self.sspfile = np.array(btable['sspfile'])
        self.ageBase = np.array(btable['age_base'])
        self.metBase = np.array(btable['Z_base'])
        self.component = np.array(btable['component'])
        self.Mstars = np.array(btable['Mstars'])
        self.YA_V = np.array(btable['YA_V'])
        self.aFe = np.array(btable['aFe'])
        self._l_ssp = l_ssp
        self._f_ssp = f_ssp
        self.nBase = len(self.ageBase)
        self.nWavelength = len(self._l_ssp)
        
    def _loadHDF5(self, base_db, base_id):
        with open_file(base_db, mode='r') as h5f:
            base = h5f.root._f_get_child(base_id)
            self.sspfile = base.base_table.cols.sspfile[:]
            self.ageBase = base.base_table.cols.age_base[:]
            self.metBase = base.base_table.cols.Z_base[:]
            self.component = base.base_table.cols.component[:]
            self.Mstars = base.base_table.cols.Mstars[:]
            self.YA_V = base.base_table.cols.YA_V[:]
            self.aFe = base.base_table.cols.aFe[:]
            self._l_ssp = base.l_ssp[:]
            self._f_ssp = base.f_ssp[...]
            self.nBase = len(self.ageBase)
            self.nWavelength = len(self._l_ssp)

    def writeHDF5(self, base_storage, base_id=None, overwrite=True):
        if base_id is None:
            base_id = self.name
        with open_file(base_storage, 'a') as h5f:
        
            if base_id not in h5f.root:
                gg = h5f.create_group('/', base_id)
            else:
                gg = h5f.root._f_get_child(base_id)
                if overwrite:
                    if 'base_table' in gg:
                        gg.base_table._f_remove()
                    if 'l_ssp' in gg:
                        gg.l_ssp._f_remove()
                    if 'f_ssp' in gg:
                        gg.f_ssp._f_remove()
            
            if 'base_table' in gg:
                raise Exception('Table /%s/base_table already exists in %s.' % (base_id, base_storage))
            if 'l_ssp' in gg:
                raise Exception('Dataset /%s/lssp already exists in %s.' % (base_id, base_storage))
            if 'f_ssp' in gg:
                raise Exception('Dataset /%s/f_ssp already exists in %s.' % (base_id, base_storage))

            base_table = h5f.create_table(gg, 'base_table', BaseTable, 'Starlight base')
            r = base_table.row
            for i in range(self.nBase):
                r['sspfile'] = self.sspfile[i]
                r['age_base'] = self.ageBase[i]
                r['Z_base'] = self.metBase[i]
                r['component'] = self.component[i]
                r['Mstars'] = self.Mstars[i]
                r['YA_V'] = self.YA_V[i]
                r['aFe'] = self.aFe[i]
                r.append()
            base_table.flush()
            h5f.create_array(gg, 'l_ssp', self._l_ssp, 'Wavelength (Angstroms)')
            h5f.create_array(gg, 'f_ssp', self._f_ssp, 'Luminosity (L_sun / M_sun / Angstrom)')
            
    
    def _calc_Fnorm(self, l_norm):
        fbase_norm = np.zeros(self.nBase)
        for i in range(self.nBase):
            fbase_norm[i] = np.interp(l_norm, self.l_obs, self.f_ssp[i])
        return fbase_norm
    
    
    def _resample_f_ssp(self, l_res):
        f_ssp_res = resample_cube(self._l_ssp, l_res, self._f_ssp.T)
        return f_ssp_res.T


    def f_syn(self, popx, v_0, v_d, A_V, redlaw='CCM'):
        if popx.ndim == 1:
            popx = popx[:, np.newaxis]
            v_0 = np.array([v_0])
            v_d = np.array([v_d])
            A_V = np.array([A_V])
        spatial_shape = popx.shape[1:]
        popx = popx.reshape(self.nBase, -1)
        A_V = A_V.ravel()
        v_0 = v_0.ravel()
        v_d = v_d.ravel()
        popx = popx / 100.0
        f_syn = np.tensordot(self.f_ssp, popx, (0, 0))
        q = calc_redlaw(self.l_obs, redlaw)
        q_norm = calc_redlaw([self._l_norm], redlaw)
        q -= q_norm
        redfactor = np.power(10.0, -0.4 * q[:, np.newaxis] * A_V)
        f_syn *= redfactor
        for i in range(f_syn.shape[-1]):
            f_syn[:, i] = gauss_velocity_smooth(self.l_obs, f_syn[:, i], v_0[i], v_d[i])
        f_syn.shape = (len(self.l_obs),) + spatial_shape
        return np.squeeze(f_syn) 
    

    def popmu_ini(self, popx, A_V, redlaw='CCM'):
        raise NotImplementedError('mass stuff not implemented.')
        

    def popmu_cor(self, popx, A_V, redlaw='CCM'):
        raise NotImplementedError('mass stuff not implemented.')
        

    def popx_trans(self, l_norm, popx, v_0, v_d, A_V, redlaw='CCM'):
        raise NotImplementedError('TODO')
        
