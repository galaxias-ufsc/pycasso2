'''
Created on 22/06/2015

@author: andre
'''

import pystarlight.util.StarlightUtils as stutil
import numpy as np

from . import flags

__all__ = ['resample_spectra', 'reshape_spectra']



def resample_spectra(spectra, l_orig, l_resam):
    '''
    FIXME: document reshape_image
    '''
    R = stutil.ReSamplingMatrixNonUniform(l_orig, l_resam)
    spectra = np.tensordot(R, spectra, (1,0))
    flagged = np.zeros_like(spectra, dtype='int') + flags.no_data
    good = (l_resam >= l_orig[0]) & (l_resam <= l_orig[-1])
    flagged[good] = 0
    return spectra, flagged


def reshape_spectra(f_obs, f_flag, center, new_shape):
    '''
    FIXME: document reshape_image
    '''
    shape = f_obs.shape
    y_axis = 1
    x_axis = 2
    y0 = new_shape[y_axis] / 2 - center[y_axis]
    yf = y0 + shape[y_axis]
    x0 = new_shape[x_axis] / 2 - center[x_axis]
    xf = x0 + shape[x_axis]
    
    res_f_obs = np.zeros((new_shape))
    res_f_obs[:, y0:yf, x0:xf] = f_obs
    res_f_flag = np.ones_like(res_f_obs, dtype='int') * flags.no_data
    res_f_flag[:, y0:yf, x0:xf] = f_flag
    
    res_center = (center[0], new_shape[1] / 2, new_shape[2] / 2)
    
    return res_f_obs, res_f_flag, res_center
    
    