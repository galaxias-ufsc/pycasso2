'''
Created on Jun 27, 2013

@author: andre
'''

import numpy as np

__all__ = ['FixKinematics', 'apply_kinematics', 'apply_kinematics_flagged']

################################################################################
def gauss_velocity_smooth(lo, fo, v0, sig, ls=None):
    '''
    Apply a gaussian velocity dispersion and displacement filter to a spectrum.
    Implements the integration presented in the page 27 of the 
    `STARLIGHT manual <http://www.starlight.ufsc.br/papers/Manual_StCv04.pdf>`_.
    
    Parameters
    ----------
    
    lo : array
        Original wavelength array.
        
    fo : array
        Original flux array, must be of same length as ``lo``.
    
    v0 : float
        Systemic velocity to apply to input spectrum.
        
    sig : float
        Velocity dispersion (sigma of the gaussian).
        
    ls : array, optional
        Wavelengths in which calculate the output spectrum.
        If not set, ``lo`` will be used.
        
        
    Returns
    -------
    
    fs : array
        Resampled and smoothed flux array, must be of same length as ``ls``
        (or ``lo`` if ``ls`` is not set).
    
   '''
    from .gauss_smooth import gauss_velocity_smooth
    lo = np.ascontiguousarray(lo)
    fo = np.ascontiguousarray(fo)
    if lo.shape != fo.shape or lo.ndim > 1:
        raise Exception('lo and fo must be unidimensional and have the same length.')
    return gauss_velocity_smooth(lo, fo, v0, sig)
################################################################################


################################################################################
class FixKinematics(object):
    '''
    FIXME: docme.
    '''
    
    def __init__(self, l_obs, v_0, v_d, nproc=None):
        self.l_obs = np.ascontiguousarray(l_obs, 'float64')
        if not np.allclose(self.l_obs, np.linspace(self.l_obs.min(), self.l_obs.max(), len(self.l_obs))):
            raise ValueError('l_obs is not equally spaced.')
        if np.isscalar(v_0):
            self.v_0 = np.array([v_0])
        else:
            self.v_0 = np.asarray(v_0)
        if np.isscalar(v_d):
            self.v_d = np.array([v_d])
        else:
            self.v_d = np.asarray(v_d)
        self.nproc = nproc
        
        
    def _getVd(self, target_vd):
        m = self.v_d < target_vd
        vd_fix = np.zeros_like(self.v_d)
        vd_fix[m] = np.sqrt(target_vd**2 - self.v_d[m]**2)
        return vd_fix
  
    
    def applyFlagged(self, flux, err, target_vd=0.0, fill='nearest', fill_val=0.0, l_cov_FWHM=-1.0):
        vd = self._getVd(target_vd)
        return apply_kinematics_flagged(self.l_obs, flux, err, -self.v_0, vd,
                                        fill, fill_val, l_cov_FWHM, self.nproc)


    def apply(self, flux, target_vd=0.0, fill='nearest', fill_val=0.0):
        vd = self._getVd(target_vd)
        return apply_kinematics(self.l_obs, flux, -self.v_0, vd, fill, fill_val, self.nproc)
################################################################################


################################################################################
def apply_kinematics_flagged(l_obs, flux, err, v_0, v_d,
                             fill='nearest', fill_val=0.0, l_cov_FWHM=-1.0, nproc=None):
    '''
    FIXME: docme.
    '''
    if np.isscalar(v_0):
        v_0 = np.array((v_0,))
    if np.isscalar(v_d):
        v_d = np.array((v_d,))
    shape = (len(l_obs),) + v_0.shape
    one_d = False
    if flux.shape == l_obs.shape:
        one_d = True
        flux = flux[:, np.newaxis]
        err = err[:, np.newaxis]
    elif flux.shape != shape:
        raise ValueError('flux has an incorrect shape: %s. Should be %s.' % flux.shape, shape)
    if isinstance(flux, np.ma.MaskedArray):
        badpix = np.ma.getmaskarray(flux)
    else:
        badpix = np.ones_like(flux, dtype='bool')

    spatial_shape = flux.shape[1:]
    N_spec = flux.shape[0]
    flux = flux.reshape(N_spec, -1)
    err = err.reshape(N_spec, -1)
    badpix = badpix.reshape(N_spec, -1)
    v_0 = v_0.ravel()
    v_d = v_d.ravel()

    params = _params_error_iter(l_obs, flux, err, badpix, v_0, v_d, fill, fill_val, l_cov_FWHM)
    out = _process(_velocity_smooth_error_wrapper, params, nproc)

    if one_d:
        flux_s, err_s, badpix_s = out[0][0], out[0][1], out[0][2]
    else:
        flux_s = np.empty_like(flux)
        err_s = np.empty_like(err)
        badpix_s = np.empty_like(badpix)
        for i, z in enumerate(out):
            flux_s[:,i], err_s[:,i], badpix_s[:,i] = z[0], z[1], z[2]
        flux_s.shape = (N_spec,) + spatial_shape
        err_s.shape = (N_spec,) + spatial_shape
        badpix_s.shape = (N_spec,) + spatial_shape
    if isinstance(flux, np.ma.MaskedArray):
        flux_s[badpix_s] = np.ma.masked
        err_s[badpix_s] = np.ma.masked
    return flux_s, err_s, badpix_s
################################################################################


################################################################################
def apply_kinematics(l_obs, flux, v_0, v_d, fill='nearest', fill_val=0.0, nproc=None):
    '''
    FIXME: docme.
    '''
    if np.isscalar(v_0):
        v_0 = np.array((v_0,))
    if np.isscalar(v_d):
        v_d = np.array((v_d,))
    shape = (len(l_obs),) + v_0.shape
    one_d = False
    if flux.shape == l_obs.shape:
        one_d = True
        flux = flux[:, np.newaxis]
    elif flux.shape != shape:
        raise ValueError('flux has an incorrect shape: %s. Should be %s.' % flux.shape, shape)

    spatial_shape = flux.shape[1:]
    N_spec = flux.shape[0]
    flux = flux.reshape(N_spec, -1)
    v_0 = v_0.ravel()
    v_d = v_d.ravel()
    params = _params_iter(l_obs, flux, v_0, v_d, fill, fill_val)
    out = _process(_velocity_smooth_wrapper, params, nproc)

    if one_d:
        return out[0]
    else:
        flux = np.array(out).T
        flux.shape = (N_spec,) + spatial_shape
        return flux
################################################################################


################################################################################
def _process(func, params, nproc=None):
    import multiprocessing
    if nproc is None:
        nproc = multiprocessing.cpu_count()
    if nproc != 1:
        pool = multiprocessing.Pool(nproc)
        out = pool.map(func, params)
        pool.close()
    else:
        out = [func(args) for args in params]
    return out
################################################################################


################################################################################
def _params_error_iter(l_obs, flux, err, badpix, v_0, v_d, fill, fill_val, l_cov_FWHM):
    '''
    Argument factory for :func:`_velocitySmoothFlag_wrapper`.
    '''
    if flux.ndim == 1:
        N_spec = 1
    else:
        N_spec = flux.shape[1]
    for i in range(N_spec):
        yield (l_obs,
               np.ascontiguousarray(flux[:,i], 'float64'),
               np.ascontiguousarray(err[:,i], 'float64'),
               np.ascontiguousarray(badpix[:,i], 'bool'),
               v_0[i], v_d[i], fill, fill_val, l_cov_FWHM)
################################################################################
    

################################################################################
def _params_iter(l_obs, flux, v_0, v_d, fill, fill_val):
    '''
    Argument factory for :func:`_velocitySmooth_wrapper`.
    '''
    if flux.ndim == 1:
        N_spec = 1
    else:
        N_spec = flux.shape[1]
    for i in range(N_spec):
        yield (l_obs,
               np.ascontiguousarray(flux[:,i], 'float64'),
               v_0[i], v_d[i], fill, fill_val)
################################################################################
    
    
################################################################################
def _velocity_smooth_error_wrapper(args):
    from .gauss_smooth import gauss_velocity_smooth_error
    l_obs, flux, err, badpix, v_0, v_d, fill, fill_val, l_cov_FWHM = args
    return gauss_velocity_smooth_error(l_obs, flux, err, badpix,
                                   v_0, v_d, n_sig=6, n_u=51,
                                   fill=fill, fill_val=fill_val,
                                   badpix_threshold=0.8, l_cov_FWHM=l_cov_FWHM)
################################################################################

################################################################################
def _velocity_smooth_wrapper(args):
    from .gauss_smooth import gauss_velocity_smooth  # @UnresolvedImport
    l_obs, flux, v_0, v_d, fill, fill_val = args
    return gauss_velocity_smooth(l_obs, flux, v_0, v_d, n_sig=6, n_u=51, fill=fill, fill_val=fill_val)
################################################################################

