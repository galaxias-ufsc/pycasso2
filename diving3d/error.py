'''
Created on 02/10/2015

@author: andre
'''

import numpy as np

__all__ = ['estimate_error']


def smooth_spectra(ll, flux, spatial_mask, fwhm):
    '''
    Apply a gaussian filter to all spectra.
    '''
    from diving3d.resampling import gaussian1d_spectra

    flux_filt = np.zeros_like(flux)
    _, Ny, Nx = flux.shape
    for j in xrange(Ny):
        for i in xrange(Nx):
            if spatial_mask[j, i]:
                flux_filt[:, j, i] = np.ma.masked
            else:
                flux_filt[:, j, i] = gaussian1d_spectra(fwhm, ll, flux[:, j, i])
    return flux_filt


def interp_spectra(ll, flux, spatial_mask):
    '''
    Linearly interpolate the spectra, filling the masked wavelengths.
    '''
    from diving3d.resampling import interp1d_spectra

    flux_filt = np.zeros_like(flux)
    _, Ny, Nx = flux.shape
    for j in xrange(Ny):
        for i in xrange(Nx):
            if spatial_mask[j, i]:
                flux_filt[:, j, i] = np.ma.masked
            else:
                flux_filt[:, j, i] = interp1d_spectra(ll, flux[:, j, i])
    return flux_filt


def rms_box_spectra(ll, flux, width=100.0, threshold=0.5):
    '''
    Measure the RMS of the spectra using a running box,
    assuming the mean value of the spectra is zero.
    '''
    dl = ll[1] - ll[0]
    r = np.ceil(width / dl / 2.0) 
    Nl = flux.shape[0]
    rms = np.ma.masked_all(flux.shape)
    for l in xrange(Nl):
        l1 = l - r
        if l1 < 0:
            l1 = 0
        l2 = l + r
        if l2 >= Nl:
            l2 = Nl - 1
        Nbox = l2 - l1
        f = flux[l1:l2]
        Ngood = (~f.mask).astype('int').sum(axis=0)
        rms[l] = np.sqrt((f * f).sum(axis=0) / (Ngood - 1)) 
        rms[l][Ngood < (threshold * Nbox)] = np.ma.masked
    return rms


def estimate_error(ll, f_res, spatial_mask, smooth_fwhm=15.0, box_width=100.0):
    '''
    Estimate the uncertainty in the spectra from the residual of a
    STARLIGHT fit. Ideally, if the fits are good, the residual is
    caused by noise. In real life, there are problems with calibration
    and model spectra, so the residual must be rectified by subtracting
    the residual smoothed using a Gaussian filter.
    
    The uncertainty is then calculated as the RMS of the rectified
    residual, in a running box of the given width.
    
    Parameters
    ----------
    ll : array
        Wavelength array.
    
    f_res : array
        Residual from STARLIGHT fits.
    
    spatial_mask : array
        2-D image of the bad spaxels.
    
    smooth_fwhm : float, optional
        Full width at half maximum of the Gaussian kernel used
        in the rectification of the residual, in angstroms.
        Default: ``15.0``
    
    box_width : float, optional
        Width, in angstroms, of the running box used to
        measure the RMS of the residual.
        Default: ``100.0``
        
    Returns
    -------
    f_err : array
        The estimated uncertainty of the spectra.
    '''
    f_res_rect = f_res - smooth_spectra(ll, f_res, spatial_mask, fwhm=smooth_fwhm)
    f_err = rms_box_spectra(ll, f_res_rect, width=box_width, threshold=0.5)
    f_err = interp_spectra(ll, f_err, spatial_mask)
    return f_err

