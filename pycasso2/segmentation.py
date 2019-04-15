'''
Created on 23 de mar de 2017

@author: andre
'''
import numpy as np
from .geometry import get_image_distance
from . import flags


def mosaic_segmentation(shape, bin_size=10):
    Ny, Nx = shape
    xb = np.arange(0, Nx, bin_size)
    yb = np.arange(0, Ny, bin_size)
    zone_bins = [[y, x] for y in yb for x in xb]
    Nzone = len(zone_bins)
    segmask = np.zeros((Nzone, Ny, Nx), dtype='int32')
    for z in range(Nzone):
        y1, x1 = zone_bins[z]
        y2 = y1 + bin_size
        x2 = x1 + bin_size
        segmask[z, y1:y2, x1:x2] = 1
    return segmask


def ring_segmentation(shape, x0, y0, pa=0.0, ba=1.0, step=5):
    dist = get_image_distance(shape, x0, y0, pa, ba)
    Nzone = int(dist.max() / step)
    segmask = np.zeros((Nzone,) + shape, dtype='int32')
    for z in range(Nzone):
        r1 = z * step
        r2 = r1 + step
        ring = (dist >= r1) & (dist < r2)
        segmask[z, ring] = 1
    return segmask


def aperture_segmentation(shape, x0, y0, pa=0.0, ba=1.0, step=5):
    dist = get_image_distance(shape, x0, y0, pa, ba)
    Nzone = int(dist.max() / step)
    segmask = np.zeros((Nzone,) + shape, dtype='int32')
    for z in range(Nzone):
        ring = dist < (z + 1) * step
        segmask[z, ring] = 1
    return segmask


def voronoi_segmentation(signal, noise, targetSN, plot=False, quiet=True):
    try:
        from voronoi.voronoi_2d_binning import voronoi_2d_binning
    except ImportError:
        raise Exception('Voronoi binning module not installed.')
    good = ~np.ma.getmaskarray(signal)
    yy, xx = np.indices(signal.shape)

    zone_num, xNode, _, _, _, _, _, _ = voronoi_2d_binning(
            xx[good], yy[good], signal[good], noise[good], targetSN,
            plot=plot, quiet=quiet)

    zones = np.ma.empty(signal.shape)
    zones[good] = zone_num
    zones[~good] = -1
    Nzone = len(xNode)
    segmask = np.zeros((Nzone,) + signal.shape, dtype='int32')
    for z in range(Nzone):
        segmask[z, zones == z] = 1
    return segmask


def prune_segmask(segmask, spatial_mask):
    '''
    Remove zones that have no data.
    '''
    segmask[:, spatial_mask] = 0
    good_zones = segmask.sum(axis=(1, 2)) > 0
    return segmask[good_zones]


def sum_spectra(segmask, f_obs, f_err, f_flag=None, cov_factor_A=0.0, cov_factor_B=1.0, cov_matrix=None):
    if not isinstance(f_obs, np.ma.MaskedArray):
        if f_flag is None:
            raise Exception('f_flag must be specified if f_obs is not a masked array.')
        good = (f_flag & flags.no_obs) == 0
        f_obs = np.where(good, f_obs, 0.0)
        f_err = np.where(good, f_err, 0.0)
    else:
        good = ~np.ma.getmaskarray(f_obs)
        f_obs = f_obs.filled(0.0)
        f_err = f_err.filled(0.0)

    N_good = np.tensordot(good.astype('float'), segmask, axes=[[1, 2], [1, 2]])

    N_pix = segmask.sum(axis=(1, 2)).astype('float')
    good_frac = N_good / N_pix
    zone_flux = np.tensordot(f_obs, segmask, axes=[[1, 2], [1, 2]])
    valid = N_good > 0
    zone_flux[valid] /= good_frac[valid]
    if cov_matrix is None:
        zone_error = np.tensordot(f_err**2, segmask, axes=[[1, 2], [1, 2]])
        zone_error[valid] /= good_frac[valid]
        np.sqrt(zone_error, out=zone_error)
        f = get_cov_factor(N_pix, cov_factor_A, cov_factor_B)
        zone_error *= f
    else:
        C  = cov_matrix[0]
        fC = cov_matrix[1]
        Cg, Cr, Ci, Cz =  C['g'],  C['r'],  C['i'],  C['z']
        fg, fr, fi, fz = fC['g'], fC['r'], fC['i'], fC['z']
        zone_error_g = calc_ferr_covMatrix(Cg, segmask, f_err)
        zone_error_r = calc_ferr_covMatrix(Cr, segmask, f_err)
        zone_error_i = calc_ferr_covMatrix(Ci, segmask, f_err)
        zone_error_z = calc_ferr_covMatrix(Cz, segmask, f_err)
        zone_error     = zone_error_g
        zone_error[fr] = zone_error_r[fr]
        zone_error[fi] = zone_error_i[fi]
        zone_error[fz] = zone_error_z[fz]
    return zone_flux, zone_error, good_frac

def calc_ferr_covMatrix(C, segmask, f_err):
    Nz = segmask.shape[0]
    Nl = f_err.shape[0]
    err = np.zeros((Nl, Nz))
    for iz in np.arange(Nz):
        i = np.where(segmask.reshape(Nz, -1)[iz] > 0)
        ij = np.meshgrid(i, i)
        C_segmask = C[ij[0], ij[1]]
        f_err_segmask = f_err.reshape(Nl, -1)[..., i].reshape(Nl, -1)
        A = np.tensordot(C_segmask.toarray(), f_err_segmask, axes=[[0], [1]])
        V = np.sum(A * f_err_segmask.T, axis=0)
        err[..., iz] = np.sqrt(V)
    return err

def spatialize(x, segmask, extensive=False):
    good_zones = ~np.ma.getmaskarray(x)
    while good_zones.ndim > 1:
        good_zones = good_zones.any(axis=0)
    sum_segmask = segmask[good_zones].sum(axis=0)
    if (sum_segmask > 1).any():
        raise Exception('Segmentation mask has overlapping regions, can not be'
                        ' spatialized.')
    if extensive:
        area = segmask.sum(axis=(1, 2)).astype('float')
        x = x / area
    if isinstance(x, np.ma.MaskedArray):
        x = x.filled(0.0)
    x_spat = np.tensordot(x, segmask, axes=(-1, 0))
    mask = sum_segmask < 1
    if x_spat.ndim == mask.ndim:
        return np.ma.array(x_spat, mask=mask)
    else:
        x_spat = np.ma.array(x_spat)
        x_spat[..., mask] = np.ma.masked
        return x_spat


def read_segmentation_map(fitsfile):
    from astropy.io import fits
    from .cube import FitsCube

    segmask = fits.getdata(fitsfile, extname=FitsCube._ext_segmask)
    return segmask


def bin_spectra(f_obs, f_err, f_flag, bin_size, cov_factor_A=0.0, cov_factor_B=1.0, cov_matrix=None):
    Nl = f_obs.shape[0]
    spatial_shape = f_obs.shape[1:]
    segmask = mosaic_segmentation(spatial_shape, bin_size)
    f_obs, f_err, good_frac = sum_spectra(segmask, f_obs, f_err, f_flag,
                                          cov_factor_A, cov_factor_B, cov_matrix)
    Ny_b = np.ceil(spatial_shape[0] / bin_size)
    Nx_b = np.ceil(spatial_shape[1] / bin_size)
    shape = (Nl, int(Ny_b), int(Nx_b))
    f_obs = f_obs.reshape(shape)
    f_err = f_err.reshape(shape)
    good_frac = good_frac.reshape(shape)
    return f_obs, f_err, good_frac


def get_cov_factor(N, A, B):
    return 1.0 + A * np.log10(N)**B


def integrate_spectra(f_obs, f_err, f_flag, mask, bin_size=1, cov_factor_A=0.0, cov_factor_B=1.0):
    segmask = np.where(mask, 0, 1)[np.newaxis, ...]
    if bin_size > 1:
        N = bin_size**2
        f_err = f_err / get_cov_factor(N, cov_factor_A, cov_factor_B)
    f_obs, f_err, good_frac = sum_spectra(segmask, f_obs, f_err, f_flag,
                                          cov_factor_A, cov_factor_B)
    f_obs = f_obs.squeeze()
    f_err = f_err.squeeze()
    good_frac = good_frac.squeeze()
    return f_obs, f_err, good_frac
