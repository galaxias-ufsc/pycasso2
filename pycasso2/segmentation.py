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


def sum_spectra(segmask, f_obs, f_err, f_flag=None, cov_factor=None):
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
    zone_error = np.tensordot(f_err**2, segmask, axes=[[1, 2], [1, 2]])
    zone_error[valid] /= good_frac[valid]
    np.sqrt(zone_error, out=zone_error)
    if cov_factor is not None:
        zone_error *= cov_factor
    return zone_flux, zone_error, good_frac


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
        return


def read_segmentation_map(fitsfile):
    from astropy.io import fits
    from .cube import FitsCube

    segmask = fits.getdata(fitsfile, extname=FitsCube._ext_segmask)
    return segmask


def bin_spectra(f_obs, f_err, f_flag, bin_size, cov_factor=None):
    Nl = f_obs.shape[0]
    spatial_shape = f_obs.shape[1:]
    segmask = mosaic_segmentation(spatial_shape, bin_size)
    f_obs, f_err, good_frac = sum_spectra(segmask, f_obs, f_err, f_flag, cov_factor)
    shape = (Nl, spatial_shape[0] // bin_size, spatial_shape[1] // bin_size)
    f_obs = f_obs.reshape(shape)
    f_err = f_err.reshape(shape)
    good_frac = good_frac.reshape(shape)
    return f_obs, f_err, good_frac
    

def get_cov_factor(N, A, B):
    return 1.0 + A * np.log10(N)**B
