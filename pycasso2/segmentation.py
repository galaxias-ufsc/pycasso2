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


def sum_spectra(segmask, f_obs, f_err, f_flag, threshold=0.5):

    N_good = np.tensordot(
            (f_flag == 0).astype('float'), segmask, axes=[[1, 2], [1, 2]])

    N_pix = segmask.sum(axis=(1, 2)).astype('float')
    good_frac = N_good / N_pix
    flagged = good_frac < threshold
    zone_flag = np.where(flagged, flags.no_data, 0)
    zone_flag |= np.where(good_frac < 1.0, flags.seg_has_badpixels, 0)
    zone_flux = np.tensordot(
            f_obs.filled(0.0), segmask, axes=[[1, 2], [1, 2]]) / good_frac
    zone_flux[flagged] = 0.0
    zone_error2 = np.tensordot(
            f_err.filled(0.0)**2, segmask, axes=[[1, 2], [1, 2]]) / good_frac
    zone_error = np.sqrt(zone_error2)
    zone_error[flagged] = 0.0
    return zone_flux, zone_error, zone_flag


def spatialize(x, segmask, extensive=False):
    good_zones = ~np.ma.getmaskarray(x)
    sum_segmask = segmask[good_zones].sum(axis=0)
    if (sum_segmask > 1).any():
        raise Exception('Segmentation mask has overlapping regions, can not be'
                        ' spatialized.')
    if extensive:
        area = segmask.sum(axis=(1, 2)).astype('float')
        x = x / area
    x_spat = np.tensordot(x.filled(0.0), segmask, axes=(-1, 0))
    mask = sum_segmask < 1
    if x_spat.ndim == mask.ndim:
        return np.ma.array(x_spat, mask=mask)
    else:
        x_spat = np.ma.array(x_spat)
        x_spat[:, mask] = np.ma.masked
        return x_spat
        return


def read_segmentation_map(fitsfile):
    from astropy.io import fits
    from .cube import FitsCube

    segmask = fits.getdata(fitsfile, extname=FitsCube._ext_segmask)
    return segmask
