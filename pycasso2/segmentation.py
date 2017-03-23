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
    for z in xrange(Nzone):
        y1, x1 = zone_bins[z]
        y2 = y1 + bin_size
        x2 = x1 + bin_size 
        segmask[z, y1:y2, x1:x2] = 1
    return segmask


def ring_segmentation(shape, x0, y0, pa=0.0, ba=1.0, step=5):
    dist = get_image_distance(shape, x0, y0, pa, ba)
    Nzone = int(dist.max() / step)
    segmask = np.zeros((Nzone,) + shape, dtype='int32')
    for z in xrange(Nzone):
        r1 = z * step
        r2 = r1 + step
        ring = (dist >= r1) & (dist < r2)
        segmask[z, ring] = 1
    return segmask


def aperture_segmentation(shape, x0, y0, pa=0.0, ba=1.0, step=5):
    dist = get_image_distance(shape, x0, y0, pa, ba)
    Nzone = int(dist.max() / step)
    segmask = np.zeros((Nzone,) + shape, dtype='int32')
    for z in xrange(Nzone):
        ring = dist < (z + 1) * step
        segmask[z, ring] = 1
    return segmask


def sum_spectra(segmask, f_obs, f_err, f_flag, threshold=0.5):
    N_lambda = f_obs.shape[0]
    N_zone = segmask.shape[0]
    N_pix = segmask.sum(axis=2).sum(axis=1).astype('float')
    shape = (N_zone, N_lambda)
    zone_flux = np.zeros(shape)
    zone_error = np.zeros(shape)
    zone_flag = np.zeros(shape, dtype='int')
    for z in xrange(N_zone):
        zmask = segmask[z]
        fo = f_obs[:, zmask]
        fe = f_err[:, zmask]
        ff = f_flag[:, zmask]
        N_good = (ff == 0).astype('float').sum(axis=1)
        good_frac = N_good / N_pix[z]
        zone_flux[z] = fo.sum(axis=1) / good_frac
        zone_error[z] = np.sqrt((fe**2).sum(axis=1) / good_frac)
        bad_lambdas = good_frac < threshold
        zone_flag[z, bad_lambdas] = np.bitwise_or.reduce(ff[bad_lambdas], axis=1)
        zone_flux[z, bad_lambdas] = 0.0
        zone_error[z, bad_lambdas] = 0.0
    return zone_flux, zone_error, zone_flag
        

def sum_spectra_vector(segmask, f_obs, f_err, f_flag, threshold=0.5):
    N_good = np.tensordot(segmask, (f_flag == 0).astype('float'), axes=[[1,2],[1,2]])
    N_pix = segmask.sum(axis=2).sum(axis=1).astype('float')
    good_frac = N_good / N_pix[:, np.newaxis]
    bad_zone = good_frac < threshold
    zone_flag = np.where(bad_zone, flags.no_data, 0)
    zone_flux = np.tensordot(segmask, f_obs.filled(0.0), axes=[[1,2],[1,2]]) / good_frac
    zone_flux[bad_zone] = 0.0
    zone_error2 = np.tensordot(segmask, f_err.filled(0.0)**2, axes=[[1,2],[1,2]]) / good_frac
    zone_error = np.sqrt(zone_error2)
    zone_flux[bad_zone] = 0.0
    return zone_flux, zone_error, zone_flag


def spatialize(x, segmask, extensive=False):
    sum_segmask = segmask.sum(axis=0) 
    if (sum_segmask > 1).any():
        raise Exception('Segmentation mask has overlapping regions, can not be spatialized.')
    if extensive:
        area = segmask.sum(axis=2).sum(axis=1).astype('float')
        if x.ndim > 1:
            area.shape = area.shape + ((1,) * (x.ndim - 1))
        x = x / area
    x_spat = np.tensordot(x, segmask, axes=(0, 0))
    return np.ma.array(x_spat, mask=sum_segmask < 1)

