'''
Created on 22 de mar de 2017

@author: andre
'''

from pycasso2 import FitsCube
from pycasso2.segmentation import mosaic_segmentation, spatialize, sum_spectra

from astropy.io import fits
import matplotlib.pyplot as plt
import sys

def read_segmentation(fitsfile):
    return fits.getdata(fitsfile)

def plot_test(f_obs_orig, f_obs_zone, segmask):
    f_obs_yx = spatialize(f_obs_zone, segmask, extensive=True)
    plt.figure()
    plt.imshow((f_obs_orig[500]))
    plt.colorbar()
    plt.figure()
    plt.imshow((f_obs_yx[500]))
    plt.colorbar()


c = FitsCube(sys.argv[1])
#segmask = ring_segmentation((c.Ny, c.Nx), c.x0, c.y0, c.pa, c.ba)
#segmask = aperture_segmentation((c.Ny, c.Nx), c.x0, c.y0, c.pa, c.ba)
segmask = mosaic_segmentation((c.Ny, c.Nx), bin_size=25)
f_obs, f_err, f_flag = sum_spectra(segmask, c.f_obs, c.f_err, c.f_flag)

cz = FitsCube()
cz._initFits(f_obs, f_err, f_flag, c._header, c._wcs, segmask)
cz.name = sys.argv[2]
cz.write('test.fits', overwrite=True)