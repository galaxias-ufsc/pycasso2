'''
Created on 24/06/2015

@author: andre
'''

from astropy.io import fits


obscube = 'data/cubes_obs/T001_208g21.fits'
redcube = 'data/cubes/T001_AV_r_d_NIT_6_fft_x_0pt15_y_0pt15_n_eq_6_bg_rec_wav_rec_pca_dop_hel.fits'

def safe_getheader(f, ext):
    with fits.open(f) as hl:
        hdu = hl[ext]
        hdu.verify('fix')
        return hdu.header

hobs = safe_getheader(obscube, 0)
hred = safe_getheader(redcube, 0)

# Unique set of headers
keys = set(hred.keys() + hobs.keys())

# Check for collisions
for k in keys:
    if k in hred and k in hobs:
        print '%010s: obs=%020s, red=%020s' % (k, hobs.get(k), hred.get(k))
        
'''
The output should look like the contents below. Only the WCS should change.
    EXTEND: obs=                True, red=                True
    SIMPLE: obs=                True, red=                True
    OBJECT: obs=             208-G21, red=             208-G21
    CCDSUM: obs=                 1 1, red=                 1 1
      GAIN: obs=                 1.0, red=                 1.0
     CD2_2: obs=   9.21106248208e-06, red=               -0.05
  GAINMULT: obs=                2.01, red=                2.01
    CTYPE2: obs=            DEC--TAN, red=              DEC-AR
    CTYPE1: obs=            RA---TAN, red=              RA--AR
   EXPTIME: obs=       1800.46078491, red=            1800.461
     NAXIS: obs=                   0, red=                   3
    BITPIX: obs=                  16, red=                 -32
     CD1_1: obs=   9.22125945997e-06, red=                0.05
    ORIGIN: obs=NOAO-IRAF FITS Image Kernel July 2003, red=NOAO-IRAF FITS Image Kernel July 2003
  IRAF-TLM: obs=06:39:33 (21/11/2008), red=06:39:34 (21/11/2008)
   RDNOISE: obs=                3.69, red=                3.69
   AIRMASS: obs=               1.189, red=               INDEF
    CRVAL2: obs=      -50.4360700117, red=                 0.0
    CRPIX1: obs=       3109.63729259, red=             31.0000
    CRPIX2: obs=        2303.5894574, red=             44.0000
    CRVAL1: obs=       113.490729853, red=                 0.0
      DATE: obs= 2008-11-21T11:39:33, red= 2008-11-21T11:42:24
'''
        
# Compute merged header, keep the reduced version if there's collision.
for k in hobs.keys():
    if k in hred or k == 'COMMENT' or k == '': continue
    try:
        hred[k] = hobs[k]
    except:
        print 'Error in card %s: %s' % (k, hobs[k])
        
        