'''
Created on 15/06/2015

@author: andre

Resample spectra in 1 \AA bins and change wavelength interval. 

'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id, read_masterlist, get_wavelength_mask
from diving3d import flags

def velocity_to_redshift(v):
    c = 299792458.0 # km/s
    return v / c

def plot_example(d3d, galaxyId, y_slice):
    import matplotlib.pyplot as plt
    plt.ioff()
    xx = d3d.x_coords
    yy = d3d.y_coords
    ll = d3d.l_obs
    plt.pcolormesh(xx, ll, d3d.f_flag[:, y_slice, :])
    plt.xlabel(r'R.A. [arcsec]')
    plt.ylabel(r'Wavelength [$\AA$]')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(ll.min(), ll.max())
    plt.title('%s - flags example @ dec = %.02f "' % (galaxyId, yy[y_slice]))
    plt.colorbar()
    plt.show()

# TODO: read resampling setup from a config file
debug = False
redcube = 'data/cubes/T001_AV_r_d_NIT_6_fft_x_0pt15_y_0pt15_n_eq_6_bg_rec_wav_rec_pca_dop_hel.fits'
resamcube = 'data/cubes/T001_resampled.fits'
masterlist = 'data/masterlist_sampleT.txt'
mask_template = 'data/Masks.gap.%s'

kwargs = dict(l_ini=4000.0,
              l_fin=7200.0,
              dl=1.0,
              width=100,
              height=100)

d3d = D3DFitsCube.from_reduced(redcube, **kwargs)

galaxyId = get_galaxy_id(redcube)
ml = read_masterlist(masterlist, galaxyId)
z = velocity_to_redshift(ml['V_hel'])
maskfile = mask_template % ml['grating']
gap_mask = get_wavelength_mask(maskfile, d3d.l_obs, z, dest='rest')
d3d.f_flag[gap_mask] |= flags.ccd_gap

if debug:
    plot_example(d3d, galaxyId, y_slice=10)

d3d.write(resamcube)
