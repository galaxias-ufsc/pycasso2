'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id, get_wavelength_mask
from diving3d.resampling import velocity2redshift

def plot_example(l_obs, f_obs, gap_mask_z, gap_mask_no_z):
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.figure()
    
    plt.subplot(111)
    plt.plot(l_obs, f_obs, 'k-')
    plt.plot(l_obs, gap_mask_z, 'r-', label='gap (rest frame of the galaxy)')
    plt.plot(l_obs, gap_mask_no_z, 'b-', label='gap (observed frame)')
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.xlim(5350.0, 5500.0)
    plt.ylim(0.1, 0.2)
    plt.legend()
    plt.show()

def get_gap_mask(l_obs, v, gap_mask_file):
    z = velocity2redshift(v)
    print 'Loading gap masks (v = %.1f km/s, z = %f)' % (v, z)
    return get_wavelength_mask(gap_mask_file, l_obs, z, dest='rest')


cube = 'data/cubes_out/T001_resampled.fits'
galaxy_id = get_galaxy_id(cube)
gap_mask_template = 'data/Masks.gap.%s'

d3d = D3DFitsCube(cube)
ml = d3d.masterlist
c = d3d.center

print 'Masterlist:'
for k, v in ml.iteritems():
    print '%010s: %020s' % (k, str(v))

print ml['V_HEL']

gap_mask_file = gap_mask_template % ml['GRATING']
gap_mask_z = get_gap_mask(d3d.l_obs, ml['V_HEL'], gap_mask_file)
gap_mask_no_z = get_gap_mask(d3d.l_obs, 0.0, gap_mask_file)

f_obs = d3d.f_obs[:, c[1], c[2]]

plot_example(d3d.l_obs, f_obs, gap_mask_z, gap_mask_no_z)
