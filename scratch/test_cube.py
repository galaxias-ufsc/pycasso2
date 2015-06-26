'''
Created on 23/06/2015

@author: andre
'''

from diving3d.cube import D3DFitsCube
from diving3d.tables import get_galaxy_id

def plot_example(d3d, galaxyId, x_slice):
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.figure()
    xx = d3d.x_coords
    yy = d3d.y_coords
    ll = d3d.l_obs
    
    plt.subplot(311)
    plt.pcolormesh(ll, yy, d3d.f_obs[:, :, x_slice].T * d3d.flux_unit, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'%s - flux [$\mathrm{erg}\ \mathrm{s}^{-1} \mathrm{cm}^{-2}\ \AA^{-1}$] @ R.A. = %.02f' % (galaxyId, xx[x_slice]))
    plt.colorbar()
    
    plt.subplot(312)
    plt.pcolormesh(ll, yy, d3d.f_syn[:, :, x_slice].T * d3d.flux_unit, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.gca().xaxis.set_ticklabels([])
    plt.xlim(ll.min(), ll.max())
    plt.title(r'flux (synthetic)')
    plt.colorbar()
    
    plt.subplot(313)
    plt.pcolormesh(ll, yy, d3d.f_flag[:, :, x_slice].T, cmap='cubehelix_r')
    plt.ylabel(r'dec. [arcsec]')
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.xlim(ll.min(), ll.max())
    plt.title('flags')
    plt.colorbar()
    
    plt.show()


cube = 'data/cubes_out/T001_resam_synth.fits'
galaxy_id = get_galaxy_id(cube)
d3d = D3DFitsCube(cube)

print 'Masterlist:'
for k, v in d3d.masterlist.iteritems():
    print '%010s: %020s' % (k, str(v))

plot_example(d3d, galaxy_id, x_slice=10)
