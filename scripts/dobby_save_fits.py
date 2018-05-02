from os import path

import h5py
import numpy as np
from astropy.table import Table

def save_fits_test(c, galname, outdir, el_dir, name_template, balLim=True, kinTies=True, model='GaussianIntELModel'):

    # Define things that depend on pixels/zones
    inZones = (len(c.f_obs.shape) == 2)

    if inZones:

        # Get dimensions
        Nl, Nz = c.f_obs.shape

    else:

        # Get dimensions
        Nl, Ny, Nx = c.f_obs.shape
        Nz = (Ny, Nx)

        
    # Get wavelenths
    ll = c.l_obs

    
    # Read the integrated spec to find the emission lines fitted
    filename = path.join(outdir, 'integ.hdf5')

    with h5py.File(filename, 'r') as f:
        El_lambda = f['elines']['lambda']
        El_name   = f['elines']['line']
        El_l0     = f['elines']['El_l0']

    Ne = len(El_lambda)
    
    El_F     = np.zeros((Ne, Ny, Nx))
    El_v0    = np.zeros((Ne, Ny, Nx))
    El_vd    = np.zeros((Ne, Ny, Nx))
    El_EW    = np.zeros((Ne, Ny, Nx))
    El_lcrms = np.zeros((Ne, Ny, Nx))
    El_vdins = np.zeros((Ne, Ny, Nx))
    El_lc    = np.zeros((Nl, Ny, Nx))
    
        
    '''
    # Get the dimensions
    Nl, Ny, Nx = c.f_obs.shape
    ll = c.l_obs
    
    # Read the central pixel to find the emission lines fitted
    iy, ix = c.y0, c.x0
    name = 'p%04i-%04i' % (iy, ix)
    filename = path.join(outdir, '%s.hdf5' % name)
        
    with h5py.File(filename, 'r') as f:
        El_lambda = f['elines']['lambda']
        El_name   = f['elines']['line']
        El_l0     = f['elines']['El_l0']
    
    Ne = len(El_lambda)
    
    El_F     = np.zeros((Ne, Ny, Nx))
    El_v0    = np.zeros((Ne, Ny, Nx))
    El_vd    = np.zeros((Ne, Ny, Nx))
    El_EW    = np.zeros((Ne, Ny, Nx))
    El_lcrms = np.zeros((Ne, Ny, Nx))
    El_lc    = np.zeros((Nl, Ny, Nx))
    
    # Reading hdf5 files
    iys, ixs = range(Ny), range(Nx)
    
    for iy in iys:
      for ix in ixs:
    
            if (c.SN_normwin[iy, ix] > 3):
    
                name = 'p%04i-%04i' % (iy, ix)
                filename = path.join(outdir, '%s.hdf5' % name)

                if (path.exists(filename)):
                    
                    print ('Reading pixel ', iy, ix)
                            
                    with h5py.File(filename, 'r') as f:
        
                        El_lc[:, iy, ix] = f['spec']['total_lc']
                        
                        for il, l in enumerate(El_lambda):
                            flag_line = (f['elines']['lambda'] == l)
                            El_F [il, iy, ix]    = f['elines']['El_F'    ][flag_line]
                            El_v0[il, iy, ix]    = f['elines']['El_v0'   ][flag_line]
                            El_vd[il, iy, ix]    = f['elines']['El_vd'   ][flag_line]
                            El_EW[il, iy, ix]    = f['elines']['El_EW'   ][flag_line]
                            El_lcrms[il, iy, ix] = f['elines']['El_lcrms'][flag_line]
                            
    # Save info about each emission line
    aux = { 'lambda': El_lambda,
            'name'  : El_name,
            'l0'    : El_l0,
            'model' : Ne*[model],
            'kinematic_ties_on' : np.array(Ne*[kinTies], dtype='int'),
            'Balmer_dec_limit'  : np.array(Ne*[balLim ], dtype='int'),
            }
        
    El_info = Table(aux)
    print(El_info)
    El_info.convert_unicode_to_bytestring()
    print(El_info)
    c._addTableExtension('El_info', data=El_info, overwrite=True)
    
    # Save fluxes, EWs, etc
    c._addExtension('El_F',     data=El_F,     wcstype='image',   overwrite=True)
    c._addExtension('El_v0',    data=El_v0,    wcstype='image',   overwrite=True)
    c._addExtension('El_vd',    data=El_vd,    wcstype='image',   overwrite=True)
    c._addExtension('El_EW',    data=El_EW,    wcstype='image',   overwrite=True)
    c._addExtension('El_lcrms', data=El_lcrms, wcstype='image',   overwrite=True)
    c._addExtension('El_lc',    data=El_lc,    wcstype='spectra', overwrite=True)
    
    # Save integrated spec info
    name = 'integ'
    filename = path.join(outdir, '%s.hdf5' % name)
        
    with h5py.File(filename, 'r') as f:
        c._addTableExtension('El_integ', data=Table(f['elines'].value), overwrite=True)
        c._addTableExtension('El_integ_lc', data=Table({'l_obs': ll, 'total_lc': f['spec']['total_lc']}), overwrite=True)
    
    c.write( path.join(el_dir, 'manga-%s.dr14.bin2.cA.Ca0c_6x16.El.fits' % galname), overwrite=True )

    '''

    
def save_fits(c, outfile, el_dir, name_template, suffix, kinTies, balLim, model):

    # Get the dimensions
    Nl, Ny, Nx = c.f_obs.shape
    ll = c.l_obs

    # Read the integrated spectra to find the emission lines fitted
    name = suffix + '.' + 'integ'
    filename = path.join(el_dir, '%s.hdf5' % name)
        
    with h5py.File(filename, 'r') as f:
        El_lambda = f['elines']['lambda']
        El_name   = f['elines']['line']
        El_l0     = f['elines']['El_l0']
    
    Ne = len(El_lambda)
    
    El_F     = np.zeros((Ne, Ny, Nx))
    El_v0    = np.zeros((Ne, Ny, Nx))
    El_vd    = np.zeros((Ne, Ny, Nx))
    El_EW    = np.zeros((Ne, Ny, Nx))
    El_lcrms = np.zeros((Ne, Ny, Nx))
    El_vdins = np.zeros((Ne, Ny, Nx))
    El_lc    = np.zeros((Nl, Ny, Nx))
    
    # Reading hdf5 files
    iys, ixs = range(Ny), range(Nx)
    
    for iy in iys:
        for ix in ixs:
    
            if (c.SN_normwin[iy, ix] > 3):
    
                name = suffix + '.' + name_template % (iy, ix)
                filename = path.join(el_dir, '%s.hdf5' % name)

                if (path.exists(filename)):
                    
                    print ('Reading pixel ', iy, ix)
                            
                    with h5py.File(filename, 'r') as f:
        
                        El_lc[:, iy, ix] = f['spec']['total_lc']
                        
                        for il, l in enumerate(El_lambda):
                            flag_line = (f['elines']['lambda'] == l)
                            El_F [il, iy, ix]    = f['elines']['El_F'    ][flag_line]
                            El_v0[il, iy, ix]    = f['elines']['El_v0'   ][flag_line]
                            El_vd[il, iy, ix]    = f['elines']['El_vd'   ][flag_line]
                            El_EW[il, iy, ix]    = f['elines']['El_EW'   ][flag_line]
                            El_vdins[il, iy, ix] = f['elines']['El_vdins'][flag_line]
                            El_lcrms[il, iy, ix] = f['elines']['El_lcrms'][flag_line]
                            
    # Save info about each emission line
    aux = { 'lambda': El_lambda,
            'name'  : El_name,
            'l0'    : El_l0,
            'model' : Ne*[model],
            'kinematic_ties_on' : np.array(Ne*[kinTies], dtype='int'),
            'Balmer_dec_limit'  : np.array(Ne*[balLim ], dtype='int'),
            }
        
    El_info = Table(aux)
    El_info.convert_unicode_to_bytestring()
    c._addTableExtension('El_info', data=El_info, overwrite=True)
    
    # Save fluxes, EWs, etc
    c._addExtension('El_F',     data=El_F,     wcstype='image',   overwrite=True)
    c._addExtension('El_v0',    data=El_v0,    wcstype='image',   overwrite=True)
    c._addExtension('El_vd',    data=El_vd,    wcstype='image',   overwrite=True)
    c._addExtension('El_EW',    data=El_EW,    wcstype='image',   overwrite=True)
    c._addExtension('El_lcrms', data=El_lcrms, wcstype='image',   overwrite=True)
    c._addExtension('El_vdins', data=El_vdins, wcstype='image',   overwrite=True)
    c._addExtension('El_lc',    data=El_lc,    wcstype='spectra', overwrite=True)
    
    # Save integrated spec info
    name = suffix + '.' + 'integ'
    filename = path.join(el_dir, '%s.hdf5' % name)
        
    with h5py.File(filename, 'r') as f:
        c._addTableExtension('El_integ', data=Table(f['elines'].value), overwrite=True)
        c._addTableExtension('El_integ_lc', data=Table({'l_obs': ll, 'total_lc': f['spec']['total_lc']}), overwrite=True)
    
    c.write(outfile, overwrite=True)

