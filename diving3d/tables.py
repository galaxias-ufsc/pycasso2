'''
Created on 15/06/2015

@author: andre
'''

from .resampling import apply_redshift

__all__ = ['read_masterlist', 'get_wavelength_mask', 'get_galaxy_id',
           'write_starlight_input']


masterlist_dtype=[('id', '|S05'),
                  ('name', '|S12'),
                  ('V_hel', 'float64'),
                  ('morph', '|S05'),
                  ('T', 'float64'),
                  ('R_e', 'float64'),
                  ('M_K', 'float64'),
                  ('n_s', 'float64'),
                  ('epsilon', 'float64'),
                  ('DL', 'float64'),
                  ('eDL', 'float64'),
                  ('EL', '|S05'),
                  ('grating', '|S04'),
                  ('cube', '|S0128'),
                  ('cube_obs', '|S0128'),
                  ]


def read_masterlist(filename, galaxy_id=None):
    '''
    Read the whole masterlist, or a single entry.
    
    Parameters
    ----------
    filename : string
        Path to the file containing the masterlist.
        
    galaxy_id : string, optional
        ID of the masterlist entry, the first column of the table.
        If set, return only the entry pointed by ``galaxy_id'``.
        Default: ``None``
        
    Returns
    -------
    masterlist : recarray
        A numpy record array containing either the whole masterlist
        or the entry pointed by ``galaxy_id``.
    '''
    import numpy as np
    ml = np.genfromtxt(filename, masterlist_dtype, skip_header=2)
    if galaxy_id is not None:
        index = np.where(ml['id'] == galaxy_id)[0]
        if len(index) == 0:
            raise Exception('Entry %s not found in masterlist %s.' % (galaxy_id, filename))
        return np.squeeze(ml[index][0])
    else:
        return ml


def get_galaxy_id(cube):
    '''
    Return the ID of the cube, which can be used to index the masterlist.
    '''
    from os.path import basename
    return basename(cube).split('_')[0]



def get_wavelength_mask(maskfile, wl, z=0.0, dest='rest'):
    '''
    Read a STARLIGHT mask file, optionally applying redshift correction,
    returning a boolean array of masked wavelengths.
    
    Parameters
    ----------
    maskfile : string
        Path to mask file.
        
    wl : array
        Wavelength array.
        
    z : float, optional.
        Redshift. Default: ``0.0`` (no redshift).
        
    dest : string, optional
        Destination frame. Either ``'rest'`` or ``'observed'``.
        Default: ``'rest'``.
    
    Returns
    -------
    wl_mask : array
        Boolean array with same shape as ``wl`` marking the masked
        wavelengths as ``True``.
    '''
    import numpy as np
    import atpy
    import pystarlight.io  # @UnusedImport
    
    t = atpy.Table(maskfile, type='starlight_mask')
    masked_wl = np.zeros(wl.shape, dtype='bool')
    for i in xrange(len(t)):
        l_low, l_upp, line_w, _ = t[i]
        if line_w > 0.0: continue
        if z > 0.0:
            l_low, l_upp = apply_redshift(np.array([l_low, l_upp]), z, dest)
        masked_wl |= (wl > l_low) & (wl < l_upp)
    return masked_wl


def write_starlight_input(wl, flux, err, flags, filename):
    from astropy.io import ascii
    import numpy as np
    
    flags = np.where(flags, 1.0, 0.0)
    if flags is not None and err is not None:
        cols = [wl, flux.data, err.data, flags]
    else:
        cols = [wl, flux]
    ascii.write(cols, filename, Writer=ascii.NoHeader)

