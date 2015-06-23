'''
Created on 15/06/2015

@author: andre
'''


__all__ = ['read_masterlist', 'get_galaxy_id']


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
       ('EL', '|S05')]


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
        return np.squeeze(ml[index][0])
    else:
        return ml


def get_galaxy_id(cube):
    from os.path import basename
    return basename(cube).split('_')[0]

