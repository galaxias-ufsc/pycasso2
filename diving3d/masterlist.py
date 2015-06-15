'''
Created on 15/06/2015

@author: andre
'''


__all__ = ['read_masterlist']


dtype=[('id', '|S05'),
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


def read_masterlist(filename):
    import numpy as np
    return np.genfromtxt(filename, dtype, skip_header=2)
