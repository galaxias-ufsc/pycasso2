'''
Save an hdf5 table with all MaNGA spaxels, similar to Daniel's.

Natalia@UFSC - 28/Feb/2018
'''

import sys
from os import path
import glob
from copy import copy

import numpy as np
from astropy.table import Table
import tables

from pycasso2 import FitsCube
from pycasso2 import flags

########################################################################
# Defining dirs

el_dir = '/Users/natalia/data/MaNGA/dr14/starlight_el/dr14.bin2.cA.CB17_7x16/'
    
########################################################################
# Selecting all files

el_files = glob.glob(el_dir + '*.fits')
el_files.sort()

# Get info from first file
c = FitsCube(el_files[0])
El_info = Table(c._getTableExtensionData('El_info')).as_array()
lines = El_info['lambda']
Nl = len(lines)

########################################################################
# Creating hdf5 file

h5file = tables.open_file('dobby_emlines.h5', mode='w', title='MaNGA emission lines fitted with dobby', compression = 'gzip', compression_opts = 4)
group_int = h5file.create_group('/', 'integrated', 'Galaxy integrated spectra information')
group_spx = h5file.create_group('/', 'spaxels', 'Individual spaxel spectra information')

# TO DO - table with galaxy info - see Daniel's
#c._HDUList[0].header['plateifu']

tables_int = {}
tables_spx = {}
tables = { 'El_F'     : 'Emission line flux',
           'El_v0'    : 'Emission line v0',                 
           'El_vd'    : 'Emission line vd',                 
           'El_vdins' : 'Emission line vd instrumental',    
           'El_EW'    : 'Emission line EW',                 
           'El_lcrms' : 'Emission line local continuum rms'
         }

# Start and fill in info table
tables_int['El_info'] = h5file.create_table(group_int, 'El_info',  El_info.dtype, 'Emission line info')
for i in range(len(El_info)):
    rr = tables_int['El_info'].row
    for col in El_info.dtype.names:
        rr[col] = El_info[col][i]
    rr.append()
tables_int['El_info'].flush()

# Start data tables
dt = np.dtype([('l%s' % line, np.float64) for line in lines])
for table, description in tables.items():
    tables_int[table] = h5file.create_table(group_int, table, dt, description)
    tables_spx[table] = h5file.create_table(group_spx, table, dt, description)

    
# Fill in data tables
for el_file in el_files:
    
    print('Reading file ', el_file)
    c = FitsCube(el_file)

    # Integrated
    El_data = Table(c._getTableExtensionData('El_integ')).as_array()
    for table in tables.keys():
        rr = tables_int[table].row
        for line in lines:
            flag_line = (c._getTableExtensionData('El_integ')['lambda'] == line)
            rr['l%s' % line] = El_data[table][flag_line]
        rr.append()
    tables_int[table].flush()
    
    # Flagged spaxels
    flag__z = (~c.synthImageMask).reshape(-1).T

    # Append spaxels
    for table in tables.keys():
        El_data = np.array(c._getSynthExtension(table).astype('float64')).reshape(Nl, -1).T
        tables_spx[table].append(El_data[flag__z])

# Flush
for table in tables.keys():
        tables_spx[table].flush()
    
h5file.close()    



