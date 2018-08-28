'''
Created on Sep 24, 2012

@author: Andre Luiz de Amorim

base2hdf5 reads the base files for STARLIGHT and converts them
to a HDF5 database.

The base files contain metadata for the elements of a base, and
the name of a file for each element containing the SED of the
SSP. These SED files are located at the directory specified by the
--base-dir option.

Each base will be stored in a group given by the base file name, or
by the group specified (see help).

Requirements:

* numpy
* pytables
'''

from os import path
import time
import argparse
from pycasso2.starlight.base import StarlightBase

parser = argparse.ArgumentParser(description='Import STARLIGHT bases to HDF5.')
parser.add_argument('base', type=str, nargs='+',
                    help='Base files. Each base will be imported to a separate group. \
                    To specify a group use group_name=base_file.')
parser.add_argument('--base-dir', dest='baseDir', help='Path to base files.')
parser.add_argument('--output', dest='output', help='Path to output HDF5 file.')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite base in output file.')
args = parser.parse_args()



for base in args.base:
    print(base)
    
    if '=' in base:
        x = base.split('=')
        group = x[0]
        baseFile = x[1]
    else:
        baseFile = base
        group = path.basename(baseFile)

    print('Reading base %s from %s...' % (group, baseFile))
    t0 = time.clock()
    base = StarlightBase(baseFile, args.baseDir)
    print('done')

    base.writeHDF5(args.output, group, overwrite=args.overwrite)
    print('Elapsed time: %.2f seconds' % (time.clock() - t0))
